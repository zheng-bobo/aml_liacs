import logging
import typing

import numpy as np

from vertical_model_evaluator import VerticalModelEvaluator


LOGGER = logging.getLogger(__name__)


class IPL(VerticalModelEvaluator):
    """
    Implements an inverse power-law (IPL) extrapolation strategy to decide
    whether a configuration warrants a full evaluation.
    """

    def __init__(
        self,
        surrogate_model,
        minimal_anchor: int,
        final_anchor: int,
        min_points_for_fit: int = 5,
    ) -> None:
        super().__init__(surrogate_model, minimal_anchor, final_anchor)
        # min_points_for_fit specifies the minimum number of observed points required to fit the IPL (Inverse Power Law) curve.
        # For example, if min_points_for_fit=3, the IPL fit will only start after performance points have been collected
        # on at least three different anchors. This parameter helps prevent premature or inaccurate curve fitting
        # caused by using too few data points.
        self.min_points_for_fit = max(3, min_points_for_fit)

    def evaluate_model(
        self,
        best_so_far: typing.Optional[float],
        configuration: typing.Dict,
    ) -> typing.List[typing.Tuple[int, float]]:
        anchors = self._build_anchor_schedule()
        results: typing.List[typing.Tuple[int, float]] = []

        if best_so_far is None:
            for anchor in anchors:
                results.append((anchor, self._evaluate_anchor(configuration, anchor)))
            return results

        for anchor in anchors:
            performance = self._evaluate_anchor(configuration, anchor)
            results.append((anchor, performance))

            if anchor == self.final_anchor:
                return results

            if len(results) >= self.min_points_for_fit:
                prediction = self._predict_final_performance(results)
                if prediction is not None:
                    LOGGER.debug(
                        "IPL prediction %f at anchor %s (best %.6f)",
                        prediction,
                        anchor,
                        best_so_far,
                    )
                    if prediction >= best_so_far:
                        return results
                    else:
                        # If the IPL prediction indicates that the configuration can potentially outperform the current best,
                        # immediately evaluate at final_anchor and return the results.
                        if all(r[0] != self.final_anchor for r in results):
                            results.append(
                                (
                                    self.final_anchor,
                                    self._evaluate_anchor(
                                        configuration, self.final_anchor
                                    ),
                                )
                            )
                        return results

    def _evaluate_anchor(self, configuration: typing.Dict, anchor: int) -> float:
        theta = dict(configuration)
        theta["anchor_size"] = anchor
        return float(self.surrogate_model.predict(theta))

    def _predict_final_performance(
        self, observations: typing.List[typing.Tuple[int, float]]
    ) -> typing.Optional[float]:
        # 步骤1：将 observation 中的 anchor 和 performance 拆分成数组
        anchors = np.array([a for a, _ in observations], dtype=float)
        performances = np.array([p for _, p in observations], dtype=float)

        # 步骤2：判断 anchor 的数量是否足够做拟合
        if anchors.size < self.min_points_for_fit:
            return None

        # 步骤3：用 performances 里的最小值作为函数的渐近线（asymptote）
        asymptote = performances.min()

        # 步骤4：计算每个 performance 与渐近线的残差
        residual = performances - asymptote

        # 步骤5：由于残差需要取对数，所以加一个小的正数，避免log(0)或负值
        residual += 1e-9

        # 步骤6：对 anchor 和残差都取对数，以便线性拟合
        log_x = np.log(anchors)
        log_residual = np.log(residual)

        # 步骤7：用最小二乘法在 log-log 空间里做线性拟合，得到斜率和截距
        slope, intercept = np.polyfit(log_x, log_residual, 1)

        # 步骤8：根据 IPL 模型推导，指数项是 -slope，系数是 e 的 intercept 次方
        exponent = -slope
        coefficient = np.exp(intercept)

        # 步骤9：按 IPL 曲线模型公式，用预测点 final_anchor 推算最终性能
        predicted = asymptote + coefficient * (self.final_anchor ** (-exponent))

        # 步骤10：返回 float 型的预测结果
        return float(predicted)

    def _build_anchor_schedule(self) -> typing.List[int]:
        # No available_anchors, build a geometric progression
        anchors = []
        anchor = self.minimal_anchor
        while anchor < self.final_anchor:
            anchors.append(anchor)
            anchor = anchor * 2
        if anchors and anchors[-1] != self.final_anchor:
            anchors.append(self.final_anchor)
        elif not anchors:
            anchors = [self.final_anchor]
        return anchors

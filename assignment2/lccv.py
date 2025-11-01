import logging
import typing

from vertical_model_evaluator import VerticalModelEvaluator


LOGGER = logging.getLogger(__name__)


class LCCV(VerticalModelEvaluator):

    @staticmethod
    def optimistic_extrapolation(
        previous_anchor: int,
        previous_performance: float,
        current_anchor: int,
        current_performance: float,
        target_anchor: int,
    ) -> float:
        """
        Does the optimistic performance. Since we are working with a simplified
        surrogate model, we can not measure the infimum and supremum of the
        distribution. Just calculate the slope between the points, and
        extrapolate this.

        :param previous_anchor: See name
        :param previous_performance: Performance at previous anchor
        :param current_anchor: See name
        :param current_performance: Performance at current anchor
        :param target_anchor: the anchor at which we want to have the
        optimistic extrapolation
        :return: The optimistic extrapolation of the performance
        """
        if current_anchor == previous_anchor:
            return current_performance

        slope = (previous_performance - current_performance) / (
            previous_anchor - current_anchor
        )
        optimistic = current_performance + (target_anchor - current_anchor) * slope
        return optimistic

    def evaluate_model(
        self, best_so_far: typing.Optional[float], configuration: typing.Dict
    ) -> typing.List[typing.Tuple[int, float]]:
        """
        Does a staged evaluation of the model, on increasing anchor sizes.
        Determines after the evaluation at every anchor an optimistic
        extrapolation. In case the optimistic extrapolation can not improve
        over the best so far, it stops the evaluation.
        In case the best so far is not determined (None), it evaluates
        immediately on the final anchor (determined by self.final_anchor)

        :param best_so_far: indicates which performance has been obtained so far
        :param configuration: A dictionary indicating the configuration

        :return: A tuple of the evaluations that have been done. Each element of
        the tuple consists of two elements: the anchor size and the estimated
        performance.
        """
        anchors = self._build_anchor_schedule()
        results: typing.List[typing.Tuple[int, float]] = []

        if best_so_far is None:
            config = dict(configuration)
            config["anchor_size"] = self.final_anchor
            performance = float(self.surrogate_model.predict(config))
            results.append((self.final_anchor, performance))
            return results

        previous_anchor = None
        previous_performance = None

        for anchor in anchors:
            config = dict(configuration)
            config["anchor_size"] = anchor
            performance = float(self.surrogate_model.predict(config))
            results.append((anchor, performance))

            if (
                previous_anchor is not None
                and best_so_far is not None
                and anchor != self.final_anchor
            ):
                optimistic = self.optimistic_extrapolation(
                    previous_anchor,
                    previous_performance,
                    anchor,
                    performance,
                    self.final_anchor,
                )
                LOGGER.debug(
                    "Optimistic extrapolation at anchor %s towards %s: %.6f (best %.6f)",
                    anchor,
                    self.final_anchor,
                    optimistic,
                    best_so_far,
                )
                # The score in the dataset is a loss value; lower is better
                if optimistic >= best_so_far:
                    break

            previous_anchor = anchor
            previous_performance = performance

        return results

    def _build_anchor_schedule(self) -> typing.List[int]:
        # Build a schedule as a geometric progression from minimal_anchor to final_anchor (inclusive)
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

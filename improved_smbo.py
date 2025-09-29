import numpy as np
import pandas as pd
import typing
import random

from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm
import ConfigSpace


class ImprovedSequentialModelBasedOptimization:
    """
    改进的SMBO实现，包含混合策略（贝叶斯优化 + 随机采样）
    """

    def __init__(self, config_space, random_ratio=0.3):
        """
        初始化改进的SMBO

        :param config_space: 配置空间
        :param random_ratio: 随机采样的比例（0.3表示30%随机，70%贝叶斯）
        """
        self.config_space = config_space
        self.runs = []
        self.theta_inc = None
        self.theta_inc_performance = None
        self.scaler = StandardScaler()
        self.model = GaussianProcessRegressor(random_state=42)
        self.random_ratio = random_ratio
        self.iteration = 0

    def initialize(
        self, capital_phi: typing.List[typing.Tuple[typing.Dict, float]]
    ) -> None:
        """初始化SMBO"""
        self.runs = list(capital_phi)

        if len(self.runs) > 0:
            best_idx = np.argmin([run[1] for run in self.runs])
            best_run = self.runs[best_idx]
            self.theta_inc = best_run[0]
            self.theta_inc_performance = best_run[1]
        else:
            self.theta_inc = None
            self.theta_inc_performance = None

    def fit_model(self) -> None:
        """训练代理模型"""
        if len(self.runs) < 2:
            return

        configs = [run[0] for run in self.runs]
        performances = [run[1] for run in self.runs]

        X = self._configs_to_matrix(configs)
        y = np.array(performances)

        self.model.fit(X, y)

    def select_configuration(self):
        """选择下一个配置（混合策略）"""
        self.iteration += 1

        # 混合策略：根据比例决定使用贝叶斯优化还是随机采样
        if random.random() < self.random_ratio:
            # 随机采样
            print(f"迭代 {self.iteration}: 使用随机采样")
            return self.config_space.sample_configuration()
        else:
            # 贝叶斯优化
            print(f"迭代 {self.iteration}: 使用贝叶斯优化")
            return self._bayesian_selection()

    def _bayesian_selection(self):
        """贝叶斯优化选择配置"""
        if len(self.runs) < 2:
            return self.config_space.sample_configuration()

        # 生成候选配置
        candidates = [self.config_space.sample_configuration() for _ in range(100)]
        X_candidates = self._configs_to_matrix(candidates)

        # 计算期望改进
        ei_values = self.expected_improvement(
            model_pipeline=self.model,
            f_star=self.theta_inc_performance,
            theta=X_candidates,
        )

        # 选择EI最大的配置
        best_idx = np.argmax(ei_values)
        return candidates[best_idx]

    def _configs_to_matrix(self, configs: list) -> np.array:
        """将配置转换为数值矩阵"""
        configs_df = pd.DataFrame([dict(config) for config in configs])
        configs_encoded = pd.get_dummies(configs_df)

        if hasattr(self.scaler, "mean_"):
            X = self.scaler.transform(configs_encoded)
        else:
            X = self.scaler.fit_transform(configs_encoded)

        return X

    @staticmethod
    def expected_improvement(
        model_pipeline, f_star: float, theta: np.array
    ) -> np.array:
        """计算期望改进"""
        mu, sigma = model_pipeline.predict(theta, return_std=True)

        improvement = f_star - mu
        sigma = np.maximum(sigma, 1e-9)
        z = improvement / sigma

        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.maximum(ei, 0.0)

        return ei

    def update_runs(self, run: typing.Tuple[typing.Dict, float]):
        """更新运行记录"""
        self.runs.append(run)

        if self.theta_inc_performance is None or run[1] < self.theta_inc_performance:
            self.theta_inc = run[0]
            self.theta_inc_performance = run[1]

        # 重新训练模型
        self.fit_model()

    def get_surrogate_predictions(self, n_samples=100):
        """获取代理模型的预测（用于可视化）"""
        if len(self.runs) < 2:
            return None, None, None

        # 生成测试配置
        test_configs = [
            self.config_space.sample_configuration() for _ in range(n_samples)
        ]
        X_test = self._configs_to_matrix(test_configs)

        # 预测
        mu, sigma = self.model.predict(X_test, return_std=True)

        return test_configs, mu, sigma

    def analyze_surrogate_quality(self):
        """分析代理模型质量"""
        if len(self.runs) < 5:
            return "数据不足，无法分析代理模型质量"

        # 计算训练误差
        configs = [run[0] for run in self.runs]
        performances = [run[1] for run in self.runs]
        X = self._configs_to_matrix(configs)

        # 预测训练数据
        mu, sigma = self.model.predict(X, return_std=True)

        # 计算误差
        mse = np.mean((performances - mu) ** 2)
        mae = np.mean(np.abs(performances - mu))

        return {
            "mse": mse,
            "mae": mae,
            "mean_uncertainty": np.mean(sigma),
            "n_training_points": len(self.runs),
        }

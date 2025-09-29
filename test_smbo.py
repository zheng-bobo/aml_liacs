import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import ConfigSpace as CS
from ConfigSpace import (
    ConfigurationSpace,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import importlib.util
import sys

# 动态导入 smbo copy.py
spec = importlib.util.spec_from_file_location("smbo_copy", "./automl-a1/smbo copy.py")
smbo_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smbo_module)
SequentialModelBasedOptimization = smbo_module.SequentialModelBasedOptimization


def objective_function(config):
    """
    目标函数：基于配置训练随机森林并返回交叉验证准确率
    """
    # 创建数据集
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42
    )

    # 从配置中提取超参数
    n_estimators = int(config["n_estimators"])
    max_depth = int(config["max_depth"]) if config["max_depth"] != "None" else None
    min_samples_split = int(config["min_samples_split"])
    min_samples_leaf = int(config["min_samples_leaf"])

    # 创建模型
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )

    # 交叉验证
    scores = cross_val_score(model, X, y, cv=3, scoring="accuracy")
    return 1 - scores.mean()  # 返回错误率（最小化问题）


def create_config_space():
    """创建配置空间"""
    cs = ConfigurationSpace()

    # 添加超参数
    cs.add_hyperparameter(UniformIntegerHyperparameter("n_estimators", 10, 200))
    cs.add_hyperparameter(UniformIntegerHyperparameter("max_depth", 3, 20))
    cs.add_hyperparameter(UniformIntegerHyperparameter("min_samples_split", 2, 20))
    cs.add_hyperparameter(UniformIntegerHyperparameter("min_samples_leaf", 1, 10))

    return cs


def run_smbo_with_visualization():
    """运行SMBO并可视化结果"""
    print("开始SMBO优化...")

    # 创建配置空间
    config_space = create_config_space()

    # 初始化SMBO
    smbo = SequentialModelBasedOptimization(config_space)

    # 生成初始配置（随机采样）
    print("生成初始配置...")
    initial_configs = []
    initial_performances = []

    for i in range(5):  # 5个初始点
        config = config_space.sample_configuration()
        performance = objective_function(config)
        initial_configs.append((config, performance))
        initial_performances.append(performance)
        print(f"初始配置 {i+1}: 错误率 = {performance:.4f}")

    # 初始化SMBO
    smbo.initialize(initial_configs)

    # 存储优化历史
    optimization_history = {
        "iterations": [],
        "best_performance": [],
        "current_performance": [],
        "configurations": [],
    }

    # 运行优化
    n_iterations = 20
    print(f"\n开始 {n_iterations} 轮优化...")

    for iteration in range(n_iterations):
        print(f"\n--- 迭代 {iteration + 1} ---")

        # 选择下一个配置
        next_config = smbo.select_configuration()

        # 评估配置
        performance = objective_function(next_config)

        # 更新SMBO
        smbo.update_runs((next_config, performance))

        # 记录历史
        optimization_history["iterations"].append(iteration + 1)
        optimization_history["best_performance"].append(smbo.theta_inc_performance)
        optimization_history["current_performance"].append(performance)
        optimization_history["configurations"].append(next_config)

        print(f"选择配置: {dict(next_config)}")
        print(f"性能: {performance:.4f}")
        print(f"当前最优: {smbo.theta_inc_performance:.4f}")

    # 可视化结果
    plot_optimization_results(optimization_history, smbo)

    return smbo, optimization_history


def plot_optimization_results(history, smbo):
    """绘制优化结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 优化曲线
    axes[0, 0].plot(
        history["iterations"],
        history["best_performance"],
        "b-",
        label="Best Performance",
        linewidth=2,
    )
    axes[0, 0].plot(
        history["iterations"],
        history["current_performance"],
        "r--",
        label="Current Performance",
        alpha=0.7,
    )
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Error Rate")
    axes[0, 0].set_title("SMBO Optimization Progress")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 性能分布
    axes[0, 1].hist(
        history["current_performance"],
        bins=10,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )
    axes[0, 1].axvline(
        smbo.theta_inc_performance,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Best: {smbo.theta_inc_performance:.4f}",
    )
    axes[0, 1].set_xlabel("Error Rate")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Performance Distribution")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 超参数重要性（简单分析）
    if len(history["configurations"]) > 0:
        # 提取n_estimators和max_depth
        n_estimators_values = [
            dict(config)["n_estimators"] for config in history["configurations"]
        ]
        max_depth_values = [
            dict(config)["max_depth"] for config in history["configurations"]
        ]

        axes[1, 0].scatter(
            n_estimators_values,
            max_depth_values,
            c=history["current_performance"],
            cmap="viridis",
            alpha=0.7,
        )
        axes[1, 0].set_xlabel("n_estimators")
        axes[1, 0].set_ylabel("max_depth")
        axes[1, 0].set_title("Hyperparameter Space Exploration")
        axes[1, 0].grid(True, alpha=0.3)

        # 添加颜色条
        scatter = axes[1, 0].scatter(
            n_estimators_values,
            max_depth_values,
            c=history["current_performance"],
            cmap="viridis",
            alpha=0.7,
        )
        plt.colorbar(scatter, ax=axes[1, 0], label="Error Rate")

    # 4. 代理模型预测（如果有足够数据）
    if len(smbo.runs) >= 5:
        try:
            # 生成测试配置
            test_configs = [smbo.config_space.sample_configuration() for _ in range(50)]
            X_test = smbo._configs_to_matrix(test_configs)

            # 预测
            mu, sigma = smbo.model.predict(X_test, return_std=True)

            axes[1, 1].scatter(range(len(mu)), mu, alpha=0.6, label="Predicted Mean")
            axes[1, 1].fill_between(
                range(len(mu)), mu - sigma, mu + sigma, alpha=0.3, label="Uncertainty"
            )
            axes[1, 1].set_xlabel("Configuration Index")
            axes[1, 1].set_ylabel("Predicted Performance")
            axes[1, 1].set_title("Surrogate Model Predictions")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        except Exception as e:
            axes[1, 1].text(
                0.5,
                0.5,
                f"Surrogate Model Error:\n{str(e)}",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )
            axes[1, 1].set_title("Surrogate Model Status")

    plt.tight_layout()
    plt.savefig(
        "/Users/zhengxuzhang/Code/zhengxz/automl-a1/smbo_optimization_results.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # 打印最终结果
    print(f"\n=== 优化完成 ===")
    print(f"最优配置: {dict(smbo.theta_inc)}")
    print(f"最优性能: {smbo.theta_inc_performance:.4f}")
    print(f"总评估次数: {len(smbo.runs)}")


if __name__ == "__main__":
    try:
        smbo, history = run_smbo_with_visualization()
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback

        traceback.print_exc()

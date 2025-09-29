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
import importlib.util

# 导入原始SMBO
spec_original = importlib.util.spec_from_file_location(
    "smbo_copy", "/Users/zhengxuzhang/Code/zhengxz/automl-a1/smbo copy.py"
)
smbo_module = importlib.util.module_from_spec(spec_original)
spec_original.loader.exec_module(smbo_module)
OriginalSMBO = smbo_module.SequentialModelBasedOptimization

# 导入改进的SMBO
from improved_smbo import ImprovedSequentialModelBasedOptimization


def objective_function(config):
    """目标函数"""
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42
    )

    n_estimators = int(config["n_estimators"])
    max_depth = int(config["max_depth"]) if config["max_depth"] != "None" else None
    min_samples_split = int(config["min_samples_split"])
    min_samples_leaf = int(config["min_samples_leaf"])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )

    scores = cross_val_score(model, X, y, cv=3, scoring="accuracy")
    return 1 - scores.mean()


def create_config_space():
    """创建配置空间"""
    cs = ConfigurationSpace()
    cs.add(UniformIntegerHyperparameter("n_estimators", 10, 200))
    cs.add(UniformIntegerHyperparameter("max_depth", 3, 20))
    cs.add(UniformIntegerHyperparameter("min_samples_split", 2, 20))
    cs.add(UniformIntegerHyperparameter("min_samples_leaf", 1, 10))
    return cs


def run_optimization(smbo_class, name, n_iterations=30):
    """运行优化并返回结果"""
    print(f"\n=== 运行 {name} ===")

    config_space = create_config_space()
    smbo = smbo_class(config_space)

    # 生成初始配置
    initial_configs = []
    for i in range(5):
        config = config_space.sample_configuration()
        performance = objective_function(config)
        initial_configs.append((config, performance))

    smbo.initialize(initial_configs)

    # 存储结果
    results = {
        "iterations": [],
        "best_performance": [],
        "current_performance": [],
        "method_used": [],
    }

    # 运行优化
    for iteration in range(n_iterations):
        next_config = smbo.select_configuration()
        performance = objective_function(next_config)
        smbo.update_runs((next_config, performance))

        results["iterations"].append(iteration + 1)
        results["best_performance"].append(smbo.theta_inc_performance)
        results["current_performance"].append(performance)

        # 记录使用的方法（仅对改进版本）
        if hasattr(smbo, "iteration"):
            if smbo.iteration <= n_iterations:
                results["method_used"].append(
                    "Bayesian" if random.random() >= smbo.random_ratio else "Random"
                )
            else:
                results["method_used"].append("Unknown")
        else:
            results["method_used"].append("Bayesian")

        print(
            f"迭代 {iteration + 1}: 性能 = {performance:.4f}, 最优 = {smbo.theta_inc_performance:.4f}"
        )

    return smbo, results


def plot_comparison(original_results, improved_results):
    """绘制对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. 优化曲线对比
    axes[0, 0].plot(
        original_results["iterations"],
        original_results["best_performance"],
        "b-",
        label="Original SMBO",
        linewidth=2,
    )
    axes[0, 0].plot(
        improved_results["iterations"],
        improved_results["best_performance"],
        "r-",
        label="Improved SMBO (Mixed)",
        linewidth=2,
    )
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Best Performance")
    axes[0, 0].set_title("Optimization Progress Comparison")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 当前性能对比
    axes[0, 1].plot(
        original_results["iterations"],
        original_results["current_performance"],
        "b--",
        label="Original SMBO",
        alpha=0.7,
    )
    axes[0, 1].plot(
        improved_results["iterations"],
        improved_results["current_performance"],
        "r--",
        label="Improved SMBO",
        alpha=0.7,
    )
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Current Performance")
    axes[0, 1].set_title("Current Performance Comparison")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 性能分布对比
    axes[0, 2].hist(
        original_results["current_performance"],
        bins=15,
        alpha=0.5,
        label="Original SMBO",
        color="blue",
    )
    axes[0, 2].hist(
        improved_results["current_performance"],
        bins=15,
        alpha=0.5,
        label="Improved SMBO",
        color="red",
    )
    axes[0, 2].set_xlabel("Performance")
    axes[0, 2].set_ylabel("Frequency")
    axes[0, 2].set_title("Performance Distribution")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. 方法使用统计（仅改进版本）
    if "method_used" in improved_results:
        method_counts = {}
        for method in improved_results["method_used"]:
            method_counts[method] = method_counts.get(method, 0) + 1

        axes[1, 0].pie(
            method_counts.values(), labels=method_counts.keys(), autopct="%1.1f%%"
        )
        axes[1, 0].set_title("Method Usage Distribution (Improved SMBO)")

    # 5. 收敛速度对比
    original_convergence = []
    improved_convergence = []

    for i in range(1, len(original_results["best_performance"])):
        original_convergence.append(
            original_results["best_performance"][i]
            - original_results["best_performance"][i - 1]
        )
        improved_convergence.append(
            improved_results["best_performance"][i]
            - improved_results["best_performance"][i - 1]
        )

    axes[1, 1].plot(
        range(1, len(original_convergence) + 1),
        original_convergence,
        "b-",
        label="Original",
    )
    axes[1, 1].plot(
        range(1, len(improved_convergence) + 1),
        improved_convergence,
        "r-",
        label="Improved",
    )
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Improvement")
    axes[1, 1].set_title("Convergence Speed")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. 最终结果对比
    final_original = original_results["best_performance"][-1]
    final_improved = improved_results["best_performance"][-1]

    methods = ["Original SMBO", "Improved SMBO"]
    performances = [final_original, final_improved]
    colors = ["blue", "red"]

    bars = axes[1, 2].bar(methods, performances, color=colors, alpha=0.7)
    axes[1, 2].set_ylabel("Final Best Performance")
    axes[1, 2].set_title("Final Performance Comparison")
    axes[1, 2].grid(True, alpha=0.3)

    # 添加数值标签
    for bar, perf in zip(bars, performances):
        axes[1, 2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{perf:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(
        "/Users/zhengxuzhang/Code/zhengxz/automl-a1/smbo_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def main():
    """主函数"""
    print("开始SMBO对比实验...")

    # 运行原始SMBO
    original_smbo, original_results = run_optimization(
        OriginalSMBO, "原始SMBO", n_iterations=25
    )

    # 运行改进的SMBO
    improved_smbo, improved_results = run_optimization(
        ImprovedSequentialModelBasedOptimization, "改进SMBO", n_iterations=25
    )

    # 分析代理模型质量
    print("\n=== 代理模型质量分析 ===")
    original_quality = (
        original_smbo.analyze_surrogate_quality()
        if hasattr(original_smbo, "analyze_surrogate_quality")
        else "无分析功能"
    )
    improved_quality = improved_smbo.analyze_surrogate_quality()

    print(f"原始SMBO: {original_quality}")
    print(f"改进SMBO: {improved_quality}")

    # 绘制对比图
    plot_comparison(original_results, improved_results)

    # 打印最终结果
    print(f"\n=== 最终结果对比 ===")
    print(f"原始SMBO最优性能: {original_smbo.theta_inc_performance:.4f}")
    print(f"改进SMBO最优性能: {improved_smbo.theta_inc_performance:.4f}")
    print(
        f"改进: {((original_smbo.theta_inc_performance - improved_smbo.theta_inc_performance) / original_smbo.theta_inc_performance * 100):.2f}%"
    )


if __name__ == "__main__":
    import random

    random.seed(42)
    np.random.seed(42)
    main()

import argparse
import ConfigSpace
import matplotlib.pyplot as plt
import pandas as pd
import os
from random_search import RandomSearch
from surrogate_model import SurrogateModel
from smbo import SequentialModelBasedOptimization
from successive_halving import SuccessiveHalving


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_space_file", type=str, default="lcdb_config_space_knn.json"
    )
    parser.add_argument(
        "--configurations_performance_file",
        type=str,
        default="./config-performances/config_performances_dataset-1457.csv",
    )

    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument("--max_anchor_size", type=int, default=1600)
    parser.add_argument("--num_iterations", type=int, default=500)

    return parser.parse_args()


def run(args):
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    random_search = RandomSearch(config_space)
    df = pd.read_csv(args.configurations_performance_file)
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df)
    results = {"random_search": [1.0]}

    for idx in range(args.num_iterations):
        theta_new = dict(random_search.select_configuration())
        theta_new["anchor_size"] = args.max_anchor_size
        performance = surrogate_model.predict(theta_new)
        # ensure to only record improvements
        results["random_search"].append(min(results["random_search"][-1], performance))
        random_search.update_runs((theta_new, performance))

    plt.plot(range(len(results["random_search"])), results["random_search"])
    plt.yscale("log")
    plt.show()


def runSMBO(args):
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)

    random_search = RandomSearch(config_space)
    smbo = SequentialModelBasedOptimization(config_space)
    smbo_random = SequentialModelBasedOptimization(config_space, random_ratio=0.3)

    df = pd.read_csv(args.configurations_performance_file)
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df)
    results = {"random_search": [1.0], "smbo": [1.0], "smbo_random": [1.0]}

    # Initialize SMBO with some initial configurations
    initial_configs = []
    for _ in range(5):  # Use 5 initial random configurations
        config = dict(random_search.select_configuration())
        config["anchor_size"] = args.max_anchor_size
        performance = surrogate_model.predict(config)
        initial_configs.append((config, performance))
    smbo.initialize(initial_configs)
    smbo_random.initialize(initial_configs)

    for idx in range(args.num_iterations):
        # Random Search
        theta_new_random = dict(random_search.select_configuration())
        theta_new_random["anchor_size"] = args.max_anchor_size
        performance_random = surrogate_model.predict(theta_new_random)
        results["random_search"].append(
            min(results["random_search"][-1], performance_random)
        )
        random_search.update_runs((theta_new_random, performance_random))

        # SMBO
        theta_new_smbo = dict(smbo.select_configuration())
        theta_new_smbo["anchor_size"] = args.max_anchor_size
        performance_smbo = surrogate_model.predict(theta_new_smbo)
        results["smbo"].append(min(results["smbo"][-1], performance_smbo))
        smbo.update_runs((theta_new_smbo, performance_smbo))

        # SMBO with 30% random sampling
        theta_smbo_random = dict(smbo_random.select_configuration())
        theta_smbo_random["anchor_size"] = args.max_anchor_size
        performance_smbo_random = surrogate_model.predict(theta_smbo_random)
        results["smbo_random"].append(
            min(results["smbo_random"][-1], performance_smbo_random)
        )
        smbo_random.update_runs((theta_smbo_random, performance_smbo_random))

    # Plot all results
    plt.figure(figsize=(12, 8))
    plt.plot(
        range(len(results["random_search"])),
        results["random_search"],
        label="RandomSearch",
        marker="o",
        linewidth=2,
    )
    plt.plot(
        range(len(results["smbo"])),
        results["smbo"],
        label="SMBO",
        marker="s",
        linewidth=2,
    )
    plt.plot(
        range(len(results["smbo_random"])),
        results["smbo_random"],
        label="SMBO (30% Random)",
        marker="^",
        linewidth=2,
    )
    plt.yscale("log")
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Performance", fontsize=12)
    plt.title("Comparison of Random Search vs SMBO vs SMBO (30% Random)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def runSuccessiveHalving(args):
    """Run the Successive Halving algorithm"""

    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)

    # Run Successive Halving on multiple datasets
    datasets = [6, 11, 1457]

    for ds_id in datasets:
        print(f"\n{'='*60}")
        print(f"Running Successive Halving on Dataset {ds_id}")
        print(f"{'='*60}")

        sh = SuccessiveHalving(
            config_space=config_space,
            dataset_id=ds_id,
            min_budget=16,
            max_budget=args.max_anchor_size,
            eta=2,
        )

        results = sh.run(n_initial_configs=50, max_rounds=8)
        sh.print_summary(results)
        sh.plot_config_scores(results)


def compare_smbo_vs_successive_halving(args):
    """
    Compare SMBO vs Successive Halving vs Random Search with fair budget allocation.

    Key insight: SMBO and Random Search evaluate at max_budget each time,
    while Successive Halving evaluates at different budget levels.
    Fair comparison requires equal TOTAL budget consumption:
        - SMBO/Random: max_budget * num_iterations
        - Successive Halving: sum of all budgets used across all evaluations
    """

    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)

    # Extract dataset_id from the configuration performance file's name
    filename = args.configurations_performance_file
    base = os.path.basename(filename)
    dataset_id_str = base.split("-")[-1].split(".")[0]
    dataset_id = int(dataset_id_str) if dataset_id_str.isdigit() else 1457

    print(f"\n{'='*80}")
    print(f"Fair Comparison: SMBO vs Successive Halving vs Random Search")
    print(f"Dataset {dataset_id} - Based on Total Budget Consumption")
    print(f"{'='*80}")

    # Load the surrogate model for performance evaluation
    df = pd.read_csv(args.configurations_performance_file)
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df)

    # ========== 1. Run Successive Halving and calculate total budget ==========
    print(f"\n[1/3] Running Successive Halving...")
    sh = SuccessiveHalving(
        config_space=config_space,
        dataset_id=dataset_id,
        min_budget=16,
        max_budget=args.max_anchor_size,
        eta=2,
    )
    sh_results = sh.run(n_initial_configs=50, max_rounds=8)

    # Calculate total budget consumed by Successive Halving
    # Each evaluation uses a specific budget (anchor_size)
    total_sh_budget = 0
    sh_budget_trajectory = [0]  # Cumulative budget
    sh_performance_trajectory = [1.0]  # Best performance so far
    current_best = 1.0

    for round_info in sh_results["round_results"]:
        budget_level = round_info["budget"]
        for config, perf in round_info["results"]:
            total_sh_budget += budget_level
            sh_budget_trajectory.append(total_sh_budget)
            current_best = min(current_best, perf)
            sh_performance_trajectory.append(current_best)

    print(f"Successive Halving finished:")
    print(f"  - Total evaluations: {sh_results['total_evaluations']}")
    print(f"  - Total budget consumed: {total_sh_budget}")
    print(f"  - Best performance: {sh_results['best_performance']:.6f}")

    # ========== 2. Run SMBO with same total budget ==========
    # For SMBO: each evaluation costs max_budget
    # So num_iterations = total_sh_budget / max_budget
    max_budget = args.max_anchor_size
    smbo_iterations = int(total_sh_budget / max_budget)

    print(f"\n[2/3] Running SMBO with same total budget...")
    print(f"  - Budget per iteration: {max_budget}")
    print(f"  - Number of iterations: {smbo_iterations}")
    print(f"  - Total budget: {smbo_iterations * max_budget}")

    random_search = RandomSearch(config_space)
    smbo = SequentialModelBasedOptimization(config_space)

    # Initialize SMBO with some random configurations
    initial_configs = []
    num_init = min(5, smbo_iterations)
    for _ in range(num_init):
        config = dict(random_search.select_configuration())
        config["anchor_size"] = max_budget
        performance = surrogate_model.predict(config)
        initial_configs.append((config, performance))
    smbo.initialize(initial_configs)

    # Track SMBO performance trajectory with cumulative budget
    smbo_budget_trajectory = [0]
    smbo_performance_trajectory = [1.0]
    current_best = 1.0
    current_budget = 0

    for config, perf in initial_configs:
        current_budget += max_budget
        smbo_budget_trajectory.append(current_budget)
        current_best = min(current_best, perf)
        smbo_performance_trajectory.append(current_best)

    # Run remaining SMBO iterations
    remaining_iterations = smbo_iterations - num_init
    for idx in range(remaining_iterations):
        theta_new = dict(smbo.select_configuration())
        theta_new["anchor_size"] = max_budget
        performance = surrogate_model.predict(theta_new)

        current_budget += max_budget
        smbo_budget_trajectory.append(current_budget)
        current_best = min(current_best, performance)
        smbo_performance_trajectory.append(current_best)

        smbo.update_runs((theta_new, performance))

    print(f"SMBO finished:")
    print(f"  - Total evaluations: {smbo_iterations}")
    print(f"  - Total budget consumed: {current_budget}")
    print(f"  - Best performance: {smbo_performance_trajectory[-1]:.6f}")

    # ========== 3. Run Random Search with same total budget ==========
    print(f"\n[3/3] Running Random Search (baseline) with same total budget...")
    print(f"  - Budget per iteration: {max_budget}")
    print(f"  - Number of iterations: {smbo_iterations}")
    print(f"  - Total budget: {smbo_iterations * max_budget}")

    random_search_baseline = RandomSearch(config_space)
    random_budget_trajectory = [0]
    random_performance_trajectory = [1.0]
    current_best = 1.0
    current_budget = 0

    for idx in range(smbo_iterations):
        theta_new = dict(random_search_baseline.select_configuration())
        theta_new["anchor_size"] = max_budget
        performance = surrogate_model.predict(theta_new)

        current_budget += max_budget
        random_budget_trajectory.append(current_budget)
        current_best = min(current_best, performance)
        random_performance_trajectory.append(current_best)

        random_search_baseline.update_runs((theta_new, performance))

    print(f"Random Search finished:")
    print(f"  - Total evaluations: {smbo_iterations}")
    print(f"  - Total budget consumed: {current_budget}")
    print(f"  - Best performance: {random_performance_trajectory[-1]:.6f}")

    # ========== Plot Comparison ==========

    # Create single figure for Performance vs Total Budget
    plt.figure(figsize=(12, 7))

    plt.plot(
        random_budget_trajectory,
        random_performance_trajectory,
        label="Random Search",
        marker="o",
        linewidth=2.5,
        alpha=0.7,
        markersize=5,
        markevery=max(1, len(random_budget_trajectory) // 20),
    )

    plt.plot(
        smbo_budget_trajectory,
        smbo_performance_trajectory,
        label="SMBO",
        marker="s",
        linewidth=2.5,
        alpha=0.7,
        markersize=5,
        markevery=max(1, len(smbo_budget_trajectory) // 20),
    )

    plt.plot(
        sh_budget_trajectory,
        sh_performance_trajectory,
        label="Successive Halving",
        marker="^",
        linewidth=2.5,
        alpha=0.7,
        markersize=5,
        markevery=max(1, len(sh_budget_trajectory) // 20),
    )

    plt.yscale("log")
    plt.xlabel("Total Budget Consumed", fontsize=14, fontweight="bold")
    plt.ylabel("Best Performance", fontsize=14, fontweight="bold")
    plt.title(
        f"Performance vs Total Budget (Dataset {dataset_id})",
        fontsize=16,
        fontweight="bold",
    )
    plt.legend(fontsize=12, loc="best")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.show()

    # ========== Print Summary ==========
    print(f"Dataset: {dataset_id}")
    print(f"Max Budget per Evaluation (SMBO/Random): {max_budget}")
    print(f"\nTotal Budget Consumed:")
    print(f"  Random Search       : {random_budget_trajectory[-1]}")
    print(f"  SMBO                : {smbo_budget_trajectory[-1]}")
    print(f"  Successive Halving  : {sh_budget_trajectory[-1]}")

    print(f"\nNumber of Evaluations:")
    print(f"  Random Search       : {len(random_performance_trajectory)-1}")
    print(f"  SMBO                : {len(smbo_performance_trajectory)-1}")
    print(f"  Successive Halving  : {len(sh_performance_trajectory)-1}")

    print(f"\nFinal Performance:")
    print(f"  Random Search       : {random_performance_trajectory[-1]:.6f}")
    print(f"  SMBO                : {smbo_performance_trajectory[-1]:.6f}")
    print(f"  Successive Halving  : {sh_performance_trajectory[-1]:.6f}")

    # Compute relative improvements
    baseline = random_performance_trajectory[-1]
    smbo_improvement = (baseline - smbo_performance_trajectory[-1]) / baseline * 100
    sh_improvement = (baseline - sh_performance_trajectory[-1]) / baseline * 100

    print(f"\nImprovement over Random Search:")
    print(f"  SMBO                : {smbo_improvement:+.2f}%")
    print(f"  Successive Halving  : {sh_improvement:+.2f}%")

    # Compare SMBO vs Successive Halving
    if smbo_performance_trajectory[-1] < sh_performance_trajectory[-1]:
        winner = "SMBO"
        improvement = (
            (sh_performance_trajectory[-1] - smbo_performance_trajectory[-1])
            / sh_performance_trajectory[-1]
            * 100
        )
        print(
            f"\n Best Algorithm: {winner} ({improvement:.2f}% better than Successive Halving)"
        )
    elif sh_performance_trajectory[-1] < smbo_performance_trajectory[-1]:
        winner = "Successive Halving"
        improvement = (
            (smbo_performance_trajectory[-1] - sh_performance_trajectory[-1])
            / smbo_performance_trajectory[-1]
            * 100
        )
        print(f"\n Best Algorithm: {winner} ({improvement:.2f}% better than SMBO)")
    else:
        print(f"\n The two algorithms perform equally well")

    return {
        "random_search": {
            "budget_trajectory": random_budget_trajectory,
            "performance_trajectory": random_performance_trajectory,
        },
        "smbo": {
            "budget_trajectory": smbo_budget_trajectory,
            "performance_trajectory": smbo_performance_trajectory,
        },
        "successive_halving": {
            "budget_trajectory": sh_budget_trajectory,
            "performance_trajectory": sh_performance_trajectory,
        },
        "dataset_id": dataset_id,
    }


if __name__ == "__main__":
    args = parse_args()
    run(args)
    runSMBO(args)
    runSuccessiveHalving(args)
    compare_smbo_vs_successive_halving(args)

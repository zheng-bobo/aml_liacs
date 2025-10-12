import argparse
import ConfigSpace
import matplotlib.pyplot as plt
import pandas as pd
from random_search import RandomSearch
from surrogate_model import SurrogateModel
from smbo import SequentialModelBasedOptimization


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


def runs(args):
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


if __name__ == "__main__":
    args = parse_args()
    # run(args)
    runs(args)

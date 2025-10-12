import argparse
import ConfigSpace
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from surrogate_model import SurrogateModel


class SuccessiveHalving:
    """
    Successive Halving implementation for hyperparameter optimization.

    Successive Halving is a resource allocation method that evaluates several configurations
    with limited resources and progressively allocates more resources to the top-performing
    model configurations.
    """

    def __init__(self, config_space, dataset_id, min_budget=16, max_budget=1024, eta=2):
        """
        Initialize Successive Halving optimizer.

        Args:
            config_space: ConfigurationSpace object defining the hyperparameter space
            dataset_id: ID of the dataset to optimize for
            min_budget: Minimum budget (anchor size) to start with
            max_budget: Maximum budget (anchor size) to use
            eta: Reduction factor (how many configurations to keep at each round)
        """
        self.config_space = config_space
        self.dataset_id = dataset_id
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.eta = eta

        # Load performance data for this dataset
        self.performance_data = self._load_performance_data()

        # Initialize SurrogateModel for this dataset
        self.surrogate_model = SurrogateModel(config_space)
        self._initialize_surrogate_model()

        # Successive Halving state
        self.current_configs = []
        self.current_budget = min_budget
        self.round = 0
        self.results = []

    def _load_performance_data(self):
        """Load performance data for the specific dataset."""
        data_file = (
            f"config-performances/config_performances_dataset-{self.dataset_id}.csv"
        )
        if not Path(data_file).exists():
            raise FileNotFoundError(f"Performance data file not found: {data_file}")

        df = pd.read_csv(data_file)
        print(
            f"Loaded performance data for dataset {self.dataset_id}: {len(df)} configurations"
        )
        return df

    def _initialize_surrogate_model(self):
        """Initialize SurrogateModel with performance data."""
        print(f"Initializing SurrogateModel for dataset {self.dataset_id}...")

        # Use a subset of data for training the surrogate model
        subset_size = min(200, len(self.performance_data))
        training_data = self.performance_data.head(subset_size).copy()

        # Fit the surrogate model with the training data
        self.surrogate_model.fit(training_data)
        print("SurrogateModel initialized successfully")

    def _get_config_performance(self, config_dict, anchor_size):
        """
        Get the performance of a configuration at a specific anchor size using SurrogateModel.

        Args:
            config_dict: Configuration dictionary
            anchor_size: Anchor size (budget)

        Returns:
            Performance score
        """
        # Use SurrogateModel to predict performance
        # Add anchor_size to the configuration for prediction
        config_with_anchor = config_dict.copy()
        config_with_anchor["anchor_size"] = anchor_size

        # Use the surrogate model to predict performance
        if self.surrogate_model.model is not None:
            predicted_performance = self.surrogate_model.predict(config_with_anchor)
            return float(predicted_performance)
        else:
            # Fallback: return a random score if model is not ready
            return np.random.uniform(0.1, 0.9)

    def _sample_initial_configurations(self, n_configs):
        """
        Sample initial configurations using random sampling from config space.

        Args:
            n_configs: Number of configurations to sample

        Returns:
            List of configuration dictionaries
        """
        configs = []
        for _ in range(n_configs):
            # Sample random configuration from config space
            config = self.config_space.sample_configuration()
            config_dict = dict(config)
            configs.append(config_dict)
        return configs

    def _evaluate_configurations(self, configs, budget):
        """
        Evaluate configurations at a specific budget level.

        Args:
            configs: List of configuration dictionaries
            budget: Current budget (anchor size)

        Returns:
            List of tuples (config, performance)
        """
        results = []
        for config in configs:
            performance = self._get_config_performance(config, budget)
            results.append((config, performance))
        return results

    def _select_top_configurations(self, results, n_keep):
        """
        Select the top n_keep configurations based on performance.

        Args:
            results: List of tuples (config, performance)
            n_keep: Number of configurations to keep

        Returns:
            List of top configuration dictionaries
        """
        # Sort by performance (lower is better for error rate)
        sorted_results = sorted(results, key=lambda x: x[1])
        top_configs = [config for config, _ in sorted_results[:n_keep]]
        return top_configs

    def run(self, n_initial_configs=100, max_rounds=10):
        """
        Run the Successive Halving optimization process.

        Args:
            n_initial_configs: Number of initial configurations to start with
            max_rounds: Maximum number of rounds to run

        Returns:
            Dictionary containing optimization results
        """
        print(f"Starting Successive Halving for dataset {self.dataset_id}")
        print(f"Initial configurations: {n_initial_configs}")
        print(f"Budget range: {self.min_budget} - {self.max_budget}")
        print(f"Reduction factor (eta): {self.eta}")

        # Initialize with random configurations
        self.current_configs = self._sample_initial_configurations(n_initial_configs)
        self.current_budget = self.min_budget
        self.round = 0

        all_results = []

        while (
            len(self.current_configs) > 1
            and self.current_budget <= self.max_budget
            and self.round < max_rounds
        ):

            print(f"\n--- Round {self.round + 1} ---")
            print(
                f"Evaluating {len(self.current_configs)} configurations at budget {self.current_budget}"
            )

            # Evaluate current configurations at current budget
            round_results = self._evaluate_configurations(
                self.current_configs, self.current_budget
            )
            all_results.extend(round_results)

            # Store results for this round
            self.results.append(
                {
                    "round": self.round + 1,
                    "budget": self.current_budget,
                    "n_configs": len(self.current_configs),
                    "results": round_results.copy(),
                }
            )

            # Find best configuration so far
            best_config, best_performance = min(round_results, key=lambda x: x[1])
            print(
                f"Best performance at budget {self.current_budget}: {best_performance:.4f}"
            )

            # Check if we should continue
            if len(self.current_configs) <= 1:
                print("Only one configuration left, stopping.")
                break

            # Calculate how many configurations to keep
            n_keep = max(1, len(self.current_configs) // self.eta)
            print(f"Keeping top {n_keep} configurations")

            # Select top configurations
            self.current_configs = self._select_top_configurations(
                round_results, n_keep
            )

            # Update SurrogateModel with current round results
            # Prepare new training data with anchor_size included
            new_training_data = []
            for config, performance in round_results:
                config_with_anchor = config.copy()
                config_with_anchor["anchor_size"] = self.current_budget
                config_with_anchor["score"] = performance
                new_training_data.append(config_with_anchor)

            # Convert to DataFrame and retrain the model
            if new_training_data:
                new_df = pd.DataFrame(new_training_data)
                # Combine with existing data and retrain
                combined_data = pd.concat(
                    [self.performance_data, new_df], ignore_index=True
                )
                self.surrogate_model.fit(combined_data)

            # Increase budget
            self.current_budget = min(self.current_budget * self.eta, self.max_budget)
            self.round += 1

        # Final evaluation at maximum budget
        if len(self.current_configs) > 0 and self.current_budget < self.max_budget:
            print(f"\n--- Final Round ---")
            print(
                f"Evaluating {len(self.current_configs)} configurations at maximum budget {self.max_budget}"
            )

            final_results = self._evaluate_configurations(
                self.current_configs, self.max_budget
            )
            all_results.extend(final_results)

            self.results.append(
                {
                    "round": self.round + 1,
                    "budget": self.max_budget,
                    "n_configs": len(self.current_configs),
                    "results": final_results,
                }
            )

            best_config, best_performance = min(final_results, key=lambda x: x[1])
            print(f"Final best performance: {best_performance:.4f}")

        # Compile final results
        final_results = {
            "dataset_id": self.dataset_id,
            "best_config": best_config,
            "best_performance": best_performance,
            "total_evaluations": len(all_results),
            "rounds_completed": self.round + 1,
            "round_results": self.results,
            "all_evaluations": all_results,
        }

        return final_results

    def print_summary(self, results):
        """Print a summary of the optimization results."""
        print(f"\n=== Successive Halving Summary for Dataset {self.dataset_id} ===")
        print(f"Best performance: {results['best_performance']:.4f}")
        print(f"Total evaluations: {results['total_evaluations']}")
        print(f"Rounds completed: {results['rounds_completed']}")

        print(f"\nBest configuration:")
        for param, value in results["best_config"].items():
            print(f"  {param}: {value}")

        print(f"\nRound-by-round results:")
        for round_info in results["round_results"]:
            round_num = round_info["round"]
            budget = round_info["budget"]
            n_configs = round_info["n_configs"]
            best_perf = min(round_info["results"], key=lambda x: x[1])[1]
            print(
                f"  Round {round_num}: Budget {budget}, {n_configs} configs, best: {best_perf:.4f}"
            )

    def plot_config_scores(self, results):
        """
        Plot a classic Successive Halving-style graph, showing how configuration performance changes with increasing budget.

        Args:
            results: Optimization result dictionary
        """
        plt.figure(figsize=(14, 8))

        # Set font for potential Chinese support, but primarily display in English
        plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        # Prepare color palette
        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        # Create a trajectory dictionary for each configuration to track full paths
        config_trajectories = {}
        budget_values = []

        # Collect all budget values
        for round_info in results["round_results"]:
            budget = round_info["budget"]
            budget_values.append(budget)

        # Process results from every round
        for round_idx, round_info in enumerate(results["round_results"]):
            round_num = round_info["round"]
            budget = round_info["budget"]
            round_results = round_info["results"]

            # Create a unique identifier for each config
            for config_idx, (config, score) in enumerate(round_results):
                # Use hash of sorted config items as unique identifier
                config_id = hash(str(sorted(config.items())))

                if config_id not in config_trajectories:
                    config_trajectories[config_id] = {
                        "budgets": [],
                        "scores": [],
                        "config": config,
                        "color": colors[config_idx % len(colors)],
                    }

                config_trajectories[config_id]["budgets"].append(budget)
                config_trajectories[config_id]["scores"].append(score)

        # Plot trajectory for each configuration
        for config_id, trajectory in config_trajectories.items():
            budgets = trajectory["budgets"]
            scores = trajectory["scores"]

            # Plot the full trajectory for this configuration
            plt.plot(
                budgets,
                scores,
                "o-",
                color=trajectory["color"],
                linewidth=2,
                markersize=4,
                alpha=0.8,
            )

        # Add vertical dashed lines indicating elimination points (except final round)
        for i, budget_val in enumerate(budget_values[:-1]):  # all except last budget
            plt.axvline(
                x=budget_val, color="black", linestyle="--", alpha=0.6, linewidth=1
            )

        # Set plot attributes
        plt.xlabel("Budget", fontsize=14, fontweight="bold")
        plt.ylabel("Score", fontsize=14, fontweight="bold", rotation=0, labelpad=20)

        plt.title(
            f"Successive Halving(dataset_id={self.dataset_id})",
            fontsize=16,
            fontweight="bold",
            color="darkgreen",
        )

        # Set axis properties
        plt.xlim(0, self.max_budget)
        plt.xticks([0] + budget_values)

        # Add a grid
        plt.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    # Example usage
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_space_file", type=str, default="lcdb_config_space_knn.json"
    )

    config_space = ConfigSpace.ConfigurationSpace.from_json(
        parser.parse_args().config_space_file
    )

    # Run Successive Halving on different datasets
    datasets = [6, 11, 1457]

    for dataset_id in datasets:
        print(f"\n{'='*60}")
        print(f"Running Successive Halving on Dataset {dataset_id}")
        print(f"{'='*60}")

        sh = SuccessiveHalving(
            config_space=config_space,
            dataset_id=dataset_id,
            min_budget=16,
            max_budget=1024,
            eta=2,
        )

        results = sh.run(n_initial_configs=50, max_rounds=8)
        sh.print_summary(results)
        sh.plot_config_scores(results)

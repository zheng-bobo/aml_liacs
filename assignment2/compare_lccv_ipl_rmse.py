"""
Compare LCCV and IPL based on prediction error (RMSE)
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ConfigSpace
from typing import List, Tuple, Dict
from lccv import LCCV
from ipl import IPL
from surrogate_model import SurrogateModel


def compare_prediction_errors(
    surrogate_model: SurrogateModel,
    config_space,
    minimal_anchor: int,
    max_anchor_size: int,
    num_configs: int = 50,
    min_points_for_fit: int = 5,
) -> Dict:
    """
    Compare LCCV and IPL based on RMSE of their predictions at different anchor points.

    :param surrogate_model: Trained surrogate model
    :param config_space: Configuration space
    :param minimal_anchor: Minimal anchor size
    :param max_anchor_size: Maximum anchor size (final anchor)
    :param num_configs: Number of configurations to test
    :param min_points_for_fit: Minimum points for IPL fitting
    :return: Dictionary containing comparison results
    """
    lccv = LCCV(surrogate_model, minimal_anchor, max_anchor_size)
    ipl = IPL(surrogate_model, minimal_anchor, max_anchor_size, min_points_for_fit)

    # Get true final performance for each configuration
    true_final_performances = []
    lccv_predictions = []  # Predictions at each anchor point
    ipl_predictions = []  # Predictions at each anchor point
    lccv_anchor_points = []  # Anchor points where predictions were made
    ipl_anchor_points = []  # Anchor points where predictions were made

    for idx in range(num_configs):
        config = dict(config_space.sample_configuration())

        # Get true final performance
        config_final = dict(config)
        config_final["anchor_size"] = max_anchor_size
        true_final = float(surrogate_model.predict(config_final))
        true_final_performances.append(true_final)

        # LCCV: Evaluate at all anchor points to get predictions
        # We need to manually evaluate at each anchor point since evaluate_model with None only returns final anchor
        anchors = lccv._build_anchor_schedule()
        lccv_results = []
        for anchor in anchors:
            config_anchor = dict(config)
            config_anchor["anchor_size"] = anchor
            performance = float(surrogate_model.predict(config_anchor))
            lccv_results.append((anchor, performance))

        # Calculate LCCV optimistic predictions at each anchor point (except the first)
        lccv_preds = []
        lccv_pred_anchors = []
        for i in range(1, len(lccv_results)):  # Start from index 1 (need previous point)
            pred = LCCV.optimistic_extrapolation(
                lccv_results[i - 1][0],
                lccv_results[i - 1][1],
                lccv_results[i][0],
                lccv_results[i][1],
                max_anchor_size,
            )
            lccv_preds.append(pred)
            lccv_pred_anchors.append(lccv_results[i][0])

        lccv_predictions.append(lccv_preds)
        lccv_anchor_points.append(lccv_pred_anchors)

        # IPL: Get predictions at each anchor point
        ipl_results = ipl.evaluate_model(None, config)
        ipl_anchors = [r[0] for r in ipl_results]
        ipl_perfs = [r[1] for r in ipl_results]

        # Calculate IPL predictions at each anchor point (after min_points_for_fit)
        ipl_preds = []
        ipl_pred_anchors = []
        for i in range(len(ipl_results)):
            if i + 1 >= min_points_for_fit:  # Enough points for fitting
                observations = ipl_results[: i + 1]
                pred = ipl._predict_final_performance(observations)
                if pred is not None:
                    ipl_preds.append(pred)
                    ipl_pred_anchors.append(ipl_results[i][0])

        ipl_predictions.append(ipl_preds)
        ipl_anchor_points.append(ipl_pred_anchors)

    # Calculate RMSE at each anchor point
    anchor_schedule = lccv._build_anchor_schedule()

    # For LCCV: Calculate RMSE at each anchor point where predictions exist
    lccv_rmse_by_anchor = {}
    for anchor in anchor_schedule[:-1]:  # Exclude final anchor
        squared_errors = []
        for idx in range(num_configs):
            if anchor in lccv_anchor_points[idx]:
                pred_idx = lccv_anchor_points[idx].index(anchor)
                pred = lccv_predictions[idx][pred_idx]
                true = true_final_performances[idx]
                squared_errors.append((pred - true) ** 2)
        if squared_errors:
            lccv_rmse_by_anchor[anchor] = np.sqrt(np.mean(squared_errors))

    # For IPL: Calculate RMSE at each anchor point where predictions exist
    ipl_rmse_by_anchor = {}
    for anchor in anchor_schedule:
        squared_errors = []
        for idx in range(num_configs):
            if anchor in ipl_anchor_points[idx]:
                pred_idx = ipl_anchor_points[idx].index(anchor)
                pred = ipl_predictions[idx][pred_idx]
                true = true_final_performances[idx]
                squared_errors.append((pred - true) ** 2)
        if squared_errors:
            ipl_rmse_by_anchor[anchor] = np.sqrt(np.mean(squared_errors))

    return {
        "lccv_rmse_by_anchor": lccv_rmse_by_anchor,
        "ipl_rmse_by_anchor": ipl_rmse_by_anchor,
        "true_final_performances": true_final_performances,
        "anchor_schedule": anchor_schedule,
    }


def visualize_comparison(results: Dict, output_file: str = "lccv_ipl_comparison_rmse.png"):
    """
    Visualize the comparison between LCCV and IPL prediction errors (RMSE).
    """
    lccv_rmse = results["lccv_rmse_by_anchor"]
    ipl_rmse = results["ipl_rmse_by_anchor"]
    anchor_schedule = results["anchor_schedule"]

    plt.figure(figsize=(12, 6))

    # Extract anchor points and RMSE values
    lccv_anchors = sorted(lccv_rmse.keys())
    lccv_rmse_values = [lccv_rmse[a] for a in lccv_anchors]

    ipl_anchors = sorted(ipl_rmse.keys())
    ipl_rmse_values = [ipl_rmse[a] for a in ipl_anchors]

    # Plot RMSE comparison
    plt.plot(
        lccv_anchors, lccv_rmse_values, "o-", label="LCCV", linewidth=2, markersize=8
    )
    plt.plot(ipl_anchors, ipl_rmse_values, "s-", label="IPL", linewidth=2, markersize=8)

    plt.xlabel("Anchor Size", fontsize=12)
    plt.ylabel("Root Mean Squared Error (RMSE)", fontsize=12)
    plt.title("LCCV vs IPL: Prediction Error Comparison (RMSE)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Comparison plot saved to {output_file}")
    plt.show()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("LCCV vs IPL Prediction Error Comparison (RMSE)")
    print("=" * 60)
    print(f"\nLCCV RMSE by anchor:")
    for anchor in lccv_anchors:
        print(f"  Anchor {anchor:5d}: RMSE = {lccv_rmse[anchor]:.6f}")

    print(f"\nIPL RMSE by anchor:")
    for anchor in ipl_anchors:
        print(f"  Anchor {anchor:5d}: RMSE = {ipl_rmse[anchor]:.6f}")

    # Overall statistics
    if lccv_rmse_values:
        print(f"\nLCCV Overall RMSE: {np.mean(lccv_rmse_values):.6f}")
    if ipl_rmse_values:
        print(f"IPL Overall RMSE: {np.mean(ipl_rmse_values):.6f}")
    print("=" * 60 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare LCCV and IPL based on prediction error (RMSE)"
    )
    parser.add_argument(
        "--config_space_file", type=str, default="lcdb_config_space_knn.json"
    )
    parser.add_argument(
        "--configurations_performance_file",
        type=str,
        default="./config-performances/config_performances_dataset-6.csv",
    )
    parser.add_argument("--minimal_anchor", type=int, default=256)
    parser.add_argument("--max_anchor_size", type=int, default=16000)
    parser.add_argument("--num_configs", type=int, default=50)
    parser.add_argument(
        "--min_points_for_fit",
        type=int,
        default=5,
        help="Minimum number of points required for IPL fitting",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file name for comparison plot (default: lccv_ipl_comparison_rmse_{dataset_name}.png)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration space and data
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    df = pd.read_csv(args.configurations_performance_file)

    # Train surrogate model
    print("Training surrogate model...")
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df, test_size=0.2, random_state=42)

    # Generate output file name
    if args.output_file is None:
        base_name = os.path.splitext(
            os.path.basename(args.configurations_performance_file)
        )[0]
        dataset_name = base_name.split("_")[-1] if "_" in base_name else base_name
        output_file = f"lccv_ipl_comparison_rmse_{dataset_name}.png"
    else:
        output_file = args.output_file

    # Compare LCCV and IPL
    print("\n" + "=" * 60)
    print("Comparing LCCV and IPL Prediction Errors (RMSE)")
    print("=" * 60)
    comparison_results = compare_prediction_errors(
        surrogate_model,
        config_space,
        args.minimal_anchor,
        args.max_anchor_size,
        num_configs=args.num_configs,
        min_points_for_fit=args.min_points_for_fit,
    )

    # Visualize comparison
    visualize_comparison(comparison_results, output_file)


if __name__ == "__main__":
    main()


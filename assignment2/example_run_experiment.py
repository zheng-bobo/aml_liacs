import argparse
import ConfigSpace
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import pandas as pd
from ipl import IPL
from lccv import LCCV
from surrogate_model import SurrogateModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_space_file", type=str, default="lcdb_config_space_knn.json"
    )
    parser.add_argument(
        "--configurations_performance_file",
        type=str,
        default="./config-performances/config_performances_dataset-6.csv",
    )

    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument("--minimal_anchor", type=int, default=256)
    parser.add_argument("--max_anchor_size", type=int, default=16000)
    parser.add_argument("--num_iterations", type=int, default=50)
    parser.add_argument(
        "--min_points_for_fit",
        type=int,
        default=5,
        help="Minimum number of points required for IPL fitting",
    )

    return parser.parse_args()


def run(args):
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    df = pd.read_csv(args.configurations_performance_file)
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df, test_size=0.2, random_state=42)

    # Generate output file name based on configurations_performance_file
    base_name = os.path.splitext(
        os.path.basename(args.configurations_performance_file)
    )[0]

    # Evaluate surrogate model on holdout set
    print("\n" + "=" * 60)
    print("Surrogate Model Evaluation Results")
    print("=" * 60)
    eval_results = surrogate_model.evaluate()
    for metric, value in eval_results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.6f}")
        else:
            print(f"{metric}: {value}")
    print("=" * 60 + "\n")

    # Visualize predictions vs actual values
    plt.figure(figsize=(10, 6))
    X_test_encoded = pd.get_dummies(surrogate_model.X_test, dummy_na=True)
    X_test_encoded = X_test_encoded.reindex(
        columns=surrogate_model.feature_columns_, fill_value=0
    )
    X_test_encoded[surrogate_model.num_cols_] = surrogate_model.imputer_.transform(
        X_test_encoded[surrogate_model.num_cols_]
    )
    y_pred = surrogate_model.model.predict(X_test_encoded)
    y_true = surrogate_model.y_test.values

    plt.scatter(y_true, y_pred, alpha=0.6, s=50)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot(
        [min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction"
    )
    plt.xlabel("Actual Performance", fontsize=12)
    plt.ylabel("Predicted Performance", fontsize=12)
    # Extract dataset name from base_name (e.g., "config_performances_dataset-6" -> "dataset-6")
    dataset_name = base_name.split("_")[-1] if "_" in base_name else base_name
    plt.title(
        f"Surrogate Model: Predictions vs Actual on {dataset_name}\n"
        f'Spearman ρ = {eval_results["Spearman Correlation"]:.4f}, '
        f'R² = {eval_results["R²"]:.4f}, '
        f'RMSE = {eval_results["RMSE"]:.4f}',
        fontsize=12,
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    surrogate_eval_file = f"surrogate_eval_{base_name}.png"
    plt.savefig(surrogate_eval_file, dpi=300, bbox_inches="tight")
    print(f"Surrogate model evaluation plot saved to {surrogate_eval_file}")
    plt.show()

    # Run LCCV
    best_so_far_lccv = None
    lccv = LCCV(surrogate_model, args.minimal_anchor, args.max_anchor_size)

    plt.figure(figsize=(10, 6))
    # Use a colormap with many distinct colors, cycling through multiple colormaps if needed
    # Combine multiple colormaps to get more distinct colors
    colors_list = []
    colormaps = [
        plt.cm.tab20,
        plt.cm.tab20b,
        plt.cm.tab20c,
        plt.cm.Set3,
        plt.cm.Pastel1,
    ]
    for cmap in colormaps:
        colors_list.extend([cmap(i) for i in range(cmap.N)])

    for idx in range(args.num_iterations):
        theta_new = dict(config_space.sample_configuration())
        result = lccv.evaluate_model(best_so_far_lccv, theta_new)
        final_result = result[-1][1]
        if best_so_far_lccv is None or final_result < best_so_far_lccv:
            best_so_far_lccv = final_result
        x_values = [i[0] for i in result]
        y_values = [i[1] for i in result]
        color = colors_list[idx % len(colors_list)]  # Cycle through colors
        plt.plot(x_values, y_values, "-o", color=color, alpha=0.7, markersize=4)

    lccv_output_file = f"lccv_{base_name}.png"
    plt.xlabel("Anchor Size")
    plt.ylabel("Performance")
    plt.title("LCCV Results")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(lccv_output_file, dpi=300, bbox_inches="tight")
    print(f"LCCV plot saved to {lccv_output_file}")
    plt.show()

    # Run IPL
    best_so_far_ipl = None
    ipl = IPL(
        surrogate_model,
        args.minimal_anchor,
        args.max_anchor_size,
        min_points_for_fit=args.min_points_for_fit,
    )

    plt.figure(figsize=(10, 6))
    # Use a colormap with many distinct colors, cycling through multiple colormaps if needed
    # Combine multiple colormaps to get more distinct colors
    colors_list = []
    colormaps = [
        plt.cm.tab20,
        plt.cm.tab20b,
        plt.cm.tab20c,
        plt.cm.Set3,
        plt.cm.Pastel1,
    ]
    for cmap in colormaps:
        colors_list.extend([cmap(i) for i in range(cmap.N)])

    for idx in range(args.num_iterations):
        theta_new = dict(config_space.sample_configuration())
        result = ipl.evaluate_model(best_so_far_ipl, theta_new)
        final_result = result[-1][1]
        if best_so_far_ipl is None or final_result < best_so_far_ipl:
            best_so_far_ipl = final_result
        x_values = [i[0] for i in result]
        y_values = [i[1] for i in result]
        color = colors_list[idx % len(colors_list)]  # Cycle through colors
        plt.plot(x_values, y_values, "-s", color=color, alpha=0.7, markersize=4)

    ipl_output_file = f"ipl_{base_name}.png"
    plt.xlabel("Anchor Size")
    plt.ylabel("Performance")
    plt.title("IPL Results")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ipl_output_file, dpi=300, bbox_inches="tight")
    print(f"IPL plot saved to {ipl_output_file}")
    plt.show()


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args())

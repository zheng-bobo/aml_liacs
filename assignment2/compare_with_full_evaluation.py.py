"""
Summarize anchor evaluations for all datasets and methods
"""
import argparse
import os
import pandas as pd
import ConfigSpace
from compare_with_full_evaluation import compare_methods
from surrogate_model import SurrogateModel


def summarize_all_datasets(
    config_space_file: str,
    datasets: list,
    minimal_anchor: int = 256,
    max_anchor_size: int = 16000,
    num_iterations: int = 50,
    min_points_for_fit: int = 5,
):
    """
    Summarize anchor evaluations for all datasets.
    """
    config_space = ConfigSpace.ConfigurationSpace.from_json(config_space_file)
    
    all_results = []
    
    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"Processing {dataset}")
        print(f"{'='*70}")
        
        config_file = f"./config-performances/config_performances_{dataset}.csv"
        if not os.path.exists(config_file):
            print(f"Warning: {config_file} not found, skipping...")
            continue
        
        # Load data and train surrogate model
        df = pd.read_csv(config_file)
        surrogate_model = SurrogateModel(config_space)
        surrogate_model.fit(df, test_size=0.2, random_state=42)
        
        # Compare methods
        results = compare_methods(
            surrogate_model,
            config_space,
            minimal_anchor,
            max_anchor_size,
            num_iterations=num_iterations,
            min_points_for_fit=min_points_for_fit,
        )
        
        all_results.append({
            "dataset": dataset,
            "full_evaluation": results["full_evaluation"]["total_evaluations"],
            "lccv": results["lccv"]["total_evaluations"],
            "ipl": results["ipl"]["total_evaluations"],
            "lccv_saved": results["lccv"]["evaluations_saved"],
            "ipl_saved": results["ipl"]["evaluations_saved"],
            "lccv_saved_ratio": results["lccv"]["evaluations_saved_ratio"],
            "ipl_saved_ratio": results["ipl"]["evaluations_saved_ratio"],
        })
    
    # Print summary table
    print("\n" + "=" * 80)
    print("Anchor Evaluations Summary: All Datasets")
    print("=" * 80)
    
    print("\nTotal Anchor Evaluations by Method and Dataset:")
    print("-" * 80)
    print(f"{'Dataset':<15} {'Full Eval':<12} {'LCCV':<12} {'IPL':<12} {'LCCV Saved':<15} {'IPL Saved':<15}")
    print("-" * 80)
    
    for result in all_results:
        print(
            f"{result['dataset']:<15} "
            f"{result['full_evaluation']:<12} "
            f"{result['lccv']:<12} "
            f"{result['ipl']:<12} "
            f"{result['lccv_saved']} ({result['lccv_saved_ratio']:.2f}%){'':<6} "
            f"{result['ipl_saved']} ({result['ipl_saved_ratio']:.2f}%)"
        )
    
    # Calculate totals
    total_full = sum(r["full_evaluation"] for r in all_results)
    total_lccv = sum(r["lccv"] for r in all_results)
    total_ipl = sum(r["ipl"] for r in all_results)
    total_lccv_saved = sum(r["lccv_saved"] for r in all_results)
    total_ipl_saved = sum(r["ipl_saved"] for r in all_results)
    
    print("-" * 80)
    print(
        f"{'TOTAL':<15} "
        f"{total_full:<12} "
        f"{total_lccv:<12} "
        f"{total_ipl:<12} "
        f"{total_lccv_saved} ({total_lccv_saved/total_full*100:.2f}%){'':<6} "
        f"{total_ipl_saved} ({total_ipl_saved/total_full*100:.2f}%)"
    )
    print("=" * 80)
    
    # Create DataFrame for better visualization
    df_summary = pd.DataFrame(all_results)
    print("\nDetailed Summary Table:")
    print(df_summary.to_string(index=False))
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Summarize anchor evaluations across all datasets"
    )
    parser.add_argument(
        "--config_space_file", type=str, default="lcdb_config_space_knn.json"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["dataset-6", "dataset-11", "dataset-1457"],
        help="List of dataset names",
    )
    parser.add_argument("--minimal_anchor", type=int, default=256)
    parser.add_argument("--max_anchor_size", type=int, default=16000)
    parser.add_argument("--num_iterations", type=int, default=50)
    parser.add_argument(
        "--min_points_for_fit",
        type=int,
        default=5,
        help="Minimum number of points required for IPL fitting",
    )
    
    args = parser.parse_args()
    
    summarize_all_datasets(
        args.config_space_file,
        args.datasets,
        args.minimal_anchor,
        args.max_anchor_size,
        args.num_iterations,
        args.min_points_for_fit,
    )


if __name__ == "__main__":
    main()


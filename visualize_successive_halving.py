#!/usr/bin/env python3
"""
Successive Halving 可视化脚本

使用seaborn创建美观的实验结果可视化图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_results(filename="successive_halving_results.json"):
    """加载实验结果"""
    if not Path(filename).exists():
        print(f"结果文件 {filename} 不存在")
        return None
    
    with open(filename, 'r') as f:
        results = json.load(f)
    
    return results

def create_analysis_dataframe(results):
    """创建分析用的DataFrame"""
    analysis_data = []
    
    for dataset_id, dataset_results in results.items():
        for round_info in dataset_results['round_summary']:
            analysis_data.append({
                'Dataset': f'Dataset {dataset_id}',
                'Round': round_info['round'],
                'Budget': round_info['budget'],
                'N_Configs': round_info['n_configs'],
                'Best_Performance': round_info['best_performance'],
                'Dataset_ID': int(dataset_id)
            })
    
    return pd.DataFrame(analysis_data)

def plot_successive_halving_results(results, save_plots=True):
    """绘制Successive Halving实验结果"""
    
    # 创建分析数据
    df = create_analysis_dataframe(results)
    
    # 设置seaborn样式
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Successive Halving Optimization Results', fontsize=20, fontweight='bold', y=0.98)
    
    # 1. 每轮最佳性能变化
    ax1 = axes[0, 0]
    sns.lineplot(data=df, x='Round', y='Best_Performance', hue='Dataset', 
                marker='o', linewidth=3, markersize=8, ax=ax1)
    ax1.set_title('Best Performance per Round', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Round Number', fontsize=12)
    ax1.set_ylabel('Best Performance (Error Rate)', fontsize=12)
    ax1.legend(title='Dataset', title_fontsize=12, fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. 配置数量变化（对数尺度）
    ax2 = axes[0, 1]
    sns.lineplot(data=df, x='Round', y='N_Configs', hue='Dataset', 
                marker='s', linewidth=3, markersize=8, ax=ax2)
    ax2.set_title('Number of Configurations per Round', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Round Number', fontsize=12)
    ax2.set_ylabel('Number of Configurations', fontsize=12)
    ax2.set_yscale('log')
    ax2.legend(title='Dataset', title_fontsize=12, fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. 预算 vs 最佳性能
    ax3 = axes[1, 0]
    sns.lineplot(data=df, x='Budget', y='Best_Performance', hue='Dataset', 
                marker='^', linewidth=3, markersize=8, ax=ax3)
    ax3.set_title('Budget vs Best Performance', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Budget (Anchor Size)', fontsize=12)
    ax3.set_ylabel('Best Performance (Error Rate)', fontsize=12)
    ax3.set_xscale('log')
    ax3.legend(title='Dataset', title_fontsize=12, fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. 性能改进热力图
    ax4 = axes[1, 1]
    
    # 计算每轮的性能改进
    improvement_data = []
    for dataset_id, dataset_results in results.items():
        dataset_name = f'Dataset {dataset_id}'
        rounds = dataset_results['round_summary']
        
        for i in range(1, len(rounds)):
            prev_perf = rounds[i-1]['best_performance']
            curr_perf = rounds[i]['best_performance']
            improvement = prev_perf - curr_perf  # 正值表示改进
            improvement_data.append({
                'Dataset': dataset_name,
                'Round': rounds[i]['round'],
                'Improvement': improvement
            })
    
    if improvement_data:
        improvement_df = pd.DataFrame(improvement_data)
        pivot_data = improvement_df.pivot(index='Dataset', columns='Round', values='Improvement')
        
        sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='RdYlGn', 
                   center=0, ax=ax4, cbar_kws={'label': 'Performance Improvement'})
        ax4.set_title('Performance Improvement per Round', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Round Number', fontsize=12)
        ax4.set_ylabel('Dataset', fontsize=12)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('successive_halving_analysis.png', dpi=300, bbox_inches='tight')
        print("分析图表已保存为 'successive_halving_analysis.png'")
    
    plt.show()

def plot_detailed_analysis(results, save_plots=True):
    """绘制详细的分析图表"""
    
    df = create_analysis_dataframe(results)
    
    # 设置样式
    sns.set_style("whitegrid")
    sns.set_palette("Set2")
    
    # 创建更详细的图表
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Detailed Successive Halving Analysis', fontsize=20, fontweight='bold', y=0.98)
    
    # 1. 性能收敛曲线
    ax1 = axes[0, 0]
    for dataset_id in df['Dataset_ID'].unique():
        dataset_data = df[df['Dataset_ID'] == dataset_id]
        sns.lineplot(data=dataset_data, x='Round', y='Best_Performance', 
                    label=f'Dataset {dataset_id}', linewidth=2.5, marker='o', markersize=6, ax=ax1)
    ax1.set_title('Performance Convergence', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Round Number', fontsize=12)
    ax1.set_ylabel('Best Performance (Error Rate)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 配置剪枝效率
    ax2 = axes[0, 1]
    for dataset_id in df['Dataset_ID'].unique():
        dataset_data = df[df['Dataset_ID'] == dataset_id]
        sns.lineplot(data=dataset_data, x='Round', y='N_Configs', 
                    label=f'Dataset {dataset_id}', linewidth=2.5, marker='s', markersize=6, ax=ax2)
    ax2.set_title('Configuration Pruning Efficiency', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Round Number', fontsize=12)
    ax2.set_ylabel('Number of Configurations', fontsize=12)
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 资源利用率
    ax3 = axes[0, 2]
    for dataset_id in df['Dataset_ID'].unique():
        dataset_data = df[df['Dataset_ID'] == dataset_id]
        # 计算资源利用率（配置数 * 预算）
        dataset_data = dataset_data.copy()
        dataset_data['Resource_Usage'] = dataset_data['N_Configs'] * dataset_data['Budget']
        sns.lineplot(data=dataset_data, x='Round', y='Resource_Usage', 
                    label=f'Dataset {dataset_id}', linewidth=2.5, marker='^', markersize=6, ax=ax3)
    ax3.set_title('Resource Usage per Round', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Round Number', fontsize=12)
    ax3.set_ylabel('Resource Usage (Configs × Budget)', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 最终性能对比
    ax4 = axes[1, 0]
    final_performances = []
    for dataset_id, dataset_results in results.items():
        final_perf = dataset_results['best_performance']
        final_performances.append({
            'Dataset': f'Dataset {dataset_id}',
            'Final_Performance': final_perf
        })
    
    final_df = pd.DataFrame(final_performances)
    sns.barplot(data=final_df, x='Dataset', y='Final_Performance', ax=ax4)
    ax4.set_title('Final Best Performance by Dataset', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Dataset', fontsize=12)
    ax4.set_ylabel('Final Best Performance (Error Rate)', fontsize=12)
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. 总评估次数对比
    ax5 = axes[1, 1]
    total_evaluations = []
    for dataset_id, dataset_results in results.items():
        total_eval = dataset_results['total_evaluations']
        total_evaluations.append({
            'Dataset': f'Dataset {dataset_id}',
            'Total_Evaluations': total_eval
        })
    
    eval_df = pd.DataFrame(total_evaluations)
    sns.barplot(data=eval_df, x='Dataset', y='Total_Evaluations', ax=ax5)
    ax5.set_title('Total Evaluations by Dataset', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Dataset', fontsize=12)
    ax5.set_ylabel('Total Number of Evaluations', fontsize=12)
    ax5.tick_params(axis='x', rotation=45)
    
    # 6. 执行时间对比
    ax6 = axes[1, 2]
    execution_times = []
    for dataset_id, dataset_results in results.items():
        exec_time = dataset_results['execution_time']
        execution_times.append({
            'Dataset': f'Dataset {dataset_id}',
            'Execution_Time': exec_time
        })
    
    time_df = pd.DataFrame(execution_times)
    sns.barplot(data=time_df, x='Dataset', y='Execution_Time', ax=ax6)
    ax6.set_title('Execution Time by Dataset', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Dataset', fontsize=12)
    ax6.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax6.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('successive_halving_detailed_analysis.png', dpi=300, bbox_inches='tight')
        print("详细分析图表已保存为 'successive_halving_detailed_analysis.png'")
    
    plt.show()

def print_summary_table(results):
    """打印结果摘要表格"""
    print("\n" + "="*80)
    print("Successive Halving 实验结果摘要")
    print("="*80)
    
    summary_data = []
    for dataset_id, dataset_results in results.items():
        summary_data.append({
            'Dataset': f'Dataset {dataset_id}',
            'Best Performance': f"{dataset_results['best_performance']:.4f}",
            'Total Evaluations': dataset_results['total_evaluations'],
            'Rounds Completed': dataset_results['rounds_completed'],
            'Execution Time (s)': f"{dataset_results['execution_time']:.2f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    print("\n最佳配置:")
    for dataset_id, dataset_results in results.items():
        print(f"\nDataset {dataset_id}:")
        best_config = dataset_results['best_config']
        for param, value in best_config.items():
            print(f"  {param}: {value}")

def main():
    """主函数"""
    print("Successive Halving 结果可视化")
    print("="*50)
    
    # 加载结果
    results = load_results()
    if results is None:
        print("请先运行 run_successive_halving.py 生成实验结果")
        return
    
    # 打印摘要表格
    print_summary_table(results)
    
    # 绘制基本分析图表
    print("\n生成基本分析图表...")
    plot_successive_halving_results(results, save_plots=True)
    
    # 绘制详细分析图表
    print("\n生成详细分析图表...")
    plot_detailed_analysis(results, save_plots=True)
    
    print("\n可视化完成!")

if __name__ == "__main__":
    main()

"""
Successive Halving实验脚本

这个脚本实现了Successive Halving算法，用于超参数优化。
Successive Halving是一种资源分配方法，通过逐步增加资源来评估配置，
并逐步剪枝表现较差的配置。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple

from successive_halving import SuccessiveHalving, load_config_space


class SuccessiveHalvingExperiment:
    """Successive Halving实验类，用于运行和分析实验"""
    
    def __init__(self, datasets: List[int] = [6, 11, 1457]):
        """
        初始化实验
        
        Args:
            datasets: 要实验的数据集ID列表
        """
        self.datasets = datasets
        self.config_space = load_config_space()
        self.results = {}
        
    def run_experiment(self, 
                      n_initial_configs: int = 50,
                      max_rounds: int = 8,
                      min_budget: int = 16,
                      max_budget: int = 1024,
                      eta: int = 2):
        """
        运行Successive Halving实验
        
        Args:
            n_initial_configs: 初始配置数量
            max_rounds: 最大轮数
            min_budget: 最小预算
            max_budget: 最大预算
            eta: 剪枝因子
        """
        print("="*80)
        print("Successive Halving 超参数优化实验")
        print("="*80)
        print(f"实验参数:")
        print(f"  数据集: {self.datasets}")
        print(f"  初始配置数: {n_initial_configs}")
        print(f"  最大轮数: {max_rounds}")
        print(f"  预算范围: {min_budget} - {max_budget}")
        print(f"  剪枝因子: {eta}")
        print("="*80)
        
        for dataset_id in self.datasets:
            print(f"\n{'='*60}")
            print(f"运行数据集 {dataset_id} 的Successive Halving")
            print(f"{'='*60}")
            
            try:
                start_time = time.time()
                
                # 创建Successive Halving实例
                sh = SuccessiveHalving(
                    config_space=self.config_space,
                    dataset_id=dataset_id,
                    min_budget=min_budget,
                    max_budget=max_budget,
                    eta=eta
                )
                
                # 运行优化
                results = sh.run(
                    n_initial_configs=n_initial_configs,
                    max_rounds=max_rounds
                )
                
                end_time = time.time()
                results['execution_time'] = end_time - start_time
                
                # 存储结果
                self.results[dataset_id] = results
                
                # 打印摘要
                sh.print_summary(results)
                print(f"执行时间: {results['execution_time']:.2f} 秒")
                
            except FileNotFoundError as e:
                print(f"错误: 数据集 {dataset_id} 的性能数据文件未找到")
                print(f"详细信息: {e}")
                continue
            except Exception as e:
                print(f"错误: 运行数据集 {dataset_id} 时发生异常")
                print(f"详细信息: {e}")
                continue
    
    def analyze_results(self):
        """分析实验结果"""
        if not self.results:
            print("没有实验结果可供分析")
            return
        
        print("\n" + "="*80)
        print("实验结果分析")
        print("="*80)
        
        # 创建分析数据框
        analysis_data = []
        for dataset_id, results in self.results.items():
            for round_info in results['round_results']:
                analysis_data.append({
                    'dataset': dataset_id,
                    'round': round_info['round'],
                    'budget': round_info['budget'],
                    'n_configs': round_info['n_configs'],
                    'best_performance': min(round_info['results'], key=lambda x: x[1])[1],
                    'avg_performance': np.mean([perf for _, perf in round_info['results']]),
                    'std_performance': np.std([perf for _, perf in round_info['results']])
                })
        
        df = pd.DataFrame(analysis_data)
        
        # 打印统计摘要
        print("\n数据集性能摘要:")
        for dataset_id in self.datasets:
            if dataset_id in self.results:
                result = self.results[dataset_id]
                print(f"\n数据集 {dataset_id}:")
                print(f"  最佳性能: {result['best_performance']:.4f}")
                print(f"  总评估次数: {result['total_evaluations']}")
                print(f"  完成轮数: {result['rounds_completed']}")
                print(f"  执行时间: {result['execution_time']:.2f} 秒")
        
        return df
    
    def plot_results(self, save_plots: bool = True):
        """绘制实验结果图表"""
        if not self.results:
            print("没有实验结果可供绘制")
            return
        
        # 创建分析数据
        analysis_data = []
        for dataset_id, results in self.results.items():
            for round_info in results['round_results']:
                analysis_data.append({
                    'dataset': dataset_id,
                    'round': round_info['round'],
                    'budget': round_info['budget'],
                    'n_configs': round_info['n_configs'],
                    'best_performance': min(round_info['results'], key=lambda x: x[1])[1],
                    'avg_performance': np.mean([perf for _, perf in round_info['results']])
                })
        
        df = pd.DataFrame(analysis_data)
        
        # 设置绘图样式
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Successive Halving Results Analysis', fontsize=16, fontweight='bold')
        
        # 1. 每轮最佳性能
        ax1 = axes[0, 0]
        for dataset_id in self.datasets:
            if dataset_id in self.results:
                dataset_data = df[df['dataset'] == dataset_id]
                ax1.plot(dataset_data['round'], dataset_data['best_performance'], 
                        marker='o', linewidth=2, label=f'Dataset {dataset_id}')
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Best Performance (Error Rate)')
        ax1.set_title('Best Performance per Round')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 配置数量变化
        ax2 = axes[0, 1]
        for dataset_id in self.datasets:
            if dataset_id in self.results:
                dataset_data = df[df['dataset'] == dataset_id]
                ax2.plot(dataset_data['round'], dataset_data['n_configs'], 
                        marker='s', linewidth=2, label=f'Dataset {dataset_id}')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Number of Configurations')
        ax2.set_title('Number of Configurations per Round')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # 3. 预算 vs 性能
        ax3 = axes[1, 0]
        for dataset_id in self.datasets:
            if dataset_id in self.results:
                dataset_data = df[df['dataset'] == dataset_id]
                ax3.plot(dataset_data['budget'], dataset_data['best_performance'], 
                        marker='^', linewidth=2, label=f'Dataset {dataset_id}')
        ax3.set_xlabel('Budget (Anchor Size)')
        ax3.set_ylabel('Best Performance (Error Rate)')
        ax3.set_title('Budget vs Best Performance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')
        
        # 4. 平均性能 vs 最佳性能
        ax4 = axes[1, 1]
        for dataset_id in self.datasets:
            if dataset_id in self.results:
                dataset_data = df[df['dataset'] == dataset_id]
                ax4.scatter(dataset_data['avg_performance'], dataset_data['best_performance'], 
                           alpha=0.7, s=60, label=f'Dataset {dataset_id}')
        ax4.set_xlabel('Average Performance (Error Rate)')
        ax4.set_ylabel('Best Performance (Error Rate)')
        ax4.set_title('Average vs Best Performance')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 添加对角线
        min_val = min(df['avg_performance'].min(), df['best_performance'].min())
        max_val = max(df['avg_performance'].max(), df['best_performance'].max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('successive_halving_results.png', dpi=300, bbox_inches='tight')
            print("图表已保存为 'successive_halving_results.png'")
        
        plt.show()
    
    def save_results(self, filename: str = "successive_halving_results.json"):
        """保存实验结果到JSON文件"""
        if not self.results:
            print("没有实验结果可供保存")
            return
        
        # 准备保存的数据
        save_data = {}
        for dataset_id, results in self.results.items():
            save_data[dataset_id] = {
                'best_config': results['best_config'],
                'best_performance': results['best_performance'],
                'total_evaluations': results['total_evaluations'],
                'rounds_completed': results['rounds_completed'],
                'execution_time': results['execution_time'],
                'round_summary': []
            }
            
            # 添加每轮摘要
            for round_info in results['round_results']:
                best_perf = min(round_info['results'], key=lambda x: x[1])[1]
                save_data[dataset_id]['round_summary'].append({
                    'round': round_info['round'],
                    'budget': round_info['budget'],
                    'n_configs': round_info['n_configs'],
                    'best_performance': best_perf
                })
        
        # 保存到文件
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"实验结果已保存到 '{filename}'")
    
    def compare_with_random_search(self):
        """与随机搜索进行比较"""
        print("\n" + "="*80)
        print("与随机搜索比较")
        print("="*80)
        
        for dataset_id in self.datasets:
            if dataset_id not in self.results:
                continue
            
            sh_result = self.results[dataset_id]
            total_evaluations = sh_result['total_evaluations']
            
            print(f"\n数据集 {dataset_id}:")
            print(f"Successive Halving:")
            print(f"  最佳性能: {sh_result['best_performance']:.4f}")
            print(f"  总评估次数: {total_evaluations}")
            print(f"  执行时间: {sh_result['execution_time']:.2f} 秒")
            
            # 这里可以添加随机搜索的比较逻辑
            # 由于我们没有随机搜索的实现，这里只是占位符
            print(f"随机搜索 (模拟):")
            print(f"  预期评估次数: {total_evaluations}")
            print(f"  预期执行时间: {sh_result['execution_time'] * 1.5:.2f} 秒 (估算)")


def main():
    """主函数"""
    # 创建实验实例
    experiment = SuccessiveHalvingExperiment(datasets=[6, 11, 1457])
    
    # 运行实验
    experiment.run_experiment(
        n_initial_configs=50,
        max_rounds=8,
        min_budget=16,
        max_budget=1024,
        eta=2
    )
    
    # 分析结果
    analysis_df = experiment.analyze_results()
    
    # 绘制图表
    experiment.plot_results(save_plots=True)
    
    # 保存结果
    experiment.save_results()
    
    # 与随机搜索比较
    experiment.compare_with_random_search()
    
    print("\n" + "="*80)
    print("实验完成!")
    print("="*80)


if __name__ == "__main__":
    main()

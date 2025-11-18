from collections import defaultdict
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import matplotlib

matplotlib.use('TkAgg')

file_path = "result_obb.txt"


def parse_simulation_results(file_path: str) -> dict:
    """
    解析模拟结果文件，返回每个组合及其出现频率的字典
    格式: {('3-0': set, '3-1/3-2': set, '0-3': set): frequency}
    """
    results = defaultdict(int)
    total_simulations = 0
    pattern = r"3-0: (.*?) \| 3-1/3-2: (.*?) \| 0-3: (.*?): (\d+)/\d+"

    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(pattern, line)
            if match:
                three_zero = set(t.strip() for t in match.group(1).split(','))
                three_one_two = set(t.strip() for t in match.group(2).split(','))
                zero_three = set(t.strip() for t in match.group(3).split(','))
                count = int(match.group(4))

                key = (frozenset(three_zero), frozenset(three_one_two), frozenset(zero_three))
                results[key] += count
                total_simulations += count

    return results, total_simulations


def evaluate_combination(combo: dict, results: dict) -> tuple:
    """
    评估组合在模拟结果中的表现

    Args:
        combo: 要评估的组合
        results: 模拟结果字典

    Returns:
        tuple: (正确数列表, 正确数>=5的概率, 正确数期望)
    """
    correct_counts = []
    total_simulations = sum(results.values())

    for (three_zero, three_one_two, zero_three), count in results.items():
        correct = 0
        correct += len(set(combo['3-0']) & set(three_zero))
        correct += len(set(combo['3-1/3-2']) & set(three_one_two))
        correct += len(set(combo['0-3']) & set(zero_three))
        correct_counts.extend([correct] * count)

    correct_counts = np.array(correct_counts)
    prob_ge5 = np.mean(correct_counts >= 5)
    expected_value = np.mean(correct_counts)

    return correct_counts, prob_ge5, expected_value


def plot_distribution(correct_counts: List[int], best_probability: float, expected_value: float):
    """
    绘制正确数分布函数

    Args:
        correct_counts: 所有组合的正确数列表
        best_probability: 最优组合的概率
        expected_value: 正确数期望
    """
    plt.figure(figsize=(10, 6))

    # 计算分布
    counts, bins = np.histogram(correct_counts, bins=range(11), density=True)

    # 绘制直方图，>=5的部分用绿色，<5的部分用红色
    colors = ['red' if x < 5 else 'green' for x in bins[:-1]]
    plt.bar(bins[:-1], counts, width=0.8, alpha=0.7, color=colors)

    # 添加最优概率线和期望线
    plt.axvline(x=5, color='r', linestyle='--', label=f'P(X>=5)={best_probability:.4f}')
    plt.axvline(x=expected_value, color='blue', linestyle=':',
                label=f'E[X]={expected_value:.2f}')

    # 设置图表属性
    plt.xlabel('X')
    plt.ylabel('Freq')
    plt.xticks(range(11))
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 保存为SVG格式
    plt.savefig('result.svg', format='svg')
    plt.show()
    plt.close()


def generate_file_paths() -> List[str]:
    """
    生成所有可能的文件路径
    格式: (0.0000-1.0000)_(1.0000-0.0000)_592.0000.txt
    """
    file_paths = []
    for i in range(11):
        alpha = round(i * 0.1, 4)
        beta = round(1 - alpha, 4)
        file_paths.append(file_path)
    return file_paths


def main():
    """主函数"""
    # 要验证的组合
    combo = {
        '3-0': {'Legacy', 'PARIVISION'},
        '3-1/3-2': {'Imperial', 'Lynn Vision', 'FlyQuest', 'fnatic', 'FaZe', 'B8'},
        '0-3': {'The Huns', 'Rare Atom'}
    }

    # 加载模拟结果
    results, total_simulations = parse_simulation_results(file_path)
    print(f"已加载 {total_simulations} 个模拟结果")

    # 评估组合
    correct_counts, prob_ge5, expected_value = evaluate_combination(combo, results)

    # 打印结果
    print(f"{combo}")
    print(f"预测正确数 >= 5 的概率 = {prob_ge5:.6f}")
    print(f"预测正确数期望 = {expected_value:.6f}")

    plot_distribution(correct_counts, prob_ge5, expected_value)


if __name__ == "__main__":
    main()

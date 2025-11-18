"""
CS2 Pick'Em —— 自动从候选池生成所有完整方案版本
"""

from collections import defaultdict
import re
import tqdm
import numpy as np
from multiprocessing import Pool, cpu_count, Manager
import time
import threading
import itertools

FILE_PATH = "result_obb.txt"

############################################################
# 1. 你只需要改这里！
#   每个列表表示“这一组可能的队伍池”
#   程序会自动：3-0 选 2，3-1/3-2 选 6，0-3 选 2
############################################################

CANDIDATE_POOL = {
    "3-0": {"FaZe", "Legacy"},
    "3-1/3-2": {"FaZe", "Legacy", "PARIVISION", "Lynn Vision", "B8",
                "GamerLegion", "fnatic", "Ninjas in Pyjamas",
                "M80", "FlyQuest", "NRG", "Imperial", "Fluxo"},
    "0-3": {"The Huns", "RED Canids"},
}

# 要求数量（可改）
SIZE_3_0 = 2
SIZE_ADV = 6
SIZE_0_3 = 2

############################################################
# 2. 枚举所有合法完整方案
############################################################

def generate_all_combinations(pool):

    candidates = []

    for c3 in itertools.combinations(pool["3-0"], SIZE_3_0):
        for c0 in itertools.combinations(pool["0-3"], SIZE_0_3):

            # 必须无重叠
            if set(c3) & set(c0):
                continue

            # 3-1/3-2 需要从池中排除 c3 和 c0
            adv_pool = pool["3-1/3-2"] - set(c3) - set(c0)

            if len(adv_pool) < SIZE_ADV:
                continue

            for adv in itertools.combinations(adv_pool, SIZE_ADV):
                combo = {
                    "3-0": set(c3),
                    "3-1/3-2": set(adv),
                    "0-3": set(c0)
                }
                candidates.append(combo)

    return candidates


############################################################
# 3. 解析模拟结果
############################################################

def parse_simulation_results(file_path: str):
    results = defaultdict(int)
    pattern = r"3-0: (.*?) \| 3-1/3-2: (.*?) \| 0-3: (.*?): (\d+)/"

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.match(pattern, line)
            if not m:
                continue
            three_zero = frozenset(t.strip() for t in m.group(1).split(",") if t.strip())
            adv = frozenset(t.strip() for t in m.group(2).split(",") if t.strip())
            zero_three = frozenset(t.strip() for t in m.group(3).split(",") if t.strip())
            count = int(m.group(4))
            results[(three_zero, adv, zero_three)] += count

    return results, sum(results.values())


############################################################
# 4. 单个组合评估函数
############################################################

def evaluate_single(candidate, results):
    correct_counts = []

    set3 = candidate["3-0"]
    setadv = candidate["3-1/3-2"]
    set0 = candidate["0-3"]

    for (r3, radv, r0), cnt in results.items():
        correct = 0
        correct += len(set3 & r3)
        correct += len(setadv & radv)
        correct += len(set0 & r0)

        correct_counts.extend([correct] * cnt)

    correct_counts = np.array(correct_counts)
    return np.mean(correct_counts >= 5)


############################################################
# 5. 多进程评估
############################################################

def process_chunk(args):
    chunk, results, shared = args

    local_best = None
    local_prob = -1

    for c in chunk:
        p = evaluate_single(c, results)

        if p > local_prob:
            local_prob = p
            local_best = c

        with shared["lock"]:
            shared["processed"] += 1

    return local_best, local_prob


############################################################
# 6. 主求解流程
############################################################

def find_best(file_path):

    # 读取模拟结果
    results, total = parse_simulation_results(file_path)
    print(f"已加载模拟结果：{total} 条\n")

    # 自动生成所有完整方案
    print("正在生成所有完整预测方案……")
    candidates = generate_all_combinations(CANDIDATE_POOL)
    N = len(candidates)
    print(f"共生成 {N} 个方案。\n")

    # 并行计算
    manager = Manager()
    shared = manager.dict()
    shared["processed"] = 0
    shared["lock"] = manager.Lock()

    ncore = max(1, cpu_count() - 1)
    chunk_size = max(1, N // ncore)
    chunks = [candidates[i:i + chunk_size] for i in range(0, N, chunk_size)]

    pbar = tqdm.tqdm(total=N, ncols=100, desc="评估中")

    def update_bar():
        last = 0
        while last < N:
            with shared["lock"]:
                cur = shared["processed"]
            if cur != last:
                pbar.n = cur
                pbar.refresh()
                last = cur
            time.sleep(0.1)

    thread = threading.Thread(target=update_bar, daemon=True)
    thread.start()

    with Pool(ncore) as pool:
        res_list = pool.map(process_chunk, [(ch, results, shared) for ch in chunks])

    pbar.close()

    best = None
    best_p = -1
    for c, p in res_list:
        if p > best_p:
            best = c
            best_p = p

    return best, best_p


############################################################
# 7. 入口
############################################################

if __name__ == "__main__":
    best, prob = find_best(FILE_PATH)

    print("\n====== 最优结果 ======")
    print(f"P(X ≥ 5) = {prob:.6f}\n")
    print("3-0 :", ", ".join(best["3-0"]))
    print("3-1/3-2 :", ", ".join(best["3-1/3-2"]))
    print("0-3 :", ", ".join(best["0-3"]))

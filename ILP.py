from collections import defaultdict
import re
import pulp

file_path = "result3.txt"

# -------------------------
# 1. 解析 result.txt
# -------------------------
def parse_simulation_results(file_path):
    results = defaultdict(int)
    pattern = r"3-0: (.*?) \| 3-1/3-2: (.*?) \| 0-3: (.*?): (\d+)/"
    total = 0

    with open(file_path, "r") as f:
        for line in f:
            m = re.match(pattern, line)
            if not m: continue

            three_zero = tuple(t.strip() for t in m.group(1).split(","))
            adv = tuple(t.strip() for t in m.group(2).split(","))
            zero_three = tuple(t.strip() for t in m.group(3).split(","))
            count = int(m.group(4))

            key = (three_zero, adv, zero_three)
            results[key] += count
            total += count

    return results, total


# -------------------------
# 2. 从 result.txt 统计概率
# -------------------------
def compute_bucket_probabilities(results, total):
    teams = set()
    for (a,b,c) in results.keys():
        teams |= set(a) | set(b) | set(c)
    teams = list(teams)

    p_3_0 = {t:0 for t in teams}
    p_adv = {t:0 for t in teams}
    p_0_3 = {t:0 for t in teams}

    for (three_zero, adv, zero_three), count in results.items():
        for t in three_zero: p_3_0[t] += count
        for t in adv: p_adv[t] += count
        for t in zero_three: p_0_3[t] += count

    # 归一化
    for t in teams:
        p_3_0[t] /= total
        p_adv[t] /= total
        p_0_3[t] /= total

    return teams, p_3_0, p_adv, p_0_3


# -------------------------
# 3. ILP 求最优 Pick'Em
# -------------------------
def solve_pickem(teams, p3, padv, p03):
    model = pulp.LpProblem("PickEmOptimal", pulp.LpMaximize)

    # 变量
    x30  = pulp.LpVariable.dicts("x3_0", teams, 0,1, pulp.LpBinary)
    xadv = pulp.LpVariable.dicts("xadv", teams, 0,1, pulp.LpBinary)
    x03  = pulp.LpVariable.dicts("x0_3", teams, 0,1, pulp.LpBinary)

    # 目标函数：最大化期望正确数
    model += pulp.lpSum(p3[t]*x30[t] + padv[t]*xadv[t] + p03[t]*x03[t] for t in teams)

    # 约束
    model += pulp.lpSum(x30[t] for t in teams) == 2      # 3-0 only 1
    model += pulp.lpSum(x03[t] for t in teams) == 2      # 0-3 only 1
    model += pulp.lpSum(xadv[t] for t in teams) == 6     # adv = 7

    for t in teams:
        model += x30[t] + xadv[t] + x03[t] <= 1  # 每队只能出现一次

    model.solve()

    # 结果
    pick_30 = [t for t in teams if x30[t].value() == 1]
    pick_adv = [t for t in teams if xadv[t].value() == 1]
    pick_03 = [t for t in teams if x03[t].value() == 1]

    best_score = pulp.value(model.objective)

    return pick_30, pick_adv, pick_03, best_score


# -------------------------
# 主程序
# -------------------------
results, total = parse_simulation_results(file_path)
teams, p3, padv, p03 = compute_bucket_probabilities(results, total)
pick30, pickadv, pick03, score = solve_pickem(teams, p3, padv, p03)

print("最优 Pick'Em（基于模拟后验）")
print("3-0:", pick30)
print("晋级:", pickadv)
print("0-3:", pick03)
print("期望正确数 =", score)

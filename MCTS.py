# beam_mcts_enhanced.py
from collections import defaultdict
import re
import random
import math
import time
import numpy as np
import matplotlib

# 使用无 GUI 的后端以在服务器 / 无界面环境运行
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import itertools

# -------------------- 配置 --------------------
FILE_PATH = "result.txt"

# Beam / MCTS 参数（可调整）
ITERATIONS = 1000        # MCTS 迭代次数
EXPLORATION_C = 1.5      # UCB 探索常数
PRINT_EVERY = 10         # 每多少次迭代打印一次
BEAM_SIZE = 2048         # 建议“大 Beam” — 你可改小一点看速度
RANDOM_SEED = 42
RANDOM_PERTURB_RATE = 0.08  # 对 beam 中的候选做随机扰动的概率（每个候选）
PERTURB_SWAP_PROB = 0.6     # 扰动时，做 swap（跨组交换）还是随机替换
PERTURB_SWAPS = 2           # 每次扰动做几次交换/替换操作

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -------------------- 解析与评估 --------------------
def parse_simulation_results(file_path: str):
    """解析模拟结果文件，返回组合频数及总次数"""
    results = defaultdict(int)
    pattern = r"3-0: (.*?) \| 3-1/3-2: (.*?) \| 0-3: (.*?): (\d+)/"
    total = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(pattern, line)
            if not match:
                continue
            three_zero = tuple(t.strip() for t in match.group(1).split(",") if t.strip())
            adv = tuple(t.strip() for t in match.group(2).split(",") if t.strip())
            zero_three = tuple(t.strip() for t in match.group(3).split(",") if t.strip())
            count = int(match.group(4))
            key = (three_zero, adv, zero_three)
            results[key] += count
            total += count
    return results, total

def compute_bucket_probabilities(results, total):
    """统计每队在各 bucket 的频率（作为启发）"""
    teams = set()
    for three_zero, adv, zero_three in results.keys():
        teams.update(three_zero)
        teams.update(adv)
        teams.update(zero_three)
    teams = sorted(teams)
    p_3_0 = {t: 0.0 for t in teams}
    p_adv = {t: 0.0 for t in teams}
    p_0_3 = {t: 0.0 for t in teams}
    for (three_zero, adv, zero_three), count in results.items():
        for team in three_zero:
            p_3_0[team] += count
        for team in adv:
            p_adv[team] += count
        for team in zero_three:
            p_0_3[team] += count
    for team in teams:
        p_3_0[team] /= total
        p_adv[team] /= total
        p_0_3[team] /= total
    return teams, p_3_0, p_adv, p_0_3

def evaluate_combination(combo, results):
    """
    评估单个组合在模拟结果中的表现：
    返回 (correct_counts_list, prob_ge5)
    """
    correct_counts = []
    total_simulations = sum(results.values())
    combo_three_zero = set(combo["3-0"])
    combo_adv = set(combo["3-1/3-2"])
    combo_zero_three = set(combo["0-3"])
    for (three_zero, adv, zero_three), count in results.items():
        correct = 0
        correct += len(combo_three_zero & set(three_zero))
        correct += len(combo_adv & set(adv))
        correct += len(combo_zero_three & set(zero_three))
        if count <= 0:
            continue
        correct_counts.extend([correct] * count)
    counts_np = np.array(correct_counts) if correct_counts else np.array([])
    prob_ge5 = float(np.mean(counts_np >= 5)) if total_simulations else 0.0
    return counts_np.tolist(), prob_ge5

def plot_distribution(correct_counts, best_probability, filename="correct_counts_distribution_enhanced.svg"):
    """绘图并保存"""
    if not correct_counts:
        return
    plt.figure(figsize=(10, 6))
    counts, bins = np.histogram(correct_counts, bins=range(12), density=True)
    colors = ["red" if bin_edge < 5 else "green" for bin_edge in bins[:-1]]
    plt.bar(bins[:-1], counts, width=0.8, alpha=0.7, color=colors)
    expected_value = np.mean(correct_counts)
    plt.axvline(x=5, color="r", linestyle="--", label=f"P(X>=5)={best_probability:.4f}")
    plt.axvline(x=expected_value, color="blue", linestyle=":", label=f"E[X]={expected_value:.2f}")
    plt.xlabel("X")
    plt.ylabel("Freq")
    plt.xticks(range(11))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, format="svg")
    plt.close()

# -------------------- Beam 生成（larger + heuristic） --------------------
def random_weighted_choice(pool, probs, k):
    chosen = set()
    # 退化处理
    pool = list(pool)
    weights = [probs.get(x, 1e-12) for x in pool]
    while len(chosen) < k:
        t = random.choices(pool, weights=weights, k=1)[0]
        chosen.add(t)
    return list(chosen)

def generate_beam_candidates(teams, p3, padv, p03, beam_size=BEAM_SIZE):
    """
    生成 beam 候选：保留启发式 top pools，但扩大 top_k，并对 adv 组合做更多采样。
    """
    n = len(teams)
    # 放宽池大小（比之前更宽）
    top30_k = min(12, n)
    top03_k = min(12, n)
    topadv_k = min(14, n)

    sorted_by_p3 = sorted(teams, key=lambda t: p3.get(t, 0.0), reverse=True)
    sorted_by_p03 = sorted(teams, key=lambda t: p03.get(t, 0.0), reverse=True)
    sorted_by_padv = sorted(teams, key=lambda t: padv.get(t, 0.0), reverse=True)

    pool_30 = sorted_by_p3[:top30_k]
    pool_03 = sorted_by_p03[:top03_k]
    pool_adv = sorted_by_padv[:topadv_k]

    candidates = []
    # 穷举 3-0 pair 和 0-3 pair（扩大 adv sampling）
    for three_zero in itertools.combinations(pool_30, 2):
        for zero_three in itertools.combinations(pool_03, 2):
            used_initial = set(three_zero) | set(zero_three)
            adv_pool = [t for t in pool_adv if t not in used_initial]
            if len(adv_pool) < 6:
                adv_pool_full = [t for t in sorted_by_padv if t not in used_initial]
                if len(adv_pool_full) < 6:
                    continue
                adv_choices_iter = itertools.combinations(adv_pool_full[:max(len(adv_pool_full), 6)], 6)
            else:
                adv_choices_iter = itertools.combinations(adv_pool, 6)

            # 增加 adv sampling 数量（比之前更多）
            adv_count = 0
            for adv in adv_choices_iter:
                cand_three = tuple(sorted(three_zero))
                cand_zero = tuple(sorted(zero_three))
                cand_adv = tuple(sorted(adv))
                # heuristic score
                eps = 1e-12
                score = 0.0
                for t in cand_three:
                    score += math.log(p3.get(t, eps) + eps)
                for t in cand_zero:
                    score += math.log(p03.get(t, eps) + eps)
                for t in cand_adv:
                    score += math.log(padv.get(t, eps) + eps)
                candidates.append((score, (cand_three, cand_adv, cand_zero)))
                adv_count += 1
                if adv_count >= 60:  # 每对 pair 的 adv 采样上限（扩大）
                    break

    # 若候选太少，随机补充
    if len(candidates) < beam_size:
        attempts = 0
        while len(candidates) < beam_size and attempts < beam_size * 6:
            three_zero = tuple(sorted(random.sample(teams, 2)))
            zero_three = tuple(sorted(random.sample([t for t in teams if t not in three_zero], 2)))
            adv_candidates = tuple(sorted(random.sample([t for t in teams if t not in three_zero and t not in zero_three], 6)))
            eps = 1e-12
            score = sum(math.log(p3.get(t, eps) + eps) for t in three_zero) + \
                    sum(math.log(padv.get(t, eps) + eps) for t in adv_candidates) + \
                    sum(math.log(p03.get(t, eps) + eps) for t in zero_three)
            candidates.append((score, (three_zero, adv_candidates, zero_three)))
            attempts += 1

    # 取 top beam_size（但不过度去重——只排除完全重复与冲突）
    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
    selected = []
    seen = set()
    for score, (three, adv, zero) in candidates:
        key = (tuple(three), tuple(adv), tuple(zero))
        if key in seen:
            continue
        # 只做基础冲突检查（三组内部应互不相交）
        if set(three) & set(adv) or set(three) & set(zero) or set(adv) & set(zero):
            continue
        selected.append((tuple(three), tuple(adv), tuple(zero)))
        seen.add(key)
        if len(selected) >= beam_size:
            break

    return selected

# -------------------- 随机扰动（对 beam 做多样性增强） --------------------
def perturb_beam_candidates(candidates, teams, rate=RANDOM_PERTURB_RATE, swaps=PERTURB_SWAPS, swap_prob=PERTURB_SWAP_PROB):
    """
    对部分 beam candidate 做随机扰动：
      - 以概率 rate 对候选进行扰动
      - 扰动操作：随机 swap（跨组交换两个队）或随机替换某组里某支队伍为未使用队伍
    返回新的 candidates list（长度不变，顺序可能改变）
    """
    new_cands = candidates.copy()
    n = len(new_cands)
    for i in range(n):
        if random.random() >= rate:
            continue
        three, adv, zero = list(new_cands[i][0:1])[0] if False else new_cands[i]  # noqa: keep signature
        three = list(new_cands[i][0])
        adv = list(new_cands[i][1])
        zero = list(new_cands[i][2])
        for _ in range(swaps):
            if random.random() < swap_prob:
                # swap: 在两个不同组之间交换两个元素
                g1, g2 = random.sample(["3-0", "3-1/3-2", "0-3"], 2)
                if g1 == "3-0":
                    a_list = three
                elif g1 == "0-3":
                    a_list = zero
                else:
                    a_list = adv
                if g2 == "3-0":
                    b_list = three
                elif g2 == "0-3":
                    b_list = zero
                else:
                    b_list = adv
                if len(a_list) == 0 or len(b_list) == 0:
                    continue
                ia = random.randrange(len(a_list))
                ib = random.randrange(len(b_list))
                a_list[ia], b_list[ib] = b_list[ib], a_list[ia]
            else:
                # replace：从未使用队伍中随机替换一支
                used = set(three + adv + zero)
                unused = [t for t in teams if t not in used]
                if not unused:
                    continue
                target_group = random.choice(["3-0", "3-1/3-2", "0-3"])
                if target_group == "3-0":
                    idx = random.randrange(len(three))
                    three[idx] = random.choice(unused)
                elif target_group == "0-3":
                    idx = random.randrange(len(zero))
                    zero[idx] = random.choice(unused)
                else:
                    idx = random.randrange(len(adv))
                    adv[idx] = random.choice(unused)
        # fix order and ensure validity (deduplicate and re-fill if needed)
        # keep unique within groups; if duplicate arose, replace duplicates with unused
        def fix_group(glist, size):
            gset = []
            for t in glist:
                if t not in gset:
                    gset.append(t)
            # fill if lacking
            used_now = set(gset)
            pool = [t for t in teams if t not in used_now]
            while len(gset) < size and pool:
                gset.append(pool.pop(0))
            return tuple(sorted(gset[:size]))
        three_t = fix_group(three, 2)
        zero_t = fix_group(zero, 2)
        adv_t = fix_group(adv, 6)
        # final conflict check: if conflicts remain (rare), skip perturbation
        if set(three_t) & set(zero_t) or set(three_t) & set(adv_t) or set(adv_t) & set(zero_t):
            continue
        new_cands[i] = (three_t, adv_t, zero_t)
    # shuffle beam a bit to avoid ordering bias
    random.shuffle(new_cands)
    return new_cands

# -------------------- MCTS（beam-limited） --------------------
class MCTSNode:
    def __init__(self, picked_3_0=frozenset(), picked_0_3=frozenset(), picked_adv=frozenset(),
                 parent=None, beam_candidates=None):
        self.picked_3_0 = picked_3_0
        self.picked_0_3 = picked_0_3
        self.picked_adv = picked_adv
        self.parent = parent
        self.children = {}
        self._untried_actions = None
        self.visits = 0
        self.total_reward = 0.0
        self.beam_candidates = beam_candidates if beam_candidates is not None else []

    def state_key(self):
        return (tuple(sorted(self.picked_3_0)), tuple(sorted(self.picked_0_3)), tuple(sorted(self.picked_adv)))

    def is_terminal(self):
        return (len(self.picked_3_0) == 2) and (len(self.picked_0_3) == 2) and (len(self.picked_adv) == 6)

    def current_group(self):
        if len(self.picked_3_0) < 2:
            return "3-0"
        if len(self.picked_0_3) < 2:
            return "0-3"
        return "adv"

    def _consistent_candidates(self):
        """
        重要更改：只检查冲突，不做子集匹配
        条件：candidate 在任何组里都不能包含已被选为其他组的队伍（无冲突）
        这比强制 candidate 包含 picked 更宽松，能带来更多可行解
        """
        res = []
        p30 = set(self.picked_3_0)
        p03 = set(self.picked_0_3)
        pad = set(self.picked_adv)
        for three, adv, zero in self.beam_candidates:
            # candidate groups must be internally disjoint (already ensured), and must not assign a team to two different groups
            # We'll allow candidate even if it doesn't include current picks, as long as it doesn't conflict:
            conflict = False
            # If any team already picked in group A appears in candidate's different group -> conflict
            if any(t in adv for t in p30) or any(t in zero for t in p30):
                conflict = True
            if any(t in three for t in p03) or any(t in adv for t in p03):
                conflict = True
            if any(t in three for t in pad) or any(t in zero for t in pad):
                conflict = True
            if not conflict:
                res.append((three, adv, zero))
        return res

    def available_actions(self, all_teams):
        if self._untried_actions is not None:
            return list(self._untried_actions)
        used = set(self.picked_3_0) | set(self.picked_0_3) | set(self.picked_adv)
        group = self.current_group()
        consistent = self._consistent_candidates()
        possible = set()
        for three, adv, zero in consistent:
            if group == "3-0":
                possible.update(set(three) - used)
            elif group == "0-3":
                possible.update(set(zero) - used)
            else:
                possible.update(set(adv) - used)
        if not possible:
            possible = set([t for t in all_teams if t not in used])
        actions = list(possible)
        self._untried_actions = set(actions)
        return actions

    def expand(self, action, all_teams):
        group = self.current_group()
        if group == "3-0":
            new_30 = frozenset(set(self.picked_3_0) | {action})
            new_node = MCTSNode(new_30, self.picked_0_3, self.picked_adv, parent=self, beam_candidates=self.beam_candidates)
        elif group == "0-3":
            new_03 = frozenset(set(self.picked_0_3) | {action})
            new_node = MCTSNode(self.picked_3_0, new_03, self.picked_adv, parent=self, beam_candidates=self.beam_candidates)
        else:
            new_adv = frozenset(set(self.picked_adv) | {action})
            new_node = MCTSNode(self.picked_3_0, self.picked_0_3, new_adv, parent=self, beam_candidates=self.beam_candidates)
        if self._untried_actions is None:
            self._untried_actions = set(self.available_actions(all_teams))
        self._untried_actions.discard(action)
        self.children[action] = new_node
        return new_node

    def best_child(self, c=EXPLORATION_C):
        choices = []
        for act, child in self.children.items():
            if child.visits == 0:
                return child
            exploit = child.total_reward / child.visits
            explore = c * math.sqrt(math.log(self.visits) / child.visits)
            choices.append((exploit + explore, child))
        return max(choices, key=lambda x: x[0])[1] if choices else None

    def tree_policy(self, all_teams):
        node = self
        while not node.is_terminal():
            actions = node.available_actions(all_teams)
            if node._untried_actions and len(node._untried_actions) > 0:
                action = random.choice(list(node._untried_actions))
                node = node.expand(action, all_teams)
            else:
                next_node = node.best_child()
                if next_node is None:
                    action = random.choice(actions)
                    node = node.expand(action, all_teams)
                else:
                    node = next_node
        return node

    def default_policy(self, results, total, all_teams):
        """
        Rollout: 如果有一致的 beam candidate（只按冲突检查），随机选一个 candidate 完整补全
        否则随机补全（fallback）
        """
        consistent = self._consistent_candidates()
        if consistent:
            three, adv, zero = random.choice(consistent)
            combo = {"3-0": list(three), "3-1/3-2": list(adv), "0-3": list(zero)}
            _, prob_ge5 = evaluate_combination(combo, results)
            return prob_ge5

        # fallback: 随机补全
        picked_3_0 = set(self.picked_3_0)
        picked_0_3 = set(self.picked_0_3)
        picked_adv = set(self.picked_adv)
        used = picked_3_0 | picked_0_3 | picked_adv
        remaining = [t for t in all_teams if t not in used]
        need_30 = 2 - len(picked_3_0)
        need_03 = 2 - len(picked_0_3)
        need_adv = 6 - len(picked_adv)
        choice = remaining.copy()
        random.shuffle(choice)
        idx = 0
        for _ in range(need_30):
            if idx < len(choice):
                picked_3_0.add(choice[idx]); idx += 1
        for _ in range(need_03):
            if idx < len(choice):
                picked_0_3.add(choice[idx]); idx += 1
        for _ in range(need_adv):
            if idx < len(choice):
                picked_adv.add(choice[idx]); idx += 1
        combo = {"3-0": list(picked_3_0), "3-1/3-2": list(picked_adv), "0-3": list(picked_0_3)}
        _, prob_ge5 = evaluate_combination(combo, results)
        return prob_ge5

    def backup(self, reward):
        node = self
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

def mcts_search_on_beam(results, total, all_teams, beam_candidates, iterations=ITERATIONS):
    root = MCTSNode(frozenset(), frozenset(), frozenset(), parent=None, beam_candidates=beam_candidates)
    best_combo = None
    best_value = -1.0
    start = time.time()

    for it in range(1, iterations + 1):
        node = root.tree_policy(all_teams)
        reward = node.default_policy(results, total, all_teams)
        node.backup(reward)
        if node.is_terminal():
            combo = {"3-0": list(node.picked_3_0), "3-1/3-2": list(node.picked_adv), "0-3": list(node.picked_0_3)}
            _, prob_ge5 = evaluate_combination(combo, results)
            if prob_ge5 > best_value:
                best_value = prob_ge5
                best_combo = combo
        if it % PRINT_EVERY == 0:
            elapsed = time.time() - start
            top_children = sorted(root.children.items(), key=lambda kv: kv[1].visits if kv[1].visits else 0, reverse=True)[:4]
            top_summary = []
            for act, ch in top_children:
                top_summary.append(f"{act}(v={ch.visits},r={ch.total_reward:.3f})")
            print(f"[MCTS] iter {it}/{iterations} elapsed {elapsed:.2f}s, root.visits={root.visits}, best_prob={best_value:.6f}, top_root_children={top_summary}")
        # periodic exhaustive scan of beam candidates to ensure fallback improvement
        if it % 500 == 0:
            for three, adv, zero in beam_candidates:
                combo = {"3-0": list(three), "3-1/3-2": list(adv), "0-3": list(zero)}
                _, probv = evaluate_combination(combo, results)
                if probv > best_value:
                    best_value = probv
                    best_combo = combo

    duration = time.time() - start
    if best_combo is None:
        # fallback to best in beam
        local_best_v = -1.0
        local_best = None
        for three, adv, zero in beam_candidates:
            combo = {"3-0": list(three), "3-1/3-2": list(adv), "0-3": list(zero)}
            _, probv = evaluate_combination(combo, results)
            if probv > local_best_v:
                local_best_v = probv
                local_best = combo
        best_combo = local_best
        best_value = local_best_v

    correct_counts, prob_ge5 = evaluate_combination(best_combo, results)
    expected_correct = float(np.mean(correct_counts)) if correct_counts else 0.0
    return best_combo, prob_ge5, expected_correct, duration

# -------------------- 主流程 --------------------
def main():
    results, total = parse_simulation_results(FILE_PATH)
    if total == 0:
        print("result.txt 中没有有效数据，无法运行。")
        return

    teams, p3, padv, p03 = compute_bucket_probabilities(results, total)
    print(f"共 {len(teams)} 支队伍，生成 Beam（size={BEAM_SIZE}）...")
    beam_candidates = generate_beam_candidates(teams, p3, padv, p03, beam_size=BEAM_SIZE)
    print(f"生成候选组合 {len(beam_candidates)} 个 (<= {BEAM_SIZE})，示例前 3 个：")
    for i, c in enumerate(beam_candidates[:3]):
        three, adv, zero = c
        print(f"  #{i+1}: 3-0={three} | adv(len={len(adv)})={adv} | 0-3={zero}")

    # 对 beam 进行随机扰动（增加多样性）
    print(f"对 beam 进行扰动：rate={RANDOM_PERTURB_RATE}, swaps={PERTURB_SWAPS}, swap_prob={PERTURB_SWAP_PROB}")
    beam_candidates = perturb_beam_candidates(beam_candidates, teams, rate=RANDOM_PERTURB_RATE,
                                              swaps=PERTURB_SWAPS, swap_prob=PERTURB_SWAP_PROB)
    print("扰动后示例前 3 个：")
    for i, c in enumerate(beam_candidates[:3]):
        three, adv, zero = c
        print(f"  #{i+1}: 3-0={three} | adv(len={len(adv)})={adv} | 0-3={zero}")

    print(f"开始在 Beam 上运行 MCTS：iterations={ITERATIONS}, exploration_c={EXPLORATION_C}, print_every={PRINT_EVERY}")
    best_combo, prob_ge5, expected_correct, duration = mcts_search_on_beam(results, total, teams, beam_candidates, iterations=ITERATIONS)

    # 绘图、打印（保持和以前输出一致）
    correct_counts, prob = evaluate_combination(best_combo, results)
    plot_distribution(correct_counts, prob, filename="correct_counts_distribution_enhanced.svg")

    print("\n最优 Pick'Em（Enhanced Beam + MCTS）")
    print(f"预测正确数 >= 5 的概率 = {prob_ge5:.6f}")
    print("  3-0 晋级:", ", ".join(sorted(best_combo["3-0"])))
    print("  3-1/3-2 晋级:", ", ".join(sorted(best_combo["3-1/3-2"])))
    print("  0-3 淘汰:", ", ".join(sorted(best_combo["0-3"])))
    print(f"期望正确数 = {expected_correct:.6f}")
    print(f"MCTS 用时: {duration:.2f} 秒（{ITERATIONS} 次迭代）")

if __name__ == "__main__":
    main()

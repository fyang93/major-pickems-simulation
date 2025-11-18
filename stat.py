"""
统计 result.txt 中每支队伍在 3-0 / 3-1&3-2 / 0-3 的出现频率
"""

import re
from collections import defaultdict

FILE_PATH = "result.txt"

def parse_and_count(file_path):

    # 计数字典
    count_3_0 = defaultdict(int)
    count_adv = defaultdict(int)
    count_0_3 = defaultdict(int)

    pattern = r"3-0: (.*?) \| 3-1/3-2: (.*?) \| 0-3: (.*?): (\d+)/"

    total_sim = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.match(pattern, line)
            if not m:
                continue

            three_zero = [t.strip() for t in m.group(1).split(",") if t.strip()]
            adv = [t.strip() for t in m.group(2).split(",") if t.strip()]
            zero_three = [t.strip() for t in m.group(3).split(",") if t.strip()]
            count = int(m.group(4))

            total_sim += count

            for t in three_zero:
                count_3_0[t] += count
            for t in adv:
                count_adv[t] += count
            for t in zero_three:
                count_0_3[t] += count

    return count_3_0, count_adv, count_0_3, total_sim


def print_table(counts, total, title):
    print(f"\n====== {title} ======")
    print(f"{'队伍':20s} {'次数':10s} {'占比(%)'}")
    print("-" * 40)
    for team, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"{team:20s} {cnt:10d} {cnt / total * 100:6.2f}")


if __name__ == "__main__":
    c3, cadv, c0, total = parse_and_count(FILE_PATH)

    print(f"模拟总次数：{total}\n")

    print_table(c3,  total, "3-0 出现频率")
    print_table(cadv, total, "3-1/3-2 出现频率")
    print_table(c0,  total, "0-3 出现频率")

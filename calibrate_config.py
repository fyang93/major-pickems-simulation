from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from config import Team, load_teams
from scipy.optimize import least_squares


@dataclass(frozen=True)
class Observation:
    team_a: Team
    team_b: Team
    probability: float


@dataclass(frozen=True)
class CalibrationResult:
    vrs_weight: float
    hltv_weight: float
    sigma: float
    hltv_exp: float
    loss: float


def predict_probability(
    team_a: Team,
    team_b: Team,
    sigma_value: float,
    hltv_exp: float,
    vrs_weight: float,
    hltv_weight: float,
) -> float:
    v1, h1 = team_a.rating[0], team_a.rating[1]
    v2, h2 = team_b.rating[0], team_b.rating[1]

    sigma_value = max(1.0, sigma_value)

    # VRS (Elo)
    p_vrs = 1.0 / (1.0 + 10.0 ** ((v2 - v1) / sigma_value))

    # HLTV
    if h1 <= 0 or h2 <= 0:
        p_hltv = 0.5
    else:
        p_hltv = 1.0 / (1.0 + (h2 / h1) ** hltv_exp)

    total_weight = max(vrs_weight + hltv_weight, 1e-9)
    return (vrs_weight * p_vrs + hltv_weight * p_hltv) / total_weight


def load_observations(csv_path: Path, team_lookup: dict[str, Team]) -> list[Observation]:
    observations: list[Observation] = []
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or reader.fieldnames[0] != "Team":
            raise ValueError("赔率文件的表头第一列必须命名为 'Team'")

        for row in reader:
            team_name = row.get("Team", "").strip()
            if not team_name:
                continue
            if team_name not in team_lookup:
                raise KeyError(f"在 JSON 中找不到队伍：{team_name}")
            team_a = team_lookup[team_name]

            for opponent, value in row.items():
                if opponent == "Team" or not value or value.strip() == "-":
                    continue
                if opponent not in team_lookup:
                    raise KeyError(f"在 JSON 中找不到对手：{opponent}")
                team_b = team_lookup[opponent]
                observations.append(Observation(team_a, team_b, float(value)))
    if not observations:
        raise ValueError("赔率文件没有有效的胜率数据，无法拟合。")
    return observations


def residuals(params, observations: Sequence[Observation]):
    """非线性最小二乘残差函数"""
    vrs_weight, sigma, hltv_exp = params
    hltv_weight = 1.0 - vrs_weight
    res = []
    for obs in observations:
        pred = predict_probability(obs.team_a, obs.team_b, sigma, hltv_exp, vrs_weight, hltv_weight)
        res.append(pred - obs.probability)
    return res


def fit_parameters(observations: Sequence[Observation], bounds=None, x0=None) -> CalibrationResult:
    """使用非线性最小二乘拟合参数"""
    if x0 is None:
        x0 = [0.5, 100.0, 1.0]  # 初始猜测
    if bounds is None:
        bounds = ([0.0, 1.0, 0.0], [1.0, 1000.0, 10.0])

    result = least_squares(residuals, x0=x0, bounds=bounds, args=(observations,))

    vrs_weight, sigma, hltv_exp = result.x
    hltv_weight = 1.0 - vrs_weight
    loss = (result.fun ** 2).mean()
    return CalibrationResult(vrs_weight, hltv_weight, sigma, hltv_exp, loss)


def summarize_result(result: CalibrationResult, observations: Sequence[Observation], samples: int = 8):
    print("\n=== 最优参数 (可粘贴到 config.py 11-15 行) ===")
    print(f"VRS_WEIGHT = {result.vrs_weight:.6f}")
    print(f"HLTV_WEIGHT = {result.hltv_weight:.6f}")
    print(f"SIGMA = {result.sigma:.6f}")
    print(f"HLTV_EXP = {result.hltv_exp:.6f}")
    print(f"\n均方误差 (MSE) = {result.loss:.8f}")

    print("对局（拟合 vs 实际）：")
    import random
    showcase = random.sample(observations, min(samples, len(observations)))
    for obs in showcase:
        pred = predict_probability(obs.team_a, obs.team_b, result.sigma, result.hltv_exp, result.vrs_weight, result.hltv_weight)
        print(f"{obs.team_a.name:>20} vs {obs.team_b.name:<20} 预测 {pred:.4f} | 真实 {obs.probability:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据部分胜率矩阵拟合 config.py 里的参数（非线性最小二乘）。")
    parser.add_argument("--teams-json", type=Path, default=Path("stage_1.json"))
    parser.add_argument("--odds-csv", type=Path, default=Path("odds.csv"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    import random
    random.seed(args.seed)

    teams = load_teams(str(args.teams_json))
    team_lookup = {team.name: team for team in teams}
    observations = load_observations(args.odds_csv, team_lookup)
    print(f"载入 {len(observations)} 条胜率数据用作拟合。")

    result = fit_parameters(observations)
    summarize_result(result, observations)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""위험 방향 Z-score, STD_FLOOR 보정, 6개 지표 평균 위험도 계산."""
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from trend_settings import STD_FLOOR, FEATURE_DIRECTIONS, FEATURE_GROUPS, RISK_Z_CAP

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if pd.isna(value):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def _denominator(feature: str, mean: float, std: float) -> float:
    """
    Z-score 분모를 계산한다.

    기존 문제:
    - std가 0은 아니지만 너무 작으면 작은 변화에도 z-score가 과도하게 커짐.
    - 특히 ttr_mean, repeated_ratio_mean 같은 값에서 민감하게 반응함.

    해결:
    - feature별 STD_FLOOR를 적용한다.
    """
    floor = STD_FLOOR.get(feature, 1e-6)

    if std is None:
        std = 0.0

    std = float(std)

    if std > floor:
        return std

    # 평균 기반 보정도 함께 고려
    mean_based_floor = abs(float(mean)) * 0.05

    return max(floor, mean_based_floor, 1e-6)


def _round_float(value: Any, digits: int = 4) -> float:
    return round(safe_float(value), digits)


def calculate_feature_score(
    feature: str,
    value: float,
    baseline_values: List[float],
    direction: str,
) -> Dict[str, float | str]:
    baseline_mean = float(np.mean(baseline_values)) if baseline_values else 0.0
    baseline_std = float(np.std(baseline_values, ddof=0)) if baseline_values else 0.0
    denom = _denominator(feature, baseline_mean, baseline_std)

    z_score = (value - baseline_mean) / denom

    if direction == "down":
        risk_z = -z_score
    else:
        risk_z = z_score

    risk_z = max(0.0, min(float(risk_z), RISK_Z_CAP))

    return {
        "value": float(value),
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "denominator": denom,
        "z_score": float(z_score),
        "risk_z": risk_z,
        "direction": direction,
        "weight": 1.0,
        "group": FEATURE_GROUPS.get(feature, "etc"),
    }


def calculate_daily_risk_score(
    feature_scores: Dict[str, Dict[str, Any]]
) -> float:
    """
    6개 Feature의 risk_z를 동일한 비중으로 평균한다.

    Daily Risk
    = 6개 Feature risk_z의 합 / 6
    """
    if not feature_scores:
        return 0.0

    target_features = list(FEATURE_DIRECTIONS.keys())

    # 반드시 6개 Feature가 모두 계산됐는지 확인
    missing_features = [
        feature
        for feature in target_features
        if feature not in feature_scores
    ]

    if missing_features:
        raise ValueError(
            "Daily Risk 계산에 필요한 Feature가 누락되었습니다: "
            f"{missing_features}"
        )

    risk_z_values = [
        safe_float(feature_scores[feature].get("risk_z", 0.0))
        for feature in target_features
    ]

    return float(sum(risk_z_values) / 6.0)

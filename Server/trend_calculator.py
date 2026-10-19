# -*- coding: utf-8 -*-
"""Rolling baseline, 위험 Z-score, EWMA와 최근 추세를 통합 계산한다."""
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from trend_settings import (
    WINDOW_SIZE, MIN_BASELINE_DAYS, TREND_DAYS, EWMA_ALPHA,
    FEATURE_DIRECTIONS, RISK_Z_THRESHOLD,
)
from risk_feature_scoring import (
    safe_float, _round_float, calculate_feature_score, calculate_daily_risk_score,
)
from risk_alerts import decide_alert_level, _make_main_reasons

def _trend_stats(values: List[float]) -> Dict[str, Any]:
    if len(values) < 2:
        return {
            "slope": 0.0,
            "increase_count": 0,
            "is_up": False,
            "window_values": [round(float(v), 4) for v in values],
        }

    clean_values = [float(v) for v in values]
    diffs = np.diff(clean_values)
    increase_count = int(np.sum(diffs > 0))

    x = np.arange(len(clean_values))
    slope = float(np.polyfit(x, clean_values, 1)[0]) if len(clean_values) >= 2 else 0.0

    # 최근 4일이면 증가 구간이 최대 3개.
    # 3개 중 2개 이상 증가하고 slope가 양수이면 상승 추세로 판단.
    is_up = bool(slope > 0 and increase_count >= max(1, len(clean_values) - 2))

    return {
        "slope": slope,
        "increase_count": increase_count,
        "is_up": is_up,
        "window_values": [round(float(v), 4) for v in clean_values],
    }


def analyze_daily_summaries(
    daily_df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    min_baseline_days: int = MIN_BASELINE_DAYS,
    trend_days: int = TREND_DAYS,
    alpha: float = EWMA_ALPHA,
    feature_directions: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    daily summary DataFrame을 입력받아 날짜별 anomaly/trend 결과를 반환한다.

    daily_df 필수 컬럼:
    - user_id
    - date
    - FEATURE_DIRECTIONS에 정의된 feature 중 일부 또는 전부
    """
    if daily_df is None or daily_df.empty:
        return []

    directions = feature_directions or FEATURE_DIRECTIONS
    required = {"user_id", "date"}
    missing = required - set(daily_df.columns)

    if missing:
        raise ValueError(f"daily_df 필수 컬럼 누락: {sorted(missing)}")

    df = daily_df.copy()
    df["date"] = df["date"].astype(str)
    df = df.sort_values(["user_id", "date"], ascending=True).reset_index(drop=True)

    # 사용할 feature를 숫자형으로 정리
    for feature in directions:
        if feature not in df.columns:
            df[feature] = 0.0
        df[feature] = pd.to_numeric(df[feature], errors="coerce").fillna(0.0)

    results: List[Dict[str, Any]] = []

    for user_id, user_df in df.groupby("user_id", sort=False):
        user_df = user_df.sort_values("date", ascending=True).reset_index(drop=True)

        user_results: List[Dict[str, Any]] = []
        prev_ewma: Optional[float] = None

        for idx, row in user_df.iterrows():
            start_idx = max(0, idx - window_size)
            baseline_df = user_df.iloc[start_idx:idx]

            if len(baseline_df) < min_baseline_days:
                continue

            feature_scores: Dict[str, Dict[str, Any]] = {}

            for feature, direction in directions.items():
                baseline_values = [
                    safe_float(v)
                    for v in baseline_df[feature].tolist()
                    if v is not None and not pd.isna(v)
                ]

                if len(baseline_values) < min_baseline_days:
                    continue

                value = safe_float(row.get(feature, 0.0))

                feature_scores[feature] = calculate_feature_score(
                    feature=feature,
                    value=value,
                    baseline_values=baseline_values,
                    direction=direction,
                )

            daily_risk_score = calculate_daily_risk_score(feature_scores)

            if prev_ewma is None:
                ewma_risk_score = daily_risk_score
            else:
                ewma_risk_score = alpha * daily_risk_score + (1.0 - alpha) * prev_ewma

            prev_ewma = ewma_risk_score

            risk_features = {
                feature: score
                for feature, score in feature_scores.items()
                if safe_float(score.get("risk_z", 0.0)) >= RISK_Z_THRESHOLD
            }

            risk_feature_count = len(risk_features)

            recent = user_results[-(trend_days - 1):] + [
                {
                    "daily_risk_score": daily_risk_score,
                    "ewma_risk_score": ewma_risk_score,
                }
            ]

            daily_trend = _trend_stats([safe_float(r["daily_risk_score"]) for r in recent])
            ewma_trend = _trend_stats([safe_float(r["ewma_risk_score"]) for r in recent])

            alert_level = decide_alert_level(
                daily_risk_score=daily_risk_score,
                ewma_risk_score=ewma_risk_score,
                risk_feature_count=risk_feature_count,
                daily_trend=daily_trend,
                ewma_trend=ewma_trend,
                history_count=len(recent),
                trend_days=trend_days,
            )

            main_reasons = _make_main_reasons(risk_features)

            result = {
                "user_id": str(user_id),
                "date": str(row.get("date")),
                "model_type": "rolling_zscore_ewma",
                "window_size": int(window_size),
                "min_baseline_days": int(min_baseline_days),
                "trend_days": int(trend_days),
                "alpha": float(alpha),
                "daily_risk_score": _round_float(daily_risk_score),
                "ewma_risk_score": _round_float(ewma_risk_score),
                "risk_feature_count": int(risk_feature_count),
                "alert_level": alert_level,
                "trend_direction": "up" if daily_trend.get("is_up") else "flat_or_down",
                "daily_trend": daily_trend,
                "ewma_trend": ewma_trend,
                "main_reasons": main_reasons,
                "feature_scores": feature_scores,
                "created_at": datetime.now().isoformat(),
            }

            user_results.append(result)
            results.append(result)

    return results


def latest_result(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not results:
        return None

    return sorted(
        results,
        key=lambda x: (x.get("user_id", ""), x.get("date", "")),
    )[-1]

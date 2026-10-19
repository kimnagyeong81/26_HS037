# -*- coding: utf-8 -*-
"""Jetson 감정 점수의 유효성 검사와 하루 감정 지표 계산."""
from __future__ import annotations
import math
from itertools import combinations
from typing import Any, Dict, Optional
import numpy as np
from trend_settings import EMOTION_COLUMNS, EMOTION_SUM_FEATURES

EMOTIONS = ("joy", "calm", "anger", "fear", "sadness")


def extract_jetson_emotion_scores(event: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Jetson 5종 감정 점수가 모두 유효하면 반환하고, 없으면 None을 반환한다.

    `emotion_source='jetson'` 표시는 새 API/Consumer를 통해 저장한 이벤트의 필수 조건이다.
    구형 서버 모델의 기록과 기준선을 섞지 않기 위한 구분값이다.
    """
    if event.get("emotion_source") != "jetson":
        return None
    if event.get("emotion_scores_available") is False:
        return None

    result = {}
    for emotion in EMOTIONS:
        raw = event.get(f"emotion_{emotion}")
        if isinstance(raw, bool) or raw is None:
            return None
        try:
            value = float(raw)
        except (ValueError, TypeError, OverflowError):
            return None
        if not math.isfinite(value) or not 0 <= value <= 100:
            return None
        result[emotion] = value
    return result


def calculate_emotion_gap_features(
    emotion_sums: Dict[str, Optional[float]],
    valid_emotion_count: int = 1,
) -> Dict[str, Optional[float]]:
    """감정 비율·혼재도·감정 사이 차이를 계산한다.

    비율/혼재도는 기존 계산과 동일하며, 감정 사이 차이는 유효 발화당 평균 점수
    기반으로 변경하여 발화 수에 따른 크기 왜곡을 방지한다.
    """
    fields = (
        "emotion_gap_mean", "emotion_gap_std", "emotion_gap_min", "emotion_gap_max",
        "positive_emotion_ratio", "negative_emotion_ratio",
        "dominant_emotion_ratio", "emotion_mixed_score",
    )
    if valid_emotion_count <= 0:
        return {key: None for key in fields}

    raw_sums = [emotion_sums.get(f"emotion_{name}_sum") for name in EMOTIONS]
    if any(value is None for value in raw_sums):
        return {key: None for key in fields}
    values = [float(value) for value in raw_sums]
    total = float(sum(values))
    if total <= 0:
        return {key: None for key in fields}

    avg = [value / valid_emotion_count for value in values]
    gaps = [abs(a - b) for a, b in combinations(avg, 2)]
    gap_mean = float(np.mean(gaps))
    joy_sum, calm_sum, anger_sum, fear_sum, sadness_sum = values
    positive_ratio = (joy_sum + calm_sum) / total
    negative_ratio = (anger_sum + fear_sum + sadness_sum) / total
    dominant_ratio = max(values) / total
    # 기존 1-gap/total의 혼재도 수치를 유지한다.
    normalized_gap = gap_mean / (total / valid_emotion_count)
    mixed_score = max(0.0, 1.0 - normalized_gap)

    return {
        "emotion_gap_mean": gap_mean,
        "emotion_gap_std": float(np.std(gaps)),
        "emotion_gap_min": float(np.min(gaps)),
        "emotion_gap_max": float(np.max(gaps)),
        "positive_emotion_ratio": float(positive_ratio),
        "negative_emotion_ratio": float(negative_ratio),
        "dominant_emotion_ratio": float(dominant_ratio),
        "emotion_mixed_score": float(mixed_score),
    }

# -*- coding: utf-8 -*-
"""기존 위험 경보 기준(normal/watch/warning/risk) 및 주요 원인 문구 생성."""
from __future__ import annotations
from typing import Any, Dict, List
from trend_settings import TREND_DAYS, FEATURE_GROUPS, FEATURE_REASON
from risk_feature_scoring import safe_float

def decide_alert_level(
    daily_risk_score: float,
    ewma_risk_score: float,
    risk_feature_count: int,
    daily_trend: Dict[str, Any],
    ewma_trend: Dict[str, Any],
    history_count: int,
    trend_days: int = TREND_DAYS,
) -> str:
    """
    alert_level 판단 기준.

    normal:
    - 위험 feature가 거의 없고 점수도 낮음.

    watch:
    - 하루 단위 위험 신호가 일부 있음.
    - 단, 지속 추세가 강하지는 않음.

    warning:
    - daily trend와 EWMA trend가 모두 상승하고,
    - 위험 feature가 3개 이상이며,
    - daily 또는 EWMA 점수가 일정 수준 이상임.

    risk:
    - warning보다 더 강한 조건.
    """

    daily_up = bool(daily_trend.get("is_up"))
    ewma_up = bool(ewma_trend.get("is_up"))

    # trend 계산에 필요한 history가 부족한 경우 보수적으로 판단
    if history_count < trend_days:
        if daily_risk_score >= 1.5 and risk_feature_count >= 3:
            return "watch"
        return "normal"

    # 강한 위험
    if (
        risk_feature_count >= 5
        and daily_risk_score >= 1.5
        and ewma_risk_score >= 1.2
        and daily_up
        and ewma_up
    ):
        return "risk"

    # 지속 악화 warning
    if (
        risk_feature_count >= 3
        and daily_up
        and ewma_up
        and (daily_risk_score >= 1.0 or ewma_risk_score >= 0.8)
    ):
        return "warning"

    # EWMA가 높더라도 feature 개수가 충분할 때만 warning
    if ewma_risk_score >= 1.5 and risk_feature_count >= 3:
        return "warning"

    # 관찰 단계
    if daily_risk_score >= 1.0 or risk_feature_count >= 2:
        return "watch"

    return "normal"


def _make_main_reasons(
    risk_features: Dict[str, Dict[str, Any]],
    max_reasons: int = 5,
) -> List[str]:
    """
    feature별 주요 원인을 생성한다.

    감정 ratio는 positive/negative가 서로 반대 관계라 동시에 많이 나올 수 있으므로,
    emotion_ratio 그룹은 최대 1개만 노출한다.
    """

    sorted_items = sorted(
        risk_features.items(),
        key=lambda item: safe_float(item[1].get("risk_z", 0.0)),
        reverse=True,
    )

    reasons: List[str] = []
    used_limited_groups = set()

    for feature, score in sorted_items:
        group = score.get("group", FEATURE_GROUPS.get(feature, "etc"))

        if group == "emotion_ratio":
            if group in used_limited_groups:
                continue
            used_limited_groups.add(group)

        reason = FEATURE_REASON.get(feature, f"{feature} 지표가 위험 방향으로 변화했습니다.")
        reasons.append(reason)

        if len(reasons) >= max_reasons:
            break

    return reasons

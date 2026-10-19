# -*- coding: utf-8 -*-
"""MongoDB 발화 이벤트를 KST 기준 일별 인지·감정 특징값으로 변환한다."""
from __future__ import annotations
import math
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from trend_settings import EMOTION_COLUMNS, EMOTION_SUM_FEATURES, DAILY_FEATURE_COLUMNS
from emotion_features import extract_jetson_emotion_scores, calculate_emotion_gap_features

KST = ZoneInfo("Asia/Seoul")
LANGUAGE_COLUMNS = (
    "demonstrative_pronoun_count", "ttr", "repeated_ratio",
    "sentence_complexity_score",
)


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """누락값·NaN·무한대를 실제 점수 0과 구별한다."""
    if value is None or isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def parse_timestamp(value: Any) -> Optional[datetime]:
    """UTC/offset 문자열을 KST로 변환한다. timezone 없는 이전 기록은 KST로 가정."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=KST)
    return parsed.astimezone(KST)


def get_date_string(value: Any) -> Optional[str]:
    """명시적 YYYY-MM-DD는 그대로 사용하고 datetime은 KST 날짜로 변환."""
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str) and len(value.strip()) == 10:
        try:
            return date.fromisoformat(value.strip()).isoformat()
        except ValueError:
            pass
    parsed = parse_timestamp(value)
    return parsed.date().isoformat() if parsed else None


def is_utterance_event(event: Dict[str, Any]) -> bool:
    """사용자 발화만 집계하고 일별 이벤트·AI 응답은 제외한다."""
    return (
        (event.get("granularity") == "utterance"
         or event.get("event_type") == "dementia_utterance_analysis")
        and event.get("speaker") in (None, "user")
    )


def make_daily_summary(
    events: List[Dict[str, Any]],
    date: str,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """하루 유효 감정 점수는 Jetson 기록만 집계, 언어 지표는 전체 사용자 발화 평균."""
    utterances = [event for event in events if is_utterance_event(event)]
    summary: Dict[str, Any] = {"date": date, "event_count": len(utterances)}
    if user_id is not None:
        summary["user_id"] = user_id

    valid = [scores for event in utterances
             if (scores := extract_jetson_emotion_scores(event)) is not None]
    summary["emotion_event_count"] = len(valid)
    summary["missing_emotion_count"] = len(utterances) - len(valid)
    for source_column, target_column in zip(EMOTION_COLUMNS, EMOTION_SUM_FEATURES):
        emotion = source_column.removeprefix("emotion_")
        summary[target_column] = (
            float(sum(item[emotion] for item in valid)) if valid else None
        )
    summary.update(calculate_emotion_gap_features(
        {column: summary[column] for column in EMOTION_SUM_FEATURES}, len(valid)
    ))
    for column in LANGUAGE_COLUMNS:
        values = [number for event in utterances
                  if (number := safe_float(event.get(column))) is not None]
        summary[f"{column}_mean"] = float(np.mean(values)) if values else None
    return summary

def group_events_by_user_and_date(
    events: Iterable[Dict[str, Any]],
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    MongoDB 발화 이벤트를 user_id와 날짜 기준으로 묶습니다.
    """
    grouped: Dict[
        str,
        Dict[str, List[Dict[str, Any]]],
    ] = {}

    for event in events:
        if not is_utterance_event(event):
            continue

        user_id = str(
            event.get("user_id", "unknown_user")
        )

        date = (
            get_date_string(event.get("date"))
            or get_date_string(event.get("timestamp"))
        )

        if date is None:
            continue

        grouped.setdefault(user_id, {})
        grouped[user_id].setdefault(date, [])
        grouped[user_id][date].append(event)

    return grouped


def build_daily_summaries(
    events: Iterable[Dict[str, Any]],
) -> pd.DataFrame:
    """
    발화 이벤트 목록을 사용자별·날짜별 Daily Feature DataFrame으로 변환합니다.
    """
    grouped = group_events_by_user_and_date(events)

    rows: List[Dict[str, Any]] = []

    for user_id, date_events in grouped.items():
        for date, day_events in date_events.items():
            rows.append(
                make_daily_summary(
                    events=day_events,
                    date=date,
                    user_id=user_id,
                )
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "user_id",
                "date",
                *DAILY_FEATURE_COLUMNS,
            ]
        )

    daily_df = pd.DataFrame(rows)

    daily_df = daily_df.sort_values(
        ["user_id", "date"],
        ascending=True,
    ).reset_index(drop=True)

    return daily_df

# -*- coding: utf-8 -*-
"""사용자별 추세 분석 실행 진입점 (기존 run_for_user/CLI 호환)."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import pandas as pd
from trend_settings import MONGO_URI, DB_NAME, EVENT_COLLECTION, ANOMALY_COLLECTION, FEATURE_DIRECTIONS
from trend_mongodb import (
    get_mongo_client, load_events_from_mongodb, list_user_ids, save_results_to_mongodb,
)
from daily_feature_builder import (
    safe_float, parse_timestamp, get_date_string,
    is_utterance_event, make_daily_summary,
    group_events_by_user_and_date, build_daily_summaries,
)
from emotion_features import calculate_emotion_gap_features
from trend_monitor import analyze_daily_summaries

def run_for_user(
    user_id: str,
    window_size: int,
    min_baseline_days: int,
    trend_days: int,
    alpha: float,
    mongo_uri: str = MONGO_URI,
    db_name: str = DB_NAME,
    event_collection: str = EVENT_COLLECTION,
    anomaly_collection: str = ANOMALY_COLLECTION,
    save_result: bool = True,
) -> List[Dict[str, Any]]:
    """
    한 사용자의 발화 이벤트를 조회하여
    Daily Feature → Rolling Z-score → EWMA 분석을 수행합니다.
    """
    events = load_events_from_mongodb(
        user_id=user_id,
        mongo_uri=mongo_uri,
        db_name=db_name,
        collection_name=event_collection,
    )

    print(
        f"[Trend] 발화 이벤트 조회: "
        f"user_id={user_id}, count={len(events)}"
    )

    daily_df = build_daily_summaries(events)

    if daily_df.empty:
        print(
            f"[Trend] daily summary 없음: "
            f"user_id={user_id}"
        )
        return []

    print(
        f"[Trend] Daily Feature 생성: "
        f"user_id={user_id}, days={len(daily_df)}"
    )

    # 6개 위험 지표가 모두 있고 Jetson 감정 점수가 유효한 날짜만 사용한다.
    # 결측치를 0으로 채우면 false alert가 발생할 수 있으므로 사전에 제외한다.
    required_features = list(FEATURE_DIRECTIONS)
    available = (daily_df["emotion_event_count"].gt(0)
                 & daily_df[required_features].apply(
                     pd.to_numeric, errors="coerce"
                 ).notna().all(axis=1))
    skipped_count = int((~available).sum())
    if skipped_count:
        print(f"[Trend] 감정·언어 지표 미충족 날짜 제외: {skipped_count}일")
    daily_df = daily_df.loc[available].copy()
    if daily_df.empty:
        return []

    results = analyze_daily_summaries(
        daily_df=daily_df,
        window_size=window_size,
        min_baseline_days=min_baseline_days,
        trend_days=trend_days,
        alpha=alpha,
    )

    if save_result:
        saved = save_results_to_mongodb(
            results=results,
            mongo_uri=mongo_uri,
            db_name=db_name,
            collection_name=anomaly_collection,
        )

        print(
            "[Trend] MongoDB anomaly_results "
            f"저장/갱신: {saved}건"
        )

    if results:
        latest = results[-1]

        print(
            "[Trend] 최신 결과 "
            f"user_id={latest.get('user_id')} "
            f"date={latest.get('date')} "
            f"alert_level={latest.get('alert_level')} "
            f"daily_risk_score="
            f"{latest.get('daily_risk_score')} "
            f"ewma_risk_score="
            f"{latest.get('ewma_risk_score')}"
        )

    else:
        print(
            "[Trend] 기준 일수 부족으로 결과 없음 "
            f"user_id={user_id}, "
            f"min_baseline_days={min_baseline_days}"
        )

    return results



def parse_args():
    from trend_cli import parse_args as _parse_args
    return _parse_args()


def validate_args(args):
    from trend_cli import validate_args as _validate_args
    return _validate_args(args)


def main():
    from trend_cli import main as _main
    return _main()


if __name__ == "__main__":
    main()

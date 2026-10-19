# -*- coding: utf-8 -*-
"""추세 분석 명령행 인자 처리와 사용자별 실행 오케스트레이션."""
from __future__ import annotations
import argparse
from trend_settings import MONGO_URI, DB_NAME, EVENT_COLLECTION, ANOMALY_COLLECTION
from trend_mongodb import list_user_ids

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rolling Z-score + EWMA "
            "패턴 변화 감지"
        )
    )

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--user-id",
        help="분석할 user_id",
    )

    group.add_argument(
        "--all-users",
        action="store_true",
        help="전체 사용자 분석",
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=7,
        help="Rolling baseline window 일수",
    )

    parser.add_argument(
        "--min-baseline-days",
        type=int,
        default=7,
        help="최소 baseline 일수",
    )

    parser.add_argument(
        "--trend-days",
        type=int,
        default=4,
        help="추세 판단 일수",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="EWMA alpha",
    )

    parser.add_argument(
        "--mongo-uri",
        default=MONGO_URI,
        help=f"MongoDB URI, 기본값: {MONGO_URI}",
    )

    parser.add_argument(
        "--db-name",
        default=DB_NAME,
        help=f"MongoDB DB 이름, 기본값: {DB_NAME}",
    )

    parser.add_argument(
        "--event-collection",
        default=EVENT_COLLECTION,
        help=(
            "발화 이벤트 컬렉션 이름, "
            f"기본값: {EVENT_COLLECTION}"
        ),
    )

    parser.add_argument(
        "--anomaly-collection",
        default=ANOMALY_COLLECTION,
        help=(
            "결과 컬렉션 이름, "
            f"기본값: {ANOMALY_COLLECTION}"
        ),
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="MongoDB 저장 생략",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """입력 옵션을 검증합니다."""
    if args.window_size < 1:
        raise ValueError(
            "--window-size는 1 이상이어야 합니다."
        )

    if args.min_baseline_days < 1:
        raise ValueError(
            "--min-baseline-days는 "
            "1 이상이어야 합니다."
        )

    if args.min_baseline_days > args.window_size:
        raise ValueError(
            "--min-baseline-days는 "
            "--window-size보다 클 수 없습니다."
        )

    if args.trend_days < 2:
        raise ValueError(
            "--trend-days는 2 이상이어야 합니다."
        )

    if not 0 < args.alpha <= 1:
        raise ValueError(
            "--alpha는 0보다 크고 "
            "1 이하여야 합니다."
        )


def main() -> None:
    from inference_trend import run_for_user  # 진입점과의 순환 import 방지
    args = parse_args()
    validate_args(args)

    if args.all_users:
        user_ids = list_user_ids(
            mongo_uri=args.mongo_uri,
            db_name=args.db_name,
            collection_name=args.event_collection,
        )
    else:
        user_ids = [args.user_id]

    if not user_ids:
        print(
            "[Trend] 분석 대상 user_id가 없습니다."
        )
        return

    total_results = 0

    for user_id in user_ids:
        results = run_for_user(
            user_id=user_id,
            window_size=args.window_size,
            min_baseline_days=args.min_baseline_days,
            trend_days=args.trend_days,
            alpha=args.alpha,
            mongo_uri=args.mongo_uri,
            db_name=args.db_name,
            event_collection=args.event_collection,
            anomaly_collection=args.anomaly_collection,
            save_result=not args.no_save,
        )

        total_results += len(results)

    print(
        f"[Trend] 전체 결과 생성: "
        f"{total_results}건"
    )

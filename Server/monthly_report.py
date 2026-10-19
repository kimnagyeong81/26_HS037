# -*- coding: utf-8 -*-
"""
monthly_report.py

실행:
    python -m uvicorn dashboard_1month:app --reload --port 8001

접속:
    http://localhost:8001/report/U001/2026/8

설명:
    - dementia_db.events의 실제 발화 분석 필드를 월간 집계합니다.
    - dementia_db.anomaly_results의 실제 Rolling Z-score / EWMA 결과를 사용합니다.
    - 감정 구슬은 별도 실제 컬렉션 emotion_gauge.marble_events를 사용합니다.
    - 임의 난수/합성 지표를 만들지 않습니다.
"""

from __future__ import annotations

import json
import math
import os

from calendar import monthrange
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Optional
import shutil
import subprocess
import tempfile
from urllib.parse import quote
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)

from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    FileResponse,
)

from fastapi.templating import Jinja2Templates
from pymongo import MongoClient
from starlette.background import BackgroundTask

# =============================================================================
# 기본 설정
# =============================================================================

app = FastAPI(
    title="치매 조기 감지 월간 리포트"
)


BASE_DIR = Path(
    __file__
).resolve().parent


TEMPLATES_DIR = (
    BASE_DIR
    / "templates"
)

REPORT_INTERNAL_BASE_URL = os.getenv(
    "REPORT_INTERNAL_BASE_URL",
    "http://127.0.0.1:8502",
).rstrip("/")

if not TEMPLATES_DIR.is_dir():

    raise RuntimeError(
        f"templates 폴더가 없습니다: "
        f"{TEMPLATES_DIR}"
    )


if not (
    TEMPLATES_DIR
    / "report.html"
).is_file():

    raise RuntimeError(
        "templates/report.html "
        "파일이 없습니다."
    )


templates = Jinja2Templates(
    directory=str(
        TEMPLATES_DIR
    )
)


# =============================================================================
# MongoDB 설정
# =============================================================================

MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb://localhost:27017/",
)


DB_NAME = os.getenv(
    "DEMENTIA_DB_NAME",
    "dementia_db",
)


EVENT_COLLECTION = os.getenv(
    "EVENT_COLLECTION",
    "events",
)


ANOMALY_COLLECTION = os.getenv(
    "ANOMALY_COLLECTION",
    "anomaly_results",
)


EMOTION_DB_NAME = os.getenv(
    "EMOTION_DB_NAME",
    "emotion_gauge",
)


MARBLE_COLLECTION = os.getenv(
    "MARBLE_COLLECTION",
    "marble_events",
)


# =============================================================================
# 기본 리포트
# =============================================================================

DEFAULT_USER_ID = "U001"

DEFAULT_YEAR = 2026

DEFAULT_MONTH = 8


# 현재 report.html에
# 주차별 감정 트렌드 영역을
# 표시하도록 설정
SHOW_WEEKLY_EMOTION_TREND = True


# =============================================================================
# 감정 설정
# =============================================================================

EMOTION_ORDER = [
    "joy",
    "calm",
    "fear",
    "sadness",
    "anger",
]


EMOTION_KO = {

    "joy":
        "기쁨",

    "calm":
        "평온",

    "fear":
        "두려움",

    "sadness":
        "슬픔",

    "anger":
        "분노",
}


EMOTION_COLORS = {

    "joy":
        "#EAB308",

    "calm":
        "#22C55E",

    "fear":
        "#A855F7",

    "sadness":
        "#3B82F6",

    "anger":
        "#EF4444",
}


# anomaly_results.feature_scores의
# 실제 feature name
FEATURE_KO = {

    "demonstrative_pronoun_count_mean":
        "지시대명사",

    "ttr_mean":
        "어휘 다양성",

    "repeated_ratio_mean":
        "반복 발화",

    "sentence_complexity_score_mean":
        "문장 복잡도",

    "negative_emotion_ratio":
        "부정 감정",

    "positive_emotion_ratio":
        "긍정 감정",
}


RISK_Z_THRESHOLD = 1.6


# =============================================================================
# Jinja2 필터
# =============================================================================

def _pct_str(
    value: float | None,
    decimals: int = 1,
) -> str:

    if value is None:

        return "-"


    if value == 0:

        return (
            f"{0:.{decimals}f}%"
        )


    return (
        f"{value * 100:+.{decimals}f}%"
    )


def _grade_ko(
    grade: str,
) -> str:

    return {

        "caution":
            "⚠ 주의",

        "watch":
            "🟡 관찰",

        "ok":
            "✓ 정상",

    }.get(
        grade,
        "-",
    )


def _grade_css(
    grade: str,
) -> str:

    return {

        "caution":
            "badge-caution",

        "watch":
            "badge-watch",

        "ok":
            "badge-ok",

    }.get(
        grade,
        "",
    )


def _emotion_ko(
    emotion: str,
) -> str:

    return EMOTION_KO.get(
        emotion,
        emotion,
    )


def _emotion_color(
    emotion: str,
) -> str:

    return EMOTION_COLORS.get(
        emotion,
        "#64748B",
    )


templates.env.filters.update({

    "pct_str":
        _pct_str,

    "grade_ko":
        _grade_ko,

    "grade_css":
        _grade_css,

    "emotion_ko":
        _emotion_ko,

    "emotion_color":
        _emotion_color,

})


templates.env.globals.update({

    "emotion_order":
        EMOTION_ORDER,

    "emotion_names":
        EMOTION_KO,

    "emotion_colors":
        EMOTION_COLORS,

})


# =============================================================================
# MongoDB 연결
# =============================================================================

def _mongo_client() -> MongoClient:

    return MongoClient(

        MONGO_URI,

        serverSelectionTimeoutMS=5000,

    )


# =============================================================================
# 공통 유틸
# =============================================================================

def _safe_float(
    value: Any,
    default: Optional[float] = None,
) -> Optional[float]:

    if (
        value is None
        or isinstance(
            value,
            bool,
        )
    ):

        return default


    try:

        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):

        return default


    if not math.isfinite(
        result
    ):

        return default


    return result


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:

    if (
        value is None
        or isinstance(
            value,
            bool,
        )
    ):

        return default


    try:

        return int(
            value
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):

        return default


def _safe_datetime(
    value: Any,
) -> Optional[datetime]:

    if isinstance(
        value,
        datetime,
    ):

        return value


    if not isinstance(
        value,
        str,
    ):

        return None


    value = value.strip()


    if not value:

        return None


    # ISO 형식
    # 2026-10-16T...
    # 2026-10-16 10:00:00
    for candidate in (

        value,

        value.replace(
            "Z",
            "+00:00",
        ),

    ):

        try:

            return (
                datetime.fromisoformat(
                    candidate
                )
            )

        except ValueError:

            pass


    # 구형 테스트 데이터 대응
    for fmt in (

        "%Y-%m-%d %H:%M:%S",

        "%Y-%m-%d %H:%M:%S.%f",

        "%Y-%m-%d",

    ):

        try:

            return datetime.strptime(
                value,
                fmt,
            )

        except ValueError:

            pass


    return None


def _event_date(
    event: dict,
) -> Optional[str]:

    value = event.get(
        "date"
    )


    if (
        isinstance(
            value,
            str,
        )
        and len(value) >= 10
    ):

        return value[:10]


    timestamp = _safe_datetime(
        event.get(
            "timestamp"
        )
    )


    if timestamp is None:

        return None


    return timestamp.strftime(
        "%Y-%m-%d"
    )


def _change(
    value: float,
    baseline: float,
) -> Optional[float]:

    if baseline == 0:

        return None


    return (
        value
        - baseline
    ) / baseline


def _grade_directional(
    value: float,
    baseline: float,
    direction: int,
) -> str:
    """
    direction = +1
        값 증가가 악화 방향

    direction = -1
        값 감소가 악화 방향
    """

    change = _change(
        value,
        baseline,
    )


    if change is None:

        return "ok"


    adverse_change = (
        change
        * direction
    )


    if adverse_change >= 0.30:

        return "caution"


    if adverse_change >= 0.15:

        return "watch"


    return "ok"


def _metric(
    value: float,
    baseline: float,
    direction: int,
    decimals: int = 3,
) -> dict:

    change = _change(
        value,
        baseline,
    )


    return {

        "value":
            round(
                value,
                decimals,
            ),

        "baseline":
            round(
                baseline,
                decimals,
            ),

        "change":
            (
                round(
                    change,
                    4,
                )
                if change is not None
                else None
            ),

        "grade":
            _grade_directional(
                value,
                baseline,
                direction,
            ),

    }


def _prev_month(
    year: int,
    month: int,
) -> tuple[int, int]:

    if month == 1:

        return (
            year - 1,
            12,
        )


    return (
        year,
        month - 1,
    )


def _week(
    day: int,
) -> int:

    # 1~7
    # 8~14
    # 15~21
    # 22~말일

    return min(
        (day - 1)
        // 7
        + 1,
        4,
    )


def _month_strings(
    year: int,
    month: int,
) -> tuple[
    str,
    str,
    str,
]:

    last_day = monthrange(
        year,
        month,
    )[1]


    prefix = (
        f"{year}-"
        f"{month:02d}-"
    )


    start = (
        f"{year}-"
        f"{month:02d}-01"
    )


    end = (
        f"{year}-"
        f"{month:02d}-"
        f"{last_day:02d}"
    )


    return (
        prefix,
        start,
        end,
    )


def _mean_numeric(
    docs: Iterable[dict],
    field: str,
) -> float:

    values = []


    for doc in docs:

        value = _safe_float(
            doc.get(
                field
            )
        )


        if value is not None:

            values.append(
                value
            )


    return (
        mean(
            values
        )
        if values
        else 0.0
    )


def _format_timestamp(
    value: Any,
) -> str:

    ts = _safe_datetime(
        value
    )


    if ts is None:

        return "-"


    ampm = (
        "오전"
        if ts.hour < 12
        else "오후"
    )


    hour12 = (
        ts.hour
        % 12
        or 12
    )


    return (

        f"{ts.month}월 "
        f"{ts.day}일 "
        f"{ampm} "
        f"{hour12}:"
        f"{ts.minute:02d}"

    )


def _validate_period(
    year: int,
    month: int,
) -> None:

    if not (
        2020
        <= year
        <= 2099
    ):

        raise HTTPException(

            status_code=400,

            detail=(
                "year 범위 오류: "
                "2020~2099"
            ),

        )


    if not (
        1
        <= month
        <= 12
    ):

        raise HTTPException(

            status_code=400,

            detail=(
                "month 범위 오류: "
                "1~12"
            ),

        )


# =============================================================================
# MongoDB events 조회
# =============================================================================

def _load_events(
    user_id: str,
    year: int,
    month: int,
) -> list[dict]:
    """
    events의 date 필드를 기준으로
    한 달 데이터를 가져옵니다.

    실제 확인된 필드 예:

        event_type
        pipeline
        granularity
        user_id
        session_id
        timestamp
        text
        created_at
        date
        utterance_id
        speaker
        source_file

        emotion_joy
        emotion_calm
        emotion_anger
        emotion_fear
        emotion_sadness

        emotion_total_score
        emotion_positive_score
        emotion_negative_score
        emotion_label
        emotion_score

        demonstrative_pronoun_count
        demonstrative_pronoun_exists
        demonstrative_pronoun_words
        demonstrative_pronoun_word_counts

        total_tokens
        unique_tokens
        ttr
        repeated_ratio
        top_repeated_words

        reliability_score
        reliability_label

        repetition_exists
        repetition_total_tokens
        repetition_unique_tokens
        repetition_ratio
        repeated_words
        most_repeated_word
        most_repeated_count

        sentence_count
        average_sentence_length
        average_word_count
        average_connective_count
        average_clause_count

        sentence_complexity_score
        sentence_complexity_label
        sentence_details

        request_id
        audio_filename
        cnn_status
        cnn_available
        cnn_error
        cnn_result
    """

    _, start, end = (
        _month_strings(
            year,
            month,
        )
    )


    client = _mongo_client()


    try:

        collection = (
            client[
                DB_NAME
            ][
                EVENT_COLLECTION
            ]
        )


        docs = list(

            collection.find({

                "user_id":
                    user_id,

                "date": {

                    "$gte":
                        start,

                    "$lte":
                        end,

                },

            }).sort(
                "timestamp",
                1,
            )

        )


        return docs


    finally:

        client.close()


# =============================================================================
# anomaly_results 조회
# =============================================================================

def _load_anomaly_docs(
    user_id: str,
    year: int,
    month: int,
) -> list[dict]:

    _, start, end = (
        _month_strings(
            year,
            month,
        )
    )


    client = _mongo_client()


    try:

        collection = (
            client[
                DB_NAME
            ][
                ANOMALY_COLLECTION
            ]
        )


        return list(

            collection.find({

                "user_id":
                    user_id,

                "date": {

                    "$gte":
                        start,

                    "$lte":
                        end,

                },

            }).sort(
                "date",
                1,
            )

        )


    finally:

        client.close()


# =============================================================================
# 구슬 이벤트 조회
# =============================================================================

def _load_marble_events(
    user_id: str,
    year: int,
    month: int,
) -> list[dict]:
    """
    감정 구슬은 dementia_db.events와
    별도 데이터입니다.

    실제 emotion_gauge.marble_events를
    사용합니다.
    """

    client = _mongo_client()


    try:

        docs = list(

            client[
                EMOTION_DB_NAME
            ][
                MARBLE_COLLECTION
            ].find({

                "user_id":
                    user_id,

            })

        )


    finally:

        client.close()


    month_prefix = (
        f"{year}-"
        f"{month:02d}-"
    )


    result = []


    for doc in docs:

        date_str = None


        if isinstance(
            doc.get(
                "date"
            ),
            str,
        ):

            date_str = (
                doc[
                    "date"
                ][:10]
            )


        else:

            ts = _safe_datetime(
                doc.get(
                    "timestamp"
                )
            )


            if ts is not None:

                date_str = (
                    ts.strftime(
                        "%Y-%m-%d"
                    )
                )


        if (
            date_str
            and date_str.startswith(
                month_prefix
            )
        ):

            result.append(
                doc
            )


    return result


# =============================================================================
# 세션 통계
# =============================================================================

def _session_statistics(
    events: list[dict],
) -> tuple[
    float,
    float,
]:
    """
    반환

    total_minutes
    avg_response_delay_sec
    """

    grouped = defaultdict(
        list
    )


    for index, event in enumerate(
        events
    ):

        ts = _safe_datetime(
            event.get(
                "timestamp"
            )
        )


        if ts is None:

            continue


        session_id = str(

            event.get(
                "session_id"
            )
            or
            f"__single__{index}"

        )


        grouped[
            session_id
        ].append(
            ts
        )


    total_minutes = 0.0

    response_gaps = []


    for timestamps in (
        grouped.values()
    ):

        timestamps.sort()


        if len(
            timestamps
        ) < 2:

            continue


        duration_sec = (

            timestamps[-1]
            -
            timestamps[0]

        ).total_seconds()


        if duration_sec > 0:

            total_minutes += (
                duration_sec
                / 60.0
            )


        for before, after in zip(

            timestamps,

            timestamps[1:],

        ):

            gap_sec = (

                after
                -
                before

            ).total_seconds()


            # 기존 리포트 기준
            # 30초 이하 차이만
            # 응답 간격으로 사용
            if (
                0
                < gap_sec
                <= 30
            ):

                response_gaps.append(
                    gap_sec
                )


    avg_delay = (

        mean(
            response_gaps
        )

        if response_gaps

        else 0.0

    )


    return (
        total_minutes,
        avg_delay,
    )


# =============================================================================
# WPM 계산
# =============================================================================

def _calculate_wpm(
    events: list[dict],
) -> float:
    """
    실제 total_tokens와
    session timestamp를 이용합니다.

    session 시작/끝 시간이 같은 경우는
    WPM 계산에서 제외합니다.
    """

    grouped = defaultdict(
        list
    )


    for index, event in enumerate(
        events
    ):

        session_id = str(

            event.get(
                "session_id"
            )
            or
            f"__single__{index}"

        )


        grouped[
            session_id
        ].append(
            event
        )


    values = []


    for session_events in (
        grouped.values()
    ):

        timestamps = [

            _safe_datetime(
                event.get(
                    "timestamp"
                )
            )

            for event
            in session_events

        ]


        timestamps = [

            timestamp

            for timestamp
            in timestamps

            if timestamp
            is not None

        ]


        if len(
            timestamps
        ) < 2:

            continue


        duration_min = (

            max(
                timestamps
            )
            -
            min(
                timestamps
            )

        ).total_seconds() / 60.0


        if duration_min <= 0:

            continue


        total_tokens = sum(

            _safe_int(
                event.get(
                    "total_tokens"
                ),
                0,
            )

            for event
            in session_events

        )


        if total_tokens <= 0:

            continue


        values.append(

            total_tokens
            /
            duration_min

        )


    return (

        mean(
            values
        )

        if values

        else 0.0

    )


# =============================================================================
# 대화 활동
# =============================================================================

def _build_activity(
    events: list[dict],
    prev_events: list[dict],
    year: int,
    month: int,
) -> dict:

    total_days = monthrange(
        year,
        month,
    )[1]


    active_dates = sorted({

        date_str

        for event
        in events

        if (
            date_str
            :=
            _event_date(
                event
            )
        )

    })


    active_day_numbers = {

        int(
            date_str[-2:]
        )

        for date_str
        in active_dates

    }


    inactive_days = [

        day

        for day
        in range(
            1,
            total_days + 1,
        )

        if day
        not in active_day_numbers

    ]


    # 최장 연속 미대화
    longest = []

    current = []


    for day in range(
        1,
        total_days + 1,
    ):

        if day in active_day_numbers:

            current = []


        else:

            current.append(
                day
            )


            if len(
                current
            ) > len(
                longest
            ):

                longest = (
                    current.copy()
                )


    total_minutes, _ = (
        _session_statistics(
            events
        )
    )


    prev_total_minutes, _ = (
        _session_statistics(
            prev_events
        )
    )


    prev_active_dates = {

        date_str

        for event
        in prev_events

        if (
            date_str
            :=
            _event_date(
                event
            )
        )

    }


    avg_daily_minutes = (

        total_minutes
        /
        len(
            active_dates
        )

        if active_dates

        else 0.0

    )


    prev_avg_minutes = (

        prev_total_minutes
        /
        len(
            prev_active_dates
        )

        if prev_active_dates

        else 0.0

    )


    session_ids = {

        str(
            event.get(
                "session_id"
            )
        )

        for event
        in events

        if event.get(
            "session_id"
        )

    }


    prev_session_ids = {

        str(
            event.get(
                "session_id"
            )
        )

        for event
        in prev_events

        if event.get(
            "session_id"
        )

    }


    return {

        "active_days":
            len(
                active_dates
            ),

        "inactive_days":
            len(
                inactive_days
            ),

        "no_conversation_days":
            len(
                inactive_days
            ),

        "inactive_dates":
            inactive_days,

        "total_days":
            total_days,

        "session_count":
            len(
                session_ids
            ),

        "total_utterances":
            len(
                events
            ),

        "avg_daily_minutes":
            round(
                avg_daily_minutes,
                1,
            ),

        "total_minutes":
            round(
                total_minutes,
                1,
            ),

        "max_gap_days":
            len(
                longest
            ),

        "gap_start":
            (
                f"{month}/"
                f"{longest[0]}"
                if longest
                else None
            ),

        "gap_end":
            (
                f"{month}/"
                f"{longest[-1]}"
                if longest
                else None
            ),

        "prev_avg_minutes":
            round(
                prev_avg_minutes,
                1,
            ),

        "prev_session_count":
            len(
                prev_session_ids
            ),

        "avg_minutes_change":
            _change(
                avg_daily_minutes,
                prev_avg_minutes,
            ),

    }


# =============================================================================
# 반복 단어 Top 3
# =============================================================================

def _top_repeated_words(
    events: list[dict],
) -> list[dict]:

    counter = Counter()


    for event in events:

        repeated_words = (
            event.get(
                "repeated_words"
            )
        )


        if isinstance(
            repeated_words,
            dict,
        ):

            for word, count in (
                repeated_words.items()
            ):

                counter[
                    str(
                        word
                    )
                ] += _safe_int(
                    count,
                    0,
                )


    return [

        {
            "word":
                word,

            "count":
                count,
        }

        for word, count
        in counter.most_common(
            3
        )

    ]


def _most_common_label(
    events: list[dict],
    field: str,
) -> str:

    values = [

        str(
            event[
                field
            ]
        )

        for event
        in events

        if event.get(
            field
        )

    ]


    if not values:

        return "-"


    return Counter(
        values
    ).most_common(
        1
    )[0][0]


# =============================================================================
# 긍정/부정 감정 비율
# =============================================================================

def _emotion_ratio(
    events: list[dict],
    positive: bool,
) -> float:

    total = sum(

        _safe_float(
            event.get(
                "emotion_total_score"
            ),
            0.0,
        )
        or 0.0

        for event
        in events

    )


    field = (

        "emotion_positive_score"

        if positive

        else "emotion_negative_score"

    )


    part = sum(

        _safe_float(
            event.get(
                field
            ),
            0.0,
        )
        or 0.0

        for event
        in events

    )


    return (

        part
        / total

        if total

        else 0.0

    )


# =============================================================================
# 언어·인지 지표
# 실제 MongoDB events 필드를 직접 사용
# =============================================================================

def _build_cognition(
    events: list[dict],
    prev_events: list[dict],
) -> dict:

    # -------------------------------------------------------------------------
    # 지시대명사
    # demonstrative_pronoun_count
    # -------------------------------------------------------------------------

    pronoun_value = _mean_numeric(
        events,
        "demonstrative_pronoun_count",
    )


    pronoun_baseline = _mean_numeric(
        prev_events,
        "demonstrative_pronoun_count",
    )


    # -------------------------------------------------------------------------
    # TTR
    # -------------------------------------------------------------------------

    ttr_value = _mean_numeric(
        events,
        "ttr",
    )


    ttr_baseline = _mean_numeric(
        prev_events,
        "ttr",
    )


    # -------------------------------------------------------------------------
    # 반복 발화
    # repetition_exists
    # -------------------------------------------------------------------------

    repetition_count = sum(

        bool(
            event.get(
                "repetition_exists"
            )
        )

        for event
        in events

    )


    prev_repetition_count = sum(

        bool(
            event.get(
                "repetition_exists"
            )
        )

        for event
        in prev_events

    )


    repetition_ratio = (

        repetition_count
        /
        len(
            events
        )

        if events

        else 0.0

    )


    prev_repetition_ratio = (

        prev_repetition_count
        /
        len(
            prev_events
        )

        if prev_events

        else 0.0

    )


    # -------------------------------------------------------------------------
    # 문장 복잡도
    # -------------------------------------------------------------------------

    complexity_value = _mean_numeric(
        events,
        "sentence_complexity_score",
    )


    complexity_baseline = _mean_numeric(
        prev_events,
        "sentence_complexity_score",
    )


    complexity_label = _most_common_label(
        events,
        "sentence_complexity_label",
    )


    # -------------------------------------------------------------------------
    # 응답 지연
    # -------------------------------------------------------------------------

    _, response_delay = (
        _session_statistics(
            events
        )
    )


    _, prev_response_delay = (
        _session_statistics(
            prev_events
        )
    )


    # -------------------------------------------------------------------------
    # WPM
    # -------------------------------------------------------------------------

    wpm_value = _calculate_wpm(
        events
    )


    wpm_baseline = _calculate_wpm(
        prev_events
    )


    cognition = {

        "demonstrative_pronoun_rate":
            _metric(

                pronoun_value,

                pronoun_baseline,

                direction=1,

                decimals=2,

            ),


        "ttr":
            _metric(

                ttr_value,

                ttr_baseline,

                direction=-1,

                decimals=3,

            ),


        "repetition":
            _metric(

                repetition_ratio,

                prev_repetition_ratio,

                direction=1,

                decimals=4,

            ),


        "response_delay_sec":
            _metric(

                response_delay,

                prev_response_delay,

                direction=1,

                decimals=1,

            ),


        "wpm":
            _metric(

                wpm_value,

                wpm_baseline,

                direction=-1,

                decimals=1,

            ),


        "sentence_complexity":
            _metric(

                complexity_value,

                complexity_baseline,

                direction=-1,

                decimals=2,

            ),

    }


    cognition[
        "demonstrative_pronoun_rate"
    ][
        "unit"
    ] = "회/발화"


    cognition[
        "repetition"
    ].update({

        "count":
            repetition_count,

        "prev_count":
            prev_repetition_count,

        "ratio":
            round(
                repetition_ratio
                * 100,
                1,
            ),

        "baseline_ratio":
            round(
                prev_repetition_ratio
                * 100,
                1,
            ),

        "total_utterances":
            len(
                events
            ),

        "top_words":
            _top_repeated_words(
                events
            ),

    })


    cognition[
        "pronoun_total_count"
    ] = sum(

        _safe_int(
            event.get(
                "demonstrative_pronoun_count"
            ),
            0,
        )

        for event
        in events

    )


    cognition[
        "sentence_complexity"
    ][
        "label"
    ] = complexity_label


    # -------------------------------------------------------------------------
    # 실제 감정 분석 필드
    # -------------------------------------------------------------------------

    positive_value = _emotion_ratio(
        events,
        positive=True,
    )


    negative_value = _emotion_ratio(
        events,
        positive=False,
    )


    prev_positive_value = _emotion_ratio(
        prev_events,
        positive=True,
    )


    prev_negative_value = _emotion_ratio(
        prev_events,
        positive=False,
    )


    cognition[
        "positive_emotion_ratio"
    ] = _metric(

        positive_value,

        prev_positive_value,

        direction=-1,

        decimals=3,

    )


    cognition[
        "negative_emotion_ratio"
    ] = _metric(

        negative_value,

        prev_negative_value,

        direction=1,

        decimals=3,

    )


    # -------------------------------------------------------------------------
    # 다른 report.html 버전과의 호환을 위한 language_cards
    # -------------------------------------------------------------------------

    cognition[
        "language_cards"
    ] = [

        {
            "key":
                "demonstrative_pronoun_rate",

            "title":
                "지시대명사",

            "display_value":
                f"{pronoun_value:.2f}",

            "display_baseline":
                f"{pronoun_baseline:.2f}",

            "display_difference":
                (
                    f"{pronoun_value - pronoun_baseline:+.2f}"
                ),

            "unit":
                "회/발화",

            "range_label":
                "발화당 평균",

            "grade":
                cognition[
                    "demonstrative_pronoun_rate"
                ][
                    "grade"
                ],

            "description":
                (
                    "실제 "
                    "demonstrative_pronoun_count의 "
                    "월평균"
                ),

            "detail":
                "",
        },


        {
            "key":
                "ttr",

            "title":
                "어휘 다양성",

            "display_value":
                f"{ttr_value:.3f}",

            "display_baseline":
                f"{ttr_baseline:.3f}",

            "display_difference":
                (
                    f"{ttr_value - ttr_baseline:+.3f}"
                ),

            "unit":
                "",

            "range_label":
                "0~1",

            "grade":
                cognition[
                    "ttr"
                ][
                    "grade"
                ],

            "description":
                "실제 ttr의 월평균",

            "detail":
                "",
        },


        {
            "key":
                "repetition",

            "title":
                "반복 발화",

            "display_value":
                f"{repetition_ratio:.3f}",

            "display_baseline":
                f"{prev_repetition_ratio:.3f}",

            "display_difference":
                (
                    f"{repetition_ratio - prev_repetition_ratio:+.3f}"
                ),

            "unit":
                "",

            "range_label":
                "0~1",

            "grade":
                cognition[
                    "repetition"
                ][
                    "grade"
                ],

            "description":
                (
                    "repetition_exists=True인 "
                    "발화의 월간 비율"
                ),

            "detail":
                (
                    f"전체 {len(events)}개 중 "
                    f"{repetition_count}개"
                ),
        },


        {
            "key":
                "sentence_complexity",

            "title":
                "문장 복잡도",

            "display_value":
                f"{complexity_value:.2f}",

            "display_baseline":
                f"{complexity_baseline:.2f}",

            "display_difference":
                (
                    f"{complexity_value - complexity_baseline:+.2f}"
                ),

            "unit":
                "",

            "range_label":
                "원본 점수",

            "grade":
                cognition[
                    "sentence_complexity"
                ][
                    "grade"
                ],

            "description":
                (
                    "실제 "
                    "sentence_complexity_score의 "
                    "월평균"
                ),

            "detail":
                (
                    f"대표 등급: "
                    f"{complexity_label}"
                ),
        },


        {
            "key":
                "negative_emotion_ratio",

            "title":
                "부정 감정분석",

            "display_value":
                f"{negative_value:.3f}",

            "display_baseline":
                f"{prev_negative_value:.3f}",

            "display_difference":
                (
                    f"{negative_value - prev_negative_value:+.3f}"
                ),

            "unit":
                "",

            "range_label":
                "0~1",

            "grade":
                cognition[
                    "negative_emotion_ratio"
                ][
                    "grade"
                ],

            "description":
                (
                    "emotion_negative_score / "
                    "emotion_total_score"
                ),

            "detail":
                "",
        },


        {
            "key":
                "positive_emotion_ratio",

            "title":
                "긍정 감정분석",

            "display_value":
                f"{positive_value:.3f}",

            "display_baseline":
                f"{prev_positive_value:.3f}",

            "display_difference":
                (
                    f"{positive_value - prev_positive_value:+.3f}"
                ),

            "unit":
                "",

            "range_label":
                "0~1",

            "grade":
                cognition[
                    "positive_emotion_ratio"
                ][
                    "grade"
                ],

            "description":
                (
                    "emotion_positive_score / "
                    "emotion_total_score"
                ),

            "detail":
                "",
        },

    ]


    return cognition


# =============================================================================
# 유효 감정 event
# =============================================================================

def _valid_emotion_events(
    events: list[dict],
) -> list[dict]:

    result = []


    for event in events:

        # 최신 live 데이터에서
        # 감정 값이 실제로 없는 경우
        if (
            event.get(
                "emotion_scores_available"
            )
            is False
        ):

            continue


        has_score = any(

            _safe_float(
                event.get(
                    f"emotion_{emotion}"
                )
            )
            is not None

            for emotion
            in EMOTION_ORDER

        )


        if has_score:

            result.append(
                event
            )


    return result


# =============================================================================
# 감정 월평균
# =============================================================================

def _build_emotion_summary(
    events: list[dict],
    prev_events: list[dict],
) -> dict:

    current = _valid_emotion_events(
        events
    )


    previous = _valid_emotion_events(
        prev_events
    )


    monthly_avg = {}


    for emotion in EMOTION_ORDER:

        field = (
            f"emotion_{emotion}"
        )


        value = _mean_numeric(
            current,
            field,
        )


        baseline = _mean_numeric(
            previous,
            field,
        )


        direction = (

            -1

            if emotion
            in (
                "joy",
                "calm",
            )

            else 1

        )


        change = _change(
            value,
            baseline,
        )


        monthly_avg[
            emotion
        ] = {

            "name":
                EMOTION_KO[
                    emotion
                ],

            "color":
                EMOTION_COLORS[
                    emotion
                ],

            "mean":
                round(
                    value,
                    1,
                ),

            "baseline":
                round(
                    baseline,
                    1,
                ),

            "change":
                (
                    round(
                        change,
                        4,
                    )
                    if change is not None
                    else None
                ),

            "grade":
                _grade_directional(

                    value,

                    baseline,

                    direction,

                ),

        }


    return {

        "monthly_avg":
            monthly_avg,

    }


# =============================================================================
# 주차별 감정 트렌드
# =============================================================================

def _build_weekly_trend(
    events: list[dict],
) -> dict:

    valid_events = (
        _valid_emotion_events(
            events
        )
    )


    weeks = []


    for week in range(
        1,
        5,
    ):

        bucket = []


        for event in valid_events:

            date_str = _event_date(
                event
            )


            if not date_str:

                continue


            if (
                _week(
                    int(
                        date_str[-2:]
                    )
                )
                ==
                week
            ):

                bucket.append(
                    event
                )


        weeks.append({

            "week":
                week,

            **{

                emotion:
                    (
                        round(

                            _mean_numeric(

                                bucket,

                                f"emotion_{emotion}",

                            ),

                            1,

                        )

                        if bucket

                        else 0.0
                    )

                for emotion
                in EMOTION_ORDER

            },

        })


    return {

        "weeks":
            weeks,

    }


# =============================================================================
# 감정 기복 달력
# =============================================================================

def _build_volatility_calendar(
    events: list[dict],
    year: int,
    month: int,
) -> dict:

    last_day = monthrange(
        year,
        month,
    )[1]


    grouped = defaultdict(
        list
    )


    for event in (
        _valid_emotion_events(
            events
        )
    ):

        date_str = _event_date(
            event
        )


        if date_str:

            grouped[
                int(
                    date_str[-2:]
                )
            ].append(
                event
            )


    calendar = {}

    detail = {}


    for day in range(
        1,
        last_day + 1,
    ):

        bucket = grouped.get(
            day,
            [],
        )


        if not bucket:

            calendar[
                str(
                    day
                )
            ] = 0.0

            continue


        std_values = {}


        for emotion in (
            "anger",
            "fear",
            "sadness",
        ):

            values = [

                value

                for event
                in bucket

                if (
                    value
                    :=
                    _safe_float(
                        event.get(
                            f"emotion_{emotion}"
                        )
                    )
                )
                is not None

            ]


            std_values[
                emotion
            ] = (

                pstdev(
                    values
                )

                if len(
                    values
                ) >= 2

                else 0.0

            )


        volatility = sum(
            std_values.values()
        )


        calendar[
            str(
                day
            )
        ] = round(
            volatility,
            1,
        )


        detail[
            day
        ] = std_values


    sorted_days = sorted(

        [

            (
                int(
                    day
                ),
                value,
            )

            for day, value
            in calendar.items()

            if value > 0

        ],

        key=lambda item:
            item[1],

        reverse=True,

    )[:3]


    top3 = []


    for day, score in (
        sorted_days
    ):

        std_values = detail.get(
            day,
            {},
        )


        dominant = (

            max(
                std_values,
                key=std_values.get,
            )

            if std_values

            else "fear"

        )


        top3.append({

            "day":
                day,

            "date_str":
                f"{month}/{day}",

            "std_score":
                score,

            "dominant_emotion":
                EMOTION_KO[
                    dominant
                ],

        })


    return {

        "calendar":
            calendar,

        "top3_volatile_days":
            top3,

    }


# =============================================================================
# 감정 급등 발화 XAI
# =============================================================================

def _build_xai(
    events: list[dict],
) -> dict:

    valid_events = (
        _valid_emotion_events(
            events
        )
    )


    top_cases = []

    rest_summary = {}


    for emotion in (
        EMOTION_ORDER
    ):

        field = (
            f"emotion_{emotion}"
        )


        ranked = sorted(

            [

                event

                for event
                in valid_events

                if _safe_float(
                    event.get(
                        field
                    )
                )
                is not None

            ],

            key=lambda event:
                (
                    _safe_float(
                        event.get(
                            field
                        ),
                        0.0,
                    )
                    or 0.0
                ),

            reverse=True,

        )


        selected = (
            ranked[:2]
        )


        for event in (
            selected
        ):

            scores = {

                name:
                    round(

                        (
                            _safe_float(

                                event.get(
                                    f"emotion_{name}"
                                ),

                                0.0,

                            )
                            or 0.0
                        ),

                        1,

                    )

                for name
                in EMOTION_ORDER

            }


            top_cases.append({

                "emotion":
                    emotion,

                "emotion_ko":
                    EMOTION_KO[
                        emotion
                    ],

                "timestamp":
                    _format_timestamp(
                        event.get(
                            "timestamp"
                        )
                    ),

                "text":
                    str(
                        event.get(
                            "text"
                        )
                        or ""
                    ),

                "scores":
                    scores,

                "event_id":
                    str(
                        event.get(
                            "_id"
                        )
                        or ""
                    ),

                "request_id":
                    event.get(
                        "request_id"
                    ),

                "audio_filename":
                    event.get(
                        "audio_filename"
                    ),

            })


        remaining = (
            ranked[2:]
        )


        rest_summary[
            emotion
        ] = {

            "count":
                len(
                    remaining
                ),

            "total_count":
                len(
                    ranked
                ),

            "shown_count":
                len(
                    selected
                ),

            "remaining_count":
                len(
                    remaining
                ),

            "sample_dates":
                [

                    (
                        _event_date(
                            event
                        )
                        or "-"
                    )[5:]

                    for event
                    in remaining[:5]

                ],

        }


    return {

        "top_cases":
            top_cases,

        "rest_summary":
            rest_summary,

    }


# =============================================================================
# 감정 구슬
# =============================================================================

def _build_marbles(
    user_id: str,
    year: int,
    month: int,
    prev_year: int,
    prev_month: int,
    active_dates: set[str],
) -> dict:

    current = _load_marble_events(
        user_id,
        year,
        month,
    )


    previous = _load_marble_events(
        user_id,
        prev_year,
        prev_month,
    )


    def count_by_emotion(
        docs: list[dict],
    ) -> dict[str, int]:

        result = {

            emotion:
                0

            for emotion
            in EMOTION_ORDER

        }


        for doc in docs:

            emotion = doc.get(
                "emotion"
            )


            if emotion in result:

                result[
                    emotion
                ] += 1


        return result


    counts = count_by_emotion(
        current
    )


    prev_counts = count_by_emotion(
        previous
    )


    total = sum(
        counts.values()
    )


    prev_total = sum(
        prev_counts.values()
    )


    danger_count = (

        counts[
            "fear"
        ]
        +
        counts[
            "sadness"
        ]

    )


    prev_danger_count = (

        prev_counts[
            "fear"
        ]
        +
        prev_counts[
            "sadness"
        ]

    )


    danger_pct = (

        danger_count
        / total

        if total

        else 0.0

    )


    prev_danger_pct = (

        prev_danger_count
        / prev_total

        if prev_total

        else 0.0

    )


    daily_counter = Counter()


    for doc in current:

        date_str = None


        if isinstance(
            doc.get(
                "date"
            ),
            str,
        ):

            date_str = (
                doc[
                    "date"
                ][:10]
            )


        else:

            ts = _safe_datetime(
                doc.get(
                    "timestamp"
                )
            )


            if ts is not None:

                date_str = (
                    ts.strftime(
                        "%Y-%m-%d"
                    )
                )


        if date_str:

            daily_counter[
                date_str
            ] += 1


    last_day = monthrange(
        year,
        month,
    )[1]


    daily_counts = {

        (
            f"{year}-"
            f"{month:02d}-"
            f"{day:02d}"
        ):
            daily_counter.get(

                (
                    f"{year}-"
                    f"{month:02d}-"
                    f"{day:02d}"
                ),

                0,

            )

        for day
        in range(
            1,
            last_day + 1,
        )

    }


    active_zero_marble_days = sum(

        (
            date_str
            in active_dates
        )
        and
        (
            count == 0
        )

        for date_str, count
        in daily_counts.items()

    )


    return {

        "counts":
            counts,

        "total":
            total,

        "danger_count":
            danger_count,

        "danger_pct":
            round(
                danger_pct,
                3,
            ),

        "prev_counts":
            prev_counts,

        "prev_total":
            prev_total,

        "prev_danger_pct":
            round(
                prev_danger_pct,
                3,
            ),

        "danger_pct_change":
            round(
                danger_pct
                -
                prev_danger_pct,
                3,
            ),

        "daily_counts":
            daily_counts,

        "events":
            current,

        "zero_marble_days":
            sum(
                count == 0
                for count
                in daily_counts.values()
            ),

        "one_marble_days":
            sum(
                count == 1
                for count
                in daily_counts.values()
            ),

        "two_marble_days":
            sum(
                count == 2
                for count
                in daily_counts.values()
            ),

        "active_zero_marble_days":
            active_zero_marble_days,

    }


# =============================================================================
# anomaly_results
# =============================================================================

def _build_anomaly(
    docs: list[dict],
) -> dict:

    empty_distribution = {

        "normal":
            0,

        "watch":
            0,

        "warning":
            0,

        "risk":
            0,

    }


    # -------------------------------------------------------------------------
    # anomaly_results 없는 경우
    # -------------------------------------------------------------------------

    if not docs:

        return {

            "daily_scores":
                {},

            "latest_date":
                None,

            "latest_ewma_risk":
                None,

            "monthly_avg_risk":
                0.0,

            "monthly_max_risk":
                0.0,

            "peak_date":
                None,

            "alert_distribution":
                empty_distribution,

            "final_alert_level":
                "normal",

            "weekly_risk":
                [

                    {
                        "week":
                            week,

                        "daily":
                            0.0,

                        "ewma":
                            0.0,
                    }

                    for week
                    in range(
                        1,
                        5,
                    )

                ],

            "top_reasons":
                [],

            "monthly_feature_avg":
                [],

            "observed_days":
                0,

            "has_data":
                False,

        }


    daily = {}


    distribution = (
        empty_distribution.copy()
    )


    feature_accum = defaultdict(
        list
    )


    # -------------------------------------------------------------------------
    # 일별 실제 anomaly 데이터
    # -------------------------------------------------------------------------

    for doc in docs:

        date_str = str(
            doc.get(
                "date"
            )
            or ""
        )[:10]


        if not date_str:

            continue


        daily_score = (

            _safe_float(
                doc.get(
                    "daily_risk_score"
                ),
                0.0,
            )
            or 0.0

        )


        ewma_score = (

            _safe_float(
                doc.get(
                    "ewma_risk_score"
                ),
                0.0,
            )
            or 0.0

        )


        level = str(
            doc.get(
                "alert_level"
            )
            or "normal"
        )


        if level not in distribution:

            level = "normal"


        distribution[
            level
        ] += 1


        daily[
            date_str
        ] = {

            "daily_risk_score":
                round(
                    daily_score,
                    4,
                ),

            "ewma_risk_score":
                round(
                    ewma_score,
                    4,
                ),

            "alert_level":
                level,

            "risk_feature_count":
                _safe_int(
                    doc.get(
                        "risk_feature_count"
                    ),
                    0,
                ),

            "trend_direction":
                str(
                    doc.get(
                        "trend_direction"
                    )
                    or
                    "flat_or_down"
                ),

            "main_reasons":
                list(
                    doc.get(
                        "main_reasons"
                    )
                    or []
                ),

        }


        # ---------------------------------------------------------------------
        # feature_scores
        # ---------------------------------------------------------------------

        feature_scores = (
            doc.get(
                "feature_scores"
            )
        )


        if isinstance(
            feature_scores,
            dict,
        ):

            for feature, values in (
                feature_scores.items()
            ):

                if not isinstance(
                    values,
                    dict,
                ):

                    continue


                risk_z = _safe_float(
                    values.get(
                        "risk_z"
                    )
                )


                if risk_z is not None:

                    feature_accum[
                        feature
                    ].append(
                        risk_z
                    )


    if not daily:

        return _build_anomaly(
            []
        )


    # -------------------------------------------------------------------------
    # 월간 위험도
    # -------------------------------------------------------------------------

    risk_values = [

        item[
            "daily_risk_score"
        ]

        for item
        in daily.values()

    ]


    monthly_avg = mean(
        risk_values
    )


    monthly_max = max(
        risk_values
    )


    peak_date = max(

        daily,

        key=lambda date_str:
            daily[
                date_str
            ][
                "daily_risk_score"
            ],

    )


    # -------------------------------------------------------------------------
    # 최고 심각도
    # -------------------------------------------------------------------------

    severity = {

        "normal":
            0,

        "watch":
            1,

        "warning":
            2,

        "risk":
            3,

    }


    final_alert_level = max(

        (

            item[
                "alert_level"
            ]

            for item
            in daily.values()

        ),

        key=lambda level:
            severity.get(
                level,
                0,
            ),

    )


    # -------------------------------------------------------------------------
    # 최신 EWMA
    # -------------------------------------------------------------------------

    latest_date = max(
        daily
    )


    latest_ewma_risk = (

        daily[
            latest_date
        ][
            "ewma_risk_score"
        ]

    )


    # -------------------------------------------------------------------------
    # 주차별 risk / EWMA
    # -------------------------------------------------------------------------

    weekly_risk = []


    for week in range(
        1,
        5,
    ):

        bucket = [

            item

            for date_str, item
            in daily.items()

            if (
                _week(
                    int(
                        date_str[-2:]
                    )
                )
                ==
                week
            )

        ]


        weekly_risk.append({

            "week":
                week,

            "daily":
                (
                    round(

                        mean(

                            item[
                                "daily_risk_score"
                            ]

                            for item
                            in bucket

                        ),

                        4,

                    )

                    if bucket

                    else 0.0
                ),

            "ewma":
                (
                    round(

                        mean(

                            item[
                                "ewma_risk_score"
                            ]

                            for item
                            in bucket

                        ),

                        4,

                    )

                    if bucket

                    else 0.0
                ),

        })


    # -------------------------------------------------------------------------
    # 주요 이상 원인
    # -------------------------------------------------------------------------

    top_reasons = []


    for _, item in sorted(

        daily.items(),

        key=lambda pair:
            pair[
                1
            ][
                "daily_risk_score"
            ],

        reverse=True,

    ):

        for reason in (
            item.get(
                "main_reasons"
            )
            or []
        ):

            if reason not in top_reasons:

                top_reasons.append(
                    reason
                )


            if len(
                top_reasons
            ) >= 5:

                break


        if len(
            top_reasons
        ) >= 5:

            break


    # -------------------------------------------------------------------------
    # feature별 평균 Risk Z
    # -------------------------------------------------------------------------

    monthly_feature_avg = []


    for feature, values in (
        feature_accum.items()
    ):

        if not values:

            continue


        avg_risk_z = mean(
            values
        )


        monthly_feature_avg.append({

            "feature":
                feature,

            "name_ko":
                FEATURE_KO.get(
                    feature,
                    feature,
                ),

            "avg_risk_z":
                round(
                    avg_risk_z,
                    4,
                ),

            "risk_days":
                sum(

                    value
                    >= RISK_Z_THRESHOLD

                    for value
                    in values

                ),

            "is_risk":
                (
                    avg_risk_z
                    >= RISK_Z_THRESHOLD
                ),

        })


    monthly_feature_avg.sort(

        key=lambda item:
            item[
                "avg_risk_z"
            ],

        reverse=True,

    )


    return {

        "daily_scores":
            daily,

        "latest_date":
            latest_date,

        "latest_ewma_risk":
            latest_ewma_risk,

        "monthly_avg_risk":
            round(
                monthly_avg,
                4,
            ),

        "monthly_max_risk":
            round(
                monthly_max,
                4,
            ),

        "peak_date":
            peak_date,

        "alert_distribution":
            distribution,

        "final_alert_level":
            final_alert_level,

        "weekly_risk":
            weekly_risk,

        "top_reasons":
            top_reasons,

        "monthly_feature_avg":
            monthly_feature_avg,

        "observed_days":
            len(
                daily
            ),

        "has_data":
            True,

    }


# =============================================================================
# CNN
# =============================================================================

def _get_cnn_value(
    result: dict,
    *keys: str,
) -> Any:

    for key in keys:

        if key in result:

            return result.get(
                key
            )


    return None


def _build_cnn_summary(
    events: list[dict],
) -> dict:
    """
    실제 MongoDB의

    cnn_status
    cnn_available
    cnn_result

    를 사용합니다.

    현재 report.html에 CNN 영역이 없어도
    context에는 넘겨둡니다.
    """

    processed = [

        event

        for event
        in events

        if (
            event.get(
                "cnn_status"
            )
            ==
            "processed"
        )
        and
        isinstance(
            event.get(
                "cnn_result"
            ),
            dict,
        )

    ]


    if not processed:

        return {

            "processed_count":
                0,

            "latest_label":
                None,

            "latest_normal_prob":
                None,

            "latest_mci_prob":
                None,

            "avg_mci_prob":
                None,

        }


    def sort_key(
        event: dict,
    ) -> datetime:

        return (

            _safe_datetime(
                event.get(
                    "timestamp"
                )
            )
            or
            datetime.min

        )


    latest = max(
        processed,
        key=sort_key,
    )


    latest_result = (

        latest.get(
            "cnn_result"
        )
        or {}

    )


    mci_values = []


    for event in processed:

        result = (

            event.get(
                "cnn_result"
            )
            or {}

        )


        value = _safe_float(

            _get_cnn_value(

                result,

                "mci_prob",

                "mci_probability",

            )

        )


        if value is not None:

            mci_values.append(
                value
            )


    return {

        "processed_count":
            len(
                processed
            ),

        "latest_label":
            _get_cnn_value(

                latest_result,

                "pred_label",

                "label",

                "prediction",

            ),

        "latest_normal_prob":
            _safe_float(

                _get_cnn_value(

                    latest_result,

                    "normal_prob",

                    "normal_probability",

                )

            ),

        "latest_mci_prob":
            _safe_float(

                _get_cnn_value(

                    latest_result,

                    "mci_prob",

                    "mci_probability",

                )

            ),

        "avg_mci_prob":
            (
                mean(
                    mci_values
                )

                if mci_values

                else None
            ),

    }


# =============================================================================
# 전체 월간 리포트 데이터 생성
# =============================================================================

def build_report_data(
    user_id: str,
    year: int,
    month: int,
) -> dict:

    _validate_period(
        year,
        month,
    )


    prev_year, prev_month = (
        _prev_month(
            year,
            month,
        )
    )


    # -------------------------------------------------------------------------
    # 실제 MongoDB events
    # -------------------------------------------------------------------------

    events = _load_events(

        user_id,

        year,

        month,

    )


    prev_events = _load_events(

        user_id,

        prev_year,

        prev_month,

    )


    if not events:

        raise ValueError(

            f"{user_id}의 "
            f"{year}-{month:02d} "
            "events 데이터가 없습니다."

        )


    # -------------------------------------------------------------------------
    # 활동
    # -------------------------------------------------------------------------

    activity = _build_activity(

        events,

        prev_events,

        year,

        month,

    )


    # -------------------------------------------------------------------------
    # 언어/인지
    # -------------------------------------------------------------------------

    cognition = _build_cognition(

        events,

        prev_events,

    )


    # -------------------------------------------------------------------------
    # 감정 월평균
    # -------------------------------------------------------------------------

    emotion_summary = (
        _build_emotion_summary(

            events,

            prev_events,

        )
    )


    # -------------------------------------------------------------------------
    # 주차별 감정
    # -------------------------------------------------------------------------

    weekly_trend = (
        _build_weekly_trend(
            events
        )
    )


    # -------------------------------------------------------------------------
    # 감정 기복
    # -------------------------------------------------------------------------

    volatility_calendar = (
        _build_volatility_calendar(

            events,

            year,

            month,

        )
    )


    # -------------------------------------------------------------------------
    # XAI 원문
    # -------------------------------------------------------------------------

    xai_utterances = (
        _build_xai(
            events
        )
    )


    active_dates = {

        date_str

        for event
        in events

        if (
            date_str
            :=
            _event_date(
                event
            )
        )

    }


    # -------------------------------------------------------------------------
    # 실제 구슬 DB
    # -------------------------------------------------------------------------

    marbles = _build_marbles(

        user_id,

        year,

        month,

        prev_year,

        prev_month,

        active_dates,

    )


    # -------------------------------------------------------------------------
    # 실제 anomaly_results
    # -------------------------------------------------------------------------

    anomaly_docs = (
        _load_anomaly_docs(

            user_id,

            year,

            month,

        )
    )


    anomaly = _build_anomaly(
        anomaly_docs
    )


    # -------------------------------------------------------------------------
    # CNN
    # -------------------------------------------------------------------------

    cnn_summary = (
        _build_cnn_summary(
            events
        )
    )


    # -------------------------------------------------------------------------
    # Meta
    # -------------------------------------------------------------------------

    weekday, last_day = (
        monthrange(
            year,
            month,
        )
    )


    overall_grade = {

        "normal":
            "정상",

        "watch":
            "관찰",

        "warning":
            "주의",

        "risk":
            "위험",

    }.get(

        anomaly[
            "final_alert_level"
        ],

        "정상",

    )


    meta = {

        "user_id":
            user_id,

        "year":
            year,

        "month":
            month,

        "total_days":
            last_day,

        "first_weekday_sunday":
            (
                weekday
                + 1
            )
            % 7,

        "period_label":
            (
                f"{year}년 "
                f"{month}월"
            ),

        "period_start":
            (
                f"{year}-"
                f"{month:02d}-01"
            ),

        "period_end":
            (
                f"{year}-"
                f"{month:02d}-"
                f"{last_day:02d}"
            ),

        "generated_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "overall_grade":
            overall_grade,

        # 더 이상 합성 데이터가 아님
        "is_demo":
            False,

        "data_source":
            "mongodb",

        "baseline_description":
            "전달 실제 데이터",

    }


    # -------------------------------------------------------------------------
    # 최종
    # -------------------------------------------------------------------------

    return {

        "meta":
            meta,

        "activity":
            activity,

        "cognition":
            cognition,

        "marbles":
            marbles,

        "emotion_summary":
            emotion_summary,

        "weekly_trend":
            weekly_trend,

        "volatility_calendar":
            volatility_calendar,

        "xai_utterances":
            xai_utterances,

        "anomaly":
            anomaly,

        "cnn_summary":
            cnn_summary,

        "presentation": {

            "show_weekly_emotion_trend":
                SHOW_WEEKLY_EMOTION_TREND,

            "language_layout":
                "cards",

            "show_relative_change":
                True,

            "emotion_order":
                EMOTION_ORDER.copy(),

            "emotion_names":
                EMOTION_KO.copy(),

            "emotion_colors":
                EMOTION_COLORS.copy(),

        },

    }

# =============================================================================
# 기본 페이지
# =============================================================================

@app.get("/")
def root():

    return RedirectResponse(

        url=(

            f"/report/"
            f"{DEFAULT_USER_ID}/"
            f"{DEFAULT_YEAR}/"
            f"{DEFAULT_MONTH}"

        )

    )


# =============================================================================
# HTML 리포트
# =============================================================================

@app.get(
    "/report/{user_id}/{year}/{month}",
    response_class=HTMLResponse,
)
def get_report(
    request: Request,
    user_id: str,
    year: int,
    month: int,
):

    _validate_period(
        year,
        month,
    )


    try:

        data = build_report_data(

            user_id,

            year,

            month,

        )


    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=(
                f"데이터 집계 오류: "
                f"{exc}"
            ),

        ) from exc


    # -------------------------------------------------------------------------
    # report.html로 전달할 context
    # -------------------------------------------------------------------------

    context = {

        "request":
            request,

        "data":
            data,

        "meta":
            data[
                "meta"
            ],

        "activity":
            data[
                "activity"
            ],

        "cognition":
            data[
                "cognition"
            ],

        "marbles":
            data[
                "marbles"
            ],

        "emotion":
            data[
                "emotion_summary"
            ],

        "weekly":
            data[
                "weekly_trend"
            ],

        "calendar":
            data[
                "volatility_calendar"
            ],

        "xai":
            data[
                "xai_utterances"
            ],

        "anomaly":
            data[
                "anomaly"
            ],

        "cnn_summary":
            data[
                "cnn_summary"
            ],

        "presentation":
            data[
                "presentation"
            ],

        "emotion_order":
            EMOTION_ORDER,

        "emotion_names":
            EMOTION_KO,

        "emotion_colors":
            EMOTION_COLORS,

        "show_weekly_emotion_trend":
            SHOW_WEEKLY_EMOTION_TREND,

    }


    # -------------------------------------------------------------------------
    # Chart.js용 JSON
    # -------------------------------------------------------------------------

    json_fields = {

        "weekly_json":
            data[
                "weekly_trend"
            ][
                "weeks"
            ],

        "calendar_json":
            data[
                "volatility_calendar"
            ][
                "calendar"
            ],

        "xai_json":
            data[
                "xai_utterances"
            ][
                "top_cases"
            ],

        "anomaly_json":
            data[
                "anomaly"
            ][
                "daily_scores"
            ],

        "weekly_risk_json":
            data[
                "anomaly"
            ][
                "weekly_risk"
            ],

        "feature_avg_json":
            data[
                "anomaly"
            ][
                "monthly_feature_avg"
            ],

        "emotion_colors_json":
            EMOTION_COLORS,

        "emotion_names_json":
            EMOTION_KO,

        "emotion_order_json":
            EMOTION_ORDER,

        "inactive_dates_json":
            data[
                "activity"
            ][
                "inactive_dates"
            ],

    }


    context.update({

        name:
            json.dumps(

                value,

                ensure_ascii=False,

                allow_nan=False,

                default=str,

            )

        for name, value
        in json_fields.items()

    })


    # -------------------------------------------------------------------------
    # HTML Template 반환
    # -------------------------------------------------------------------------

    try:

        return templates.TemplateResponse(

            request=request,

            name="report.html",

            context=context,

        )


    except TypeError:

        return templates.TemplateResponse(

            "report.html",

            context,

        )


# =============================================================================
# PDF 리포트
# =============================================================================

@app.get(
    "/report/{user_id}/{year}/{month}/pdf",
    response_class=FileResponse,
)
def get_report_pdf(
    user_id: str,
    year: int,
    month: int,
):
    """
    기존 HTML 월간 리포트를
    Headless Chromium으로 렌더링한 뒤
    PDF 파일로 반환합니다.
    """

    _validate_period(
        year,
        month,
    )


    # -------------------------------------------------------------------------
    # 실제 해당 월 데이터 존재 여부 확인
    # -------------------------------------------------------------------------

    try:

        build_report_data(

            user_id,

            year,

            month,

        )


    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=(
                f"PDF 데이터 집계 오류: "
                f"{exc}"
            ),

        ) from exc


    # -------------------------------------------------------------------------
    # Chromium 실행 파일 확인
    # -------------------------------------------------------------------------

    chromium_path = (

        shutil.which(
            "chromium"
        )

        or

        shutil.which(
            "chromium-browser"
        )

    )


    if not chromium_path:

        raise HTTPException(

            status_code=500,

            detail=(
                "Chromium 실행 파일을 "
                "찾을 수 없습니다."
            ),

        )


    # -------------------------------------------------------------------------
    # Chromium이 접근할 내부 HTML 리포트 주소
    # -------------------------------------------------------------------------

    safe_user_id = quote(

        str(
            user_id
        ),

        safe="",

    )


    report_url = (

        f"{REPORT_INTERNAL_BASE_URL}"
        f"/report/"
        f"{safe_user_id}/"
        f"{year}/"
        f"{month}"

    )


    # -------------------------------------------------------------------------
    # 임시 PDF 경로 생성
    # -------------------------------------------------------------------------

    tmp_file = tempfile.NamedTemporaryFile(

        prefix=(

            f"merry_report_"
            f"{year}_"
            f"{month:02d}_"

        ),

        suffix=".pdf",

        delete=False,

    )


    pdf_path = tmp_file.name

    tmp_file.close()


    # -------------------------------------------------------------------------
    # Headless Chromium 명령
    # -------------------------------------------------------------------------

    command = [

        chromium_path,

        "--headless",

        "--no-sandbox",

        "--disable-gpu",

        "--disable-dev-shm-usage",

        # JS/Chart.js 렌더링 대기
        "--run-all-compositor-stages-before-draw",

        "--virtual-time-budget=7000",

        # 기본 날짜/URL 헤더·푸터 제거
        "--no-pdf-header-footer",

        f"--print-to-pdf={pdf_path}",

        report_url,

    ]


    # -------------------------------------------------------------------------
    # PDF 생성
    # -------------------------------------------------------------------------

    try:

        result = subprocess.run(

            command,

            stdout=subprocess.PIPE,

            stderr=subprocess.PIPE,

            text=True,

            timeout=60,

        )


    except subprocess.TimeoutExpired:

        try:

            os.remove(
                pdf_path
            )

        except OSError:

            pass


        raise HTTPException(

            status_code=504,

            detail=(
                "PDF 생성 시간이 "
                "초과되었습니다."
            ),

        )


    # -------------------------------------------------------------------------
    # 결과 검증
    # -------------------------------------------------------------------------

    if (

        result.returncode != 0

        or

        not os.path.isfile(
            pdf_path
        )

        or

        os.path.getsize(
            pdf_path
        ) == 0

    ):

        try:

            os.remove(
                pdf_path
            )

        except OSError:

            pass


        error_message = (

            result.stderr[-2000:]

            if result.stderr

            else
            "알 수 없는 Chromium 오류"

        )


        raise HTTPException(

            status_code=500,

            detail=(

                "PDF 생성 실패: "
                f"{error_message}"

            ),

        )


    # -------------------------------------------------------------------------
    # 사용자에게 내려줄 파일명
    # -------------------------------------------------------------------------

    download_name = (

        f"Merry_AI_"
        f"{user_id}_"
        f"{year}_"
        f"{month:02d}_"
        f"monthly_report.pdf"

    )


    # -------------------------------------------------------------------------
    # 다운로드 완료 후 임시 파일 삭제
    # -------------------------------------------------------------------------

    return FileResponse(

        path=pdf_path,

        media_type="application/pdf",

        filename=download_name,

        background=BackgroundTask(

            os.remove,

            pdf_path,

        ),

    )
    # -------------------------------------------------------------------------
    # Chart.js용 JSON
    # -------------------------------------------------------------------------

    json_fields = {

        "weekly_json":
            data[
                "weekly_trend"
            ][
                "weeks"
            ],

        "calendar_json":
            data[
                "volatility_calendar"
            ][
                "calendar"
            ],

        "xai_json":
            data[
                "xai_utterances"
            ][
                "top_cases"
            ],

        "anomaly_json":
            data[
                "anomaly"
            ][
                "daily_scores"
            ],

        "weekly_risk_json":
            data[
                "anomaly"
            ][
                "weekly_risk"
            ],

        "feature_avg_json":
            data[
                "anomaly"
            ][
                "monthly_feature_avg"
            ],

        "emotion_colors_json":
            EMOTION_COLORS,

        "emotion_names_json":
            EMOTION_KO,

        "emotion_order_json":
            EMOTION_ORDER,

        "inactive_dates_json":
            data[
                "activity"
            ][
                "inactive_dates"
            ],

    }


    context.update({

        name:
            json.dumps(

                value,

                ensure_ascii=False,

                allow_nan=False,

                default=str,

            )

        for name, value
        in json_fields.items()

    })


    # Starlette/FastAPI 버전 호환
    try:

        return templates.TemplateResponse(

            request=request,

            name="report.html",

            context=context,

        )


    except TypeError:

        return templates.TemplateResponse(

            "report.html",

            context,

        )


# =============================================================================
# JSON 리포트
# =============================================================================

@app.get(
    "/report/{user_id}/{year}/{month}/json",
    response_class=JSONResponse,
)
def get_report_json(
    user_id: str,
    year: int,
    month: int,
):

    _validate_period(
        year,
        month,
    )


    try:

        data = build_report_data(

            user_id,

            year,

            month,

        )


    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=(
                f"데이터 집계 오류: "
                f"{exc}"
            ),

        ) from exc


    # ObjectId / datetime 등을
    # JSON-safe 형태로 변환
    serializable = json.loads(

        json.dumps(

            data,

            ensure_ascii=False,

            allow_nan=False,

            default=str,

        )

    )


    return JSONResponse(

        content=serializable,

        media_type=(
            "application/json; "
            "charset=utf-8"
        ),

    )


# =============================================================================
# 실제 MongoDB 필드 확인
# =============================================================================

@app.get(
    "/schema/{user_id}",
    response_class=JSONResponse,
)
def get_schema(
    user_id: str,
):
    """
    사용자의 최근 events 문서를 최대 200건 확인하여
    최상위 MongoDB 필드의 합집합을 반환합니다.

    예:
        http://localhost:8001/schema/U001
    """

    client = _mongo_client()


    try:

        collection = (
            client[
                DB_NAME
            ][
                EVENT_COLLECTION
            ]
        )


        docs = list(

            collection.find({

                "user_id":
                    user_id,

            })

            .sort(
                "created_at",
                -1,
            )

            .limit(
                200
            )

        )


    finally:

        client.close()


    if not docs:

        raise HTTPException(

            status_code=404,

            detail=(
                f"{user_id}의 "
                "event가 없습니다."
            ),

        )


    types_by_field = defaultdict(
        set
    )


    for doc in docs:

        for key, value in (
            doc.items()
        ):

            types_by_field[
                key
            ].add(
                type(
                    value
                ).__name__
            )


    fields = {

        key:
            sorted(
                types
            )

        for key, types
        in sorted(
            types_by_field.items()
        )

    }


    return JSONResponse(

        content={

            "user_id":
                user_id,

            "document_count_checked":
                len(
                    docs
                ),

            "field_count":
                len(
                    fields
                ),

            "fields":
                fields,

        }

    )
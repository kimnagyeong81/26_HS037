# -*- coding: utf-8 -*-
"""
Merry AI - 서버 인지 분석 통합 모듈

역할:
1. Jetson에서 전달한 감정 점수 검증
2. 지시대명사 분석
3. 어휘다양성 분석
4. 반복 단어 분석
5. 문장 복잡도 분석
6. 분석 결과를 MongoDB 저장용 이벤트로 변환

운영 원칙:
- STT, LLM, 감정 분석은 Jetson에서 수행
- 서버에서는 언어·인지 분석만 수행
- MongoDB 저장은 analysis_consumer.py에서 담당
- 서버에서 감정 모델 및 LSTM 모델을 실행하지 않음
"""

from __future__ import annotations

import logging
import math
import os

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pymongo import ASCENDING, DESCENDING, MongoClient

from pronoun_analyzer import analyze_demonstrative_pronouns
from lexical_diversity_analyzer import analyze_single_utterance
from repetition_analyzer import analyze_repetition
from sentence_complexity_analyzer import analyze_sentence_complexity

from event_builder import build_utterance_event


# ============================================================
# 1. 환경 설정
# ============================================================

logger = logging.getLogger(__name__)

MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb://localhost:27017/"
)

DB_NAME = "dementia_db"

DEFAULT_SESSION_ID = "S001"

# Jetson에서 전달하는 5종 감정
EMOTIONS = (
    "joy",
    "calm",
    "anger",
    "fear",
    "sadness"
)

# 감정 점수가 없는 경우 null 처리할 이벤트 필드
EMOTION_EVENT_FIELDS = (
    "emotion_joy",
    "emotion_calm",
    "emotion_anger",
    "emotion_fear",
    "emotion_sadness",
    "emotion_total_score",
    "emotion_positive_score",
    "emotion_negative_score",
    "emotion_score"
)


# ============================================================
# 2. MongoDB 인덱스 준비
# ============================================================

def ensure_indexes() -> None:
    """
    인지 분석 결과 조회에 필요한 MongoDB 인덱스를 생성한다.

    실제 데이터 저장은 analysis_consumer.py가 담당한다.
    """

    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5000
    )

    try:
        db = client[DB_NAME]

        events = db["events"]

        events.create_index([
            ("user_id", ASCENDING),
            ("timestamp", ASCENDING)
        ])

        events.create_index([
            ("user_id", ASCENDING),
            ("date", ASCENDING)
        ])

        events.create_index([
            ("user_id", ASCENDING),
            ("event_type", ASCENDING)
        ])

        events.create_index([
            ("user_id", ASCENDING),
            ("source_file", ASCENDING)
        ])

        # request_id 고유 인덱스는 Consumer에서 관리한다.

        anomaly = db["anomaly_results"]

        anomaly.create_index([
            ("user_id", ASCENDING),
            ("date", ASCENDING)
        ])

        anomaly.create_index([
            ("user_id", ASCENDING),
            ("created_at", DESCENDING)
        ])

    finally:
        client.close()


# ============================================================
# 3. Jetson 감정 점수 검증
# ============================================================

def normalize_emotion_scores(
    scores: Optional[Dict[str, Any]]
) -> Optional[Dict[str, float]]:
    """
    Jetson에서 전달한 5종 감정 점수를 검증한다.

    감정 점수 범위: 0~100

    감정 점수가 전달되지 않은 경우 None을 반환한다.
    서버에서 감정 모델을 실행하지 않는다.
    """

    if scores is None or scores == {}:
        return None

    if (
        not isinstance(scores, dict)
        or set(scores) != set(EMOTIONS)
    ):
        raise ValueError(
            "emotion_scores는 "
            "joy/calm/anger/fear/sadness "
            "5종이 모두 필요합니다."
        )

    normalized = {}

    for emotion in EMOTIONS:

        raw = scores[emotion]

        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
        ):
            raise ValueError(
                f"{emotion} 감정 점수는 숫자여야 합니다."
            )

        number = float(raw)

        if (
            not math.isfinite(number)
            or not 0 <= number <= 100
        ):
            raise ValueError(
                f"{emotion} 감정 점수는 "
                "0~100 범위여야 합니다."
            )

        normalized[emotion] = number

    return normalized


# ============================================================
# 4. 언어·인지 분석
# ============================================================

def run_language_cognitive_analysis(
    text: str
) -> Dict[str, Dict[str, Any]]:
    """
    STT 발화에 대해 4가지 언어 지표를 분석한다.

    1. 지시대명사
    2. 어휘다양성
    3. 반복 단어
    4. 문장 복잡도
    """

    # 1. 지시대명사
    pronoun_result = analyze_demonstrative_pronouns(
        text
    )

    # 2. 어휘다양성
    lexical_result = analyze_single_utterance(
        text
    )

    # 3. 반복 단어
    repetition_result = analyze_repetition(
        text
    )

    # 4. 문장 복잡도
    complexity_result = analyze_sentence_complexity(
        text
    )

    return {
        "pronoun_result": pronoun_result,
        "lexical_result": lexical_result,
        "repetition_result": repetition_result,
        "complexity_result": complexity_result,
    }


# ============================================================
# 5. 발화별 인지 분석 메인 함수
# ============================================================

def analyze(
    user_id: str,
    text: str,
    scores: Optional[Dict[str, Any]] = None,
    source_file: Optional[str] = None,
    timestamp: Optional[Any] = None,
    session_id: Optional[str] = None,
    date: Optional[str] = None,
    utterance_id: Optional[str] = None,
    speaker: Optional[str] = None,
    risk_label: Optional[int] = None,
    use_external_scores: bool = True,
    save_to_mongo: bool = False,
    enable_lstm_inference: bool = False,
) -> Dict[str, Any]:
    """
    Jetson에서 수신한 발화 1건에 대한 인지 분석.

    처리 순서:
    1. 입력 데이터 검증
    2. Jetson 감정 점수 검증
    3. 서버 언어 분석
    4. 분석 이벤트 생성
    5. Consumer에 결과 반환

    MongoDB에 직접 저장하지 않는다.
    """

    # --------------------------------------------------------
    # 5-1. 운영 모드 확인
    # --------------------------------------------------------

    if not use_external_scores:
        raise ValueError(
            "서버 감정 분석 기능은 제거되었습니다. "
            "use_external_scores=True를 사용하세요."
        )

    if save_to_mongo:
        raise ValueError(
            "MongoDB 저장은 "
            "analysis_consumer.py에서 처리합니다."
        )

    if enable_lstm_inference:
        raise ValueError(
            "이 모듈에서는 LSTM 추론을 실행하지 않습니다."
        )

    # --------------------------------------------------------
    # 5-2. 입력 데이터 검증
    # --------------------------------------------------------

    text = str(text or "").strip()

    if not text:
        return {
            "ok": False,
            "error": "empty_text",
            "message": "분석할 텍스트가 비어 있습니다."
        }

    if not str(user_id or "").strip():
        return {
            "ok": False,
            "error": "missing_user_id",
            "message": "user_id가 없습니다."
        }

    session_id = session_id or DEFAULT_SESSION_ID

    # --------------------------------------------------------
    # 5-3. Jetson 감정 점수 검증
    # --------------------------------------------------------

    emotion_scores = normalize_emotion_scores(
        scores
    )

    has_emotion = emotion_scores is not None

    # --------------------------------------------------------
    # 5-4. 언어·인지 분석
    # --------------------------------------------------------

    analysis_results = run_language_cognitive_analysis(
        text
    )

    # --------------------------------------------------------
    # 5-5. Timestamp 정규화
    # --------------------------------------------------------

    if isinstance(timestamp, datetime):
        timestamp_str = timestamp.isoformat()

    elif timestamp is not None:
        timestamp_str = str(timestamp)

    else:
        timestamp_str = datetime.now(
            timezone.utc
        ).isoformat()

    # --------------------------------------------------------
    # 5-6. 분석 이벤트 생성
    # --------------------------------------------------------

    event = build_utterance_event(

        user_id=str(user_id),

        session_id=session_id,

        date=date,

        utterance_id=utterance_id,

        timestamp=timestamp_str,

        speaker=speaker or "user",

        text=text,

        # Jetson 감정 점수
        emotion_result=emotion_scores,

        # 서버 언어 분석 결과
        pronoun_result=analysis_results[
            "pronoun_result"
        ],

        lexical_result=analysis_results[
            "lexical_result"
        ],

        repetition_result=analysis_results[
            "repetition_result"
        ],

        complexity_result=analysis_results[
            "complexity_result"
        ],

        source_file=source_file,

        risk_label=risk_label
    )

    # --------------------------------------------------------
    # 5-7. 감정 점수 출처 기록
    # --------------------------------------------------------

    event["emotion_source"] = (
        "jetson" if has_emotion else "missing"
    )

    event["emotion_scores_available"] = has_emotion

    # --------------------------------------------------------
    # 5-8. 감정 점수 누락 처리
    # --------------------------------------------------------

    if not has_emotion:

        # event_builder는 누락된 감정 점수를 기본 0으로
        # 생성하므로 실제 0점과 구분하도록 수정한다.

        for field in EMOTION_EVENT_FIELDS:
            event[field] = None

        event["emotion_label"] = "unknown"

    logger.info(
        "[인지 분석 완료] user_id=%s "
        "emotion_available=%s",
        user_id,
        has_emotion
    )

    # --------------------------------------------------------
    # 5-9. Consumer에 결과 반환
    # --------------------------------------------------------

    return {

        "ok": True,

        "event": event,

        "event_id": None,

        "risk_result": None,

        "saved_to_mongo": False
    }


# ============================================================
# 6. 기존 호출 코드 호환용 함수
# ============================================================

def analyze_utterance(
    *args: Any,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    기존 analyze_utterance() 호출부와의 호환성 유지
    """
    return analyze(*args, **kwargs)
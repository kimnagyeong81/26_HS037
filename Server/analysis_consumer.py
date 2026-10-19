# -*- coding: utf-8 -*-

"""
Merry AI - 서버 인지 분석 Consumer

역할:
1. Kafka에서 Jetson STT 발화 수신
2. Jetson 감정 분석 점수 수신
3. 서버에서 CNN 음성 기반 Normal/MCI 분석
4. 서버에서 언어·인지 분석 수행
5. MongoDB에 분석 결과 저장
6. Rolling Z-score + EWMA 추세 분석
7. 분석 결과를 Kafka Result Topic에 발행

중요:
- LLM 답변 생성은 Jetson에서 수행
- 감정 분석은 Jetson에서 수행
- 서버에서는 감정 모델을 다시 실행하지 않음
- CNN 모델은 Consumer 시작 시 1회만 로드
- CNN 실패가 텍스트 기반 인지 분석을 중단시키지 않음
- CNN 결과의 MongoDB 통합은 7단계에서 수행
"""

from __future__ import annotations

import json
import logging
import math
import os
import traceback

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from kafka import KafkaConsumer, KafkaProducer

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

from kafka_k8s_runtime import ensure_kafka_topic
from dementia_module import analyze, ensure_indexes
from inference_trend import run_for_user
from cnn_module import CNNEnsemble


# ============================================================
# 1. 환경 설정
# ============================================================

KAFKA_SERVER = os.getenv(
    "KAFKA_SERVER",
    "localhost:9092",
)

RAW_TOPIC = os.getenv(
    "RAW_TOPIC",
    "dementia-raw-utterance-topic",
)

RESULT_TOPIC = os.getenv(
    "RESULT_TOPIC",
    "dementia-analysis-result-topic",
)

GROUP_ID = os.getenv(
    "GROUP_ID",
    "dementia-analysis-consumer-group",
)

MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb://localhost:27017/",
)

DB_NAME = os.getenv(
    "DB_NAME",
    "dementia_db",
)

EVENT_COLLECTION = os.getenv(
    "EVENT_COLLECTION",
    "events",
)


# CNN 사용 여부
CNN_ENABLED = (
    os.getenv("CNN_ENABLED", "true").lower()
    == "true"
)

# cnn_module.py가 참조하는 모델 디렉터리
CNN_MODEL_DIR = Path(
    os.getenv(
        "CNN_MODEL_DIR",
        "/Server/models",
    )
)

# API와 Consumer가 공유하는 음성 디렉터리
AUDIO_STORAGE_DIR = Path(
    os.getenv(
        "AUDIO_STORAGE_DIR",
        "/data/utterances",
    )
)


# ============================================================
# 2. 인지 변화 추세 분석 설정
# ============================================================

TREND_WINDOW_SIZE = 7
TREND_MIN_BASELINE_DAYS = 7
TREND_DAYS = 4
TREND_ALPHA = 0.3


# Jetson에서 전달하는 5종 감정
EMOTIONS = (
    "joy",
    "calm",
    "anger",
    "fear",
    "sadness",
)


logger = logging.getLogger(__name__)


# ============================================================
# 3. CNN Ensemble
# ============================================================

_cnn: Optional[CNNEnsemble] = None


def initialize_cnn() -> bool:
    """
    Consumer 시작 시 CNN Ensemble 모델을 한 번만 로드한다.

    성공:
        True

    실패:
        False

    CNN 초기화에 실패하더라도 Consumer 전체를 종료하지 않는다.
    텍스트 기반 인지 분석은 계속 수행한다.
    """

    global _cnn

    if not CNN_ENABLED:
        logger.warning(
            "[CNN] CNN_ENABLED=false - "
            "CNN 분석을 사용하지 않습니다."
        )
        _cnn = None
        return False

    logger.info(
        "[CNN] Ensemble 초기화 시작"
    )

    logger.info(
        "[CNN] 모델 디렉터리: %s",
        CNN_MODEL_DIR,
    )

    try:
        if not CNN_MODEL_DIR.exists():
            raise FileNotFoundError(
                f"CNN 모델 디렉터리가 없습니다: "
                f"{CNN_MODEL_DIR}"
            )

        model_files = sorted(
            CNN_MODEL_DIR.glob("*.pth")
        )

        logger.info(
            "[CNN] 발견한 .pth 모델 수: %d",
            len(model_files),
        )

        for model_file in model_files:
            logger.info(
                "[CNN] 모델 파일: %s",
                model_file.name,
            )

        # cnn_module.py의 DEFAULT_MODEL_CONFIGS 사용
        #
        # CNN_MODEL_DIR 환경변수:
        # /Server/models
        #
        # search_030 / 044 / 048 모델을
        # cnn_module.py에서 자동으로 로드한다.
        _cnn = CNNEnsemble()

        logger.info(
            "[CNN] Ensemble 초기화 완료 "
            "models=%d",
            len(_cnn.models),
        )

        return True

    except Exception as exc:
        _cnn = None

        logger.exception(
            "[CNN] Ensemble 초기화 실패: %s",
            exc,
        )

        return False


def close_cnn() -> None:
    """
    Consumer 종료 시 CNN 모델 메모리를 정리한다.
    """

    global _cnn

    if _cnn is None:
        return

    try:
        _cnn.close()

        logger.info(
            "[CNN] Ensemble 종료 완료"
        )

    except Exception:
        logger.exception(
            "[CNN] Ensemble 종료 중 오류"
        )

    finally:
        _cnn = None


# ============================================================
# 4. 공통 유틸리티
# ============================================================

def now_iso() -> str:
    """
    현재 시각을 UTC ISO 8601 형식으로 반환한다.
    """

    return datetime.now(
        timezone.utc
    ).isoformat()


def json_default(
    obj: Any,
) -> Any:
    """
    분석 결과에 포함될 수 있는
    numpy 값이나 datetime을 JSON으로 변환한다.
    """

    if isinstance(
        obj,
        (datetime, date),
    ):
        return obj.isoformat()

    if hasattr(
        obj,
        "item",
    ):
        return obj.item()

    raise TypeError(
        f"JSON 변환 불가: "
        f"{type(obj).__name__}"
    )


def validate_audio_path(
    audio_path: Any,
) -> tuple[Optional[Path], Optional[str]]:
    """
    Kafka payload의 audio_path를 검증한다.

    허용 경로:
        /data/utterances/...

    반환:
        (Path, None)
        또는
        (None, error)
    """

    if audio_path is None:
        return None, "audio_path_missing"

    audio_path_string = str(
        audio_path
    ).strip()

    if not audio_path_string:
        return None, "audio_path_empty"

    try:
        root = AUDIO_STORAGE_DIR.resolve()

        candidate = Path(
            audio_path_string
        ).resolve()

    except Exception as exc:
        return (
            None,
            f"audio_path_resolve_failed: {exc}",
        )

    try:
        candidate.relative_to(root)

    except ValueError:
        return (
            None,
            (
                "audio_path_outside_storage: "
                f"{candidate}"
            ),
        )

    if not candidate.exists():
        return (
            None,
            (
                "audio_file_not_found: "
                f"{candidate}"
            ),
        )

    if not candidate.is_file():
        return (
            None,
            (
                "audio_path_not_file: "
                f"{candidate}"
            ),
        )

    suffix = candidate.suffix.lower()

    if suffix not in {
        ".flac",
        ".wav",
    }:
        return (
            None,
            (
                "unsupported_audio_format: "
                f"{suffix}"
            ),
        )

    return candidate, None


def run_cnn_analysis(
    audio_path: Any,
) -> tuple[
    Optional[Dict[str, Any]],
    str,
    Optional[str],
]:
    """
    CNN 음성 분석을 실행한다.

    반환:
        cnn_result
        cnn_status
        cnn_error

    CNN 오류가 발생해도 예외를 외부로 던지지 않는다.
    텍스트 인지 분석은 계속 진행한다.
    """

    if not CNN_ENABLED:
        return (
            None,
            "disabled",
            None,
        )

    if _cnn is None:
        return (
            None,
            "unavailable",
            "CNN Ensemble이 초기화되지 않았습니다.",
        )

    audio_file, audio_error = validate_audio_path(
        audio_path
    )

    if audio_error is not None:
        logger.warning(
            "[CNN] 음성 파일 검증 실패: %s",
            audio_error,
        )

        return (
            None,
            "skipped",
            audio_error,
        )

    try:
        logger.info(
            "[CNN] 추론 시작: %s",
            audio_file,
        )

        result = _cnn.predict(
            str(audio_file)
        )

        logger.info(
            "[CNN] 추론 완료 "
            "label=%s "
            "normal=%.4f "
            "mci=%.4f",
            result.get(
                "pred_label"
            ),
            float(
                result.get(
                    "normal_prob",
                    0.0,
                )
            ),
            float(
                result.get(
                    "mci_prob",
                    0.0,
                )
            ),
        )

        return (
            result,
            "processed",
            None,
        )

    except Exception as exc:
        logger.exception(
            "[CNN] 추론 실패: %s",
            exc,
        )

        return (
            None,
            "failed",
            str(exc),
        )

def attach_cnn_result_to_event(
    event: Dict[str, Any],
    *,
    cnn_result: Optional[Dict[str, Any]],
    cnn_status: Optional[str],
    cnn_error: Optional[str],
    audio_path: Any,
) -> None:
    """
    CNN 분석 결과를 MongoDB event에 추가한다.
    """

    event["cnn_status"] = cnn_status

    event["cnn_available"] = (
        cnn_result is not None
    )

    event["cnn_result"] = (
        cnn_result
        if cnn_result is not None
        else None
    )

    event["cnn_error"] = cnn_error

    if audio_path:
        event["audio_filename"] = Path(
            str(audio_path)
        ).name
    else:
        event["audio_filename"] = None

# ============================================================
# 5. Kafka Consumer 생성
# ============================================================

def make_consumer() -> KafkaConsumer:

    ensure_kafka_topic(
        bootstrap_server=KAFKA_SERVER,
        topic=RAW_TOPIC,
    )

    return KafkaConsumer(
        RAW_TOPIC,

        bootstrap_servers=KAFKA_SERVER,

        group_id=GROUP_ID,

        # 기존 Consumer Group의 offset이 있으면
        # 이어서 처리한다.
        #
        # 처음 생성된 그룹이라면
        # 가장 오래된 메시지부터 처리한다.
        auto_offset_reset="earliest",

        # MongoDB 저장 및 Kafka Result Topic 발행 후
        # 직접 offset을 commit한다.
        enable_auto_commit=False,

        # CNN 처리가 오래 걸릴 수 있으므로
        # 한 번에 하나의 메시지만 처리
        max_poll_records=1,

        # CNN CPU 추론 시간을 고려
        max_poll_interval_ms=900000,

        value_deserializer=lambda raw: (
            json.loads(
                raw.decode("utf-8")
            )
        ),
    )


# ============================================================
# 6. Kafka Producer 생성
# ============================================================

def make_producer() -> KafkaProducer:

    ensure_kafka_topic(
        bootstrap_server=KAFKA_SERVER,
        topic=RESULT_TOPIC,
    )

    return KafkaProducer(
        bootstrap_servers=KAFKA_SERVER,

        value_serializer=lambda value: (
            json.dumps(
                value,
                ensure_ascii=False,
                default=json_default,
            ).encode("utf-8")
        ),

        acks="all",

        retries=5,

        linger_ms=10,
    )


# ============================================================
# 7. Jetson 감정 점수 검증
# ============================================================

def normalize_emotion_scores(
    value: Any,
) -> Optional[Dict[str, float]]:
    """
    Jetson에서 전달한 감정 점수를 검증한다.

    감정 종류:
    - joy
    - calm
    - anger
    - fear
    - sadness

    점수 범위:
    0 ~ 100

    감정 분석에 실패한 경우 None을 반환한다.
    """

    if value is None:
        return None

    if (
        not isinstance(
            value,
            dict,
        )
        or set(value) != set(EMOTIONS)
    ):
        raise ValueError(
            "emotion_scores에는 "
            "joy/calm/anger/fear/sadness가 "
            "모두 필요합니다."
        )

    scores: Dict[str, float] = {}

    for emotion in EMOTIONS:

        score = value[emotion]

        if (
            isinstance(
                score,
                bool,
            )
            or not isinstance(
                score,
                (int, float),
            )
        ):
            raise ValueError(
                f"{emotion} 점수는 "
                "숫자여야 합니다."
            )

        score = float(score)

        if (
            not math.isfinite(score)
            or not 0 <= score <= 100
        ):
            raise ValueError(
                f"{emotion} 점수는 "
                "0~100 범위여야 합니다."
            )

        scores[emotion] = score

    return scores


# ============================================================
# 8. 추세 분석 결과 정리
# ============================================================

def extract_latest_trend_result(
    results: list,
) -> Optional[Dict[str, Any]]:

    if not results:
        return None

    item = results[-1]

    return {
        "user_id": item.get(
            "user_id"
        ),

        "date": item.get(
            "date"
        ),

        "alert_level": item.get(
            "alert_level"
        ),

        "trend_direction": item.get(
            "trend_direction"
        ),

        "daily_risk_score": item.get(
            "daily_risk_score"
        ),

        "ewma_risk_score": item.get(
            "ewma_risk_score"
        ),

        "main_reasons": item.get(
            "main_reasons",
            [],
        ),

        "model_type": item.get(
            "model_type"
        ),
    }


# ============================================================
# 9. Rolling Z-score + EWMA 추세 분석
# ============================================================

def run_trend(
    user_id: str,
) -> Optional[Dict[str, Any]]:
    """
    MongoDB에 저장된 사용자 분석 데이터를 기반으로
    개인별 인지 변화 추세를 분석한다.
    """

    results = run_for_user(
        user_id=user_id,

        window_size=(
            TREND_WINDOW_SIZE
        ),

        min_baseline_days=(
            TREND_MIN_BASELINE_DAYS
        ),

        trend_days=(
            TREND_DAYS
        ),

        alpha=(
            TREND_ALPHA
        ),

        save_result=True,
    )

    return extract_latest_trend_result(
        results
    )


# ============================================================
# 10. 실패 결과 생성
# ============================================================

def failed_result(
    payload: dict,
    error: str,
    *,
    cnn_result: Optional[Dict[str, Any]] = None,
    cnn_status: Optional[str] = None,
    cnn_error: Optional[str] = None,
) -> Dict[str, Any]:

    return {
        "ok": False,

        "status": "failed",

        "request_id": payload.get(
            "request_id"
        ),

        "error": error,

        "cnn_result": cnn_result,

        "cnn_status": cnn_status,

        "cnn_error": cnn_error,

        "processed_at": now_iso(),
    }


# ============================================================
# 11. Kafka 메시지 분석
# ============================================================

def process_message(
    payload: Dict[str, Any],
    collection,
) -> Dict[str, Any]:
    """
    Jetson에서 전달된 메시지를 처리한다.

    처리 흐름:

    1. 메시지 검증
    2. Jetson 감정 점수 확인
    3. request_id 중복 확인
    4. 공유 스토리지의 FLAC 확인
    5. CNN Normal/MCI 분석
    6. 언어·인지 분석
    7. MongoDB 중복 저장 방지
    8. 인지 변화 추세 분석
    9. Kafka Result Topic용 결과 생성

    서버에서 감정 모델이나 LLM은 실행하지 않는다.

    CNN 오류는 텍스트 인지 분석을 중단시키지 않는다.
    """

    # --------------------------------------------------------
    # 11-1. 입력 검증
    # --------------------------------------------------------

    if not isinstance(
        payload,
        dict,
    ):
        return failed_result(
            {},
            "invalid_payload",
        )

    request_id = str(
        payload.get(
            "request_id"
        )
        or ""
    ).strip()

    text = str(
        payload.get(
            "text"
        )
        or ""
    ).strip()

    user_id = str(
        payload.get(
            "user_id"
        )
        or ""
    ).strip()

    session_id = str(
        payload.get(
            "session_id"
        )
        or "S001"
    ).strip()

    audio_path = payload.get(
        "audio_path"
    )

    if not request_id:
        return failed_result(
            payload,
            "missing_request_id",
        )

    if not user_id:
        return failed_result(
            payload,
            "missing_user_id",
        )

    if not text:
        return failed_result(
            payload,
            "empty_text",
        )

    # 사용자 발화만 인지 분석 대상으로 사용
    if (
        payload.get(
            "speaker",
            "user",
        )
        != "user"
    ):
        return {
            "ok": True,

            "status": "skipped",

            "request_id": request_id,

            "message": (
                "사용자 발화가 아니므로 "
                "인지 분석 대상에서 제외했습니다."
            ),

            "processed_at": now_iso(),
        }

    # --------------------------------------------------------
    # 11-2. Jetson 감정 분석 결과 추출
    # --------------------------------------------------------

    try:
        emotion_scores = (
            normalize_emotion_scores(
                payload.get(
                    "emotion_scores"
                )
            )
        )

    except ValueError as exc:
        return failed_result(
            payload,
            str(exc),
        )

    has_emotion = (
        emotion_scores is not None
    )

    logger.info(
        "[분석 시작] "
        "request_id=%s "
        "user_id=%s "
        "emotion_available=%s "
        "audio_path=%s",
        request_id,
        user_id,
        has_emotion,
        audio_path,
    )

    # --------------------------------------------------------
    # 11-3. 중복 메시지 확인
    # --------------------------------------------------------

    existing = collection.find_one(
        {
            "request_id": request_id
        }
    )

    duplicate = (
        existing is not None
    )

    cnn_result = None
    cnn_status = None
    cnn_error = None

    # --------------------------------------------------------
    # 11-4. 신규 메시지만 CNN + 인지 분석
    # --------------------------------------------------------

    if existing is None:

        # ----------------------------------------------------
        # 11-4-1. CNN 음성 분석
        # ----------------------------------------------------

        (
            cnn_result,
            cnn_status,
            cnn_error,
        ) = run_cnn_analysis(
            audio_path
        )

        # CNN 실패는 전체 요청 실패로 처리하지 않는다.
        #
        # CNN 결과는 현재 Kafka Result Topic에만 포함한다.
        #
        # MongoDB event에 저장하는 작업은
        # 전체 8단계 중 7단계에서 수행한다.

        # ----------------------------------------------------
        # 11-4-2. 언어·인지 분석
        # ----------------------------------------------------

        # 중요:
        #
        # 기존 dementia_module.py는
        # scores=None일 경우 서버 감정 모델을
        # 실행할 수 있다.
        #
        # 감정 점수가 없다면 {}를 전달하여
        # 서버 감정 모델이 실행되지 않도록 한다.

        analysis_result = analyze(
            user_id=user_id,

            session_id=session_id,

            utterance_id=payload.get(
                "utterance_id"
            ),

            timestamp=payload.get(
                "timestamp"
            ),

            date=payload.get(
                "date"
            ),

            speaker="user",

            text=text,

            # Jetson 감정 분석 결과 사용
            scores=(
                emotion_scores
                if has_emotion
                else {}
            ),

            # 서버 자체 감정 분석 비활성화
            use_external_scores=True,

            source_file="jetson_stt",

            # Consumer에서 직접
            # 중복 방지 후 MongoDB 저장
            save_to_mongo=False,

            # LSTM 자동 실행 사용하지 않음
            enable_lstm_inference=False,
        )

        if not analysis_result.get(
            "ok"
        ):
            return failed_result(
                payload,

                analysis_result.get(
                    "error",
                    "analysis_failed",
                ),

                cnn_result=cnn_result,
                cnn_status=cnn_status,
                cnn_error=cnn_error,
            )

        # ----------------------------------------------------
        # 11-5. 분석 결과 정리
        # ----------------------------------------------------

        event = analysis_result[
            "event"
        ]

        event[
            "request_id"
        ] = request_id
        
        attach_cnn_result_to_event(
            event,
            cnn_result=cnn_result,
            cnn_status=cnn_status,
            cnn_error=cnn_error,
            audio_path=audio_path,
        )

        event[
            "emotion_source"
        ] = (
            "jetson"
            if has_emotion
            else "missing"
        )

        event[
            "emotion_scores_available"
        ] = has_emotion

        # 감정 점수 누락 시
        # 실제 0점으로 오인하지 않도록
        # MongoDB 이벤트에 None 저장
        if not has_emotion:

            for key in EMOTIONS:
                event[
                    f"emotion_{key}"
                ] = None

            event[
                "emotion_label"
            ] = "unknown"

            for field in (
                "emotion_score",
                "emotion_total_score",
                "emotion_positive_score",
                "emotion_negative_score",
            ):
                event[
                    field
                ] = None

        # ----------------------------------------------------
        # 11-6. MongoDB 저장
        # ----------------------------------------------------

        try:
            write = collection.update_one(
                {
                    "request_id": (
                        request_id
                    )
                },

                {
                    "$setOnInsert": (
                        event
                    )
                },

                upsert=True,
            )

            event_id = (
                str(
                    write.upserted_id
                )
                if write.upserted_id
                else None
            )

            duplicate = (
                write.upserted_id
                is None
            )

        except DuplicateKeyError:

            duplicate = True
            event_id = None

        # 동시에 동일 request_id가 들어온 경우 등
        # 기존 ID를 다시 조회
        if event_id is None:

            existing = collection.find_one(
                {
                    "request_id": (
                        request_id
                    )
                },

                {
                    "_id": 1
                },
            )

            event_id = (
                str(
                    existing["_id"]
                )
                if existing
                else None
            )

    else:

        # 이미 MongoDB에 저장된 request_id는
        # CNN 추론까지 다시 수행하지 않는다.
        event_id = str(
            existing["_id"]
        )

        cnn_status = (
            "skipped_duplicate"
        )

        logger.info(
            "[중복 메시지] "
            "이미 저장된 발화: %s",
            request_id,
        )

    # --------------------------------------------------------
    # 11-7. 인지 변화 추세 분석
    # --------------------------------------------------------

    trend_result = None
    trend_error = None

    run_trend_flag = (
        payload.get(
            "run_trend",
            True,
        )
        is True
    )

    if (
        run_trend_flag
        and has_emotion
        and not duplicate
    ):

        try:
            trend_result = run_trend(
                user_id
            )

        except Exception as exc:

            trend_error = str(
                exc
            )

            logger.exception(
                "[Trend] 분석 실패: "
                "user_id=%s",
                user_id,
            )

    # --------------------------------------------------------
    # 11-8. 최종 처리 결과
    # --------------------------------------------------------

    return {
        "ok": True,

        "status": "processed",

        "request_id": request_id,

        "user_id": user_id,

        "session_id": session_id,

        "event_id": event_id,

        "saved_to_mongo": True,

        "duplicate": duplicate,

        "emotion_scores_available": (
            has_emotion
        ),

        "audio_path": (
            str(audio_path)
            if audio_path
            else None
        ),

        # --------------------------------------------
        # CNN 결과
        # --------------------------------------------

        "cnn_status": cnn_status,

        "cnn_result": cnn_result,

        "cnn_error": cnn_error,

        # --------------------------------------------
        # Trend 결과
        # --------------------------------------------

        "trend_result": trend_result,

        "trend_error": trend_error,

        "processed_at": now_iso(),
    }


# ============================================================
# 12. Consumer 메인 실행
# ============================================================

def main() -> None:

    logging.basicConfig(
        level=logging.INFO,

        format=(
            "%(asctime)s "
            "[%(levelname)s] "
            "%(message)s"
        ),
    )

    logger.info(
        "[Consumer] 시작"
    )

    logger.info(
        "[Consumer] MongoDB 인덱스 준비"
    )

    ensure_indexes()

    # MongoDB 연결
    mongo_client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5000,
    )

    consumer = None
    producer = None

    try:

        collection = mongo_client[
            DB_NAME
        ][
            EVENT_COLLECTION
        ]

        mongo_client.admin.command(
            "ping"
        )

        logger.info(
            "[Consumer] MongoDB 연결 완료"
        )

        # request_id 기준
        # 중복 저장 방지 인덱스
        collection.create_index(
            "request_id",

            unique=True,

            partialFilterExpression={
                "request_id": {
                    "$type": "string"
                }
            },
        )

        # ----------------------------------------------------
        # CNN 모델은 Consumer 시작 시 단 한 번 로드
        # ----------------------------------------------------

        initialize_cnn()

        # ----------------------------------------------------
        # Kafka 연결
        # ----------------------------------------------------

        consumer = make_consumer()

        producer = make_producer()

        logger.info(
            "[Consumer] 분석 Consumer 시작"
        )

        logger.info(
            "[Consumer] 입력 Topic: %s",
            RAW_TOPIC,
        )

        logger.info(
            "[Consumer] 결과 Topic: %s",
            RESULT_TOPIC,
        )

        logger.info(
            "[Consumer] CNN 상태: %s",
            (
                "READY"
                if _cnn is not None
                else "UNAVAILABLE"
            ),
        )

        # ----------------------------------------------------
        # Kafka 메시지 처리 루프
        # ----------------------------------------------------

        for record in consumer:

            logger.info(
                "[Kafka] 메시지 수신 "
                "topic=%s "
                "partition=%s "
                "offset=%s",
                record.topic,
                record.partition,
                record.offset,
            )

            # MongoDB 장애 또는
            # Kafka 결과 발행 실패 시
            # offset을 commit하지 않는다.
            #
            # 재시작하면 해당 메시지를
            # 다시 처리할 수 있다.

            result = process_message(
                record.value,
                collection,
            )

            # 분석 결과 Kafka 전송
            producer.send(
                RESULT_TOPIC,
                value=result,
            ).get(
                timeout=30
            )

            # 분석 + Result Topic 발행이
            # 모두 성공한 뒤 offset 확정
            consumer.commit()

            logger.info(
                "[분석 완료] "
                "request_id=%s "
                "status=%s "
                "event_id=%s "
                "duplicate=%s "
                "cnn_status=%s",
                result.get(
                    "request_id"
                ),
                result.get(
                    "status"
                ),
                result.get(
                    "event_id"
                ),
                result.get(
                    "duplicate"
                ),
                result.get(
                    "cnn_status"
                ),
            )

    except KeyboardInterrupt:

        logger.info(
            "[Consumer] 종료 요청"
        )

    except Exception:

        logger.exception(
            "[Consumer] 처리 오류 발생. "
            "Kafka offset 미확정 상태입니다."
        )

        raise

    finally:

        if consumer is not None:
            consumer.close(
                autocommit=False
            )

        if producer is not None:
            producer.close(
                timeout=10
            )

        close_cnn()

        mongo_client.close()

        logger.info(
            "[Consumer] 종료 완료"
        )


# ============================================================
# 13. 실행 진입점
# ============================================================

if __name__ == "__main__":
    main()
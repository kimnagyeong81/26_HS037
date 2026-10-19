# -*- coding: utf-8 -*-
"""Merry AI - Jetson 발화/감정 결과 수신 전용 FastAPI 서버.

Jetson에서 STT 및 감정 분석이 완료된 데이터를 받아 Kafka에 적재합니다.
서버에서는 LLM 답변 생성이나 감정 모델 추론을 하지 않습니다.
분석 및 MongoDB 저장은 별도 analysis_consumer.py 프로세스가 담당합니다.
"""

from __future__ import annotations

import json
import logging
import hmac
import math
import os
import signal
import subprocess
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo
from fastapi import (
    FastAPI,
    BackgroundTasks,
    UploadFile,
    File,
    Form,
    Header,
    status,
)
from kafka import KafkaProducer
from pydantic import BaseModel, Field
import shutil
from kafka_k8s_runtime import ensure_kafka_topic

logger = logging.getLogger(__name__)

HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", "8000"))
KAFKA_SERVER = os.getenv("KAFKA_SERVER", "localhost:9092")
RAW_TOPIC = os.getenv("RAW_TOPIC", "dementia-raw-utterance-topic")
START_CONSUMER = os.getenv("START_CONSUMER", "false").lower() == "true"
# 설정 시 요청 헤더 X-API-Key로 인증. 사설망 외 공개 시 반드시 인증/TLS를 적용하세요.
MERRY_API_KEY = os.getenv("MERRY_API_KEY", "").strip()
KST = ZoneInfo("Asia/Seoul")
EMOTION_KEYS = {"joy", "calm", "anger", "fear", "sadness"}
UTTERANCE_AUDIO_DIR = Path(
    os.getenv(
        "UTTERANCE_AUDIO_DIR",
        "/data/utterances",
    )
)
_producer: Optional[KafkaProducer] = None
_producer_lock = threading.Lock()
_consumer_process: Optional[subprocess.Popen] = None
_consumer_log_file = None


class UtteranceRequest(BaseModel):
    type: str = Field(default="utterance")

    request_id: Optional[str] = None
    user_id: str = Field(default="user_001")
    session_id: str = Field(default="session_001")
    utterance_id: Optional[str] = None

    timestamp: Optional[str] = None
    date: Optional[str] = None
    speaker: str = Field(default="user")
    text: str

    emotion_scores: Optional[Dict[str, Any]] = None

    run_trend: bool = Field(default=True)



def validate_emotion_scores(scores: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """Jetson의 최신 5감정 점수(0~100)만 허용합니다."""
    if scores is None:
        return None  # 감정 추론 실패 시 텍스트 기반 분석은 계속 가능
    if set(scores) != EMOTION_KEYS:
        raise ValueError(f"emotion_scores에는 5개 키가 모두 필요합니다: {sorted(EMOTION_KEYS)}")
    if any(not math.isfinite(v) or not (0.0 <= v <= 100.0) for v in scores.values()):
        raise ValueError("감정 점수는 0~100 사이의 유한한 숫자여야 합니다.")
    return scores


class QueueResponse(BaseModel):
    ok: bool
    status: str
    message: str

    request_id: Optional[str] = None
    topic: Optional[str] = None
    queued_at: Optional[str] = None
    error: Optional[str] = None


def normalize_payload(request: UtteranceRequest) -> Dict[str, Any]:
    """Jetson의 요청을 analysis_consumer.py에서 사용할 Kafka 메시지로 정규화합니다."""
    text = request.text.strip()
    emotion_scores = validate_emotion_scores(request.emotion_scores)
    if not text:
        raise ValueError("분석할 발화(text)가 비어 있습니다.")

    now = datetime.now(timezone.utc)
    if request.timestamp:
        try:
            event_time = datetime.fromisoformat(request.timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("timestamp는 ISO 8601 형식이어야 합니다.") from exc
        if event_time.tzinfo is None:
            # 타임존 없는 Jetson 시각은 한국 현지시각으로 간주합니다.
            event_time = event_time.replace(tzinfo=KST)
    else:
        event_time = now

    local_date = event_time.astimezone(KST).date().isoformat()
    if request.date:
        try:
            date.fromisoformat(request.date)
        except ValueError as exc:
            raise ValueError("date는 YYYY-MM-DD 형식이어야 합니다.") from exc
        local_date = request.date

    return {
        "request_id": request.request_id or str(uuid.uuid4()),
        "message_type": request.type,
        "received_at": now.isoformat(),
        "source": "jetson_fastapi",
        "user_id": request.user_id,
        "session_id": request.session_id,
        "utterance_id": request.utterance_id,
        "timestamp": event_time.isoformat(),
        "date": local_date,
        "speaker": request.speaker,
        "text": request.text.strip(),
        "emotion_scores": request.emotion_scores,
        "run_trend": request.run_trend
    }


def get_producer() -> KafkaProducer:
    global _producer
    with _producer_lock:
        if _producer is None:
            _producer = KafkaProducer(
                bootstrap_servers=KAFKA_SERVER,
                value_serializer=lambda value: json.dumps(value, ensure_ascii=False).encode("utf-8"),
                acks="all",
                retries=5,
                linger_ms=10,
            )
    return _producer


def send_to_kafka(payload: Dict[str, Any]) -> None:
    """브로커의 적재 확인을 받고 나서만 클라이언트에 202를 반환합니다."""
    future = get_producer().send(RAW_TOPIC, value=payload)
    future.get(timeout=30)


def close_producer() -> None:
    global _producer
    with _producer_lock:
        if _producer is not None:
            try:
                _producer.flush(timeout=10)
                _producer.close(timeout=10)
            finally:
                _producer = None


def start_analysis_consumer() -> None:
    """로컬 단독 실행 시에만 사용; Kubernetes에서는 Consumer를 별도 Pod로 실행합니다."""
    global _consumer_process, _consumer_log_file
    base_dir = Path(__file__).resolve().parent
    script = base_dir / "analysis_consumer.py"
    if not script.is_file():
        raise FileNotFoundError(f"Consumer 실행 파일 없음: {script}")
    if _consumer_process is not None and _consumer_process.poll() is None:
        return
    _consumer_log_file = open(base_dir / "analysis_consumer.log", "a", encoding="utf-8")
    _consumer_process = subprocess.Popen(
        [sys.executable, str(script)],
        cwd=str(base_dir),
        stdout=_consumer_log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    logger.info("Consumer 시작: pid=%s", _consumer_process.pid)


def stop_analysis_consumer() -> None:
    global _consumer_process, _consumer_log_file
    if _consumer_process is not None and _consumer_process.poll() is None:
        try:
            os.killpg(os.getpgid(_consumer_process.pid), signal.SIGTERM)
            _consumer_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _consumer_process.kill()
            _consumer_process.wait(timeout=5)
        except ProcessLookupError:
            pass
    _consumer_process = None
    if _consumer_log_file is not None:
        _consumer_log_file.close()
        _consumer_log_file = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_kafka_topic(bootstrap_server=KAFKA_SERVER, topic=RAW_TOPIC, timeout_sec=60)
    if START_CONSUMER:
        start_analysis_consumer()
    try:
        yield
    finally:
        close_producer()
        stop_analysis_consumer()


app = FastAPI(
    title="Merry AI Cognitive Analysis Receiver",
    description="Jetson STT 텍스트 + 5종 감정 점수 → Kafka (서버 LLM/감정 재분석 없음)",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "service": "Merry AI Cognitive Analysis Receiver",
        "topic": RAW_TOPIC,
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health")
def health():
    # 프로세스 생존 확인용이며 Kafka/MongoDB 처리 완료를 보장하지 않습니다.
    return {"ok": True, "status": "running", "server_time": datetime.now(timezone.utc).isoformat()}


@app.post("/utterance", response_model=QueueResponse, status_code=status.HTTP_202_ACCEPTED)
def receive_utterance(request: UtteranceRequest, x_api_key: Optional[str] = Header(default=None)):
    if MERRY_API_KEY and not hmac.compare_digest(x_api_key or "", MERRY_API_KEY):
        raise HTTPException(status_code=401, detail="API 인증에 실패했습니다.")

    try:
        payload = normalize_payload(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        send_to_kafka(payload)
    except Exception:
        logger.exception("Kafka 메시지 전송 실패: request_id=%s", payload["request_id"])
        raise HTTPException(status_code=503, detail="Kafka 메시지 적재에 실패했습니다.")

    return QueueResponse(
        ok=True,
        status="queued",
        message="Kafka 적재 완료 (인지 분석 결과는 별도 Consumer가 처리합니다.)",
        request_id=payload["request_id"],
        topic=RAW_TOPIC,
        queued_at=datetime.now(timezone.utc).isoformat(),
    )

@app.post(
    "/utterance/audio",
    response_model=QueueResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def receive_utterance_audio(
    background_tasks: BackgroundTasks,
    metadata: str = Form(...),
    audio: UploadFile = File(...),
):
    """
    Jetson에서 STT metadata + 실제 음성 파일을 수신한다.

    multipart/form-data

    metadata:
        UtteranceRequest JSON 문자열

    audio:
        FLAC / WAV 파일
    """

    saved_path = None

    try:
        # ---------------------------------------------------------------------
        # 1. metadata JSON -> Pydantic
        # ---------------------------------------------------------------------

        request = UtteranceRequest.model_validate_json(
            metadata
        )

        payload = normalize_payload(
            request
        )

        if not payload["text"]:
            return QueueResponse(
                ok=False,
                status="rejected",
                message="text가 비어 있어 Kafka에 전송하지 않았습니다.",
                request_id=payload["request_id"],
                error="empty_text",
            )

        # ---------------------------------------------------------------------
        # 2. 확장자 검사
        # ---------------------------------------------------------------------

        original_name = (
            audio.filename
            or "audio.flac"
        )

        suffix = Path(
            original_name
        ).suffix.lower()

        allowed_suffixes = {
            ".flac",
            ".wav",
        }

        if suffix not in allowed_suffixes:
            return QueueResponse(
                ok=False,
                status="rejected",
                message=(
                    "지원하지 않는 음성 형식입니다."
                ),
                request_id=payload["request_id"],
                error=(
                    f"unsupported_audio_format:{suffix}"
                ),
            )

        # ---------------------------------------------------------------------
        # 3. 저장 디렉터리 생성
        # ---------------------------------------------------------------------

        UTTERANCE_AUDIO_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # request_id를 파일명으로 사용
        # Jetson 파일명을 그대로 신뢰하지 않음.
        saved_path = (
            UTTERANCE_AUDIO_DIR
            / (
                f"{payload['request_id']}"
                f"{suffix}"
            )
        )

        # ---------------------------------------------------------------------
        # 4. 실제 음성 파일 저장
        # ---------------------------------------------------------------------

        with saved_path.open(
            "wb"
        ) as output_file:
            shutil.copyfileobj(
                audio.file,
                output_file,
            )

        # ---------------------------------------------------------------------
        # 5. Kafka payload에는 서버 경로를 기록
        # ---------------------------------------------------------------------

        payload["audio_path"] = str(
            saved_path
        )

        payload["audio_filename"] = (
            saved_path.name
        )

        payload["audio_content_type"] = (
            audio.content_type
        )

        # ---------------------------------------------------------------------
        # 6. Kafka 적재
        # ---------------------------------------------------------------------

        send_to_kafka(
            payload
        )


        print(
            "[API] 음성 수신 완료 "
            f"request_id={payload['request_id']} "
            f"path={saved_path}"
        )

        return QueueResponse(
            ok=True,
            status="queued",
            message=(
                "음성 저장 및 Kafka 큐 적재 완료"
            ),
            request_id=payload[
                "request_id"
            ],
            topic=RAW_TOPIC,
            queued_at=(
                datetime.now().isoformat()
            ),
        )

    except Exception as exc:

        # Kafka 전송 전에 실패했는데
        # 불완전한 파일이 남았다면 삭제
        if (
            saved_path is not None
            and saved_path.exists()
        ):
            try:
                saved_path.unlink()
            except Exception:
                pass

        return QueueResponse(
            ok=False,
            status="error",
            message=(
                "음성 수신 처리 중 오류 발생"
            ),
            error=str(exc),
        )

    finally:
        try:
            await audio.close()
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, reload=False)

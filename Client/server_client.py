# -*- coding: utf-8 -*-
"""
Merry AI - Jetson 서버 전송 모듈

역할:
1. STT 텍스트와 Jetson 감정 점수를 전송 큐에 등록
2. 백그라운드 스레드에서 FastAPI POST 요청
3. 네트워크 오류 시 재시도
4. 서버 장애가 Jetson의 대화를 방해하지 않도록 처리

서버는 LLM 및 감정 분석을 수행하지 않는다.
"""

from __future__ import annotations

import logging
import math
import os
import queue
import threading
import time
import uuid
import json
from pathlib import Path
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo
import requests

logger = logging.getLogger(__name__)
KST = ZoneInfo("Asia/Seoul")
SERVER_URL = os.getenv(
    "MERRY_SERVER_URL", "http://100.84.183.27:32081"
).strip().rstrip("/")

API_KEY = os.getenv("MERRY_API_KEY", "").strip()

REQUEST_TIMEOUT = 4
MAX_RETRIES = 3
QUEUE_SIZE = 100

EMOTIONS = (
    "joy",
    "calm",
    "anger",
    "fear",
    "sadness",
)

KST = ZoneInfo("Asia/Seoul")

_send_queue = queue.Queue(maxsize=QUEUE_SIZE)
_stop_signal = object()

_worker = None
_worker_lock = threading.Lock()


# ============================================================
# 1. 감정 점수 정규화
# ============================================================

def normalize_scores(scores):
    """Jetson 감정 점수를 JSON 전송용 float로 변환."""

    if not isinstance(scores, dict):
        return None

    try:
        result = {
            emotion: float(scores[emotion])
            for emotion in EMOTIONS
        }

        if any(
            not math.isfinite(value) or not 0 <= value <= 100
            for value in result.values()
        ):
            raise ValueError("감정 점수가 0~100 범위를 벗어났습니다.")

        return result

    except (KeyError, TypeError, ValueError):
        logger.warning(
            "[Server] 감정 점수 검증 실패. "
            "감정값 없이 발화를 전송합니다."
        )
        return None


# ============================================================
# 2. 서버 HTTP 전송
# ============================================================

# def _send_payload(payload):
#     """FastAPI에 데이터를 전송하며 실패 시 재시도."""

#     headers = {
#         "Content-Type": "application/json"
#     }

#     if API_KEY:
#         headers["X-API-Key"] = API_KEY

#     data = json.dumps(
#         payload,
#         ensure_ascii=False
#     ).encode("utf-8")

#     for attempt in range(1, MAX_RETRIES + 1):

#         try:
#             request = Request(
#                 url=f"{SERVER_URL}/utterance",
#                 data=data,
#                 headers=headers,
#                 method="POST",
#             )

#             with urlopen(
#                 request,
#                 timeout=REQUEST_TIMEOUT
#             ) as response:

#                 if response.status != 202:
#                     logger.warning(
#                         "[Server] 예상하지 못한 HTTP 상태: %s",
#                         response.status
#                     )
#                     return

#                 logger.info(
#                     "[Server] Kafka 적재 확인: request_id=%s",
#                     payload["request_id"]
#                 )

#                 return

#         except HTTPError as exc:
#             # 잘못된 요청·인증 실패는 재시도해도 해결되지 않음
#             if 400 <= exc.code < 500 and exc.code != 429:
#                 logger.error(
#                     "[Server] HTTP %s: request_id=%s",
#                     exc.code,
#                     payload["request_id"]
#                 )
#                 return

#             error = exc

#         except (URLError, TimeoutError, OSError) as exc:
#             error = exc

#         logger.warning(
#             "[Server] 전송 실패 (%s/%s): %s",
#             attempt,
#             MAX_RETRIES,
#             error
#         )

#         if attempt < MAX_RETRIES:
#             time.sleep(2 ** (attempt - 1))

#     logger.error(
#         "[Server] 최종 전송 실패: request_id=%s",
#         payload["request_id"]
#     )
def _send_payload(payload: dict) -> bool:
    """
    서버로 인지 분석 데이터를 전송한다.

    audio_path가 있으면:
        POST /utterance/audio
        multipart/form-data

    audio_path가 없으면:
        기존 POST /utterance
        application/json
    """

    if not SERVER_URL:
        logger.warning(
            "[Server] MERRY_SERVER_URL이 설정되지 않았습니다."
        )
        return False

    request_id = payload.get("request_id")

    headers = {}

    if API_KEY:
        headers["X-API-Key"] = API_KEY

    # -------------------------------------------------------------------------
    # 음성 파일이 존재하는 경우
    # -------------------------------------------------------------------------

    audio_path_value = payload.get("audio_path")

    if audio_path_value:

        audio_path = Path(audio_path_value)

        if not audio_path.exists():
            logger.error(
                "[Server] 음성 파일이 없습니다: %s",
                audio_path,
            )
            return False

        if not audio_path.is_file():
            logger.error(
                "[Server] audio_path가 파일이 아닙니다: %s",
                audio_path,
            )
            return False

        # Jetson 로컬 경로 자체는 서버가 사용할 수 없으므로
        # metadata에서는 제거한다.
        metadata = dict(payload)
        metadata.pop(
            "audio_path",
            None,
        )

        metadata["audio_filename"] = audio_path.name

        url = (
            f"{SERVER_URL}/utterance/audio"
        )

        suffix = audio_path.suffix.lower()

        if suffix == ".flac":
            content_type = "audio/flac"
        elif suffix == ".wav":
            content_type = "audio/wav"
        else:
            content_type = (
                "application/octet-stream"
            )

        for attempt in range(
            1,
            MAX_RETRIES + 1,
        ):
            try:
                logger.info(
                    "[Server] 음성 업로드 시도 "
                    "request_id=%s attempt=%s/%s "
                    "file=%s",
                    request_id,
                    attempt,
                    MAX_RETRIES,
                    audio_path,
                )

                with audio_path.open("rb") as audio_file:

                    files = {
                        "audio": (
                            audio_path.name,
                            audio_file,
                            content_type,
                        )
                    }

                    form_data = {
                        "metadata": json.dumps(
                            metadata,
                            ensure_ascii=False,
                        )
                    }

                    response = requests.post(
                        url,
                        data=form_data,
                        files=files,
                        headers=headers,
                        timeout=REQUEST_TIMEOUT,
                    )

                if response.status_code in (200, 202):

                    try:
                        result = response.json()
                    except Exception:
                        result = {}

                    logger.info(
                        "[Server] 음성 + Kafka 적재 확인: "
                        "request_id=%s "
                        "server_request_id=%s",
                        request_id,
                        result.get("request_id"),
                    )

                    return True

                logger.warning(
                    "[Server] 음성 업로드 실패: "
                    "status=%s body=%s",
                    response.status_code,
                    response.text[:500],
                )

            except requests.RequestException as exc:

                logger.warning(
                    "[Server] 음성 업로드 통신 실패 "
                    "attempt=%s/%s: %s",
                    attempt,
                    MAX_RETRIES,
                    exc,
                )

            except Exception:
                logger.exception(
                    "[Server] 음성 업로드 중 예외"
                )

            if attempt < MAX_RETRIES:
                time.sleep(1)

        return False

    # -------------------------------------------------------------------------
    # 음성 파일이 없는 경우
    # 기존 JSON API fallback
    # -------------------------------------------------------------------------

    url = (
        f"{SERVER_URL}/utterance"
    )

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        try:

            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 202:

                logger.info(
                    "[Server] Kafka 적재 확인: "
                    "request_id=%s",
                    request_id,
                )

                return True

            logger.warning(
                "[Server] HTTP 실패: "
                "status=%s body=%s",
                response.status_code,
                response.text[:500],
            )

        except requests.RequestException as exc:

            logger.warning(
                "[Server] 통신 실패 "
                "attempt=%s/%s: %s",
                attempt,
                MAX_RETRIES,
                exc,
            )

        except Exception:
            logger.exception(
                "[Server] 전송 중 예외"
            )

        if attempt < MAX_RETRIES:
            time.sleep(1)

    return False

# ============================================================
# 3. 백그라운드 전송 스레드
# ============================================================

def _worker_loop():

    while True:
        payload = _send_queue.get()

        try:
            if payload is _stop_signal:
                return

            _send_payload(payload)

        except Exception:
            logger.exception(
                "[Server] 백그라운드 전송 오류"
            )

        finally:
            _send_queue.task_done()


def _ensure_worker():
    """최초 전송 시 백그라운드 스레드 시작."""

    global _worker

    with _worker_lock:

        if _worker is None or not _worker.is_alive():

            _worker = threading.Thread(
                target=_worker_loop,
                name="merry-server-sender",
                daemon=True,
            )

            _worker.start()


# ============================================================
# 4. 서버 전송 큐 등록
# ============================================================

def enqueue_analysis(
    *,
    user_id,
    session_id,
    text,
    audio_path=None,
    timestamp=None,
    emotion_scores=None,
    utterance_id=None,
    run_trend=True,
):
    """
    발화 데이터를 서버 전송 큐에 등록한다.

    실제 HTTP 전송은 별도 스레드에서 처리된다.

    Returns:
        True: 전송 큐 등록 성공
        False: 등록 실패
    """

    if not SERVER_URL:
        logger.warning(
            "[Server] MERRY_SERVER_URL이 설정되지 않았습니다."
        )
        return False

    if not str(text or "").strip():
        return False

    # UTC timestamp를 한국시간 날짜로 변환
    try:
        dt = datetime.fromisoformat(
            timestamp.replace("Z", "+00:00")
        )

        if dt.tzinfo is None:
            raise ValueError("타임스탬프에 시간대가 없습니다.")

        local_date = dt.astimezone(
            KST
        ).date().isoformat()

    except (TypeError, AttributeError, ValueError):
        logger.error(
            "[Server] 타임스탬프 형식 오류: %r",
            timestamp
        )
        return False

    # 요청 ID는 생성 시 한 번만 결정한다.
    # 같은 요청을 재시도할 때도 동일한 ID를 사용한다.
    request_id = str(uuid.uuid4())


    # -------------------------------------------------------------------------
    # timestamp / date 정규화
    # -------------------------------------------------------------------------

    if timestamp:
        try:
            parsed_timestamp = datetime.fromisoformat(
                str(timestamp).replace(
                    "Z",
                    "+00:00",
                )
            )

            if parsed_timestamp.tzinfo is None:
                parsed_timestamp = parsed_timestamp.replace(
                    tzinfo=timezone.utc
                )

        except (ValueError, TypeError):
            logger.warning(
                "[Server] timestamp 파싱 실패: %r",
                timestamp,
            )

            parsed_timestamp = datetime.now(
                timezone.utc
            )

    else:
        parsed_timestamp = datetime.now(
            timezone.utc
        )

    timestamp_value = (
        parsed_timestamp.isoformat()
    )

    date_value = (
        parsed_timestamp
        .astimezone(KST)
        .date()
        .isoformat()
    )

    payload = {
        "type": "utterance",
        "request_id": str(uuid.uuid4()),

        "user_id": user_id,
        "session_id": session_id,
        "utterance_id": utterance_id,

        "timestamp": timestamp,
        "date": date_value,

        "speaker": "user",
        "text": text,

        "audio_path": (
            str(audio_path)
            if audio_path
            else None
        ),

        "emotion_scores": emotion_scores,

        "run_trend": bool(run_trend),
    }

    try:
        _ensure_worker()
        _send_queue.put_nowait(payload)

        logger.info(
            "[Server] 전송 큐 등록: request_id=%s",
            request_id
        )

        return True

    except queue.Full:
        logger.error(
            "[Server] 전송 큐가 가득 찼습니다. "
            "데이터를 등록하지 못했습니다."
        )
        return False


# ============================================================
# 5. 프로그램 종료 처리
# ============================================================

def close_sender(timeout=5):
    """프로그램 종료 시 남은 전송 작업을 제한된 시간 동안 처리."""

    if _worker is None or not _worker.is_alive():
        return

    try:
        _send_queue.put(
            _stop_signal,
            timeout=1
        )

        _worker.join(timeout=timeout)

        if _worker.is_alive():
            logger.warning(
                "[Server] 종료 대기 시간 초과. "
                "일부 데이터가 전송되지 않았을 수 있습니다."
            )

    except queue.Full:
        logger.warning(
            "[Server] 전송 큐가 가득 차 종료 신호를 전달하지 못했습니다."
        )
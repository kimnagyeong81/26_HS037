# -*- coding: utf-8 -*-
"""
Merry AI 서버 LLM 클라이언트.

Jetson에서는 LLM 모델을 직접 실행하지 않는다.

처리 흐름
Jetson STT
    ↓
get_general_response()
    ↓
HTTP POST /chat
    ↓
Server FastAPI
    ↓
Server GPU LLM
    ↓
merry_response 반환
    ↓
Jetson TTS

기존 main.py와의 호환성을 위해
get_general_response(), release_llm() 인터페이스를 유지한다.
"""

from __future__ import annotations

import json
import logging
import socket
from typing import Any
from urllib import error, request

from config import (
    DEFAULT_SESSION_ID,
    DEFAULT_USER_ID,
    LLM_API_KEY,
    LLM_CHAT_ENDPOINT,
    LLM_FALLBACK_RESPONSE,
    LLM_HEALTH_ENDPOINT,
    LLM_HEALTH_TIMEOUT_SEC,
    LLM_REQUEST_TIMEOUT_SEC,
    LLM_SERVER_URL,
)


logger = logging.getLogger(__name__)


# =============================================================================
# URL / Header
# =============================================================================

def _make_url(endpoint: str) -> str:
    """
    LLM 서버의 전체 URL을 생성한다.
    """

    if not LLM_SERVER_URL:
        return ""

    endpoint = (endpoint or "").strip()

    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint

    return f"{LLM_SERVER_URL}{endpoint}"


def _make_headers() -> dict[str, str]:
    """
    LLM 서버 요청에 사용할 HTTP Header를 생성한다.
    """

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json; charset=utf-8",
    }

    if LLM_API_KEY:
        headers["X-API-Key"] = LLM_API_KEY

    return headers


# =============================================================================
# HTTP Utility
# =============================================================================

def _read_json_response(
    response,
) -> dict[str, Any]:
    """
    HTTP 응답 Body를 JSON dict로 변환한다.
    """

    raw_body = response.read().decode(
        "utf-8",
        errors="replace",
    )

    if not raw_body:
        return {}

    try:
        data = json.loads(raw_body)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"서버 응답이 JSON 형식이 아닙니다: {raw_body[:300]!r}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"서버 응답 JSON이 객체(dict)가 아닙니다: {type(data).__name__}"
        )

    return data


# =============================================================================
# Health Check
# =============================================================================

def check_llm_server() -> bool:
    """
    서버 LLM API가 접근 가능한지 확인한다.

    서버가 연결되지 않더라도 False만 반환하며
    Merry 프로그램 자체를 종료시키지는 않는다.
    """

    url = _make_url(
        LLM_HEALTH_ENDPOINT
    )

    if not url:
        logger.warning(
            "[LLM Client] "
            "MERRY_LLM_SERVER_URL이 설정되지 않았습니다."
        )
        return False

    req = request.Request(
        url=url,
        method="GET",
        headers=_make_headers(),
    )

    try:
        with request.urlopen(
            req,
            timeout=LLM_HEALTH_TIMEOUT_SEC,
        ) as response:

            if response.status != 200:
                logger.warning(
                    "[LLM Client] "
                    "health check 실패: HTTP %s",
                    response.status,
                )
                return False

            data = _read_json_response(
                response
            )

        server_ok = (
            data.get("ok") is True
            or data.get("status") == "running"
        )

        if server_ok:
            logger.info(
                "[LLM Client] "
                "서버 health check 정상"
            )

            return True

        logger.warning(
            "[LLM Client] "
            "서버 health 응답이 정상 상태가 아닙니다: %s",
            data,
        )

        return False

    except error.HTTPError as exc:
        logger.warning(
            "[LLM Client] "
            "health check HTTP 오류: %s",
            exc.code,
        )

    except error.URLError as exc:
        logger.warning(
            "[LLM Client] "
            "health check 연결 실패: %s",
            exc.reason,
        )

    except (TimeoutError, socket.timeout):
        logger.warning(
            "[LLM Client] "
            "health check timeout: %.1f초",
            LLM_HEALTH_TIMEOUT_SEC,
        )

    except Exception:
        logger.exception(
            "[LLM Client] "
            "health check 중 예상하지 못한 오류"
        )

    return False


# =============================================================================
# LLM Response
# =============================================================================

def get_general_response(
    user_text: str,
    user_id: str = DEFAULT_USER_ID,
    session_id: str = DEFAULT_SESSION_ID,
) -> str:
    """
    사용자의 발화를 서버 LLM에 전달하고
    Merry 응답 문자열을 반환한다.

    Request:
    {
        "user_id": "U001",
        "session_id": "S001",
        "text": "오늘 기분이 좋아.",
        "save": false
    }

    Response:
    {
        "ok": true,
        "user_id": "U001",
        "session_id": "S001",
        "user_text": "오늘 기분이 좋아.",
        "merry_response": "...",
        "server_time": "..."
    }
    """

    user_text = (
        user_text
        or ""
    ).strip()

    if not user_text:
        return (
            "네, 말씀해 주세요. "
            "제가 여기 있어요."
        )

    url = _make_url(
        LLM_CHAT_ENDPOINT
    )

    if not url:
        logger.error(
            "[LLM Client] "
            "LLM 서버 URL이 설정되지 않았습니다. "
            "MERRY_LLM_SERVER_URL을 확인하세요."
        )

        return LLM_FALLBACK_RESPONSE

    payload = {
        "user_id": str(user_id),
        "session_id": str(session_id),
        "text": user_text,

        # 최종 대화 내용은 Jetson MongoDB에서 저장하므로
        # 서버 LLM의 별도 JSON 저장은 사용하지 않는다.
        "save": False,
    }

    body = json.dumps(
        payload,
        ensure_ascii=False,
    ).encode("utf-8")

    req = request.Request(
        url=url,
        data=body,
        method="POST",
        headers=_make_headers(),
    )

    logger.info(
        "[LLM Client] "
        "서버 LLM 요청 시작: "
        "user_id=%s session_id=%s",
        user_id,
        session_id,
    )

    try:
        with request.urlopen(
            req,
            timeout=LLM_REQUEST_TIMEOUT_SEC,
        ) as response:

            status_code = response.status

            data = _read_json_response(
                response
            )

        if status_code != 200:
            logger.error(
                "[LLM Client] "
                "서버 LLM 응답 오류: HTTP %s",
                status_code,
            )

            return LLM_FALLBACK_RESPONSE

        # 서버 내부에서 LLM 처리 실패
        if data.get("ok") is not True:
            logger.error(
                "[LLM Client] "
                "서버 LLM 처리 실패: %s",
                data.get("error"),
            )

            server_fallback = str(
                data.get("merry_response")
                or ""
            ).strip()

            if server_fallback:
                return server_fallback

            return LLM_FALLBACK_RESPONSE

        reply = str(
            data.get("merry_response")
            or ""
        ).strip()

        if not reply:
            logger.warning(
                "[LLM Client] "
                "서버에서 빈 LLM 응답을 반환했습니다."
            )

            return LLM_FALLBACK_RESPONSE

        logger.info(
            "[LLM Client] "
            "서버 LLM 응답 수신 완료"
        )

        return reply

    except error.HTTPError as exc:
        try:
            error_body = exc.read().decode(
                "utf-8",
                errors="replace",
            )

        except Exception:
            error_body = ""

        logger.error(
            "[LLM Client] "
            "HTTP 오류: status=%s body=%s",
            exc.code,
            error_body[:500],
        )

    except error.URLError as exc:
        logger.error(
            "[LLM Client] "
            "서버 연결 실패: %s",
            exc.reason,
        )

    except (TimeoutError, socket.timeout):
        logger.error(
            "[LLM Client] "
            "서버 응답 timeout: %.1f초",
            LLM_REQUEST_TIMEOUT_SEC,
        )

    except ValueError as exc:
        logger.error(
            "[LLM Client] "
            "서버 응답 형식 오류: %s",
            exc,
        )

    except Exception:
        logger.exception(
            "[LLM Client] "
            "LLM 요청 중 예상하지 못한 오류"
        )

    return LLM_FALLBACK_RESPONSE


# =============================================================================
# Release
# =============================================================================

def release_llm() -> None:
    """
    기존 main.py 인터페이스 호환용 함수.

    LLM 모델은 서버에서 관리하므로
    Jetson에서 해제할 LLM 모델은 없다.
    """

    logger.info(
        "[LLM Client] 종료"
    )
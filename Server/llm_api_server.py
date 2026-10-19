# -*- coding: utf-8 -*-
"""
Merry AI LLM API Server.

구조
Jetson
    ↓ HTTP POST /chat
FastAPI
    ↓
llm_module.py
    ↓
llama-cpp-python
    ↓
RTX 3060 Ti
    ↓
Merry response

주의
- llama.cpp 모델 객체는 프로세스당 하나만 유지한다.
- uvicorn workers는 반드시 1로 사용한다.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from llm_module import (
    clear_history,
    get_general_response,
    get_model_status,
    is_model_loaded,
    load_llm_model,
    release_llm,
    save_result,
)


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


# =============================================================================
# Request / Response Models
# =============================================================================

class ChatRequest(BaseModel):
    user_id: str = Field(
        default="U001",
    )

    session_id: str = Field(
        default="S001",
    )

    text: str

    # Jetson은 로컬 MongoDB에서 대화를 저장하므로
    # 기본적으로 False 사용
    save: bool = Field(
        default=False,
    )


class ChatResponse(BaseModel):
    ok: bool

    user_id: str
    session_id: str

    user_text: str
    merry_response: str

    server_time: str

    error: Optional[str] = None


class HealthResponse(BaseModel):
    ok: bool
    status: str
    model_loaded: bool
    model: dict
    server_time: str


class HistoryClearRequest(BaseModel):
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class HistoryClearResponse(BaseModel):
    ok: bool
    deleted_sessions: int
    server_time: str


# =============================================================================
# Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 시작 시 모델을 미리 GPU에 로드한다.

    Jetson의 첫 요청에서 모델 로딩 시간을 기다리지 않도록
    서버 시작 시 RTX GPU에 GGUF를 올린다.
    """

    logger.info(
        "[LLM API] 서버 시작"
    )

    try:
        load_llm_model()

        logger.info(
            "[LLM API] LLM preload 완료"
        )

    except Exception:
        logger.exception(
            "[LLM API] LLM preload 실패"
        )

        # 여기서는 서버 자체를 죽이지 않는다.
        # /health에서 model_loaded=False로 확인 가능.

    yield

    logger.info(
        "[LLM API] 서버 종료"
    )

    try:
        release_llm()

    except Exception:
        logger.exception(
            "[LLM API] LLM release 실패"
        )


# =============================================================================
# FastAPI
# =============================================================================

app = FastAPI(
    title="Merry AI LLM API",
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# Root
# =============================================================================

@app.get("/")
def root():
    return {
        "service": "merry-llm",
        "status": "running",
    }


# =============================================================================
# Health
# =============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():
    loaded = is_model_loaded()

    return HealthResponse(
        ok=loaded,
        status=(
            "running"
            if loaded
            else "model_not_loaded"
        ),
        model_loaded=loaded,
        model=get_model_status(),
        server_time=datetime.now(
            timezone.utc
        ).isoformat(),
    )


# =============================================================================
# Chat
# =============================================================================

@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
):
    """
    Jetson에서 STT 결과를 받아
    서버 GPU LLM으로 Merry 응답을 생성한다.
    """

    user_id = (
        request.user_id
        or "U001"
    ).strip()

    session_id = (
        request.session_id
        or "S001"
    ).strip()

    user_text = (
        request.text
        or ""
    ).strip()

    server_time = datetime.now(
        timezone.utc
    ).isoformat()

    if not user_text:
        return ChatResponse(
            ok=False,
            user_id=user_id,
            session_id=session_id,
            user_text="",
            merry_response=(
                "네, 말씀해 주세요. "
                "제가 여기 있어요."
            ),
            server_time=server_time,
            error="empty_text",
        )

    logger.info(
        "[LLM API] /chat "
        "user_id=%s session_id=%s text=%r",
        user_id,
        session_id,
        user_text[:100],
    )

    try:
        merry_response = get_general_response(
            user_text=user_text,
            user_id=user_id,
            session_id=session_id,
        )

        if request.save:
            try:
                save_result(
                    user_id=user_id,
                    session_id=session_id,
                    user_text=user_text,
                    merry_response=merry_response,
                )

            except Exception:
                logger.exception(
                    "[LLM API] 결과 저장 실패"
                )

        logger.info(
            "[LLM API] /chat 완료 "
            "user_id=%s session_id=%s",
            user_id,
            session_id,
        )

        return ChatResponse(
            ok=True,
            user_id=user_id,
            session_id=session_id,
            user_text=user_text,
            merry_response=merry_response,
            server_time=server_time,
            error=None,
        )

    except Exception as exc:
        logger.exception(
            "[LLM API] /chat 처리 실패"
        )

        return ChatResponse(
            ok=False,
            user_id=user_id,
            session_id=session_id,
            user_text=user_text,
            merry_response=(
                "죄송해요, 지금은 잠시 대답이 어려워요. "
                "조금만 있다가 다시 말씀해 주세요."
            ),
            server_time=server_time,
            error=str(exc),
        )


# =============================================================================
# History
# =============================================================================

@app.post(
    "/history/clear",
    response_model=HistoryClearResponse,
)
def history_clear(
    request: HistoryClearRequest,
):
    """
    user_id/session_id 기준 서버 메모리 대화 히스토리 삭제.

    둘 다 None이면 전체 삭제.
    """

    deleted = clear_history(
        user_id=request.user_id,
        session_id=request.session_id,
    )

    return HistoryClearResponse(
        ok=True,
        deleted_sessions=deleted,
        server_time=datetime.now(
            timezone.utc
        ).isoformat(),
    )


# =============================================================================
# Standalone
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "llm_api_server:app",
        host="0.0.0.0",
        port=8100,

        # GPU 모델이 프로세스마다 중복 로드되지 않도록 반드시 1
        workers=1,

        reload=False,
    )
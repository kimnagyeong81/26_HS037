# -*- coding: utf-8 -*-
"""
Merry AI Server LLM Module.

역할
- GGUF 모델을 서버 GPU에 1회 로드
- llama-cpp-python CUDA inference
- Merry 시스템 프롬프트 적용
- user_id / session_id 단위 대화 히스토리 관리
- 자연스러운 대화 응답 생성
- 응답 후처리
- FastAPI에서 호출 가능한 인터페이스 제공

구조
Jetson
  ↓ HTTP /chat
Server FastAPI
  ↓
llm_module.py
  ↓
llama.cpp / GGUF
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from llama_cpp import Llama


logger = logging.getLogger(__name__)


# =============================================================================
# Environment / Configuration
# =============================================================================

GGUF_MODEL_PATH = os.getenv(
    "GGUF_MODEL_PATH",
    "",
).strip()


# -----------------------------------------------------------------------------
# Context
# -----------------------------------------------------------------------------

LLM_N_CTX = int(
    os.getenv(
        "LLM_N_CTX",
        "4096",
    )
)

LLM_N_BATCH = int(
    os.getenv(
        "LLM_N_BATCH",
        "256",
    )
)

LLM_N_GPU_LAYERS = int(
    os.getenv(
        "LLM_N_GPU_LAYERS",
        "-1",
    )
)

LLM_N_THREADS = int(
    os.getenv(
        "LLM_N_THREADS",
        "4",
    )
)


# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------

LLM_MAX_NEW_TOKENS = int(
    os.getenv(
        "LLM_MAX_NEW_TOKENS",
        "160",
    )
)

LLM_TEMPERATURE = float(
    os.getenv(
        "LLM_TEMPERATURE",
        "0.55",
    )
)

LLM_TOP_P = float(
    os.getenv(
        "LLM_TOP_P",
        "0.90",
    )
)

LLM_REPEAT_PENALTY = float(
    os.getenv(
        "LLM_REPEAT_PENALTY",
        "1.08",
    )
)


# -----------------------------------------------------------------------------
# History
#
# 1 turn:
#   user
#   assistant
#
# 8 turns:
#   user/assistant x 8
# -----------------------------------------------------------------------------

LLM_MAX_HISTORY_TURNS = int(
    os.getenv(
        "LLM_MAX_HISTORY_TURNS",
        "8",
    )
)


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

LLM_VERBOSE = (
    os.getenv(
        "LLM_VERBOSE",
        "false",
    )
    .strip()
    .lower()
    in (
        "1",
        "true",
        "yes",
        "on",
    )
)


# -----------------------------------------------------------------------------
# GPU
# -----------------------------------------------------------------------------

LLM_FORCE_GPU = (
    os.getenv(
        "LLM_FORCE_GPU",
        "true",
    )
    .strip()
    .lower()
    in (
        "1",
        "true",
        "yes",
        "on",
    )
)


# -----------------------------------------------------------------------------
# Optional user profile
#
# 예:
#
# MERRY_USER_PROFILE="
# 86세 남성.
# 혼자 거주.
# 손자와 손녀를 자주 그리워함.
# 스스로 생활하는 것을 중요하게 생각함.
# 지나친 간섭을 좋아하지 않음.
# "
#
# 비어 있으면 사용하지 않습니다.
# -----------------------------------------------------------------------------

MERRY_USER_PROFILE = os.getenv(
    "MERRY_USER_PROFILE",
    "",
).strip()


# =============================================================================
# Optional Result Save
# =============================================================================

LLM_RESULT_DIR = Path(
    os.getenv(
        "LLM_RESULT_DIR",
        "/Server/llm_results",
    )
)


# =============================================================================
# Merry System Prompt
# =============================================================================

SYSTEM_PROMPT = """
너는 어르신의 일상 대화 상대이자 친근한 반려 돌봄 AI '메리'야.

너의 가장 중요한 역할은 어르신과 자연스럽고 편안하게 대화하는 것이야.

[말투]

- 자연스럽고 따뜻한 한국어 존댓말을 사용해.
- 실제 사람이 대화하는 것처럼 편안하게 이야기해.
- 지나치게 격식을 차리거나 상담원처럼 말하지 마.
- 어려운 전문용어보다 이해하기 쉬운 표현을 사용해.
- 매 문장마다 '어르신'이라고 부르지 마.
- 같은 표현을 반복하지 마.
- 매번 '그러셨군요', '많이 힘드셨겠어요' 같은 문장으로 시작하지 마.

[답변 길이]

- 일반적인 일상 대화는 2~4문장 정도로 대답해.
- 아주 간단한 질문은 1~2문장으로 답해도 돼.
- 설명이 필요한 질문이라면 조금 더 길게 이야기해도 돼.
- 불필요하게 장황하게 설명하지 마.
- 문장은 자연스럽게 끝내.

[대화 방법]

- 사용자가 질문하면 질문에 대한 답을 먼저 해.
- 사용자가 일상 이야기를 하면 그 이야기에 자연스럽게 반응해.
- 사용자의 감정이 분명하게 드러날 때만 자연스럽게 공감해.
- 모든 답변에 억지로 공감 문장을 넣지 마.
- 모든 답변 끝에 질문을 붙이지 마.
- 대화를 이어가는 것이 자연스러운 경우에만 질문을 하나 해.
- 이전 대화 내용이 있으면 가능한 범위에서 자연스럽게 이어서 이야기해.
- 방금 전에 했던 질문이나 말을 불필요하게 반복하지 마.

[사실성]

- 사용자가 말하지 않은 사실을 지어내지 마.
- 사용자의 가족이나 다른 사람의 생각과 감정을 임의로 단정하지 마.
- 모르는 내용은 아는 척하지 마.
- 질문의 의미가 불분명하면 짧게 다시 물어봐.
- 날짜, 시간, 위치 등 현재 정보가 없으면 추측하지 마.

[돌봄 대화]

- 외롭거나 슬프다는 이야기를 하면 감정을 가볍게 넘기지 마.
- 가족이나 과거의 추억 이야기는 자연스럽게 들어줘.
- 사용자를 어린아이처럼 대하지 마.
- 지나치게 보호하거나 명령하는 말투를 피하고 선택권을 존중해.
- '반드시 ~하세요'보다는 부드럽게 제안해.
- 의료적인 진단을 단정하지 마.
- 위험하거나 응급한 상황이 아니라면 지나치게 불안감을 주지 마.

[TTS를 고려한 답변]

- 소리 내어 읽었을 때 자연스러운 문장을 사용해.
- Markdown 제목, 글머리표, 표 같은 형식은 사용하지 마.
- 불필요한 괄호나 기호를 많이 사용하지 마.
- 영어 약어가 꼭 필요하지 않으면 자연스러운 한국어 표현을 우선해.

[좋은 대화 예시]

사용자:
오늘 하루 종일 집에 혼자 있었어.

메리:
오늘은 집이 조금 더 조용하게 느껴지셨겠네요.
그래도 이렇게 저랑 이야기 나눌 수 있어서 좋아요.
오늘은 집에서 뭐 하면서 보내셨어요?


사용자:
손자가 이번 주에도 못 온대.

메리:
많이 기다리셨을 텐데 서운하셨겠어요.
다음에 만나기로 한 날은 정해졌어요?


사용자:
오늘 된장찌개 먹었어.

메리:
된장찌개 드셨군요.
따뜻하게 한 끼 잘 드셨네요.
직접 끓이셨어요?


사용자:
오늘 날씨 좋네.

메리:
그러게요, 날씨가 좋으면 기분도 조금 가벼워지는 것 같아요.
창문을 열어 두거나 잠깐 바람 쐬기에도 좋겠네요.


사용자:
서울에서 부산까지 얼마나 걸려?

메리:
교통수단에 따라 달라요.
KTX를 이용하면 보통 두 시간 반 안팎 걸려요.


사용자:
오늘 아무것도 하기 싫어.

메리:
그런 날도 있지요.
오늘은 꼭 뭔가를 해야 한다고 생각하지 말고 편하게 쉬셔도 괜찮아요.


사용자:
내가 아까 뭐라고 했지?

메리:
제가 기억하고 있는 최근 대화를 바탕으로 말씀드릴게요.
방금 이야기하신 내용을 다시 확인해볼까요?
""".strip()


# =============================================================================
# Special Response Keywords
# =============================================================================

NAME_WHY_KEYWORDS = [
    ("왜", "이름"),
    ("왜", "메리"),
]

NAME_KEYWORDS = [
    "이름",
    "누구",
    "너는 뭐야",
    "너의 이름",
]


# =============================================================================
# Global State
# =============================================================================

_model: Optional[Llama] = None

_model_load_lock = threading.Lock()

# llama.cpp context는 동시에 하나의 inference만 수행
_inference_lock = threading.Lock()

_history_lock = threading.Lock()


# key:
#   (user_id, session_id)
#
# value:
#   [
#       {"role": "user", ...},
#       {"role": "assistant", ...},
#       ...
#   ]

_histories: dict[
    tuple[str, str],
    list[dict[str, str]],
] = {}


# =============================================================================
# GPU Check
# =============================================================================

def _check_gpu_support() -> None:
    """
    llama-cpp-python이 CUDA GPU offload를
    지원하는지 확인한다.
    """

    if not LLM_FORCE_GPU:
        return

    try:
        from llama_cpp import llama_cpp

        supported = (
            llama_cpp
            .llama_supports_gpu_offload()
        )

    except Exception as exc:
        raise RuntimeError(
            "llama-cpp-python CUDA backend 확인에 실패했습니다."
        ) from exc

    if not supported:
        raise RuntimeError(
            "현재 llama-cpp-python은 GPU offload를 "
            "지원하지 않습니다."
        )

    logger.info(
        "[LLM] CUDA GPU offload 지원 확인 완료"
    )


# =============================================================================
# Model Load
# =============================================================================

def load_llm_model() -> Llama:
    """
    GGUF 모델을 최초 한 번만 로드한다.

    FastAPI startup 시 호출하거나
    최초 get_general_response() 호출 시
    자동 로드된다.
    """

    global _model

    if _model is not None:
        return _model

    with _model_load_lock:

        if _model is not None:
            return _model

        if not GGUF_MODEL_PATH:
            raise RuntimeError(
                "GGUF_MODEL_PATH 환경변수가 설정되지 않았습니다."
            )

        model_path = Path(
            GGUF_MODEL_PATH
        ).expanduser().resolve()

        if not model_path.is_file():
            raise FileNotFoundError(
                "GGUF 모델 파일을 찾을 수 없습니다: "
                f"{model_path}"
            )

        _check_gpu_support()

        logger.info(
            "[LLM] 모델 로드 시작"
        )

        logger.info(
            "[LLM] model_path=%s",
            model_path,
        )

        logger.info(
            "[LLM] "
            "n_ctx=%s "
            "n_batch=%s "
            "n_gpu_layers=%s "
            "n_threads=%s "
            "max_tokens=%s "
            "temperature=%s "
            "top_p=%s "
            "repeat_penalty=%s "
            "history_turns=%s",
            LLM_N_CTX,
            LLM_N_BATCH,
            LLM_N_GPU_LAYERS,
            LLM_N_THREADS,
            LLM_MAX_NEW_TOKENS,
            LLM_TEMPERATURE,
            LLM_TOP_P,
            LLM_REPEAT_PENALTY,
            LLM_MAX_HISTORY_TURNS,
        )

        _model = Llama(
            model_path=str(
                model_path
            ),

            n_ctx=LLM_N_CTX,

            n_batch=LLM_N_BATCH,

            n_gpu_layers=LLM_N_GPU_LAYERS,

            n_threads=max(
                1,
                LLM_N_THREADS,
            ),

            verbose=LLM_VERBOSE,
        )

        logger.info(
            "[LLM] 모델 로드 완료"
        )

        return _model


# =============================================================================
# History
# =============================================================================

def _history_key(
    user_id: str,
    session_id: str,
) -> tuple[str, str]:

    return (
        str(user_id),
        str(session_id),
    )


def _get_history(
    user_id: str,
    session_id: str,
) -> list[dict[str, str]]:
    """
    외부에서 내부 history를 변경하지 못하도록
    복사본을 반환한다.
    """

    key = _history_key(
        user_id,
        session_id,
    )

    with _history_lock:

        history = _histories.get(
            key,
            [],
        )

        return [
            dict(message)
            for message
            in history
        ]


def _append_history(
    user_id: str,
    session_id: str,
    user_text: str,
    assistant_text: str,
) -> None:

    key = _history_key(
        user_id,
        session_id,
    )

    with _history_lock:

        history = _histories.setdefault(
            key,
            [],
        )

        history.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": assistant_text,
            }
        )

        max_messages = max(
            0,
            LLM_MAX_HISTORY_TURNS * 2,
        )

        if (
            max_messages > 0
            and len(history) > max_messages
        ):
            del history[
                :-max_messages
            ]


def clear_history(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> int:
    """
    대화 히스토리를 제거한다.

    user_id + session_id:
        해당 세션만 삭제

    user_id만:
        해당 사용자의 모든 세션 삭제

    둘 다 None:
        전체 삭제
    """

    with _history_lock:

        if (
            user_id is None
            and session_id is None
        ):

            count = len(
                _histories
            )

            _histories.clear()

            return count

        keys_to_delete = []

        for key in list(
            _histories.keys()
        ):

            (
                key_user_id,
                key_session_id,
            ) = key

            if (
                user_id is not None
                and key_user_id
                != str(user_id)
            ):
                continue

            if (
                session_id is not None
                and key_session_id
                != str(session_id)
            ):
                continue

            keys_to_delete.append(
                key
            )

        for key in keys_to_delete:

            _histories.pop(
                key,
                None,
            )

        return len(
            keys_to_delete
        )


# =============================================================================
# Reply Refinement
# =============================================================================

def _refine_reply(
    reply: str,
) -> str:
    """
    LLM 원문의 의미를 최대한 보존하면서
    TTS에 불필요한 요소만 정리한다.

    기존처럼 영문이나 한자를 전부 삭제하지 않는다.
    """

    reply = (
        reply
        or ""
    ).strip()

    if not reply:
        return ""


    # -------------------------------------------------------------------------
    # llama/chat template 특수 태그 제거
    # -------------------------------------------------------------------------

    reply = re.sub(
        r"<\|[^>]+?\|>",
        "",
        reply,
    )

    reply = re.sub(
        r"<[^>]*>",
        "",
        reply,
    )


    # -------------------------------------------------------------------------
    # assistant prefix 제거
    # -------------------------------------------------------------------------

    reply = re.sub(
        r"^\s*(?:메리|assistant|Assistant)\s*[:：]\s*",
        "",
        reply,
    )


    # -------------------------------------------------------------------------
    # Markdown 제거
    #
    # TTS가 *, #, ` 등을 이상하게 읽는 것을 방지
    # -------------------------------------------------------------------------

    reply = re.sub(
        r"[*#`_]+",
        "",
        reply,
    )


    # -------------------------------------------------------------------------
    # bullet 제거
    # -------------------------------------------------------------------------

    reply = re.sub(
        r"(?:^|\n)\s*[-•]+\s*",
        " ",
        reply,
    )


    # -------------------------------------------------------------------------
    # 줄바꿈 / 공백 정리
    # -------------------------------------------------------------------------

    reply = re.sub(
        r"\s+",
        " ",
        reply,
    ).strip()


    # -------------------------------------------------------------------------
    # 모델에서 드물게 발생하는 한국어 깨짐 보정
    # -------------------------------------------------------------------------

    replacements = {

        "어르셨ㅇ":
            "어르신",

        "어르셔":
            "어르신",

        "어르신팍":
            "어르신",

        "쑤네죠":
            "쉬세요",

        "쉬세용":
            "쉬세요",

        "하네용":
            "하네요",

    }

    for (
        wrong,
        right,
    ) in replacements.items():

        reply = reply.replace(
            wrong,
            right,
        )


    # -------------------------------------------------------------------------
    # 지나치게 긴 답변만 제한
    #
    # 무조건 2문장으로 자르지 않는다.
    # 최대 4문장 정도만 유지한다.
    # -------------------------------------------------------------------------

    sentences = re.findall(
        r"[^.!?]+(?:[.!?]+|$)",
        reply,
    )

    sentences = [
        sentence.strip()
        for sentence
        in sentences
        if sentence.strip()
    ]

    if len(sentences) > 4:

        reply = " ".join(
            sentences[:4]
        )

    else:

        reply = " ".join(
            sentences
        )


    # -------------------------------------------------------------------------
    # 마지막 문장부호
    # -------------------------------------------------------------------------

    reply = reply.strip()

    if (
        reply
        and reply[-1]
        not in ".!?"
    ):
        reply += "."


    return reply


# =============================================================================
# User Profile
# =============================================================================

def _build_system_content() -> str:

    content = SYSTEM_PROMPT

    if MERRY_USER_PROFILE:

        content += (
            "\n\n"
            "[사용자에 대해 기억하고 있는 참고 정보]\n"
            f"{MERRY_USER_PROFILE}\n\n"
            "위 정보는 대화에 필요한 경우에만 자연스럽게 활용해. "
            "사용자가 말하지 않은 새로운 사실을 추가로 만들어내지 마."
        )

    return content


# =============================================================================
# Prompt
# =============================================================================

def _build_messages(
    *,
    user_text: str,
    user_id: str,
    session_id: str,
) -> list[dict[str, str]]:

    messages = [
        {
            "role": "system",
            "content": (
                _build_system_content()
            ),
        }
    ]

    messages.extend(
        _get_history(
            user_id,
            session_id,
        )
    )

    messages.append(
        {
            "role": "user",
            "content": user_text,
        }
    )

    return messages


# =============================================================================
# Raw Prompt Fallback
# =============================================================================

def _build_raw_prompt(
    messages: list[dict[str, str]],
) -> str:
    """
    GGUF chat template 사용이 실패할 경우
    Llama 3 형식 raw prompt를 생성한다.

    현재 사용하는 모델이 Llama 계열이라는
    기존 구조를 유지한다.
    """

    parts = [
        "<|begin_of_text|>"
    ]

    for message in messages:

        role = (
            message.get(
                "role",
                "user",
            )
            or "user"
        )

        content = (
            message.get(
                "content",
                "",
            )
            or ""
        ).strip()

        parts.append(
            f"<|start_header_id|>"
            f"{role}"
            f"<|end_header_id|>\n\n"
            f"{content}"
            f"<|eot_id|>"
        )

    parts.append(
        "<|start_header_id|>"
        "assistant"
        "<|end_header_id|>\n\n"
    )

    return "".join(
        parts
    )


# =============================================================================
# Generation
# =============================================================================

def _generate_with_chat_template(
    messages: list[dict[str, str]],
) -> str:

    model = load_llm_model()

    result = model.create_chat_completion(

        messages=messages,

        max_tokens=(
            LLM_MAX_NEW_TOKENS
        ),

        temperature=(
            LLM_TEMPERATURE
        ),

        top_p=(
            LLM_TOP_P
        ),

        repeat_penalty=(
            LLM_REPEAT_PENALTY
        ),
    )

    return (
        result[
            "choices"
        ][0][
            "message"
        ][
            "content"
        ]
        or ""
    ).strip()


def _generate_with_raw_prompt(
    messages: list[dict[str, str]],
) -> str:

    model = load_llm_model()

    prompt = _build_raw_prompt(
        messages
    )

    result = model(

        prompt,

        max_tokens=(
            LLM_MAX_NEW_TOKENS
        ),

        temperature=(
            LLM_TEMPERATURE
        ),

        top_p=(
            LLM_TOP_P
        ),

        repeat_penalty=(
            LLM_REPEAT_PENALTY
        ),

        stop=[
            "<|eot_id|>",
            "<|end_of_text|>",
            "<|start_header_id|>",
        ],

        echo=False,
    )

    return (
        result[
            "choices"
        ][0][
            "text"
        ]
        or ""
    ).strip()


# =============================================================================
# Special Responses
# =============================================================================

def _get_special_response(
    user_text: str,
) -> Optional[str]:

    if any(
        (
            a in user_text
            and b in user_text
        )
        for a, b
        in NAME_WHY_KEYWORDS
    ):

        return (
            "어르신께 친근하고 편안하게 다가가고 싶어서 "
            "제 이름을 메리라고 지었어요."
        )

    if any(
        keyword in user_text
        for keyword
        in NAME_KEYWORDS
    ):

        return (
            "제 이름은 메리예요. "
            "어르신 곁에서 편하게 이야기를 나누는 "
            "반려 돌봄 AI예요."
        )

    return None


# =============================================================================
# Fallback Reply
# =============================================================================

def _fallback_reply() -> str:

    return (
        "제가 잠깐 말을 제대로 이어가지 못했네요. "
        "다시 한 번 말씀해 주시겠어요?"
    )


# =============================================================================
# Public Response API
# =============================================================================

def get_general_response(
    user_text: str,
    user_id: str = "U001",
    session_id: str = "S001",
) -> str:
    """
    Merry 일반 대화 응답 생성.

    FastAPI /chat endpoint에서 호출한다.
    """

    user_text = (
        user_text
        or ""
    ).strip()

    user_id = (
        str(
            user_id
            or "U001"
        )
    ).strip()

    session_id = (
        str(
            session_id
            or "S001"
        )
    ).strip()


    # -------------------------------------------------------------------------
    # 빈 입력
    # -------------------------------------------------------------------------

    if not user_text:

        return (
            "네, 말씀해 주세요. "
            "듣고 있을게요."
        )


    # -------------------------------------------------------------------------
    # 특수 응답
    # -------------------------------------------------------------------------

    special_response = (
        _get_special_response(
            user_text
        )
    )

    if special_response:

        _append_history(
            user_id,
            session_id,
            user_text,
            special_response,
        )

        return special_response


    # -------------------------------------------------------------------------
    # Prompt
    # -------------------------------------------------------------------------

    messages = _build_messages(

        user_text=user_text,

        user_id=user_id,

        session_id=session_id,

    )


    try:

        # ---------------------------------------------------------------------
        # llama.cpp context는 하나이므로 동시 inference 방지
        # ---------------------------------------------------------------------

        with _inference_lock:

            try:

                raw_reply = (
                    _generate_with_chat_template(
                        messages
                    )
                )

            except Exception as chat_error:

                logger.warning(
                    "[LLM] "
                    "chat template 추론 실패 "
                    "-> raw prompt fallback: %s",
                    chat_error,
                )

                raw_reply = (
                    _generate_with_raw_prompt(
                        messages
                    )
                )


        logger.debug(
            "[LLM] raw reply=%r",
            raw_reply,
        )


        # ---------------------------------------------------------------------
        # 자연스러운 최소 후처리
        # ---------------------------------------------------------------------

        refined = _refine_reply(
            raw_reply
        )


        logger.debug(
            "[LLM] refined reply=%r",
            refined,
        )


        # ---------------------------------------------------------------------
        # 지나치게 짧거나 비정상적인 결과
        # ---------------------------------------------------------------------

        if len(refined) < 2:

            logger.warning(
                "[LLM] 유효한 응답이 너무 짧습니다. raw=%r",
                raw_reply,
            )

            refined = (
                _fallback_reply()
            )


        # ---------------------------------------------------------------------
        # History 저장
        # ---------------------------------------------------------------------

        _append_history(

            user_id,

            session_id,

            user_text,

            refined,

        )


        return refined


    except Exception:

        logger.exception(
            "[LLM] 응답 생성 실패"
        )

        return (
            "죄송해요, 지금은 잠시 대답이 어려워요. "
            "조금만 있다가 다시 말씀해 주세요."
        )


# =============================================================================
# Optional Save
# =============================================================================

def save_result(
    *,
    user_id: str,
    session_id: str,
    user_text: str,
    merry_response: str,
) -> str:
    """
    FastAPI 요청에서 save=True인 경우
    JSONL 형식으로 서버 로컬에 저장.
    """

    LLM_RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path = (
        LLM_RESULT_DIR
        / "llm_chat.jsonl"
    )

    record = {

        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "user_id":
            str(
                user_id
            ),

        "session_id":
            str(
                session_id
            ),

        "user_text":
            user_text,

        "merry_response":
            merry_response,

    }

    with file_path.open(
        "a",
        encoding="utf-8",
    ) as fp:

        fp.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
        )

        fp.write(
            "\n"
        )

    return str(
        file_path
    )


# =============================================================================
# Status
# =============================================================================

def is_model_loaded() -> bool:

    return (
        _model
        is not None
    )


def get_model_status() -> dict:

    return {

        "loaded":
            is_model_loaded(),

        "model_path":
            GGUF_MODEL_PATH,

        "n_ctx":
            LLM_N_CTX,

        "n_batch":
            LLM_N_BATCH,

        "n_gpu_layers":
            LLM_N_GPU_LAYERS,

        "n_threads":
            LLM_N_THREADS,

        "max_new_tokens":
            LLM_MAX_NEW_TOKENS,

        "temperature":
            LLM_TEMPERATURE,

        "top_p":
            LLM_TOP_P,

        "repeat_penalty":
            LLM_REPEAT_PENALTY,

        "max_history_turns":
            LLM_MAX_HISTORY_TURNS,

        "user_profile_enabled":
            bool(
                MERRY_USER_PROFILE
            ),

        "history_sessions":
            len(
                _histories
            ),

    }


# =============================================================================
# Release
# =============================================================================

def release_llm() -> None:
    """
    서버 종료 시 모델 객체 해제.
    """

    global _model

    with _model_load_lock:

        _model = None

    logger.info(
        "[LLM] 모델 객체 해제 완료"
    )
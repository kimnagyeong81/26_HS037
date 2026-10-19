# -*- coding: utf-8 -*-

"""
Merry AI 최종 실행 파일.

최종 구조
- audio_module.py      : STT + TTS + 마이크 + Wake Word
- llm_module.py        : 서버 LLM HTTP Client
- emotion_pipeline.py  : 감정분석 + 게이지 + 개인화 + ESP32
- server_client.py     : 서버 인지 분석 데이터 비동기 전송
- config.py            : 설정

Jetson 역할
- STT
- TTS
- 감정 분석 / 게이지
- ESP32 제어
- 서버 LLM 호출
- 서버 인지 분석 API(/utterance) 전송

인지 분석 데이터는 Jetson 로컬 MongoDB에 저장하지 않고
서버의 FastAPI -> Kafka -> Consumer -> MongoDB 흐름으로 전달한다.
"""

from __future__ import annotations

import logging
import random
import threading
from datetime import datetime, timezone
from pathlib import Path

from audio_module import (
    AudioSystem,
    WakeWordState,
    is_wake_word,
    strip_wake_word,
)

from config import (
    DEFAULT_SESSION_ID,
    DEFAULT_USER_ID,
    LOG_FILE,
    LOG_LEVEL,
    PARALLEL_LLM_EMOTION,
)

from emotion_pipeline import (
    EmotionPipeline,
    format_emotion_summary,
)

from llm_module import (
    check_llm_server,
    get_general_response,
    release_llm,
)

from server_client import (
    close_sender,
    enqueue_analysis,
)


logger = logging.getLogger(__name__)


WAKE_RESPONSES = [
    "네, 저는 항상 여기 있어요!",
    "네, 메리 여기 있어요! 무슨 일이세요?",
    "네, 어르신! 불러주셔서 감사해요.",
    "여기 있어요, 어르신! 뭐가 필요하세요?",
]


# =============================================================================
# Logging
# =============================================================================

def configure_logging() -> None:
    level = getattr(
        logging,
        LOG_LEVEL.upper(),
        logging.INFO,
    )

    handlers: list[logging.Handler] = [
        logging.StreamHandler()
    ]

    if LOG_FILE:
        log_path = Path(LOG_FILE)

        log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        handlers.append(
            logging.FileHandler(
                log_path,
                encoding="utf-8",
            )
        )

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


# =============================================================================
# Empty Emotion Result
# =============================================================================

def _empty_emotion_result() -> dict:
    return {
        "scores": None,
        "triggered": [],
        "events": [],
        "gauge_snapshot": {},
        "thresholds_snapshot": {},
        "phase_at_time": None,
        "new_thresholds": None,
        "error": "pipeline_error",
    }


# =============================================================================
# AI Tasks
# =============================================================================

def _run_ai_tasks(
    *,
    user_text: str,
    audio_path: str,
    timestamp: datetime,
    emotion_pipeline: EmotionPipeline,
) -> tuple[str, dict]:
    """
    LLM은 서버에서 실행하고,
    감정 분석은 Jetson에서 실행한다.

    PARALLEL_LLM_EMOTION=False:
        Jetson 감정 분석 완료
            ↓
        Server LLM 요청

    PARALLEL_LLM_EMOTION=True:
        Jetson 감정 분석
            +
        Server LLM 요청
        동시 실행
    """

    # -------------------------------------------------------------------------
    # 순차 실행
    # -------------------------------------------------------------------------

    if not PARALLEL_LLM_EMOTION:
        emotion_result = emotion_pipeline.process_text(
            user_text,
            source_file=audio_path,
            session_id=DEFAULT_SESSION_ID,
            timestamp=timestamp,
        )

        reply = get_general_response(
            user_text,
            user_id=DEFAULT_USER_ID,
            session_id=DEFAULT_SESSION_ID,
        )

        return reply, emotion_result

    # -------------------------------------------------------------------------
    # 병렬 실행
    # -------------------------------------------------------------------------

    result_box = {
        "reply": (
            "죄송해요, 지금은 잠시 대답이 어려워요. "
            "조금만 있다가 다시 말씀해 주세요."
        ),
        "emotion_result": _empty_emotion_result(),
    }

    def llm_worker() -> None:
        try:
            result_box["reply"] = get_general_response(
                user_text,
                user_id=DEFAULT_USER_ID,
                session_id=DEFAULT_SESSION_ID,
            )

        except Exception:
            logger.exception(
                "[LLM Client] 서버 LLM 요청 실패"
            )

    def emotion_worker() -> None:
        try:
            result_box["emotion_result"] = (
                emotion_pipeline.process_text(
                    user_text,
                    source_file=audio_path,
                    session_id=DEFAULT_SESSION_ID,
                    timestamp=timestamp,
                )
            )

        except Exception:
            logger.exception(
                "감정 파이프라인 오류"
            )

    t_llm = threading.Thread(
        target=llm_worker,
        daemon=True,
        name="merry-llm-client",
    )

    t_emotion = threading.Thread(
        target=emotion_worker,
        daemon=True,
        name="merry-emotion",
    )

    t_llm.start()
    t_emotion.start()

    t_llm.join()
    t_emotion.join()

    return (
        result_box["reply"],
        result_box["emotion_result"],
    )


# =============================================================================
# Turn Processing
# =============================================================================

def process_turn(
    user_text: str,
    wake: WakeWordState,
    audio_path: str,
    *,
    audio: AudioSystem,
    emotion_pipeline: EmotionPipeline,
) -> bool:

    # =========================================================================
    # 1. 종료 키워드
    # =========================================================================

    if any(
        keyword in user_text
        for keyword in (
            "종료",
            "그만",
            "끝내자",
        )
    ):
        farewell = (
            "네, 오늘 대화 즐거웠어요. "
            "언제든 불러주세요!"
        )

        print(
            f"merry : {farewell}"
        )

        audio.speak(
            farewell
        )

        return False

    # =========================================================================
    # 2. Wake Word
    # =========================================================================

    if is_wake_word(user_text):
        response = random.choice(
            WAKE_RESPONSES
        )

        print(
            f"\n[Wake Word 감지] '{user_text}'"
        )

        print(
            f"merry : {response}"
        )

        audio.speak(
            response
        )

        wake.activate()

        trailing_command = strip_wake_word(
            user_text
        )

        # 호출어만 들어온 경우
        if not trailing_command:
            return True

        # Wake Word 중복 감지 방지
        if is_wake_word(
            trailing_command
        ):
            logger.warning(
                "[Wake Word] trailing command가 다시 "
                "wake word로 판정되어 무시: %r",
                trailing_command,
            )

            return True

        # 호출어 뒤에 실제 명령이 있는 경우
        print(
            f"[inline 명령 감지] "
            f"'{trailing_command}'"
        )

        user_text = trailing_command

    # =========================================================================
    # 3. Wake 활성 상태 확인
    # =========================================================================

    if not wake.is_active():
        print(
            "[대기 모드] "
            f"wake word 없는 발화 무시: '{user_text}'"
        )

        return True

    # =========================================================================
    # 4. 발화 시각
    # =========================================================================

    turn_timestamp = datetime.now(
        timezone.utc
    )

    # =========================================================================
    # 5. AI 처리
    #
    # Jetson
    #   - 감정 분석
    #   - 감정 게이지
    #
    # Server
    #   - LLM 답변 생성
    # =========================================================================

    try:
        reply, emotion_result = _run_ai_tasks(
            user_text=user_text,
            audio_path=audio_path,
            timestamp=turn_timestamp,
            emotion_pipeline=emotion_pipeline,
        )

    except Exception:
        logger.exception(
            "턴 처리 오류"
        )

        reply = (
            "죄송해요, 지금은 잠시 대답이 어려워요. "
            "조금만 있다가 다시 말씀해 주세요."
        )

        emotion_result = (
            _empty_emotion_result()
        )

    scores = emotion_result.get(
        "scores"
    )

    triggered = emotion_result.get(
        "triggered",
        [],
    )

    print(
        f"merry : {reply}"
    )

    if scores:
        print(
            "        "
            f"{format_emotion_summary(scores)}"
        )

    if triggered:
        print(
            "        "
            f"[구슬 배출] {triggered}"
        )

    # =========================================================================
    # 6. 서버 인지 분석 데이터 전송
    #
    # Jetson
    #   ↓
    # server_client.py
    #   ↓
    # FastAPI /utterance
    #   ↓
    # Kafka
    #   ↓
    # Consumer
    #   ↓
    # Server MongoDB
    #
    # 로컬 MongoDB 저장을 선행하지 않는다.
    # 서버 전송은 백그라운드 Worker에서 비동기로 처리된다.
    # =========================================================================
    # =========================================================================
    # 6. 서버 인지 분석 데이터 전송
    # =========================================================================

    print("[DEBUG] enqueue_analysis 진입")
    print("[DEBUG] user_text =", repr(user_text))
    print("[DEBUG] timestamp =", turn_timestamp.isoformat())

    try:
        queued = enqueue_analysis(
            user_id=DEFAULT_USER_ID,
            session_id=DEFAULT_SESSION_ID,
            utterance_id=None,
            timestamp=turn_timestamp.isoformat(),
            text=user_text,
            audio_path=audio_path,
            emotion_scores=scores,
            run_trend=True,
        )

        print("[DEBUG] enqueue_analysis 반환 =", queued)

        if queued:
            logger.info(
                "[Server] 분석 전송 준비: text=%r, audio=%s",
                user_text,
                audio_path,
            )
        else:
            logger.warning(
                "[Server] 인지 분석 데이터 전송 큐 등록 실패"
            )

    except Exception as e:
        print(
            "[DEBUG] enqueue_analysis 예외:",
            repr(e),
        )

        logger.exception(
            "[Server] 인지 분석 데이터 전송 등록 실패"
        )
    # logger.info(
    #     "[Server] 인지 분석 데이터 전송 큐 등록 시도: text=%r",
    #     user_text,
    # )

    # try:
    #     queued = enqueue_analysis(
    #         user_id=DEFAULT_USER_ID,
    #         session_id=DEFAULT_SESSION_ID,
    #         utterance_id=None,
    #         timestamp=turn_timestamp.isoformat(),
    #         text=user_text,
    #         emotion_scores=scores,
    #         run_trend=True,
    #     )

    #     if queued:
    #         logger.info(
    #             "[Server] 인지 분석 데이터 전송 큐 등록 완료"
    #         )

    #     else:
    #         logger.warning(
    #             "[Server] 인지 분석 데이터 전송 큐 등록 실패"
    #         )

    # except Exception:
    #     logger.exception(
    #         "[Server] 인지 분석 데이터 전송 등록 실패"
    #     )

    # =========================================================================
    # 7. TTS
    # =========================================================================

    audio.speak(
        reply
    )

    # LLM / TTS 시간이 길어도
    # 응답이 끝난 시점부터 다시 대화 시간을 부여한다.
    wake.keep_alive()

    return True


# =============================================================================
# Main
# =============================================================================

def run() -> None:
    configure_logging()

    wake = WakeWordState()
    audio = AudioSystem()

    emotion_pipeline = None

    try:
        # =====================================================================
        # Server LLM 연결 확인
        # =====================================================================

        logger.info(
            "[LLM Client] 서버 LLM 연결 확인"
        )

        if check_llm_server():
            logger.info(
                "[LLM Client] "
                "서버 LLM 연결 정상"
            )

        else:
            logger.warning(
                "[LLM Client] "
                "서버 LLM 연결 확인 실패. "
                "MERRY_LLM_SERVER_URL 또는 "
                "서버 상태를 확인하세요."
            )

        # =====================================================================
        # Emotion Pipeline 초기화
        # =====================================================================

        emotion_pipeline = EmotionPipeline(
            DEFAULT_USER_ID
        )

        print(
            "\n"
            + "=" * 55
        )

        print(
            "Merry AI System 시작 "
            "(Wake Word: '메리야')"
        )

        print(
            "'메리야'라고 부르면 "
            "메리가 응답합니다."
        )

        print(
            "종료하려면 "
            "Ctrl+C를 누르세요."
        )

        print(
            "=" * 55
            + "\n"
        )

        # =====================================================================
        # Greeting
        # =====================================================================

        greeting = (
            "안녕하세요 어르신! "
            "'메리야'라고 불러주시면 "
            "언제든 달려갈게요!"
        )

        print(
            f"merry : {greeting}"
        )

        audio.speak(
            greeting
        )

        # =====================================================================
        # Whisper 선로딩
        # =====================================================================

        logger.info(
            "[STT] "
            "마이크 시작 전 Whisper 선로딩"
        )

        audio.preload_stt()

        # =====================================================================
        # Microphone Loop
        # =====================================================================

        for (
            segment,
            is_silent,
        ) in audio.mic_audio_stream(
            wake
        ):

            if is_silent:
                continue

            print(
                "recognizing...",
                end="\r",
            )

            user_text, audio_path = (
                audio.transcribe(
                    segment
                )
            )

            if not user_text:
                continue

            print(
                f"user  : "
                f"{user_text}          "
            )

            should_continue = process_turn(
                user_text,
                wake,
                audio_path,
                audio=audio,
                emotion_pipeline=emotion_pipeline,
            )

            print(
                "-" * 55
            )

            if not should_continue:
                break

    except KeyboardInterrupt:
        print(
            "\n시스템 종료"
        )

    finally:
        # =====================================================================
        # 서버 인지 분석 전송 Worker 종료
        # =====================================================================

        try:
            close_sender(
                timeout=5
            )

        except Exception:
            logger.exception(
                "[Server] "
                "전송 Worker 종료 실패"
            )

        # =====================================================================
        # Emotion Pipeline 종료
        # =====================================================================

        if emotion_pipeline is not None:
            try:
                emotion_pipeline.close()

            except Exception:
                logger.exception(
                    "감정 파이프라인 종료 실패"
                )

        # =====================================================================
        # Audio 종료
        # =====================================================================

        try:
            audio.close()

        except Exception:
            logger.exception(
                "AudioSystem 종료 실패"
            )

        # =====================================================================
        # LLM Client 종료
        # =====================================================================

        try:
            release_llm()

        except Exception:
            logger.exception(
                "[LLM Client] 종료 처리 실패"
            )


if __name__ == "__main__":
    run()
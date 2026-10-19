# -*- coding: utf-8 -*-
"""
Merry AI 전역 설정.

운영 구조
- Jetson
  - STT
  - TTS
  - 감정 분석 / 게이지
  - ESP32
  - 로컬 MongoDB
- Server
  - LLM
  - Kafka 기반 인지 분석

LLM은 Jetson에서 직접 실행하지 않고
HTTP API를 통해 서버에서 실행한다.
"""

from __future__ import annotations

import os
from pathlib import Path


# -----------------------------------------------------------------------------
# 공통
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_USER_ID = "U001"
DEFAULT_SESSION_ID = "S001"

LOG_LEVEL = "INFO"
LOG_FILE = None  # 예: str(BASE_DIR / "logs" / "merry_ai.log")


# -----------------------------------------------------------------------------
# 감정
# -----------------------------------------------------------------------------

EMOTIONS = [
    "joy",
    "calm",
    "anger",
    "fear",
    "sadness",
]

EMOTION_KO = {
    "joy": "기쁨",
    "calm": "평온",
    "anger": "분노",
    "fear": "두려움",
    "sadness": "슬픔",
}

MARBLE_COLORS = {
    "joy": "YELLOW",
    "calm": "GREEN",
    "anger": "RED",
    "fear": "PURPLE",
    "sadness": "BLUE",
}


DEFAULT_THRESHOLDS = {
    "joy": 180.0,
    "calm": 180.0,
    "anger": 180.0,
    "fear": 180.0,
    "sadness": 180.0,
}


BASELINE_CONFIG = {
    "min_utterances": 100,
    "min_days": 3,
}

THRESHOLD_MULTIPLIER = 15.0
THRESHOLD_MIN = 300.0

THRESHOLD_MAX = 3000.0

# threshold가 갑자기 변하지 않도록 더 천천히 반영
EMA_ALPHA = 0.02

# 구슬 이벤트 분석 시 최근 발화 ID를 함께 연결
CONTEXT_WINDOW_SIZE = 3


# -----------------------------------------------------------------------------
# 감정 모델
# -----------------------------------------------------------------------------

EMOTION_MODEL_NAME = (
    "LimYeri/"
    "HowRU-KoELECTRA-Emotion-Classifier"
)

EMOTION_MAX_LENGTH = 128
EMOTION_SCORE_MODE = "softmax"

# 감정 분석은 Jetson CPU에서 실행
EMOTION_DEVICE = "cpu"
GAUGE_MIN_SCORE = 30.0

# -----------------------------------------------------------------------------
# STT / 마이크
# -----------------------------------------------------------------------------

STT_MODEL_PATH = str(
    BASE_DIR
    / "whisper-small-ko-final"
    / "best_model"
)

FLAC_SAVE_DIR = str(
    BASE_DIR / "stt_flac"
)

STT_TEXT_SAVE_DIR = str(
    BASE_DIR / "stt_results"
)

SAVE_STT_TEXT = True


# sounddevice에서는 PulseAudio를 통해 입력
MIC_DEVICE_INDEX = None
MIC_DEVICE_NAME = "pulse"
MIC_ALSA_HINT = ""

# 실제 STT 마이크는 USB Headphone Set으로 고정
MIC_PULSE_SOURCE = (
    "alsa_input.usb-0c76_USB_Headphone_Set-00.mono-fallback"
)

INPUT_SAMPLE_RATE = 48000
AUDIO_CHANNELS = 1
AUDIO_DTYPE = "int16"

STT_SAMPLE_RATE = 16000

WAKE_CHUNK_SECONDS = 2
ACTIVE_CHUNK_SECONDS = 10

SILENCE_THRESHOLD = 0.0015
AUDIO_QUEUE_MAX_CHUNKS = 8

WAKE_WORDS = [
    "메리야",
    "메리 야",
    "메리아",
    "메리야,",
    "메리야.",
    "메리야!",
    "메리야?",
    "메리",
    "매리야",
    "매리아",
]

WAKE_TIMEOUT_SECONDS = 30


# 중요: 현재 ImportError가 발생한 설정
STT_DEVICE = "cuda"
STT_USE_FP16 = True

STT_MIN_TEXT_LENGTH = 3
STT_REPEAT_MIN_WORDS = 5
STT_REPEAT_UNIQUE_RATIO = 0.30


# -----------------------------------------------------------------------------
# TTS
# -----------------------------------------------------------------------------

TTS_LANGUAGE = "KR"
TTS_SPEED = 1.0

MIC_DRAIN_SECONDS = 0.4

TTS_UNLOAD_AFTER_SPEAK = False

TTS_PULSE_SINK = ""

TTS_OUTPUT_SAMPLE_RATE = 48000
TTS_OUTPUT_CHANNELS = 2

TTS_PRE_SILENCE_SECONDS = float(
    os.getenv(
        "MERRY_TTS_PRE_SILENCE_SECONDS",
        "0.40",
    )
)


# 실제 음성 첫 부분의 급격한 파형 변화를 방지
TTS_FADE_IN_SECONDS = float(
    os.getenv(
        "MERRY_TTS_FADE_IN_SECONDS",
        "0.005",
    )
)


# 음성 종료 시 click/pop 방지
TTS_FADE_OUT_SECONDS = float(
    os.getenv(
        "MERRY_TTS_FADE_OUT_SECONDS",
        "0.01",
    )
)


# PulseAudio / PipeWire Pulse compatibility layer
TTS_PLAYER_COMMAND = "paplay"


# 재생 직전에 pactl로 sink resume 시도
TTS_RESUME_SINK_BEFORE_PLAY = True

# -----------------------------------------------------------------------------
# LLM / 감정 병렬 처리
# -----------------------------------------------------------------------------

# 현재는 안정성 확인을 위해 False 유지
# 서버 LLM 연결이 안정화된 뒤 True로 변경 가능
PARALLEL_LLM_EMOTION = False


# -----------------------------------------------------------------------------
# Transformers
# -----------------------------------------------------------------------------

# 로컬/캐시 모델만 사용하려면 True
TRANSFORMERS_LOCAL_ONLY = False


# =============================================================================
# Server LLM
# =============================================================================

# 예:
# export MERRY_LLM_SERVER_URL="http://100.84.183.27:31000"
#
# 실제 NodePort가 확정되기 전까지 기본값은 빈 문자열로 둔다.
LLM_SERVER_URL = os.getenv(
    "MERRY_LLM_SERVER_URL",
    "http://100.84.183.27:31033"
).strip().rstrip("/")


# FastAPI endpoint
LLM_CHAT_ENDPOINT = "/chat"
LLM_HEALTH_ENDPOINT = "/health"


# LLM 요청 timeout
LLM_REQUEST_TIMEOUT_SEC = float(
    os.getenv(
        "MERRY_LLM_TIMEOUT_SEC",
        "8",
    )
)


# health check timeout
LLM_HEALTH_TIMEOUT_SEC = float(
    os.getenv(
        "MERRY_LLM_HEALTH_TIMEOUT_SEC",
        "2",
    )
)


# 향후 LLM API 인증을 적용할 경우 사용
LLM_API_KEY = os.getenv(
    "MERRY_LLM_API_KEY",
    "",
).strip()


# 서버 장애 또는 timeout 발생 시 사용할 기본 응답
LLM_FALLBACK_RESPONSE = (
    "죄송해요, 지금은 잠시 대답이 어려워요. "
    "조금만 있다가 다시 말씀해 주세요."
)


# -----------------------------------------------------------------------------
# MongoDB
# -----------------------------------------------------------------------------

MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "emotion_gauge"

MONGO_SERVER_SELECTION_TIMEOUT_MS = 3000

COLLECTIONS = {
    "users": "users",
    "utterances": "utterances",
    "marble_events": "marble_events",
    "daily_summaries": "daily_summaries",
}


# -----------------------------------------------------------------------------
# ESP32
# -----------------------------------------------------------------------------

SERIAL_PORT = "/dev/ttyACM0"
SERIAL_BAUD = 115200
SERIAL_TIMEOUT = 1
SERIAL_RETRY_SEC = 5


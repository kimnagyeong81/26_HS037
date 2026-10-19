# -*- coding: utf-8 -*-

"""
오디오 통합 모듈.

기존 stt_module.py + tts_module.py를 통합한다.

- 마이크 입력 / 리샘플링
- Whisper STT
- FLAC + STT 텍스트 저장
- Wake word 상태
- MeloTTS 재생
- TTS 중 마이크 인터록

오디오 라우팅:
- STT: USB Headphone Set 마이크를 PulseAudio default source로 고정
- STT: sounddevice의 pulse 입력 장치를 사용
- TTS: 특정 스피커에 고정하지 않고 현재 PulseAudio default sink 사용
- TTS: USB / Bluetooth 기본 출력 전환을 자동 반영
- TTS: 48kHz / Stereo / PCM16
- TTS: sink SUSPENDED -> resume
- TTS: pre-silence / fade-in / fade-out
- TTS: paplay 기반 출력

목적:
- STT가 Jetson platform-sound가 아니라 USB 마이크를 사용하도록 보장
- TTS는 현재 시스템 기본 출력(USB 또는 Bluetooth)을 따라가도록 구성
- 출력 장치가 SUSPENDED 상태에서 깨어날 때 생기는 시작부 기계음/click을 완화
"""

from __future__ import annotations

import gc
import logging
import math
import re
import shutil
import subprocess
import tempfile
import time

from pathlib import Path
from queue import Empty, Full, Queue

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

from scipy.signal import resample_poly

from config import (
    ACTIVE_CHUNK_SECONDS,
    AUDIO_CHANNELS,
    AUDIO_DTYPE,
    AUDIO_QUEUE_MAX_CHUNKS,
    FLAC_SAVE_DIR,
    INPUT_SAMPLE_RATE,
    MIC_ALSA_HINT,
    MIC_DEVICE_INDEX,
    MIC_DEVICE_NAME,
    MIC_PULSE_SOURCE,
    MIC_DRAIN_SECONDS,
    SAVE_STT_TEXT,
    SILENCE_THRESHOLD,
    STT_DEVICE,
    STT_MIN_TEXT_LENGTH,
    STT_MODEL_PATH,
    STT_REPEAT_MIN_WORDS,
    STT_REPEAT_UNIQUE_RATIO,
    STT_SAMPLE_RATE,
    STT_TEXT_SAVE_DIR,
    STT_USE_FP16,
    TRANSFORMERS_LOCAL_ONLY,
    TTS_LANGUAGE,
    TTS_SPEED,
    TTS_UNLOAD_AFTER_SPEAK,

    # TTS output
    TTS_PULSE_SINK,
    TTS_OUTPUT_SAMPLE_RATE,
    TTS_OUTPUT_CHANNELS,
    TTS_PRE_SILENCE_SECONDS,
    TTS_FADE_IN_SECONDS,
    TTS_FADE_OUT_SECONDS,
    TTS_PLAYER_COMMAND,
    TTS_RESUME_SINK_BEFORE_PLAY,

    WAKE_CHUNK_SECONDS,
    WAKE_TIMEOUT_SECONDS,
    WAKE_WORDS,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Wake Word
# =============================================================================


class WakeWordState:
    """별도 timer thread 없이 monotonic deadline으로 wake 상태를 관리한다."""

    def __init__(
        self,
        timeout_seconds: float = WAKE_TIMEOUT_SECONDS,
    ):
        self.timeout_seconds = float(
            timeout_seconds
        )

        self._active = False
        self._active_until = 0.0

    def activate(self) -> None:
        self._active = True

        self._active_until = (
            time.monotonic()
            + self.timeout_seconds
        )

        print(
            "[Wake Word] 활성화됨 — 말씀해 주세요!"
        )

    def deactivate(
        self,
        *,
        announce: bool = True,
    ) -> None:

        was_active = self._active

        self._active = False
        self._active_until = 0.0

        if announce and was_active:
            print(
                "[Wake Word] 대기 모드로 전환"
            )

    def is_active(self) -> bool:

        if not self._active:
            return False

        if (
            time.monotonic()
            > self._active_until
        ):

            self._active = False
            self._active_until = 0.0

            print(
                "[Wake Word] 타임아웃 — 대기 모드로 전환"
            )

            return False

        return True

    def keep_alive(self) -> None:

        if self._active:

            self._active_until = (
                time.monotonic()
                + self.timeout_seconds
            )


def is_wake_word(
    text: str,
) -> bool:

    normalized = (
        text
        or ""
    ).strip().lower()

    if not normalized:
        return False

    if any(
        normalized.startswith(
            word.lower()
        )
        for word in WAKE_WORDS
    ):
        return True

    return bool(
        re.match(
            r"^메리[\s\W]*야?",
            normalized,
        )
    )


def strip_wake_word(
    text: str,
) -> str:
    """
    문장 앞의 wake word만 제거하고
    뒤의 실제 명령을 반환한다.
    """

    original = (
        text
        or ""
    ).strip()

    if not original:
        return ""

    lowered = original.lower()

    for word in sorted(
        WAKE_WORDS,
        key=len,
        reverse=True,
    ):

        candidate = word.strip()

        if lowered.startswith(
            candidate.lower()
        ):

            remainder = original[
                len(candidate):
            ]

            return remainder.lstrip(
                " \t,!.?~"
            )

    return re.sub(
        r"^(?:메리|매리|merry)"
        r"(?:\s*야|아)?"
        r"[\s,!.?~]*",
        "",
        original,
        flags=re.IGNORECASE,
    ).strip()


# =============================================================================
# Audio System
# =============================================================================


class AudioSystem:

    def __init__(self):

        self._speaking = False

        # ---------------------------------------------------------------------
        # STT
        # ---------------------------------------------------------------------

        self._stt_processor = None
        self._stt_model = None

        self._stt_device = (
            self._resolve_torch_device(
                STT_DEVICE
            )
        )

        self._stt_dtype = (
            torch.float32
        )

        # ---------------------------------------------------------------------
        # TTS
        # ---------------------------------------------------------------------

        self._tts_model = None
        self._tts_speaker_id = None

        # 실제 재생 시점의 PulseAudio sink를 확인해 사용
        self._tts_output_sink: str | None = None

        # ---------------------------------------------------------------------
        # Microphone
        # ---------------------------------------------------------------------

        self._mic_device: int | None = None

        Path(
            FLAC_SAVE_DIR
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        if SAVE_STT_TEXT:

            Path(
                STT_TEXT_SAVE_DIR
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

    # =========================================================================
    # Torch Device
    # =========================================================================

    @staticmethod
    def _resolve_torch_device(
        requested: str,
    ) -> torch.device:

        requested = (
            requested
            or "cpu"
        ).lower()

        if (
            requested.startswith("cuda")
            and not torch.cuda.is_available()
        ):

            logger.warning(
                "CUDA 사용 불가 -> CPU fallback"
            )

            requested = "cpu"

        return torch.device(
            requested
        )

    # =========================================================================
    # 마이크 / 인터록
    # =========================================================================

    def _set_speaking(
        self,
        value: bool,
    ) -> None:

        self._speaking = bool(
            value
        )

        state = (
            "마이크 OFF (TTS 출력 중)"
            if value
            else "마이크 ON (수신 대기)"
        )

        print(
            f"[Audio I/O] {state}"
        )

    def is_speaking(
        self,
    ) -> bool:

        return self._speaking

    # =========================================================================
    # PulseAudio Input Source
    # =========================================================================

    @staticmethod
    def _list_pulse_sources() -> list[str]:
        """
        현재 PulseAudio source 이름 목록을 반환한다.
        """

        pactl_path = shutil.which("pactl")

        if not pactl_path:
            logger.warning(
                "[Audio] pactl을 찾을 수 없습니다."
            )
            return []

        try:
            result = subprocess.run(
                [
                    pactl_path,
                    "list",
                    "short",
                    "sources",
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3,
            )
        except Exception:
            logger.exception(
                "[Audio] PulseAudio source 목록 조회 실패"
            )
            return []

        if result.returncode != 0:
            logger.warning(
                "[Audio] PulseAudio source 목록 조회 실패: %s",
                (result.stderr or "").strip(),
            )
            return []

        sources: list[str] = []

        for line in (result.stdout or "").splitlines():
            parts = line.split()

            if len(parts) >= 2:
                sources.append(parts[1])

        return sources

    @staticmethod
    def _get_default_pulse_source() -> str | None:
        """
        현재 PulseAudio default source를 반환한다.
        """

        pactl_path = shutil.which("pactl")

        if not pactl_path:
            return None

        try:
            result = subprocess.run(
                [
                    pactl_path,
                    "get-default-source",
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3,
            )
        except Exception:
            logger.exception(
                "[Audio] default PulseAudio source 조회 실패"
            )
            return None

        if result.returncode != 0:
            return None

        source = (result.stdout or "").strip()

        return source if source else None

    @classmethod
    def _set_pulse_input_source(cls) -> None:
        """
        STT 입력을 config.py의 MIC_PULSE_SOURCE로 고정한다.

        sounddevice의 'pulse' 입력은 PulseAudio default source를
        사용하므로 InputStream을 열기 전에 USB 마이크를
        default source로 지정한다.
        """

        source = (MIC_PULSE_SOURCE or "").strip()

        if not source:
            logger.info(
                "[Audio] MIC_PULSE_SOURCE가 비어 있어 "
                "현재 PulseAudio default source를 사용합니다."
            )
            return

        pactl_path = shutil.which("pactl")

        if not pactl_path:
            logger.warning(
                "[Audio] pactl을 찾을 수 없어 "
                "STT source를 고정하지 못했습니다."
            )
            return

        available = cls._list_pulse_sources()

        if source not in available:
            logger.warning(
                "[Audio] 설정된 STT PulseAudio source가 없습니다: %s",
                source,
            )
            logger.warning(
                "[Audio] 사용 가능한 source: %s",
                available,
            )
            return

        # USB 마이크 mute 해제
        subprocess.run(
            [
                pactl_path,
                "set-source-mute",
                source,
                "0",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # PulseAudio 기본 입력을 USB 마이크로 고정
        result = subprocess.run(
            [
                pactl_path,
                "set-default-source",
                source,
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            logger.warning(
                "[Audio] default source 설정 실패: %s",
                (result.stderr or "").strip(),
            )
            return

        current = cls._get_default_pulse_source()

        logger.info(
            "[Audio] STT PulseAudio source 고정: %s",
            current or source,
        )

    # =========================================================================
    # Input Device
    # =========================================================================

    def _resolve_input_device(
        self,
    ) -> int:
        """
        사용할 입력 마이크를 결정한다.

        우선순위:
        1. MIC_DEVICE_INDEX
        2. MIC_DEVICE_NAME + MIC_ALSA_HINT
        3. MIC_DEVICE_NAME
        4. USB Headphone Set / USB Audio 자동 탐색
        5. ALSA 물리 입력 장치
        6. 시스템 default 입력
        """

        if self._mic_device is not None:
            return self._mic_device

        # sounddevice의 pulse 입력이 실제 USB 마이크를 보도록
        # PulseAudio default source를 먼저 고정한다.
        self._set_pulse_input_source()

        devices = sd.query_devices()

        def is_input_device(
            idx: int,
        ) -> bool:

            try:

                return (
                    int(
                        devices[idx][
                            "max_input_channels"
                        ]
                    )
                    > 0
                )

            except (
                IndexError,
                TypeError,
                KeyError,
            ):

                return False

        def select_device(
            idx: int,
            reason: str,
        ) -> int:

            if (
                idx < 0
                or idx >= len(devices)
            ):

                raise RuntimeError(
                    "잘못된 마이크 "
                    f"device index: {idx}"
                )

            info = devices[idx]

            if (
                info[
                    "max_input_channels"
                ]
                <= 0
            ):

                raise RuntimeError(
                    f"device {idx}는 "
                    "입력 장치가 아닙니다: "
                    f"{info['name']}"
                )

            sd.check_input_settings(
                device=idx,
                channels=AUDIO_CHANNELS,
                dtype=AUDIO_DTYPE,
                samplerate=INPUT_SAMPLE_RATE,
            )

            self._mic_device = idx

            logger.info(
                "마이크 선택: "
                "device=%s "
                "name=%s "
                "input_channels=%s "
                "default_sr=%s "
                "reason=%s",
                idx,
                info["name"],
                info[
                    "max_input_channels"
                ],
                info[
                    "default_samplerate"
                ],
                reason,
            )

            return idx

        # ---------------------------------------------------------------------
        # MIC_DEVICE_INDEX
        # ---------------------------------------------------------------------

        if MIC_DEVICE_INDEX is not None:

            idx = int(
                MIC_DEVICE_INDEX
            )

            if not is_input_device(
                idx
            ):

                raise RuntimeError(
                    "MIC_DEVICE_INDEX="
                    f"{idx}는 사용할 수 있는 "
                    "입력 장치가 아닙니다."
                )

            return select_device(
                idx,
                "MIC_DEVICE_INDEX",
            )

        # ---------------------------------------------------------------------
        # NAME + ALSA hint
        # ---------------------------------------------------------------------

        configured_name = (
            MIC_DEVICE_NAME
            or ""
        ).strip().lower()

        alsa_hint = (
            MIC_ALSA_HINT
            or ""
        ).strip().lower()

        # ---------------------------------------------------------------------
        # PulseAudio / default 입력을 명시적으로 선택
        # ---------------------------------------------------------------------

        if configured_name in {
            "pulse",
            "default",
        }:

            for idx, info in enumerate(
                devices
            ):

                if (
                    info[
                        "max_input_channels"
                    ]
                    <= 0
                ):
                    continue

                name = str(
                    info["name"]
                ).strip().lower()

                if name == configured_name:

                    logger.info(
                        "[Audio] 입력 장치 선택: "
                        "device=%s name=%s",
                        idx,
                        info["name"],
                    )

                    return select_device(
                        idx,
                        f"{configured_name} 입력",
                    )

            raise RuntimeError(
                f"sounddevice에서 '{configured_name}' "
                "입력 장치를 찾지 못했습니다."
            )

        if (
            configured_name
            and alsa_hint
        ):

            for (
                idx,
                info,
            ) in enumerate(
                devices
            ):

                if (
                    info[
                        "max_input_channels"
                    ]
                    <= 0
                ):
                    continue

                name = str(
                    info["name"]
                ).lower()

                if (
                    configured_name
                    in name
                    and alsa_hint
                    in name
                ):

                    return select_device(
                        idx,
                        "MIC_DEVICE_NAME + MIC_ALSA_HINT",
                    )

        # ---------------------------------------------------------------------
        # NAME
        # ---------------------------------------------------------------------

        if configured_name:

            for (
                idx,
                info,
            ) in enumerate(
                devices
            ):

                if (
                    info[
                        "max_input_channels"
                    ]
                    <= 0
                ):
                    continue

                name = str(
                    info["name"]
                ).lower()

                if (
                    configured_name
                    in name
                ):

                    return select_device(
                        idx,
                        "MIC_DEVICE_NAME",
                    )

        # ---------------------------------------------------------------------
        # USB 자동 탐색
        # ---------------------------------------------------------------------

        usb_keywords = (
            "usb headphone set",
            "usb audio",
        )

        for keyword in usb_keywords:

            for (
                idx,
                info,
            ) in enumerate(
                devices
            ):

                if (
                    info[
                        "max_input_channels"
                    ]
                    <= 0
                ):
                    continue

                name = str(
                    info["name"]
                ).lower()

                if (
                    keyword
                    not in name
                ):
                    continue

                try:

                    return select_device(
                        idx,
                        (
                            "USB 자동 탐색: "
                            f"{keyword}"
                        ),
                    )

                except Exception as exc:

                    logger.warning(
                        "USB 마이크 후보 사용 불가: "
                        "device=%s name=%s error=%s",
                        idx,
                        info["name"],
                        exc,
                    )

        # ---------------------------------------------------------------------
        # ALSA fallback
        # ---------------------------------------------------------------------

        for (
            idx,
            info,
        ) in enumerate(
            devices
        ):

            if (
                info[
                    "max_input_channels"
                ]
                <= 0
            ):
                continue

            name = str(
                info["name"]
            )

            name_lower = (
                name.lower()
            )

            if (
                "hw:"
                not in name_lower
            ):
                continue

            if (
                "nvidia jetson"
                in name_lower
            ):
                continue

            try:

                return select_device(
                    idx,
                    "ALSA 물리 입력 fallback",
                )

            except Exception:

                continue

        # ---------------------------------------------------------------------
        # Default
        # ---------------------------------------------------------------------

        default_input = (
            sd.default.device[0]
        )

        if (
            default_input is not None
            and int(
                default_input
            ) >= 0
            and is_input_device(
                int(
                    default_input
                )
            )
        ):

            logger.warning(
                "USB/ALSA 물리 마이크를 찾지 못해 "
                "default 입력 장치를 사용합니다: "
                "device=%s",
                default_input,
            )

            return select_device(
                int(
                    default_input
                ),
                "system default fallback",
            )

        available = []

        for (
            idx,
            info,
        ) in enumerate(
            devices
        ):

            if (
                info[
                    "max_input_channels"
                ]
                > 0
            ):

                available.append(
                    f"{idx}: {info['name']}"
                )

        raise RuntimeError(
            "사용 가능한 입력 마이크를 "
            "찾지 못했습니다.\n"
            + "\n".join(
                available
            )
        )

    # =========================================================================
    # STT
    # =========================================================================

    def _load_stt(
        self,
    ) -> None:

        if (
            self._stt_model
            is not None
        ):
            return

        from transformers import (
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )

        logger.info(
            "[STT] 모델 로드 중: "
            "%s (device=%s)",
            STT_MODEL_PATH,
            self._stt_device,
        )

        self._stt_processor = (
            WhisperProcessor.from_pretrained(
                STT_MODEL_PATH,
                local_files_only=(
                    TRANSFORMERS_LOCAL_ONLY
                ),
            )
        )

        kwargs = {

            "local_files_only":
                TRANSFORMERS_LOCAL_ONLY,

            "low_cpu_mem_usage":
                True,
        }

        if (
            self._stt_device.type
            == "cuda"
            and STT_USE_FP16
        ):

            kwargs[
                "torch_dtype"
            ] = torch.float16

        self._stt_model = (
            WhisperForConditionalGeneration
            .from_pretrained(
                STT_MODEL_PATH,
                **kwargs,
            )
            .to(
                self._stt_device
            )
        )

        self._stt_model.eval()

        self._stt_dtype = (
            next(
                self._stt_model.parameters()
            ).dtype
        )

        if (
            getattr(
                self._stt_model,
                "generation_config",
                None,
            )
            is not None
        ):

            (
                self
                ._stt_model
                .generation_config
                .forced_decoder_ids
            ) = None

        if hasattr(
            self._stt_model.config,
            "forced_decoder_ids",
        ):

            (
                self
                ._stt_model
                .config
                .forced_decoder_ids
            ) = None

        logger.info(
            "[STT] 모델 로드 완료 "
            "(dtype=%s)",
            self._stt_dtype,
        )

    def preload_stt(
        self,
    ) -> None:

        self._load_stt()

    @staticmethod
    def _clean_transcript(
        text: str,
    ) -> str:

        text = re.sub(
            r"([?.!,])(?:\s*\1)+",
            r"\1",
            (
                text
                or ""
            ),
        ).strip()

        if not text:
            return ""

        words = text.split()

        if (
            len(words)
            >= STT_REPEAT_MIN_WORDS
        ):

            unique_ratio = (
                len(
                    set(
                        words
                    )
                )
                / len(
                    words
                )
            )

            if (
                unique_ratio
                <= STT_REPEAT_UNIQUE_RATIO
            ):

                logger.info(
                    "[STT 무시] 반복 환각 가능성: %s",
                    text,
                )

                return ""

        if (
            len(text)
            < STT_MIN_TEXT_LENGTH
        ):

            logger.info(
                "[STT 무시] 너무 짧은 발화: %s",
                text,
            )

            return ""

        return text

    @staticmethod
    def _save_flac(
        speech_array: np.ndarray,
    ) -> str:

        timestamp = (
            time.strftime(
                "%Y-%m-%d_%H-%M-%S"
            )
        )

        millis = int(
            (
                time.time()
                % 1
            )
            * 1000
        )

        filename = (
            f"{timestamp}_"
            f"{millis:03d}.flac"
        )

        filepath = (
            Path(
                FLAC_SAVE_DIR
            )
            / filename
        )

        sf.write(
            str(
                filepath
            ),
            speech_array,
            STT_SAMPLE_RATE,
            format="FLAC",
        )

        logger.info(
            "[STT] FLAC 저장: %s",
            filepath,
        )

        return str(
            filepath
        )

    @staticmethod
    def _save_stt_text(
        audio_path: str,
        raw_text: str,
    ) -> str | None:

        if not SAVE_STT_TEXT:
            return None

        result_path = (
            Path(
                STT_TEXT_SAVE_DIR
            )
            / (
                Path(
                    audio_path
                ).stem
                + ".txt"
            )
        )

        result_path.write_text(
            raw_text,
            encoding="utf-8",
        )

        logger.info(
            "[STT] 텍스트 저장: %s",
            result_path,
        )

        return str(
            result_path
        )

    def transcribe(
        self,
        speech_array: np.ndarray,
    ) -> tuple[str, str]:

        audio_path = (
            self._save_flac(
                speech_array
            )
        )

        self._load_stt()

        processed = (
            self._stt_processor(
                speech_array,
                sampling_rate=(
                    STT_SAMPLE_RATE
                ),
                return_tensors="pt",
                return_attention_mask=True,
            )
        )

        input_features = (
            processed
            .input_features
            .to(
                device=(
                    self._stt_device
                ),
                dtype=(
                    self._stt_dtype
                ),
            )
        )

        attention_mask = (
            getattr(
                processed,
                "attention_mask",
                None,
            )
        )

        if (
            attention_mask
            is not None
        ):

            attention_mask = (
                attention_mask.to(
                    self._stt_device
                )
            )

        generate_kwargs = {

            "input_features":
                input_features,

            "language":
                "korean",

            "task":
                "transcribe",
        }

        if (
            attention_mask
            is not None
        ):

            generate_kwargs[
                "attention_mask"
            ] = attention_mask

        with torch.inference_mode():

            predicted_ids = (
                self._stt_model.generate(
                    **generate_kwargs
                )
            )

        raw_text = (
            self
            ._stt_processor
            .batch_decode(
                predicted_ids,
                skip_special_tokens=True,
            )[0]
            .strip()
        )

        self._save_stt_text(
            audio_path,
            raw_text,
        )

        clean_text = (
            self._clean_transcript(
                raw_text
            )
        )

        del processed
        del input_features
        del predicted_ids

        if (
            attention_mask
            is not None
        ):

            del attention_mask

        if (
            self._stt_device.type
            == "cuda"
        ):

            torch.cuda.empty_cache()

        return (
            clean_text,
            audio_path,
        )

    # =========================================================================
    # Microphone stream
    # =========================================================================

    def mic_audio_stream(
        self,
        wake: WakeWordState,
    ):

        self.preload_stt()

        device = (
            self._resolve_input_device()
        )

        audio_queue: Queue[
            np.ndarray
        ] = Queue(
            maxsize=(
                AUDIO_QUEUE_MAX_CHUNKS
            )
        )

        def audio_callback(
            indata,
            frames,
            time_info,
            status,
        ):

            if status:

                logger.warning(
                    "[Audio callback] %s",
                    status,
                )

            if self.is_speaking():
                return

            audio = (
                indata.astype(
                    np.float32,
                    copy=False,
                )
            )

            if np.issubdtype(
                indata.dtype,
                np.integer,
            ):

                dtype_info = (
                    np.iinfo(
                        indata.dtype
                    )
                )

                scale = max(
                    abs(
                        float(
                            dtype_info.min
                        )
                    ),
                    abs(
                        float(
                            dtype_info.max
                        )
                    ),
                )

                if scale > 0:

                    audio = (
                        audio
                        / scale
                    )

            try:

                audio_queue.put_nowait(
                    audio.copy()
                )

            except Full:

                try:

                    audio_queue.get_nowait()

                except Empty:

                    pass

                try:

                    audio_queue.put_nowait(
                        audio.copy()
                    )

                except Full:

                    pass

        buffer = np.empty(
            0,
            dtype=np.float32,
        )

        print(
            "[Audio] "
            f"마이크 device={device}"
        )

        print(
            "[Audio] 입력="
            f"{INPUT_SAMPLE_RATE}Hz "
            "→ STT="
            f"{STT_SAMPLE_RATE}Hz"
        )

        with sd.InputStream(
            device=device,
            samplerate=(
                INPUT_SAMPLE_RATE
            ),
            channels=(
                AUDIO_CHANNELS
            ),
            dtype=(
                AUDIO_DTYPE
            ),
            # 100ms 단위 callback.
            # 기존 1초 block보다 입력 반응성과 queue 안정성이 좋다.
            blocksize=max(
                1,
                int(INPUT_SAMPLE_RATE * 0.1),
            ),
            callback=(
                audio_callback
            ),
        ):

            while True:

                chunk = (
                    audio_queue.get()
                )

                mono = (
                    chunk[:, 0]
                    if chunk.ndim > 1
                    else chunk
                )

                if (
                    INPUT_SAMPLE_RATE
                    != STT_SAMPLE_RATE
                ):

                    mono = (
                        resample_poly(
                            mono,
                            STT_SAMPLE_RATE,
                            INPUT_SAMPLE_RATE,
                        )
                        .astype(
                            np.float32,
                            copy=False,
                        )
                    )

                buffer = (
                    np.concatenate(
                        (
                            buffer,
                            mono,
                        )
                    )
                )

                target_samples = (
                    STT_SAMPLE_RATE
                    * (
                        ACTIVE_CHUNK_SECONDS
                        if wake.is_active()
                        else WAKE_CHUNK_SECONDS
                    )
                )

                if (
                    buffer.size
                    < target_samples
                ):

                    continue

                segment = buffer[
                    :target_samples
                ]

                buffer = buffer[
                    target_samples:
                ]

                mean_level = float(
                    np.abs(
                        segment
                    ).mean()
                )

                peak_level = float(
                    np.abs(
                        segment
                    ).max()
                )

                is_silent = (
                    mean_level
                    < SILENCE_THRESHOLD
                )

                logger.debug(
                    "[MIC] mean=%.6f peak=%.6f "
                    "threshold=%.6f silent=%s",
                    mean_level,
                    peak_level,
                    SILENCE_THRESHOLD,
                    is_silent,
                )

                yield (
                    segment,
                    is_silent,
                )

                buffer = np.empty(
                    0,
                    dtype=np.float32,
                )

                while True:

                    try:

                        audio_queue.get_nowait()

                    except Empty:

                        break

    # =========================================================================
    # TTS Model
    # =========================================================================

    def _load_tts(
        self,
    ) -> None:

        if (
            self._tts_model
            is not None
        ):

            return

        logger.info(
            "[TTS] 모델 로드 중"
        )

        original_torch_load = (
            torch.load
        )

        def compatible_torch_load(
            *args,
            **kwargs,
        ):

            kwargs.setdefault(
                "weights_only",
                False,
            )

            return original_torch_load(
                *args,
                **kwargs,
            )

        torch.load = (
            compatible_torch_load
        )

        try:

            from melo.api import (
                TTS as MeloTTS,
            )

            self._tts_model = (
                MeloTTS(
                    language=(
                        TTS_LANGUAGE
                    )
                )
            )

            self._tts_speaker_id = (
                self
                ._tts_model
                .hps
                .data
                .spk2id[
                    TTS_LANGUAGE
                ]
            )

        finally:

            torch.load = (
                original_torch_load
            )

        logger.info(
            "[TTS] 모델 로드 완료"
        )

    def release_tts(
        self,
    ) -> None:

        self._tts_model = None
        self._tts_speaker_id = None

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    # =========================================================================
    # PulseAudio sink
    # =========================================================================

    @staticmethod
    def _get_default_pulse_sink() -> str | None:
        """
        pactl에서 현재 default sink를 가져온다.
        """

        pactl_path = shutil.which(
            "pactl"
        )

        if not pactl_path:
            return None

        try:

            result = subprocess.run(
                [
                    pactl_path,
                    "get-default-sink",
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3,
            )

        except Exception:

            logger.exception(
                "[TTS] default PulseAudio sink 확인 실패"
            )

            return None

        if (
            result.returncode
            != 0
        ):

            return None

        sink = (
            result.stdout
            or ""
        ).strip()

        return (
            sink
            if sink
            else None
        )

    @staticmethod
    def _list_pulse_sinks() -> list[str]:
        """
        현재 PulseAudio sink 이름 목록을 반환한다.
        """

        pactl_path = shutil.which(
            "pactl"
        )

        if not pactl_path:
            return []

        try:

            result = subprocess.run(
                [
                    pactl_path,
                    "list",
                    "short",
                    "sinks",
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3,
            )

        except Exception:

            logger.exception(
                "[TTS] PulseAudio sink 목록 확인 실패"
            )

            return []

        if (
            result.returncode
            != 0
        ):

            return []

        sinks: list[str] = []

        for line in (
            result.stdout
            or ""
        ).splitlines():

            parts = line.split()

            if len(parts) >= 2:

                sinks.append(
                    parts[1]
                )

        return sinks

    def _resolve_tts_output_sink(
        self,
    ) -> str | None:
        """
        config.py의 TTS_PULSE_SINK가 지정되어 있으면 우선 사용한다.

        TTS_PULSE_SINK가 빈 문자열이면 현재 시스템 default sink를 사용한다.
        따라서 Bluetooth를 기본 출력으로 바꾸면 다음 TTS부터 Bluetooth로 출력된다.
        """

        # 캐시되어 있어도 USB 장치가 제거되었을 수 있으므로
        # 실제 sink 목록은 다시 확인
        available = (
            self._list_pulse_sinks()
        )

        configured = (
            TTS_PULSE_SINK
            or ""
        ).strip()

        if (
            configured
            and configured
            in available
        ):

            self._tts_output_sink = (
                configured
            )

            return configured

        if configured:

            logger.warning(
                "[TTS] 설정된 PulseAudio sink를 "
                "찾지 못했습니다: %s",
                configured,
            )

        default_sink = (
            self._get_default_pulse_sink()
        )

        if (
            default_sink
            and (
                not available
                or default_sink
                in available
            )
        ):

            logger.warning(
                "[TTS] default sink로 fallback: %s",
                default_sink,
            )

            self._tts_output_sink = (
                default_sink
            )

            return default_sink

        logger.warning(
            "[TTS] 사용할 PulseAudio sink를 "
            "확정하지 못했습니다."
        )

        self._tts_output_sink = None

        return None

    @staticmethod
    def _resume_pulse_sink(
        sink: str | None,
    ) -> None:
        """
        SUSPENDED 상태의 USB sink를 명시적으로 resume한다.

        바로 이어서 paplay를 실행하므로
        장치가 다시 suspend되기 전에 출력 스트림이 연결된다.
        """

        if (
            not TTS_RESUME_SINK_BEFORE_PLAY
            or not sink
        ):

            return

        pactl_path = shutil.which(
            "pactl"
        )

        if not pactl_path:

            logger.warning(
                "[TTS] pactl을 찾을 수 없어 "
                "sink resume을 건너뜁니다."
            )

            return

        try:

            result = subprocess.run(
                [
                    pactl_path,
                    "suspend-sink",
                    sink,
                    "0",
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3,
            )

        except Exception:

            logger.exception(
                "[TTS] sink resume 실행 실패"
            )

            return

        if (
            result.returncode
            != 0
        ):

            logger.warning(
                "[TTS] sink resume 실패: %s",
                (
                    result.stderr
                    or ""
                ).strip(),
            )

            return

        logger.info(
            "[TTS] PulseAudio sink resume: %s",
            sink,
        )

        # resume 직후 paplay 연결 전 아주 짧은 여유
        time.sleep(
            0.05
        )

    # =========================================================================
    # TTS Audio Processing
    # =========================================================================

    @staticmethod
    def _resample_tts_audio(
        data: np.ndarray,
        source_rate: int,
        target_rate: int,
    ) -> np.ndarray:

        source_rate = int(
            source_rate
        )

        target_rate = int(
            target_rate
        )

        if source_rate <= 0:

            raise RuntimeError(
                "잘못된 TTS source sample rate: "
                f"{source_rate}"
            )

        if target_rate <= 0:

            raise RuntimeError(
                "잘못된 TTS target sample rate: "
                f"{target_rate}"
            )

        if (
            source_rate
            == target_rate
        ):

            return data.astype(
                np.float32,
                copy=False,
            )

        divisor = math.gcd(
            source_rate,
            target_rate,
        )

        up = (
            target_rate
            // divisor
        )

        down = (
            source_rate
            // divisor
        )

        logger.info(
            "[TTS] 리샘플링: "
            "%sHz -> %sHz "
            "(up=%s down=%s)",
            source_rate,
            target_rate,
            up,
            down,
        )

        result = resample_poly(
            data,
            up,
            down,
            axis=0,
        )

        return result.astype(
            np.float32,
            copy=False,
        )

    @staticmethod
    def _convert_tts_channels(
        data: np.ndarray,
    ) -> np.ndarray:
        """
        현재 USB output:
        2ch stereo

        MeloTTS mono 출력을 좌/우 동일하게 복제한다.
        """

        if (
            TTS_OUTPUT_CHANNELS
            != 2
        ):

            raise RuntimeError(
                "현재 AudioSystem은 "
                "TTS_OUTPUT_CHANNELS=2를 기준으로 합니다."
            )

        if data.ndim == 1:

            return np.column_stack(
                (
                    data,
                    data,
                )
            ).astype(
                np.float32,
                copy=False,
            )

        if data.ndim != 2:

            raise RuntimeError(
                "지원하지 않는 TTS WAV 차원: "
                f"{data.shape}"
            )

        channels = (
            data.shape[1]
        )

        if channels == 1:

            return np.repeat(
                data,
                2,
                axis=1,
            ).astype(
                np.float32,
                copy=False,
            )

        if channels == 2:

            return data.astype(
                np.float32,
                copy=False,
            )

        logger.warning(
            "[TTS] %s채널 입력 WAV -> "
            "첫 두 채널만 사용",
            channels,
        )

        return data[
            :,
            :2,
        ].astype(
            np.float32,
            copy=False,
        )

    @staticmethod
    def _apply_tts_fades(
        data: np.ndarray,
        samplerate: int,
    ) -> np.ndarray:

        if (
            data.size
            == 0
        ):

            return data

        result = data.copy()

        # ---------------------------------------------------------------------
        # fade in
        # ---------------------------------------------------------------------

        fade_in_samples = min(
            int(
                samplerate
                * TTS_FADE_IN_SECONDS
            ),
            len(
                result
            ),
        )

        if (
            fade_in_samples
            > 1
        ):

            fade = np.linspace(
                0.0,
                1.0,
                fade_in_samples,
                dtype=np.float32,
            )

            if (
                result.ndim
                == 2
            ):

                fade = fade[
                    :,
                    None,
                ]

            result[
                :fade_in_samples
            ] *= fade

        # ---------------------------------------------------------------------
        # fade out
        # ---------------------------------------------------------------------

        fade_out_samples = min(
            int(
                samplerate
                * TTS_FADE_OUT_SECONDS
            ),
            len(
                result
            ),
        )

        if (
            fade_out_samples
            > 1
        ):

            fade = np.linspace(
                1.0,
                0.0,
                fade_out_samples,
                dtype=np.float32,
            )

            if (
                result.ndim
                == 2
            ):

                fade = fade[
                    :,
                    None,
                ]

            result[
                -fade_out_samples:
            ] *= fade

        return result

    @staticmethod
    def _add_tts_pre_silence(
        data: np.ndarray,
        samplerate: int,
    ) -> np.ndarray:
        """
        출력 sink가 SUSPENDED -> RUNNING으로 바뀌는 동안
        실제 목소리가 재생되지 않도록 앞부분에 무음을 넣는다.
        """

        samples = int(
            samplerate
            * TTS_PRE_SILENCE_SECONDS
        )

        if samples <= 0:

            return data

        if data.ndim == 1:

            silence = np.zeros(
                samples,
                dtype=np.float32,
            )

        else:

            silence = np.zeros(
                (
                    samples,
                    data.shape[1],
                ),
                dtype=np.float32,
            )

        return np.concatenate(
            (
                silence,
                data,
            ),
            axis=0,
        )

    @classmethod
    def _prepare_tts_playback_file(
        cls,
        source_path: str,
        output_path: str,
    ) -> None:
        """
        MeloTTS WAV
             ↓
        float32
             ↓
        48kHz
             ↓
        stereo
             ↓
        fade in / out
             ↓
        400ms pre-silence
             ↓
        WAV PCM16
        """

        data, source_rate = (
            sf.read(
                source_path,
                dtype="float32",
                always_2d=False,
            )
        )

        if (
            data is None
            or data.size == 0
        ):

            raise RuntimeError(
                "MeloTTS가 생성한 WAV가 비어 있습니다."
            )

        logger.info(
            "[TTS] 원본 WAV: "
            "rate=%s shape=%s duration=%.3fs",
            source_rate,
            data.shape,
            (
                len(
                    data
                )
                / float(
                    source_rate
                )
            ),
        )

        data = np.nan_to_num(
            data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(
            np.float32,
            copy=False,
        )

        # ---------------------------------------------------------------------
        # 48 kHz
        # ---------------------------------------------------------------------

        data = (
            cls._resample_tts_audio(
                data,
                source_rate,
                TTS_OUTPUT_SAMPLE_RATE,
            )
        )

        # ---------------------------------------------------------------------
        # stereo
        # ---------------------------------------------------------------------

        data = (
            cls._convert_tts_channels(
                data
            )
        )

        # ---------------------------------------------------------------------
        # clip
        # ---------------------------------------------------------------------

        data = np.clip(
            data,
            -1.0,
            1.0,
        ).astype(
            np.float32,
            copy=False,
        )

        # ---------------------------------------------------------------------
        # fade
        # ---------------------------------------------------------------------

        data = (
            cls._apply_tts_fades(
                data,
                TTS_OUTPUT_SAMPLE_RATE,
            )
        )

        # ---------------------------------------------------------------------
        # SUSPENDED wake-up 흡수용 silence
        # ---------------------------------------------------------------------

        data = (
            cls._add_tts_pre_silence(
                data,
                TTS_OUTPUT_SAMPLE_RATE,
            )
        )

        # ---------------------------------------------------------------------
        # PCM16
        # ---------------------------------------------------------------------

        sf.write(
            output_path,
            data,
            TTS_OUTPUT_SAMPLE_RATE,
            format="WAV",
            subtype="PCM_16",
        )

        logger.info(
            "[TTS] Playback WAV: "
            "rate=%s "
            "channels=%s "
            "pre_silence=%.0fms "
            "fade_in=%.0fms "
            "fade_out=%.0fms",
            TTS_OUTPUT_SAMPLE_RATE,
            TTS_OUTPUT_CHANNELS,
            (
                TTS_PRE_SILENCE_SECONDS
                * 1000
            ),
            (
                TTS_FADE_IN_SECONDS
                * 1000
            ),
            (
                TTS_FADE_OUT_SECONDS
                * 1000
            ),
        )

    # =========================================================================
    # TTS Playback
    # =========================================================================

    def _play_tts_with_paplay(
        self,
        wav_path: str,
    ) -> None:

        player_path = shutil.which(
            TTS_PLAYER_COMMAND
        )

        if not player_path:

            raise RuntimeError(
                "'paplay' 명령을 찾을 수 없습니다."
            )

        sink = (
            self._resolve_tts_output_sink()
        )

        logger.info(
            "[TTS] 출력 sink=%s",
            sink
            or "<default>",
        )

        # ---------------------------------------------------------------------
        # SUSPENDED USB sink 강제 resume
        # ---------------------------------------------------------------------

        self._resume_pulse_sink(
            sink
        )

        # ---------------------------------------------------------------------
        # paplay command
        # ---------------------------------------------------------------------

        command = [
            player_path,
        ]

        if sink:

            command.extend(
                [
                    "--device",
                    sink,
                ]
            )

        command.append(
            wav_path
        )

        logger.info(
            "[TTS] paplay 시작: %s",
            " ".join(
                command
            ),
        )

        try:

            result = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

        except Exception as exc:

            raise RuntimeError(
                "paplay 실행 실패"
            ) from exc

        if (
            result.returncode
            != 0
        ):

            stderr = (
                result.stderr
                or ""
            ).strip()

            raise RuntimeError(
                "paplay 재생 실패 "
                f"(returncode={result.returncode}): "
                f"{stderr}"
            )

        logger.info(
            "[TTS] paplay 재생 완료"
        )

    # =========================================================================
    # TTS Speak
    # =========================================================================

    def speak(
        self,
        text: str,
    ) -> None:

        if (
            not text
            or not text.strip()
        ):

            return

        text = (
            text.strip()
        )

        self._load_tts()

        # MeloTTS 생성부터 playback 종료까지
        # 마이크 수집 중지
        self._set_speaking(
            True
        )

        raw_path = None
        playback_path = None

        try:

            # -----------------------------------------------------------------
            # MeloTTS RAW WAV
            # -----------------------------------------------------------------

            with tempfile.NamedTemporaryFile(
                prefix="merry_tts_raw_",
                suffix=".wav",
                delete=False,
            ) as tmp:

                raw_path = (
                    tmp.name
                )

            # -----------------------------------------------------------------
            # 실제 playback WAV
            # -----------------------------------------------------------------

            with tempfile.NamedTemporaryFile(
                prefix="merry_tts_play_",
                suffix=".wav",
                delete=False,
            ) as tmp:

                playback_path = (
                    tmp.name
                )

            # -----------------------------------------------------------------
            # TTS
            # -----------------------------------------------------------------

            logger.info(
                "[TTS] 음성 생성 시작: %s",
                text,
            )

            self._tts_model.tts_to_file(
                text,
                self._tts_speaker_id,
                raw_path,
                speed=TTS_SPEED,
                quiet=True,
            )

            logger.info(
                "[TTS] MeloTTS WAV 생성 완료"
            )

            # -----------------------------------------------------------------
            # 현재 기본 출력 sink용 재생 파일로 변환
            # -----------------------------------------------------------------

            self._prepare_tts_playback_file(
                raw_path,
                playback_path,
            )

            # -----------------------------------------------------------------
            # PulseAudio playback
            # -----------------------------------------------------------------

            self._play_tts_with_paplay(
                playback_path
            )

            # -----------------------------------------------------------------
            # 출력 tail 마이크 유입 방지
            # -----------------------------------------------------------------

            if (
                MIC_DRAIN_SECONDS
                > 0
            ):

                time.sleep(
                    MIC_DRAIN_SECONDS
                )

        except Exception:

            logger.exception(
                "[TTS] 음성 생성/재생 실패"
            )

            raise

        finally:

            for path in (
                raw_path,
                playback_path,
            ):

                if not path:
                    continue

                try:

                    Path(
                        path
                    ).unlink(
                        missing_ok=True
                    )

                except Exception:

                    logger.exception(
                        "TTS 임시 파일 삭제 실패: %s",
                        path,
                    )

            self._set_speaking(
                False
            )

            if (
                TTS_UNLOAD_AFTER_SPEAK
            ):

                self.release_tts()

    # =========================================================================
    # Close
    # =========================================================================

    def close(
        self,
    ) -> None:

        self.release_tts()

        self._stt_processor = None
        self._stt_model = None

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()
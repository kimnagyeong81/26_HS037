# -*- coding: utf-8 -*-

"""

감정 분석 + 게이지 + 개인화 + ESP32 제어를 하나로 묶은 통합 파이프라인.



기존 파일 통합 대상

- emotion_analyzer.py

- gauge_engine.py

- personalization_engine.py

- serial_controller.py

- emotion_gauge_pipeline.py



핵심 개선

- 사용자 상태를 한 번만 읽고 메모리에서 일관되게 관리.

- gauge / personalization의 threshold 불일치 제거.

- 발화당 MongoDB 상태 write를 1회로 축소.

- 감정 모델은 지연 로딩.

- ESP32는 실제 구슬 배출 시 처음 연결(지연 연결).

"""



from __future__ import annotations



import logging

import threading

import time

from collections import deque

from datetime import datetime, timezone



import torch



from config import (

    BASELINE_CONFIG,

    DEFAULT_THRESHOLDS,

    EMA_ALPHA,

    EMOTION_DEVICE,

    EMOTION_MODEL_NAME,

    EMOTION_KO,

    EMOTION_MAX_LENGTH,

    EMOTION_SCORE_MODE,

    EMOTIONS,

    MARBLE_COLORS,

    SERIAL_BAUD,

    SERIAL_PORT,

    SERIAL_RETRY_SEC,

    SERIAL_TIMEOUT,

    THRESHOLD_MAX,

    THRESHOLD_MIN,

    GAUGE_MIN_SCORE,

    THRESHOLD_MULTIPLIER,


    TRANSFORMERS_LOCAL_ONLY,

)

from mongo_client import get_user, save_user_state



logger = logging.getLogger(__name__)


HOWRU_LABEL_TO_EMOTION = {
    "기쁨": "joy",
    "설렘": "joy",
    "평범함": "calm",
    "분노": "anger",
    "불쾌함": "anger",
    "두려움": "fear",
    "슬픔": "sadness",
}
HOWRU_IGNORED_LABELS = {"놀라움"}
HOWRU_EXPECTED_LABELS = set(HOWRU_LABEL_TO_EMOTION) | HOWRU_IGNORED_LABELS
TARGET_EMOTIONS = {"anger", "fear", "joy", "calm", "sadness"}





def format_emotion_summary(scores: dict | None) -> str:

    if not scores:

        return "[감정분석] 결과 없음"



    top = max(scores, key=scores.get)

    parts = " · ".join(

        f"{EMOTION_KO.get(emotion, emotion)} {score:.1f}"

        for emotion, score in scores.items()

    )

    return (

        f"[감정분석] 최고: {EMOTION_KO.get(top, top)}({scores[top]:.1f})"

        f" | 전체: {parts}"

    )





class EmotionAnalyzer:

    """HowRU KoELECTRA 8감정을 프로젝트용 5감정으로 변환한다."""



    def __init__(self):

        self.tokenizer = None

        self.model = None

        self.id2label: dict[int, str] = {}

        self.device = self._resolve_device()



    @staticmethod

    def _resolve_device() -> torch.device:

        requested = EMOTION_DEVICE.lower()

        if requested == "auto":

            requested = "cuda" if torch.cuda.is_available() else "cpu"

        if requested.startswith("cuda") and not torch.cuda.is_available():

            logger.warning("감정 모델 CUDA 요청했지만 사용 불가 -> CPU")

            requested = "cpu"

        return torch.device(requested)



    def _load(self) -> None:

        if self.model is not None:

            return



        from transformers import AutoModelForSequenceClassification, AutoTokenizer



        if set(EMOTIONS) != TARGET_EMOTIONS:

            raise RuntimeError(

                "config.EMOTIONS는 정확히 "

                "['anger', 'fear', 'joy', 'calm', 'sadness'] 5종이어야 합니다. "

                f"현재 값: {EMOTIONS}"

            )



        logger.info(

            "감정 모델 로드 중: %s (device=%s)",

            EMOTION_MODEL_NAME,

            self.device,

        )

        self.tokenizer = AutoTokenizer.from_pretrained(

            EMOTION_MODEL_NAME,

            local_files_only=TRANSFORMERS_LOCAL_ONLY,

        )



        kwargs = {

            "local_files_only": TRANSFORMERS_LOCAL_ONLY,

            "low_cpu_mem_usage": False,

        }

        if self.device.type == "cuda":

            kwargs["torch_dtype"] = torch.float16



        self.model = AutoModelForSequenceClassification.from_pretrained(

            EMOTION_MODEL_NAME,

            **kwargs,

        )

        self.model.to(self.device)

        self.model.eval()



        num_labels = getattr(self.model.config, "num_labels", None)

        if num_labels != 8:

            raise RuntimeError(f"HowRU 감정 모델 라벨 수가 8이 아닙니다: {num_labels}")



        raw_id2label = getattr(self.model.config, "id2label", None) or {}

        self.id2label = {int(idx): str(label) for idx, label in raw_id2label.items()}

        loaded_labels = set(self.id2label.values())

        if loaded_labels != HOWRU_EXPECTED_LABELS:

            raise RuntimeError(

                "HowRU 모델의 id2label이 예상과 다릅니다. "

                f"expected={sorted(HOWRU_EXPECTED_LABELS)}, "

                f"loaded={sorted(loaded_labels)}"

            )



        logger.info(

            "감정 모델 로드 완료: HowRU 8감정 -> 프로젝트 5감정 변환 활성화"

        )



    def _to_scores(self, probs: list[float]) -> dict:

        # 최종 결과에는 프로젝트에서 쓰는 5개 감정만 남긴다.

        target_probs = {emotion: 0.0 for emotion in EMOTIONS}

        surprise_prob = 0.0



        for idx, prob in enumerate(probs):

            source_label = self.id2label.get(idx)

            target_emotion = HOWRU_LABEL_TO_EMOTION.get(source_label)



            if target_emotion in target_probs:

                target_probs[target_emotion] += float(prob)

            elif source_label in HOWRU_IGNORED_LABELS:

                surprise_prob += float(prob)



        # '놀라움'을 임의로 다른 감정에 넣지 않는다.

        # 확률이 높으면 디버깅 시 확인할 수 있도록 로그만 남긴다.

        if surprise_prob >= 0.30:

            logger.debug(

                "HowRU '놀라움' 확률이 높아 5감정 점수에서 제외됨: %.1f%%",

                surprise_prob * 100.0,

            )



        if EMOTION_SCORE_MODE == "max_norm":

            max_p = max(target_probs.values()) if target_probs else 0.0

            if max_p <= 0:

                return {emotion: 0.0 for emotion in EMOTIONS}

            return {

                emotion: round(prob / max_p * 100.0, 1)

                for emotion, prob in target_probs.items()

            }



        # 기본 softmax 비율 모드.

        # 놀라움 확률을 제외했기 때문에 합계가 100보다 작을 수 있다.

        return {

            emotion: round(prob * 100.0, 1)

            for emotion, prob in target_probs.items()

        }



    def analyze(self, text: str) -> dict | None:

        if not text or not text.strip():

            return None



        try:

            self._load()

            inputs = self.tokenizer(

                text,

                return_tensors="pt",

                padding=True,

                truncation=True,

                max_length=min(EMOTION_MAX_LENGTH, 512),

            ).to(self.device)



            with torch.inference_mode():

                logits = self.model(**inputs).logits

                probs = torch.softmax(logits, dim=1)[0]



            scores = self._to_scores(probs.float().cpu().tolist())

            del inputs, logits, probs

            return scores

        except Exception:

            logger.exception("감정 분석 실패: %s", text[:60])

            return None



    def close(self) -> None:

        self.tokenizer = None

        self.model = None

        self.id2label = {}

        if torch.cuda.is_available():

            torch.cuda.empty_cache()




class SerialController:

    """

    ESP32 연결은 실제 구슬 배출이 필요할 때까지 미룬다.

    연결 실패 시 단일 daemon timer로 재연결한다.

    """



    def __init__(self):

        self.port = None

        self._queue: deque[str] = deque()

        self._lock = threading.Lock()

        self._retry_timer: threading.Timer | None = None

        self._closed = False



    def _is_connected(self) -> bool:

        return self.port is not None and self.port.is_open



    def _connect_locked(self) -> bool:

        if self._closed or self._is_connected():

            return self._is_connected()



        try:

            import serial



            self.port = serial.Serial(

                SERIAL_PORT,

                SERIAL_BAUD,

                timeout=SERIAL_TIMEOUT,

            )

            time.sleep(2.0)  # ESP32 serial open 시 reset 대기

            self.port.reset_input_buffer()

            logger.info("ESP32 연결 성공: %s @ %sbps", SERIAL_PORT, SERIAL_BAUD)

            self._flush_queue_locked()

            return True

        except Exception as exc:

            self._safe_close_port_locked()

            logger.warning("ESP32 연결 실패: %s", exc)

            self._schedule_retry_locked()

            return False



    def _safe_close_port_locked(self) -> None:

        if self.port is not None:

            try:

                self.port.close()

            except Exception:

                pass

        self.port = None



    def _schedule_retry_locked(self) -> None:

        if self._closed:

            return

        if self._retry_timer is not None and self._retry_timer.is_alive():

            return



        def _retry():

            with self._lock:

                self._retry_timer = None

                self._connect_locked()



        timer = threading.Timer(SERIAL_RETRY_SEC, _retry)

        timer.daemon = True

        self._retry_timer = timer

        timer.start()



    def _send_command_locked(self, command: str) -> bool:

        if not self._is_connected():

            return False

        try:

            self.port.write(command.encode("utf-8"))

            self.port.flush()

            ack = self.port.readline().decode(errors="replace").strip()

            logger.info("ESP32 명령: %s / ACK=%s", command.strip(), ack or "없음")

            return True

        except Exception:

            logger.exception("ESP32 전송 실패")

            self._safe_close_port_locked()

            self._schedule_retry_locked()

            return False



    def _flush_queue_locked(self) -> None:

        while self._queue and self._is_connected():

            command = self._queue[0]

            if not self._send_command_locked(command):

                return

            self._queue.popleft()



    def send_marble(self, emotion: str) -> None:

        color = MARBLE_COLORS.get(emotion, "WHITE")

        command = f"MARBLE:{color}\n"



        with self._lock:

            if not self._is_connected():

                self._connect_locked()



            if self._is_connected() and self._send_command_locked(command):

                return



            self._queue.append(command)

            logger.warning("ESP32 미연결 -> 명령 큐 보관: %s", command.strip())

            self._schedule_retry_locked()



    def close(self) -> None:

        with self._lock:

            self._closed = True

            if self._retry_timer is not None:

                self._retry_timer.cancel()

                self._retry_timer = None

            self._safe_close_port_locked()

            self._queue.clear()





class EmotionPipeline:

    """감정 시스템의 단일 상태 관리자."""



    def __init__(self, user_id: str):

        self.user_id = user_id

        user = get_user(user_id)



        self.gauges = dict(user["current_gauges"])

        self.thresholds = dict(user["thresholds"])

        self.phase = user["phase"]

        self.baseline_scores = dict(user["baseline_scores"])

        self.utterance_count = int(user["utterance_count"])

        self.baseline_start = user["baseline_start"]

        self.baseline_days = int(user.get("baseline_days", 0))



        self.analyzer = EmotionAnalyzer()

        self.serial = SerialController()



        logger.info(

            "[%s] EmotionPipeline 초기화: phase=%s gauges=%s thresholds=%s",

            user_id,

            self.phase,

            self.gauges,

            self.thresholds,

        )



    def _elapsed_days(self) -> int:

        start = self.baseline_start

        if start.tzinfo is None:

            start = start.replace(tzinfo=timezone.utc)

        return max((datetime.now(timezone.utc) - start).days, 0)



    def _update_gauges(self, scores: dict) -> list[dict]:

        """게이지를 누적하고 배출 이벤트 메타데이터를 반환한다."""

        events: list[dict] = []



        for emotion in EMOTIONS:

            score = float(
                scores.get(emotion, 0.0)
            )

            if score < GAUGE_MIN_SCORE:
                continue

            before_reset = round(
                self.gauges[emotion] + score,
                2,
            )

            threshold = float(self.thresholds[emotion])



            if before_reset >= threshold:

                remainder = round(before_reset - threshold, 2)

                self.gauges[emotion] = remainder

                events.append(

                    {

                        "emotion": emotion,

                        "color": MARBLE_COLORS[emotion],

                        "gauge_at_trigger": before_reset,

                        "threshold_at_time": threshold,

                        "gauge_remainder": remainder,

                    }

                )

            else:

                self.gauges[emotion] = before_reset



        return events



    def _calculate_personalized_thresholds(self) -> dict:

        thresholds = {}

        for emotion in EMOTIONS:

            avg = self.baseline_scores[emotion] / max(self.utterance_count, 1)

            raw = avg * THRESHOLD_MULTIPLIER

            if raw <= 0:

                raw = DEFAULT_THRESHOLDS[emotion]

            thresholds[emotion] = round(

                max(THRESHOLD_MIN, min(THRESHOLD_MAX, raw)),

                2,

            )

        return thresholds



    def _update_personalization(self, scores: dict) -> dict | None:

        """현재 thresholds도 즉시 갱신해 DB/메모리 불일치를 없앤다."""

        self.baseline_days = self._elapsed_days()



        if self.phase == "baseline":

            for emotion in EMOTIONS:

                self.baseline_scores[emotion] += float(scores.get(emotion, 0.0))

            self.utterance_count += 1



            complete = (

                self.utterance_count >= BASELINE_CONFIG["min_utterances"]

                and self.baseline_days >= BASELINE_CONFIG["min_days"]

            )

            if not complete:

                return None



            self.thresholds = self._calculate_personalized_thresholds()

            self.phase = "personalized"

            logger.info("[%s] 개인화 Phase 전환: %s", self.user_id, self.thresholds)

            return dict(self.thresholds)



        # personalized: EMA로 기준을 갱신하고 즉시 gauge 임계치에도 반영

        previous_count = max(self.utterance_count, 1)

        old_thresholds = dict(self.thresholds)

        self.utterance_count += 1



        new_thresholds = {}

        for emotion in EMOTIONS:

            old_avg = self.baseline_scores[emotion] / previous_count

            new_avg = (

                (1.0 - EMA_ALPHA) * old_avg

                + EMA_ALPHA * float(scores.get(emotion, 0.0))

            )

            self.baseline_scores[emotion] = new_avg * self.utterance_count



            raw = new_avg * THRESHOLD_MULTIPLIER

            if raw <= 0:

                raw = old_thresholds[emotion]

            clamped = round(

                max(THRESHOLD_MIN, min(THRESHOLD_MAX, raw)),

                2,

            )



            new_thresholds[emotion] = clamped



        self.thresholds = new_thresholds

        return dict(new_thresholds)



    def _persist_state(self) -> None:

        save_user_state(

            self.user_id,

            gauges=self.gauges,

            thresholds=self.thresholds,

            phase=self.phase,

            baseline_scores=self.baseline_scores,

            utterance_count=self.utterance_count,

            baseline_days=self.baseline_days,

        )



    def process_text(

        self,

        text: str,

        *,

        source_file: str = "mic",

        session_id: str = "S001",

        timestamp: datetime | None = None,

    ) -> dict:

        if not text or not text.strip():

            return {

                "scores": None,

                "triggered": [],

                "events": [],

                "gauge_snapshot": dict(self.gauges),

                "thresholds_snapshot": dict(self.thresholds),

                "phase_at_time": self.phase,

                "new_thresholds": None,

                "error": "empty_text",

            }



        scores = self.analyzer.analyze(text)

        if not scores or sum(scores.values()) <= 0:

            # 모델 실패/0점 fallback을 개인화 데이터로 학습하지 않는다.

            return {

                "scores": scores,

                "triggered": [],

                "events": [],

                "gauge_snapshot": dict(self.gauges),

                "thresholds_snapshot": dict(self.thresholds),

                "phase_at_time": self.phase,

                "new_thresholds": None,

                "error": "emotion_analysis_failed",

            }



        phase_at_time = self.phase

        thresholds_at_time = dict(self.thresholds)



        events = self._update_gauges(scores)

        new_thresholds = self._update_personalization(scores)

        self._persist_state()



        for event in events:

            self.serial.send_marble(event["emotion"])

            logger.info(

                "[구슬 배출] %s gauge=%.1f threshold=%.1f remainder=%.1f",

                EMOTION_KO.get(event["emotion"], event["emotion"]),

                event["gauge_at_trigger"],

                event["threshold_at_time"],

                event["gauge_remainder"],

            )



        result = {

            "scores": scores,

            "triggered": [e["emotion"] for e in events],

            "events": events,

            "gauge_snapshot": dict(self.gauges),

            "thresholds_snapshot": thresholds_at_time,

            "current_thresholds": dict(self.thresholds),

            "phase_at_time": phase_at_time,

            "current_phase": self.phase,

            "new_thresholds": new_thresholds,

            "source_file": source_file,

            "session_id": session_id,

            "timestamp": timestamp or datetime.now(timezone.utc),

            "error": None,

        }



        logger.info(format_emotion_summary(scores))

        logger.debug("게이지 상태: %s", self.get_status())

        return result



    def get_status(self) -> dict:

        return {

            "user_id": self.user_id,

            "phase": self.phase,

            "gauges": dict(self.gauges),

            "thresholds": dict(self.thresholds),

            "fill_ratio": {

                emotion: round(

                    self.gauges[emotion] / max(self.thresholds[emotion], 1e-9) * 100.0,

                    1,

                )

                for emotion in EMOTIONS

            },

            "utterance_count": self.utterance_count,

            "baseline_days": self.baseline_days,

        }



    def close(self) -> None:

        self.serial.close()

        self.analyzer.close()

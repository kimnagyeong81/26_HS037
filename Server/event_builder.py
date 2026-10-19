# event_builder.py
"""
치매 분석 파이프라인용 event 생성 모듈

하이브리드 분석 구조:
- 발화별 event: dementia_utterance_analysis
- 하루 통합 event: dementia_daily_analysis

기존 build_event()는 유지하고, 메타데이터를 추가로 받을 수 있도록 확장했다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional


EMOTION_KEYS = ["joy", "calm", "anger", "fear", "sadness"]


def safe_float(value: Any, default: float = 0.0) -> float:
    """숫자 변환 실패 시 기본값을 반환한다."""
    try:
        if value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """정수 변환 실패 시 기본값을 반환한다."""
    try:
        if value is None:
            return default
        return int(value)
    except (ValueError, TypeError):
        return default


def _to_iso_timestamp(timestamp: Any) -> str:
    """timestamp를 ISO 문자열로 정규화한다."""
    if isinstance(timestamp, datetime):
        return timestamp.isoformat()
    if timestamp is None:
        return datetime.now().isoformat()
    return str(timestamp)


# =============================================================================
# 감정 payload
# =============================================================================

def build_emotion_payload(emotion_result: Optional[Dict]) -> Dict:
    """
    치매 파이프라인 전용 감정 결과를 event용 payload로 정리한다.
    """
    payload = {
        "emotion_joy": 0.0,
        "emotion_calm": 0.0,
        "emotion_anger": 0.0,
        "emotion_fear": 0.0,
        "emotion_sadness": 0.0,
        "emotion_total_score": 0.0,
        "emotion_positive_score": 0.0,
        "emotion_negative_score": 0.0,
    }

    if not emotion_result:
        payload["emotion_label"] = "unknown"
        payload["emotion_score"] = 0.0
        return payload

    for emotion in EMOTION_KEYS:
        src_value = emotion_result.get(emotion, emotion_result.get(f"emotion_{emotion}", 0.0))
        payload[f"emotion_{emotion}"] = safe_float(src_value)

    positive = payload["emotion_joy"] + payload["emotion_calm"]
    negative = payload["emotion_anger"] + payload["emotion_fear"] + payload["emotion_sadness"]
    total = positive + negative

    payload["emotion_positive_score"] = round(positive, 4)
    payload["emotion_negative_score"] = round(negative, 4)
    payload["emotion_total_score"] = round(total, 4)

    emotion_scores = {
        "joy": payload["emotion_joy"],
        "calm": payload["emotion_calm"],
        "anger": payload["emotion_anger"],
        "fear": payload["emotion_fear"],
        "sadness": payload["emotion_sadness"],
    }
    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
    payload["emotion_label"] = dominant_emotion
    payload["emotion_score"] = emotion_scores[dominant_emotion]

    return payload


# =============================================================================
# 지시대명사 payload
# =============================================================================

def build_pronoun_payload(pronoun_result: Optional[Dict]) -> Dict:
    """지시대명사 분석 결과를 event용 payload로 정리한다."""
    if not pronoun_result:
        return {
            "demonstrative_pronoun_count": 0,
            "demonstrative_pronoun_exists": False,
            "demonstrative_pronoun_words": [],
            "demonstrative_pronoun_word_counts": {},
        }

    return {
        "demonstrative_pronoun_count": safe_int(pronoun_result.get("count", 0)),
        "demonstrative_pronoun_exists": bool(pronoun_result.get("exists", False)),
        "demonstrative_pronoun_words": pronoun_result.get("words", []),
        "demonstrative_pronoun_word_counts": pronoun_result.get("word_counts", {}),
    }


# =============================================================================
# 어휘 다양성 payload
# =============================================================================

def build_lexical_diversity_payload(lexical_result: Optional[Dict]) -> Dict:
    """어휘다양성 분석 결과를 event용 payload로 정리한다."""
    if not lexical_result:
        return {
            "num_utterances": 0,
            "window_size": None,
            "total_tokens": 0,
            "unique_tokens": 0,
            "ttr": 0.0,
            "repeated_ratio": 0.0,
            "top_repeated_words": [],
            "reliability_score": 0.0,
            "reliability_label": "unknown",
            "length_interpretation": "",
            "ttr_interpretation": "",
        }

    return {
        "num_utterances": safe_int(lexical_result.get("num_utterances", 1)),
        "window_size": lexical_result.get("window_size"),
        "total_tokens": safe_int(lexical_result.get("total_tokens", 0)),
        "unique_tokens": safe_int(lexical_result.get("unique_tokens", 0)),
        "ttr": safe_float(lexical_result.get("ttr", 0.0)),
        "repeated_ratio": safe_float(lexical_result.get("repeated_ratio", 0.0)),
        "top_repeated_words": lexical_result.get("top_repeated_words", []),
        "reliability_score": safe_float(lexical_result.get("reliability_score", 0.0)),
        "reliability_label": lexical_result.get("reliability_label", "unknown"),
        "length_interpretation": lexical_result.get("length_interpretation", ""),
        "ttr_interpretation": lexical_result.get("ttr_interpretation", ""),
    }


# =============================================================================
# 반복 표현 payload
# =============================================================================

def build_repetition_payload(repetition_result: Optional[Dict]) -> Dict:
    """반복 분석 결과를 event용 payload로 정리한다."""
    if not repetition_result:
        return {
            "repetition_exists": False,
            "repetition_total_tokens": 0,
            "repetition_unique_tokens": 0,
            "repetition_ratio": 0.0,
            "repeated_words": {},
            "most_repeated_word": None,
            "most_repeated_count": 0,
        }

    return {
        "repetition_exists": bool(repetition_result.get("repetition_exists", False)),
        "repetition_total_tokens": safe_int(repetition_result.get("total_tokens", 0)),
        "repetition_unique_tokens": safe_int(repetition_result.get("unique_tokens", 0)),
        "repetition_ratio": safe_float(repetition_result.get("repetition_ratio", 0.0)),
        "repeated_words": repetition_result.get("repeated_words", {}),
        "most_repeated_word": repetition_result.get("most_repeated_word"),
        "most_repeated_count": safe_int(repetition_result.get("most_repeated_count", 0)),
    }


# =============================================================================
# 문장 복잡도 payload
# =============================================================================

def build_sentence_complexity_payload(complexity_result: Optional[Dict]) -> Dict:
    """문장 복잡도 분석 결과를 event용 payload로 정리한다."""
    if not complexity_result:
        return {
            "sentence_count": 0,
            "average_sentence_length": 0.0,
            "average_word_count": 0.0,
            "average_connective_count": 0.0,
            "average_clause_count": 0.0,
            "sentence_complexity_score": 0.0,
            "sentence_complexity_label": "unknown",
            "sentence_details": [],
        }

    return {
        "sentence_count": safe_int(complexity_result.get("sentence_count", 0)),
        "average_sentence_length": safe_float(complexity_result.get("average_sentence_length", 0.0)),
        "average_word_count": safe_float(complexity_result.get("average_word_count", 0.0)),
        "average_connective_count": safe_float(complexity_result.get("average_connective_count", 0.0)),
        "average_clause_count": safe_float(complexity_result.get("average_clause_count", 0.0)),
        "sentence_complexity_score": safe_float(complexity_result.get("overall_complexity_score", 0.0)),
        "sentence_complexity_label": complexity_result.get("overall_complexity_level", "unknown"),
        "sentence_details": complexity_result.get("sentence_details", []),
    }


# =============================================================================
# 최종 event 생성
# =============================================================================

def build_event(
    user_id: str,
    session_id: str,
    timestamp: Any,
    text: str,
    emotion_result: Optional[Dict] = None,
    pronoun_result: Optional[Dict] = None,
    lexical_result: Optional[Dict] = None,
    repetition_result: Optional[Dict] = None,
    complexity_result: Optional[Dict] = None,
    source_file: Optional[str] = None,
    risk_label: Optional[int] = None,
    event_type: str = "dementia_utterance_analysis",
    granularity: str = "utterance",
    date: Optional[str] = None,
    utterance_id: Optional[str] = None,
    speaker: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict:
    """
    최종 통합 event 생성.

    기존 LSTM 학습/추론 호환을 위해 주요 feature는 flat field로 유지하고,
    하이브리드 처리를 위해 event_type, granularity, date, utterance_id 등을 추가한다.
    """
    event = {
        "event_type": event_type,
        "pipeline": "dementia",
        "granularity": granularity,
        "user_id": user_id,
        "session_id": session_id,
        "timestamp": _to_iso_timestamp(timestamp),
        "text": str(text).strip() if text is not None else "",
        "created_at": datetime.now().isoformat(),
    }

    if date is not None:
        event["date"] = str(date)

    if utterance_id is not None:
        event["utterance_id"] = str(utterance_id)

    if speaker is not None:
        event["speaker"] = str(speaker)

    if source_file is not None:
        event["source_file"] = source_file

    event.update(build_emotion_payload(emotion_result))
    event.update(build_pronoun_payload(pronoun_result))
    event.update(build_lexical_diversity_payload(lexical_result))
    event.update(build_repetition_payload(repetition_result))
    event.update(build_sentence_complexity_payload(complexity_result))

    if risk_label is not None:
        event["risk_label"] = safe_int(risk_label)

    if metadata:
        event["metadata"] = metadata

    return event


def build_utterance_event(
    user_id: str,
    session_id: str,
    date: Optional[str],
    utterance_id: Optional[str],
    timestamp: Any,
    speaker: Optional[str],
    text: str,
    emotion_result: Optional[Dict] = None,
    pronoun_result: Optional[Dict] = None,
    lexical_result: Optional[Dict] = None,
    repetition_result: Optional[Dict] = None,
    complexity_result: Optional[Dict] = None,
    source_file: Optional[str] = None,
    risk_label: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict:
    """발화 1건 분석 event 생성."""
    return build_event(
        event_type="dementia_utterance_analysis",
        granularity="utterance",
        user_id=user_id,
        session_id=session_id,
        date=date,
        utterance_id=utterance_id,
        timestamp=timestamp,
        speaker=speaker,
        text=text,
        emotion_result=emotion_result,
        pronoun_result=pronoun_result,
        lexical_result=lexical_result,
        repetition_result=repetition_result,
        complexity_result=complexity_result,
        source_file=source_file,
        risk_label=risk_label,
        metadata=metadata,
    )


def build_daily_event(
    user_id: str,
    session_id: str,
    date: str,
    daily_text: str,
    total_utterances: int,
    first_timestamp: Optional[Any],
    last_timestamp: Optional[Any],
    emotion_result: Optional[Dict] = None,
    pronoun_result: Optional[Dict] = None,
    lexical_result: Optional[Dict] = None,
    repetition_result: Optional[Dict] = None,
    complexity_result: Optional[Dict] = None,
    source_file: Optional[str] = None,
    risk_label: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict:
    """하루 전체 통합 분석 event 생성."""
    daily_metadata = {
        "total_utterances": safe_int(total_utterances),
        "daily_text_length": len(str(daily_text or "")),
        "first_timestamp": _to_iso_timestamp(first_timestamp) if first_timestamp else None,
        "last_timestamp": _to_iso_timestamp(last_timestamp) if last_timestamp else None,
    }

    if metadata:
        daily_metadata.update(metadata)

    event = build_event(
        event_type="dementia_daily_analysis",
        granularity="daily",
        user_id=user_id,
        session_id=session_id,
        date=date,
        timestamp=last_timestamp or first_timestamp or datetime.now(),
        text=daily_text,
        emotion_result=emotion_result,
        pronoun_result=pronoun_result,
        lexical_result=lexical_result,
        repetition_result=repetition_result,
        complexity_result=complexity_result,
        source_file=source_file,
        risk_label=risk_label,
        metadata=daily_metadata,
    )

    event["total_utterances"] = daily_metadata["total_utterances"]
    event["daily_text_length"] = daily_metadata["daily_text_length"]
    event["first_timestamp"] = daily_metadata["first_timestamp"]
    event["last_timestamp"] = daily_metadata["last_timestamp"]

    return event

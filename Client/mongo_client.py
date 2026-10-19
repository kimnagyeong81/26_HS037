# -*- coding: utf-8 -*-
"""
MongoDB 접근 전담 모듈.

핵심 변경
- MongoClient를 프로세스 동안 1회 재사용한다.
- 매 함수 호출마다 ping하지 않는다.
- 사용자 상태(gauge + personalization)를 한 번의 update로 저장한다.
- 한 발화를 utterances에 중복 저장하지 않고 conversation_turn 문서 1개로 통일한다.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import PyMongoError

from config import (
    COLLECTIONS,
    CONTEXT_WINDOW_SIZE,
    DEFAULT_THRESHOLDS,
    EMOTIONS,
    MONGO_DB,
    MONGO_SERVER_SELECTION_TIMEOUT_MS,
    MONGO_URI,
)

logger = logging.getLogger(__name__)

_client: MongoClient | None = None
_db = None


def get_db():
    global _client, _db

    if _client is not None and _db is not None:
        return _db

    try:
        _client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=MONGO_SERVER_SELECTION_TIMEOUT_MS,
        )
        _client.admin.command("ping")  # 최초 연결 때만 확인
        _db = _client[MONGO_DB]
        _init_indexes()
        logger.info("MongoDB 연결 성공: %s", MONGO_URI)
        return _db
    except PyMongoError:
        logger.exception("MongoDB 연결 실패")
        if _client is not None:
            _client.close()
        _client = None
        _db = None
        raise


def close_db() -> None:
    global _client, _db
    if _client is not None:
        _client.close()
    _client = None
    _db = None


def _init_indexes() -> None:
    db = _db
    if db is None:
        return

    db[COLLECTIONS["users"]].create_index("user_id", unique=True)
    db[COLLECTIONS["utterances"]].create_index(
        [("user_id", ASCENDING), ("timestamp", DESCENDING)]
    )
    db[COLLECTIONS["utterances"]].create_index(
        [("user_id", ASCENDING), ("session_id", ASCENDING), ("timestamp", DESCENDING)]
    )
    db[COLLECTIONS["marble_events"]].create_index(
        [("user_id", ASCENDING), ("timestamp", DESCENDING)]
    )
    db[COLLECTIONS["marble_events"]].create_index(
        [("user_id", ASCENDING), ("emotion", ASCENDING)]
    )
    db[COLLECTIONS["daily_summaries"]].create_index(
        [("user_id", ASCENDING), ("date", DESCENDING)],
        # unique=True,
    )


def _default_user(user_id: str) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "user_id": user_id,
        "phase": "baseline",
        "baseline_start": now,
        "baseline_days": 0,
        "current_gauges": {e: 0.0 for e in EMOTIONS},
        "thresholds": dict(DEFAULT_THRESHOLDS),
        "baseline_scores": {e: 0.0 for e in EMOTIONS},
        "utterance_count": 0,
        "created_at": now,
        "last_updated": now,
    }


def get_user(user_id: str) -> dict:
    try:
        col = get_db()[COLLECTIONS["users"]]
        user = col.find_one({"user_id": user_id})
        if user is None:
            user = _default_user(user_id)
            col.insert_one(user)
            logger.info("신규 사용자 생성: %s", user_id)
        return user
    except PyMongoError:
        logger.exception("[get_user] MongoDB 오류")
        raise


def save_user_state(
    user_id: str,
    *,
    gauges: dict,
    thresholds: dict,
    phase: str,
    baseline_scores: dict,
    utterance_count: int,
    baseline_days: int,
) -> None:
    """한 발화 처리 후 변경된 감정 시스템 상태를 1회 DB write로 동기화한다."""
    try:
        get_db()[COLLECTIONS["users"]].update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "current_gauges": dict(gauges),
                    "thresholds": dict(thresholds),
                    "phase": phase,
                    "baseline_scores": dict(baseline_scores),
                    "utterance_count": int(utterance_count),
                    "baseline_days": int(baseline_days),
                    "last_updated": datetime.now(timezone.utc),
                }
            },
            upsert=True,
        )
    except PyMongoError:
        logger.exception("[save_user_state] MongoDB 오류")


def save_conversation_turn(
    *,
    user_id: str,
    session_id: str,
    user_text: str,
    merry_response: str,
    emotion_scores: dict | None,
    source_file: str | None = None,
    gauge_snapshot: dict | None = None,
    thresholds_snapshot: dict | None = None,
    phase_at_time: str | None = None,
    triggered_emotions: list[str] | None = None,
    timestamp: datetime | None = None,
):
    """
    한 대화 턴을 딱 한 번 저장한다.

    기존 구조의 save_utterance() + save_conversation_turn() 중복을 제거하고
    텍스트/LLM 응답/감정/게이지를 하나의 문서에 모은다.
    """
    doc = {
        "type": "conversation_turn",
        "user_id": user_id,
        "session_id": session_id,
        "timestamp": timestamp or datetime.now(timezone.utc),
        "text": user_text,
        "merry_response": merry_response,
        "emotion_scores": emotion_scores or {},
        "source_file": source_file,
        "gauge_snapshot": gauge_snapshot or {},
        "thresholds_snapshot": thresholds_snapshot or {},
        "phase_at_time": phase_at_time,
        "triggered_emotions": list(triggered_emotions or []),
    }

    try:
        result = get_db()[COLLECTIONS["utterances"]].insert_one(doc)
        return result.inserted_id
    except PyMongoError:
        logger.exception("[save_conversation_turn] MongoDB 오류")
        return None


def save_marble_events(
    *,
    user_id: str,
    events: list[dict],
    utterance_id=None,
    timestamp: datetime | None = None,
) -> None:
    """
    구슬 이벤트를 저장하면서 최근 CONTEXT_WINDOW_SIZE개 turn ID를 연결한다.
    과거 구조처럼 동일 발화를 다시 insert하지 않으면서도 트리거 전후 맥락을 추적할 수 있다.
    """
    if not events:
        return

    try:
        db = get_db()
        recent_turns = list(
            db[COLLECTIONS["utterances"]]
            .find(
                {"user_id": user_id, "type": "conversation_turn"},
                {"_id": 1},
            )
            .sort("timestamp", DESCENDING)
            .limit(CONTEXT_WINDOW_SIZE)
        )
        context_ids = [doc["_id"] for doc in reversed(recent_turns)]

        ts = timestamp or datetime.now(timezone.utc)
        docs = []
        for event in events:
            docs.append(
                {
                    "user_id": user_id,
                    "timestamp": ts,
                    "emotion": event["emotion"],
                    "color": event["color"],
                    "gauge_at_trigger": event["gauge_at_trigger"],
                    "threshold_at_time": event["threshold_at_time"],
                    "gauge_remainder": event.get("gauge_remainder"),
                    "utterance_id": utterance_id,
                    "context_utterance_ids": context_ids,
                }
            )

        db[COLLECTIONS["marble_events"]].insert_many(docs)
    except PyMongoError:
        logger.exception("[save_marble_events] MongoDB 오류")


def create_daily_summary(
    user_id: str,
    date_str: str,
    stats: dict,
    marble_counts: dict,
    utterance_count: int,
) -> None:
    doc = {
        "user_id": user_id,
        "date": date_str,
        "stats": stats,
        "marble_counts": marble_counts,
        "utterance_count": utterance_count,
        "created_at": datetime.now(timezone.utc),
    }
    try:
        get_db()[COLLECTIONS["daily_summaries"]].update_one(
            {"user_id": user_id, "date": date_str},
            {"$set": doc},
            upsert=True,
        )
    except PyMongoError:
        logger.exception("[create_daily_summary] MongoDB 오류")


def get_recent_daily_summaries(user_id: str, days: int = 7) -> list:
    try:
        return list(
            get_db()[COLLECTIONS["daily_summaries"]]
            .find({"user_id": user_id})
            .sort("date", DESCENDING)
            .limit(days)
        )
    except PyMongoError:
        logger.exception("[get_recent_daily_summaries] MongoDB 오류")
        return []


def build_and_save_daily_summary(user_id: str, date_str: str) -> None:
    """conversation_turn 문서만 집계해 중복 집계를 방지한다."""
    try:
        date_start = datetime.strptime(date_str, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        date_end = date_start.replace(hour=23, minute=59, second=59, microsecond=999999)

        db = get_db()
        utterances = list(
            db[COLLECTIONS["utterances"]].find(
                {
                    "user_id": user_id,
                    "type": "conversation_turn",
                    "timestamp": {"$gte": date_start, "$lte": date_end},
                }
            )
        )
        marbles = list(
            db[COLLECTIONS["marble_events"]].find(
                {
                    "user_id": user_id,
                    "timestamp": {"$gte": date_start, "$lte": date_end},
                }
            )
        )

        scored = [
            u for u in utterances
            if isinstance(u.get("emotion_scores"), dict) and u["emotion_scores"]
        ]
        if not scored:
            logger.info("[%s] %s 감정 점수 발화 없음, summary 생략", user_id, date_str)
            return

        stats = {}
        for emotion in EMOTIONS:
            values = np.array(
                [float(u["emotion_scores"].get(emotion, 0.0)) for u in scored],
                dtype=np.float32,
            )
            stats[emotion] = {
                "mean": round(float(values.mean()), 2),
                "std": round(float(values.std()), 2),
                "min": round(float(values.min()), 2),
                "max": round(float(values.max()), 2),
            }

        marble_counts = {e: 0 for e in EMOTIONS}
        for marble in marbles:
            emotion = marble.get("emotion")
            if emotion in marble_counts:
                marble_counts[emotion] += 1

        create_daily_summary(
            user_id=user_id,
            date_str=date_str,
            stats=stats,
            marble_counts=marble_counts,
            utterance_count=len(utterances),
        )
    except (PyMongoError, ValueError):
        logger.exception("[build_and_save_daily_summary] 오류")

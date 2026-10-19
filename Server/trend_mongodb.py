# -*- coding: utf-8 -*-
"""발화 이벤트 조회, 사용자 목록 조회, 이상 징후 결과 저장을 위한 MongoDB 접근."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from trend_settings import MONGO_URI, DB_NAME, EVENT_COLLECTION, ANOMALY_COLLECTION

def get_mongo_client(mongo_uri: str = MONGO_URI):
    """MongoDB Client를 생성합니다."""
    from pymongo import MongoClient

    return MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=5000,
    )


def load_events_from_mongodb(
    user_id: Optional[str] = None,
    mongo_uri: str = MONGO_URI,
    db_name: str = DB_NAME,
    collection_name: str = EVENT_COLLECTION,
) -> List[Dict[str, Any]]:
    """
    MongoDB dementia_db.events에서 발화 단위 이벤트만 조회합니다.
    """
    query: Dict[str, Any] = {
        "$or": [
            {"granularity": "utterance"},
            {
                "event_type":
                "dementia_utterance_analysis"
            },
        ]
    }

    if user_id:
        query["user_id"] = user_id

    client = get_mongo_client(mongo_uri)

    try:
        collection = client[db_name][collection_name]

        cursor = collection.find(query).sort(
            "timestamp",
            1,
        )

        return list(cursor)

    finally:
        client.close()


def list_user_ids(
    mongo_uri: str = MONGO_URI,
    db_name: str = DB_NAME,
    collection_name: str = EVENT_COLLECTION,
) -> List[str]:
    """발화 이벤트가 존재하는 전체 사용자 ID를 조회합니다."""
    client = get_mongo_client(mongo_uri)

    try:
        collection = client[db_name][collection_name]

        query = {
            "$or": [
                {"granularity": "utterance"},
                {
                    "event_type":
                    "dementia_utterance_analysis"
                },
            ]
        }

        user_ids = collection.distinct(
            "user_id",
            query,
        )

        return sorted(
            str(user_id)
            for user_id in user_ids
            if user_id
        )

    finally:
        client.close()


def save_results_to_mongodb(
    results: List[Dict[str, Any]],
    mongo_uri: str = MONGO_URI,
    db_name: str = DB_NAME,
    collection_name: str = ANOMALY_COLLECTION,
) -> int:
    """
    Rolling Z-score + EWMA 결과를 anomaly_results에 저장하거나 갱신합니다.
    """
    if not results:
        return 0

    client = get_mongo_client(mongo_uri)

    try:
        collection = client[db_name][collection_name]

        collection.create_index(
            [
                ("user_id", 1),
                ("date", 1),
                ("model_type", 1),
            ],
            unique=True,
        )

        collection.create_index(
            [
                ("user_id", 1),
                ("created_at", -1),
            ]
        )

        saved_count = 0

        for result in results:
            model_type = result.get(
                "model_type",
                "rolling_zscore_ewma",
            )

            collection.update_one(
                {
                    "user_id": result.get("user_id"),
                    "date": result.get("date"),
                    "model_type": model_type,
                },
                {
                    "$set": result,
                },
                upsert=True,
            )

            saved_count += 1

        return saved_count

    finally:
        client.close()

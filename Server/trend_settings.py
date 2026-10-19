# -*- coding: utf-8 -*-
"""MongoDB 연결·감정 피처·Z-score/EWMA 공통 설정."""
from __future__ import annotations
import os
from typing import Dict

MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017/")
DB_NAME = os.getenv("DEMENTIA_DB_NAME", "dementia_db")
EVENT_COLLECTION = os.getenv("EVENT_COLLECTION", "events")
ANOMALY_COLLECTION = os.getenv("ANOMALY_COLLECTION", "anomaly_results")

EMOTION_COLUMNS = [
    "emotion_joy",
    "emotion_calm",
    "emotion_anger",
    "emotion_fear",
    "emotion_sadness",
]

EMOTION_SUM_FEATURES = [
    "emotion_joy_sum",
    "emotion_calm_sum",
    "emotion_anger_sum",
    "emotion_fear_sum",
    "emotion_sadness_sum",
]

DAILY_FEATURE_COLUMNS = [
    "emotion_joy_sum",
    "emotion_calm_sum",
    "emotion_anger_sum",
    "emotion_fear_sum",
    "emotion_sadness_sum",
    "emotion_gap_mean",
    "emotion_gap_std",
    "emotion_gap_min",
    "emotion_gap_max",
    "positive_emotion_ratio",
    "negative_emotion_ratio",
    "dominant_emotion_ratio",
    "emotion_mixed_score",
    "demonstrative_pronoun_count_mean",
    "ttr_mean",
    "repeated_ratio_mean",
    "sentence_complexity_score_mean",
    "event_count",
]


MIN_BASELINE_DAYS = 7
WINDOW_SIZE = 7
TREND_DAYS = 4
EWMA_ALPHA = 0.3

# risk_z가 이 값 이상일 때 의미 있는 위험 feature로 본다.
RISK_Z_THRESHOLD = 1.6

# 개별 feature의 risk_z 상한값
RISK_Z_CAP = 5.0


# =============================================================================
# feature 정의
# =============================================================================

# up: 값이 증가하면 위험
# down: 값이 감소하면 위험
FEATURE_DIRECTIONS = {
    "demonstrative_pronoun_count_mean": "up",
    "repeated_ratio_mean": "up",
    "negative_emotion_ratio": "up",

    "ttr_mean": "down",
    "sentence_complexity_score_mean": "down",
    "positive_emotion_ratio": "down",
}

# feature별 표준편차 최소값
# std가 너무 작으면 작은 변화에도 z-score가 과도하게 커지는 것을 방지한다.
STD_FLOOR = {
    "demonstrative_pronoun_count_mean": 0.20,
    "repeated_ratio_mean": 0.03,
    "negative_emotion_ratio": 0.05,
    "positive_emotion_ratio": 0.05,
    "ttr_mean": 0.03,
    "sentence_complexity_score_mean": 0.50,
}

# 모든 Feature를 동일한 비중으로 반영
FEATURE_WEIGHTS = {
    "demonstrative_pronoun_count_mean": 1.0,
    "repeated_ratio_mean": 1.0,
    "negative_emotion_ratio": 1.0,

    "ttr_mean": 1.0,
    "sentence_complexity_score_mean": 1.0,
    "positive_emotion_ratio": 1.0,
}

# main_reasons 중복 방지를 위한 그룹
FEATURE_GROUPS = {
    "demonstrative_pronoun_count_mean": "language",
    "repeated_ratio_mean": "language",
    "ttr_mean": "language",
    "sentence_complexity_score_mean": "language",

    "negative_emotion_ratio": "emotion_ratio",
    "positive_emotion_ratio": "emotion_ratio",
}

FEATURE_REASON: Dict[str, str] = {
    "demonstrative_pronoun_count_mean": "지시대명사 사용량이 개인 baseline 대비 증가했습니다.",
    "repeated_ratio_mean": "어휘 반복 비율이 개인 baseline 대비 증가했습니다.",
    "negative_emotion_ratio": "부정 감정 비율이 개인 baseline 대비 증가했습니다.",
    "ttr_mean": "어휘 다양성(TTR)이 개인 baseline 대비 감소했습니다.",
    "sentence_complexity_score_mean": "문장 복잡도 점수가 개인 baseline 대비 감소했습니다.",
    "positive_emotion_ratio": "긍정 감정 비율이 개인 baseline 대비 감소했습니다.",
}


# =============================================================================

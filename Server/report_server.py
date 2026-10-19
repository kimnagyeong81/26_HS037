# -*- coding: utf-8 -*-
"""
Merry AI 월간 인지·감정 분석 리포트

역할:
1. MongoDB 사용자 발화 기록 조회
2. 월간 언어·감정 분석 결과 집계
3. Rolling Z-score / EWMA 결과 조회
4. HTML 월간 리포트 생성
5. JSON API 제공

데이터 출처:
- dementia_db.events
- dementia_db.anomaly_results

"""

from __future__ import annotations

import math
import re

from calendar import monthrange
from collections import Counter
from datetime import date
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
)

from jinja2 import (
    Environment,
    BaseLoader,
    select_autoescape,
)

from pymongo import MongoClient

from daily_feature_builder import (
    build_daily_summaries,
    get_date_string,
    is_utterance_event,
)

from trend_settings import (
    MONGO_URI,
    DB_NAME,
    EVENT_COLLECTION,
    ANOMALY_COLLECTION,
)


# ============================================================
# 1. FastAPI 설정
# ============================================================

app = FastAPI(
    title="Merry AI 월간 인지·감정 리포트",
    version="1.0.0",
)


EMOTION_LABELS = {
    "joy": "기쁨",
    "calm": "평온",
    "anger": "분노",
    "fear": "두려움",
    "sadness": "슬픔",
}


FEATURE_LABELS = {
    "demonstrative_pronoun_count_mean": "지시대명사",
    "ttr_mean": "어휘다양성(TTR)",
    "repeated_ratio_mean": "반복 발화 비율",
    "sentence_complexity_score_mean": "문장 복잡도",
    "positive_emotion_ratio": "긍정 감정 비율",
    "negative_emotion_ratio": "부정 감정 비율",
}


# ============================================================
# 2. HTML 템플릿
# ============================================================

# 별도 templates/report.html 없이 동작하도록
# HTML 템플릿을 코드에 포함했습니다.

HTML = r"""
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport"
      content="width=device-width,initial-scale=1">

<title>Merry AI 월간 리포트 — {{ meta.user_id }}</title>

<style>
:root {
    color-scheme: light;
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: #f5f6fa;
    font-family: system-ui, -apple-system,
                 "Noto Sans KR", sans-serif;
    color: #1c2840;
}

main {
    max-width: 1080px;
    margin: 32px auto;
    padding: 0 22px;
}

header {
    background: #1e3155;
    color: white;
    padding: 30px;
    border-radius: 18px;
    margin-bottom: 20px;
}

header small {
    color: #d7e0ef;
}

h1 {
    margin: 8px 0;
    font-size: 29px;
}

h2 {
    font-size: 19px;
    margin: 0 0 18px;
}

p.note {
    color: #526177;
    font-size: 13px;
    line-height: 1.7;
}

.grid {
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(165px, 1fr));
    gap: 12px;
}

.card,
.section {
    background: white;
    border: 1px solid #e5e8f1;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 16px;
}

.metric {
    font-size: 27px;
    font-weight: 750;
    margin-top: 8px;
}

.muted {
    color: #697991;
    font-size: 13px;
}

table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
}

th,
td {
    padding: 11px 8px;
    text-align: left;
    border-bottom: 1px solid #ebedf3;
}

th {
    color: #53657d;
    background: #f8f9fc;
}

ul {
    padding-left: 22px;
    line-height: 1.75;
}

.bar {
    height: 11px;
    background: #edf0f5;
    border-radius: 9px;
    overflow: hidden;
}

.bar > span {
    display: block;
    height: 100%;
    background: #4976c8;
    border-radius: 9px;
}

blockquote {
    margin: 10px 0;
    padding: 12px 16px;
    background: #f6f8fc;
    border-left: 4px solid #8299c4;
    border-radius: 4px;
}

footer {
    padding: 28px 0;
    color: #6f7c90;
    font-size: 12px;
}

@media print {
    body {
        background: white;
    }

    main {
        margin: 0;
        max-width: none;
    }

    .section,
    .card,
    header {
        break-inside: avoid;
    }
}
</style>
</head>

<body>
<main>

<header>
    <small>
        Merry AI · 월간 인지·언어 지표 리포트
    </small>

    <h1>
        {{ meta.year }}년
        {{ meta.month }}월 분석 리포트
    </h1>

    <div>
        사용자 {{ meta.user_id }}
        · {{ meta.first_day }}
        ~ {{ meta.last_day }}
    </div>
</header>

{% if not activity.utterance_count %}

<section class="section">
    <h2>분석 데이터 없음</h2>
    <p>
        선택한 기간에 사용자 발화 기록이 없습니다.
    </p>
</section>

{% else %}

<div class="grid">

    <div class="card">
        <div class="muted">발화 기록</div>
        <div class="metric">
            {{ activity.utterance_count }}건
        </div>
    </div>

    <div class="card">
        <div class="muted">활동 일수</div>
        <div class="metric">
            {{ activity.active_days }}일
        </div>
    </div>

    <div class="card">
        <div class="muted">유효 감정 분석</div>
        <div class="metric">
            {{ activity.emotion_event_count }}건
        </div>
    </div>

    <div class="card">
        <div class="muted">추세 분석 결과</div>
        <div class="metric">
            {{ anomaly.count }}건
        </div>
    </div>

</div>

<section class="section">

    <h2>월간 언어·인지 지표</h2>

    <p class="note">
        각 지표는 일별 평균을 월간 평균한 값입니다.
        활동 일수와 발화 수에 따라 해석이 달라질 수 있습니다.
    </p>

    <table>
        <thead>
            <tr>
                <th>지표</th>
                <th>월간 평균</th>
            </tr>
        </thead>

        <tbody>
            {% for item in cognition.features %}
            <tr>
                <td>{{ item.label }}</td>
                <td>{{ item.display }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>

</section>

<section class="section">

    <h2>감정 분석</h2>

    <p class="note">
        Jetson에서 계산한 유효 감정 점수만 사용하며,
        누락값은 0점으로 처리하지 않습니다.
    </p>

    {% for item in emotion.scores %}

    <div style="
        display:grid;
        grid-template-columns:72px 85px 1fr;
        align-items:center;
        gap:10px;
        margin:12px 0;
    ">

        <span>{{ item.label }}</span>
        <strong>{{ item.display }}</strong>

        <div class="bar">
            <span style="width:{{ item.width }}%"></span>
        </div>

    </div>

    {% endfor %}

    <p class="note">
        점수 범위는 0~100입니다.
        감정 점수는 진단 결과가 아닙니다.
    </p>

</section>

<section class="section">

    <h2>주차별 활동 및 위험 점수</h2>

    <table>
        <thead>
            <tr>
                <th>주차</th>
                <th>발화 수</th>
                <th>분석 가능 일수</th>
                <th>최신 EWMA</th>
            </tr>
        </thead>

        <tbody>
            {% for w in weekly %}
            <tr>
                <td>{{ w.label }}</td>
                <td>{{ w.utterances }}</td>
                <td>{{ w.days }}</td>
                <td>{{ w.ewma_display }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>

</section>

<section class="section">

    <h2>Rolling Z-score / EWMA</h2>

    {% if anomaly.latest %}

    <div class="grid">

        <div class="card">
            <div class="muted">최근 경보 단계</div>
            <div class="metric" style="font-size:20px">
                {{ anomaly.latest.alert_level }}
            </div>
        </div>

        <div class="card">
            <div class="muted">최신 위험 점수</div>
            <div class="metric">
                {{ anomaly.latest.daily_risk_display }}
            </div>
        </div>

        <div class="card">
            <div class="muted">최신 EWMA</div>
            <div class="metric">
                {{ anomaly.latest.ewma_display }}
            </div>
        </div>

    </div>

    {% if anomaly.reasons %}
        <h3>주요 변화 징후</h3>

        <ul>
            {% for reason in anomaly.reasons %}
                <li>{{ reason }}</li>
            {% endfor %}
        </ul>
    {% endif %}

    {% else %}

    <p class="note">
        이 기간에는 기준선 부족 또는
        데이터 미수집으로 추세 분석 결과가 없습니다.
    </p>

    {% endif %}

</section>

<section class="section">

    <h2>최근 발화 예시</h2>

    <p class="note">
        서버에 실제 저장된 사용자 발화 중 최대 5건입니다.
    </p>

    {% for utterance in utterances %}

    <blockquote>
        {{ utterance.text }}
        <br>
        <small class="muted">
            {{ utterance.date }}
        </small>
    </blockquote>

    {% endfor %}

</section>

{% endif %}

<section class="section">

    <h2>구슬 배출·음성 CNN 분석</h2>

    <p class="note">
        이 서버의 dementia_db에서 해당 데이터의
        저장·연동이 확인되지 않아,
        임의의 수치를 생성하지 않았습니다.
        Jetson의 로컬 데이터 동기화와
        CNN 추론 결과 저장이 필요합니다.
    </p>

</section>

<footer>
    본 보고서는 수집된 발화의 통계적 변화에 대한
    참고 자료이며 치매 진단이나 의료적 판단을
    대체하지 않습니다.
</footer>

</main>
</body>
</html>
"""


TEMPLATE = Environment(
    loader=BaseLoader(),
    autoescape=select_autoescape(["html"]),
).from_string(HTML)


# ============================================================
# 3. 숫자 변환 유틸리티
# ============================================================

def _num(value: Any, digits: int = 3) -> float | None:
    """None, NaN, 무한대를 실제 0점과 구분."""

    try:
        if value is None or isinstance(value, bool):
            return None

        number = float(value)

        return (
            round(number, digits)
            if math.isfinite(number)
            else None
        )

    except (ValueError, TypeError, OverflowError):
        return None


def _display(value: Any, digits: int = 3) -> str:
    """화면 표시용 숫자 변환."""

    value = _num(value, digits)

    if value is None:
        return "미집계"

    return f"{value:.{digits}f}"


# ============================================================
# 4. 요청 검증
# ============================================================

def _validate(
    user_id: str,
    year: int,
    month: int,
) -> None:

    if not re.fullmatch(
        r"[A-Za-z0-9_.-]{1,64}",
        user_id,
    ):
        raise HTTPException(
            status_code=400,
            detail="user_id 형식 오류",
        )

    if not 2020 <= year <= 2099:
        raise HTTPException(
            status_code=400,
            detail="연도 범위 오류",
        )

    if not 1 <= month <= 12:
        raise HTTPException(
            status_code=400,
            detail="월 범위 오류",
        )


# ============================================================
# 5. 월간 리포트 데이터 집계
# ============================================================

def build_report_data(
    user_id: str,
    year: int,
    month: int,
) -> dict[str, Any]:
    """
    실제 MongoDB 데이터로 월간 리포트 생성.

    1. 발화 이벤트 조회
    2. 일별 특징값 집계
    3. 월간 감정·언어 지표 생성
    4. 이상 징후 결과 조회
    5. 주차별 결과 통합
    """

    _validate(user_id, year, month)

    first = date(year, month, 1)

    last = date(
        year,
        month,
        monthrange(year, month)[1],
    )

    # --------------------------------------------------------
    # 5-1. MongoDB 조회
    # --------------------------------------------------------

    with MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5000,
    ) as client:

        db = client[DB_NAME]

        cursor = db[EVENT_COLLECTION].find(
            {
                "user_id": user_id,
                "$or": [
                    {"granularity": "utterance"},
                    {
                        "event_type":
                        "dementia_utterance_analysis"
                    },
                ],
            },
            {"_id": 0},
        )

        month_events = []

        for event in cursor:

            if not is_utterance_event(event):
                continue

            day = (
                get_date_string(event.get("date"))
                or get_date_string(event.get("timestamp"))
            )

            if (
                day
                and first.isoformat()
                <= day
                <= last.isoformat()
            ):
                event["date"] = day
                month_events.append(event)

        # 기존 Rolling Z-score / EWMA 결과 조회
        anomaly_rows = list(
            db[ANOMALY_COLLECTION].find(
                {
                    "user_id": user_id,
                    "model_type": "rolling_zscore_ewma",
                    "date": {
                        "$gte": first.isoformat(),
                        "$lte": last.isoformat(),
                    },
                },
                {"_id": 0},
            ).sort("date", 1)
        )

    # --------------------------------------------------------
    # 5-2. 일별 특징값 생성
    # --------------------------------------------------------

    daily = build_daily_summaries(
        month_events
    )

    daily_rows = (
        daily.to_dict("records")
        if not daily.empty
        else []
    )

    valid_emotion = sum(
        int(d.get("emotion_event_count", 0))
        for d in daily_rows
    )

    # --------------------------------------------------------
    # 5-3. 언어·인지 분석 지표
    # --------------------------------------------------------

    features = []

    for key, label in FEATURE_LABELS.items():

        values = [
            _num(d.get(key))
            for d in daily_rows
        ]

        numbers = [
            value
            for value in values
            if value is not None
        ]

        average = (
            sum(numbers) / len(numbers)
            if numbers
            else None
        )

        features.append({
            "key": key,
            "label": label,
            "value": _num(average),
            "display": _display(average),
        })

    # --------------------------------------------------------
    # 5-4. Jetson 감정 점수 월간 평균
    # --------------------------------------------------------

    emotion_items = []

    for key, label in EMOTION_LABELS.items():

        total = sum(
            _num(
                d.get(f"emotion_{key}_sum")
            ) or 0
            for d in daily_rows
        )

        mean = (
            total / valid_emotion
            if valid_emotion
            else None
        )

        emotion_items.append({
            "label": label,
            "value": _num(mean),
            "display": _display(mean, 1),
            "width": max(
                0.0,
                min(100.0, mean or 0),
            ),
        })

    # --------------------------------------------------------
    # 5-5. Rolling Z-score / EWMA 결과 정규화
    # --------------------------------------------------------

    normalized_anomalies = []

    for row in anomaly_rows:

        normalized_anomalies.append({
            "date": str(row.get("date", "")),

            "alert_level": str(
                row.get("alert_level", "-")
            ),

            "daily_risk_score": _num(
                row.get("daily_risk_score")
            ),

            "daily_risk_display": _display(
                row.get("daily_risk_score")
            ),

            "ewma_risk_score": _num(
                row.get("ewma_risk_score")
            ),

            "ewma_display": _display(
                row.get("ewma_risk_score")
            ),

            "trend_direction": str(
                row.get("trend_direction", "-")
            ),

            "main_reasons": [
                str(x)
                for x in (
                    row.get("main_reasons") or []
                )
            ],
        })

    # --------------------------------------------------------
    # 5-6. 주차별 결과
    # --------------------------------------------------------

    weekly = []

    for week_number in range(1, 5):

        week_days = [
            d
            for d in daily_rows
            if min(
                (int(d["date"][-2:]) - 1) // 7 + 1,
                4,
            ) == week_number
        ]

        week_results = [
            a
            for a in normalized_anomalies
            if min(
                (int(a["date"][-2:]) - 1) // 7 + 1,
                4,
            ) == week_number
        ]

        weekly.append({
            "label": f"{week_number}주차",

            "days": len(week_days),

            "utterances": sum(
                int(d["event_count"])
                for d in week_days
            ),

            "ewma_display": (
                week_results[-1]["ewma_display"]
                if week_results
                else "미집계"
            ),
        })

    # --------------------------------------------------------
    # 5-7. 주요 이상 징후 원인
    # --------------------------------------------------------

    reasons = Counter(
        reason
        for anomaly in normalized_anomalies
        for reason in anomaly["main_reasons"]
    )

    # --------------------------------------------------------
    # 5-8. 최근 사용자 발화
    # --------------------------------------------------------

    utterances = sorted(
        month_events,
        key=lambda e: str(
            e.get("timestamp", "")
        ),
        reverse=True,
    )[:5]

    # --------------------------------------------------------
    # 5-9. 최종 리포트 반환
    # --------------------------------------------------------

    return {
        "meta": {
            "user_id": user_id,
            "year": year,
            "month": month,
            "first_day": first.isoformat(),
            "last_day": last.isoformat(),
            "is_demo": False,
        },

        "activity": {
            "utterance_count": len(month_events),
            "active_days": len(daily_rows),
            "emotion_event_count": valid_emotion,
            "missing_emotion_count": (
                len(month_events) - valid_emotion
            ),
        },

        "cognition": {
            "features": features,
        },

        "emotion": {
            "scores": emotion_items,
        },

        "weekly": weekly,

        "anomaly": {
            "count": len(normalized_anomalies),

            "latest": (
                normalized_anomalies[-1]
                if normalized_anomalies
                else None
            ),

            "reasons": [
                name
                for name, _ in reasons.most_common(5)
            ],

            "daily_scores": normalized_anomalies,
        },

        "utterances": [
            {
                "date": e["date"],
                "text": str(
                    e.get("text", "")
                ),
            }
            for e in utterances
        ],

        "marbles": {
            "status": "not_synced",
        },

        "cnn": {
            "status": "not_integrated",
        },
    }


# ============================================================
# 6. 서버 상태 확인
# ============================================================

@app.get("/health")
def health():
    """FastAPI 프로세스 생존 여부 확인."""

    return {
        "ok": True,
        "service": "merry-report",
    }


@app.get("/ready")
def ready():
    """실제 MongoDB 연결 여부 확인."""

    try:
        with MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=3000,
        ) as client:

            client.admin.command("ping")

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="MongoDB unavailable",
        )

    return {
        "ok": True,
        "mongo": "connected",
    }


# ============================================================
# 7. 기본 경로
# ============================================================

@app.get("/")
def root():

    today = date.today()

    return RedirectResponse(
        url=(
            f"/report/U001/"
            f"{today.year}/{today.month}"
        )
    )


# ============================================================
# 8. HTML 월간 리포트
# ============================================================

@app.get(
    "/report/{user_id}/{year}/{month}",
    response_class=HTMLResponse,
)
def report_html(
    user_id: str,
    year: int,
    month: int,
):

    try:
        data = build_report_data(
            user_id,
            year,
            month,
        )

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="리포트 데이터 조회 실패",
        )

    return HTMLResponse(
        TEMPLATE.render(**data)
    )


# ============================================================
# 9. JSON 월간 리포트
# ============================================================

@app.get(
    "/report/{user_id}/{year}/{month}/json"
)
def report_json(
    user_id: str,
    year: int,
    month: int,
):

    try:
        data = build_report_data(
            user_id,
            year,
            month,
        )

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="리포트 데이터 조회 실패",
        )

    return JSONResponse(
        jsonable_encoder(data)
    )
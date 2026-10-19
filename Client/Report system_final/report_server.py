"""
report_server.py
───────
FastAPI 서버. 아래 URL로 접속하면 완성된 HTML 리포트를 반환합니다.

    GET /report/{user_id}/{year}/{month}
    예:  GET /report/U001/2026/4

실행:
    uvicorn main:app --reload --port 8000
"""

import json
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

# report_query.py 가 같은 디렉터리에 있어야 합니다
from report_query import build_report_data

app = FastAPI(title="치매 조기 감지 월간 리포트")

# 템플릿 디렉터리 설정
# main.py 와 같은 위치의 templates/ 폴더를 사용합니다.
# report.html 은 반드시 templates/ 폴더 안에 있어야 합니다.
BASE_DIR      = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"

if not TEMPLATES_DIR.exists():
    raise RuntimeError(
        f"templates/ 폴더가 없습니다: {TEMPLATES_DIR}\n"
        "mkdir templates 후 report.html 을 templates/ 안에 넣어주세요."
    )

if not (TEMPLATES_DIR / "report.html").exists():
    raise RuntimeError(
        f"templates/report.html 파일이 없습니다.\n"
        "다운로드받은 report.html 을 templates/ 폴더 안에 넣어주세요."
    )

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ── 커스텀 Jinja2 필터 ──────────────────────────────────────────
def _pct_str(value: float | None, decimals: int = 1) -> str:
    """0.34 → '+34.0%' / -0.29 → '−29.0%'"""
    if value is None:
        return "-"
    pct = value * 100
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.{decimals}f}%"

def _grade_ko(grade: str) -> str:
    return {"caution": "⚠ 주의", "watch": "🟡 관찰", "ok": "✓ 정상"}.get(grade, "-")

def _grade_css(grade: str) -> str:
    return {"caution": "badge-caution", "watch": "badge-watch", "ok": "badge-ok"}.get(grade, "")

def _emotion_ko(emotion_en: str) -> str:
    return {"joy":"기쁨","calm":"평온","anger":"분노","fear":"두려움","sadness":"슬픔"}.get(emotion_en, emotion_en)

templates.env.filters["pct_str"]   = _pct_str
templates.env.filters["grade_ko"]  = _grade_ko
templates.env.filters["grade_css"] = _grade_css
templates.env.filters["emotion_ko"] = _emotion_ko


# ── 라우트 ──────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/report/user_001/2026/5")

@app.get("/report/{user_id}/{year}/{month}", response_class=HTMLResponse)
async def get_report(request: Request, user_id: str, year: int, month: int):
    """
    MongoDB에서 데이터를 집계한 뒤
    templates/report.html 에 주입해서 완성된 HTML을 반환합니다.
    """
    if not (2020 <= year <= 2099):
        raise HTTPException(status_code=400, detail="year 범위 오류")
    if not (1 <= month <= 12):
        raise HTTPException(status_code=400, detail="month 범위 오류")

    try:
        data = build_report_data(user_id=user_id, year=year, month=month)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터 집계 오류: {e}")

    context = {
        "request": request,
        "data":    data,
        "meta":       data["meta"],
        "activity":   data["activity"],
        "cognition":  data["cognition"],
        "marbles":    data["marbles"],
        "emotion":    data["emotion_summary"],
        "weekly":     data["weekly_trend"],
        "calendar":   data["volatility_calendar"],
        "xai":        data["xai_utterances"],
        "anomaly":    data["anomaly"],
        "weekly_json":       json.dumps(data["weekly_trend"]["weeks"],                  ensure_ascii=False),
        "calendar_json":     json.dumps(data["volatility_calendar"]["calendar"],        ensure_ascii=False),
        "xai_json":          json.dumps(data["xai_utterances"]["top_cases"],            ensure_ascii=False),
        "anomaly_json":      json.dumps(data["anomaly"]["daily_scores"],               ensure_ascii=False),
        "weekly_risk_json":  json.dumps(data["anomaly"]["weekly_risk"],                ensure_ascii=False),
        "feature_avg_json":  json.dumps(data["anomaly"]["monthly_feature_avg"],        ensure_ascii=False),  # ← 신규
    }

    # Starlette 0.28+ 은 (request, name, context) 순서
    # 구버전은 (name, context) 순서 — 양쪽 모두 호환
    try:
        return templates.TemplateResponse(request=request, name="report.html", context=context)
    except TypeError:
        return templates.TemplateResponse("report.html", context)

@app.get("/report/{user_id}/{year}/{month}/json", response_class=JSONResponse)
async def get_report_json(user_id: str, year: int, month: int):
    """
    HTML 대신 원본 JSON을 반환합니다.
    디버깅·프론트엔드 개발용.
    """
    try:
        data = build_report_data(user_id=user_id, year=year, month=month)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse(content=data, media_type="application/json; charset=utf-8")
 
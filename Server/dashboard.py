# -*- coding: utf-8 -*-

"""

Merry AI 인지·언어 분석 대시보드



기능:

1. 사용자별 인지·언어 분석 결과 조회

2. Jetson 감정 분석 결과 시각화

3. Rolling Z-score / EWMA 결과 표시

4. 일별 특징값 및 추세 그래프

5. 발화 이벤트 검색 및 상세 조회

6. 월간 HTML 리포트 생성 메뉴



월간 리포트는 별도의 report_server.py에서 제공합니다.

"""



from __future__ import annotations



import math

import os



from datetime import date, datetime

from typing import Any, List, Optional
from urllib.parse import quote, urlsplit

from zoneinfo import ZoneInfo



import pandas as pd

import streamlit as st



from pymongo import MongoClient, DESCENDING



from daily_feature_builder import (

    build_daily_summaries,

    get_date_string,

    is_utterance_event,

    parse_timestamp,

)



from trend_mongodb import (

    load_events_from_mongodb,

)



from trend_settings import (

    MONGO_URI,

    DB_NAME,

    ANOMALY_COLLECTION,

    DAILY_FEATURE_COLUMNS,

)





# ============================================================

# 1. 환경 설정

# ============================================================



KST = ZoneInfo("Asia/Seoul")



# 브라우저에서 접근 가능한 report-server 주소

# Kubernetes Pod의 내부 localhost가 아닌

# 실제 서버 IP 또는 외부 접근 가능한 도메인을 사용합니다.



# ============================================================
# 월간 리포트 외부 접근 설정
# ============================================================

# Kubernetes Service가 자동 생성하는 MERRY_REPORT_PORT와
# 이름이 충돌하지 않도록 별도 환경변수 사용
REPORT_PORT = int(
    os.getenv(
        "MERRY_REPORT_HTTP_PORT",
        "8502",
    )
)

# 기본적으로 report-server는 HTTP 사용
REPORT_SCHEME = os.getenv(
    "MERRY_REPORT_SCHEME",
    "http",
).strip().lower()

if REPORT_SCHEME not in {
    "http",
    "https",
}:
    REPORT_SCHEME = "http"


# 필요할 때만 공개 URL을 직접 지정
#
# 예:
# MERRY_REPORT_PUBLIC_URL=http://공인IP:8502
#
# 빈 문자열이면 현재 대시보드에 접속한 Host를 기준으로
# 자동으로 8502 주소를 생성한다.
REPORT_PUBLIC_URL_OVERRIDE = os.getenv(
    "MERRY_REPORT_PUBLIC_URL",
    "",
).strip().rstrip("/")

# ============================================================

# 2. Streamlit 기본 설정

# ============================================================

st.set_page_config(

    page_title="Merry AI Dashboard",

    page_icon="📊",

    layout="wide",

)

st.title("Merry AI 인지·언어 분석 대시보드")

# ============================================================

# 3. 공통 유틸리티

# ============================================================



def normalize_complexity_score(

    value: Any,

) -> Optional[float]:

    """

    sentence_complexity_score를 대시보드 표시용

    0~1 값으로 정규화한다.



    원본 기준:

      < 10  : 단순

      < 20  : 보통

      >= 20 : 복잡



    따라서 20점을 1.0으로 사용하고

    그 이상은 1.0으로 제한한다.

    """



    try:

        if value is None or isinstance(value, bool):

            return None



        score = float(value)



        if not math.isfinite(score):

            return None



        return round(

            min(

                max(score / 20.0, 0.0),

                1.0,

            ),

            3,

        )



    except (

        TypeError,

        ValueError,

        OverflowError,

    ):

        return None

def format_float(

    value: Any,

    ndigits: int = 3,

):

    """

    누락 지표를 0이나 NaN으로 표시하지 않는다.

    """



    try:

        if value is None or isinstance(value, bool):

            return "-"



        number = float(value)



        if math.isfinite(number):

            return round(number, ndigits)



        return "-"



    except (

        TypeError,

        ValueError,

        OverflowError,

    ):

        return "-"


def get_report_public_url() -> str:
    """
    브라우저에서 접근 가능한 월간 리포트 URL을 결정한다.

    우선순위:
    1. MERRY_REPORT_PUBLIC_URL 직접 지정
    2. 현재 Streamlit에 접속한 Host 사용
    3. 내부 서버 IP fallback
    """

    # --------------------------------------------------------
    # 1. 운영자가 공개 URL을 직접 지정한 경우
    # --------------------------------------------------------

    if REPORT_PUBLIC_URL_OVERRIDE:
        return REPORT_PUBLIC_URL_OVERRIDE

    # --------------------------------------------------------
    # 2. 현재 브라우저가 접속한 Dashboard Host 사용
    # --------------------------------------------------------

    try:

        headers = st.context.headers

        forwarded_host = (
            headers.get("x-forwarded-host")
            or ""
        )

        host_header = (
            forwarded_host
            or headers.get("host")
            or ""
        )

        # 프록시가 여러 Host를 넣는 경우 첫 번째 값 사용
        host_header = (
            host_header
            .split(",")[0]
            .strip()
        )

        if host_header:

            parsed = urlsplit(
                f"//{host_header}"
            )

            hostname = parsed.hostname

            if hostname:

                # IPv6 대응
                if ":" in hostname:

                    return (
                        f"{REPORT_SCHEME}://"
                        f"[{hostname}]:{REPORT_PORT}"
                    )

                return (
                    f"{REPORT_SCHEME}://"
                    f"{hostname}:{REPORT_PORT}"
                )

    except Exception:

        # st.context 사용 불가능 시 fallback
        pass

    # --------------------------------------------------------
    # 3. 내부망 fallback
    # --------------------------------------------------------

    return (
        f"{REPORT_SCHEME}://"
        f"192.168.219.104:{REPORT_PORT}"
    )


def load_events() -> List[dict]:

    """

    추세 분석과 동일한 DB 조회를 재사용한다.



    사용자 발화만 대시보드에 표시한다.

    """



    return [

        event

        for event in load_events_from_mongodb()

        if is_utterance_event(event)

    ]





def load_latest_anomaly_result(

    user_id: str,

    start_date: date,

    end_date: date,

) -> Optional[dict]:

    """

    선택한 사용자·기간의 최신 추세 분석 결과.

    """



    client = MongoClient(

        MONGO_URI,

        serverSelectionTimeoutMS=5000,

    )



    try:

        collection = client[

            DB_NAME

        ][

            ANOMALY_COLLECTION

        ]



        return collection.find_one(

            {

                "user_id": user_id,



                "model_type": "rolling_zscore_ewma",



                "date": {

                    "$gte": start_date.isoformat(),

                    "$lte": end_date.isoformat(),

                },

            },



            sort=[

                ("date", DESCENDING),

                ("created_at", DESCENDING),

            ],

        )



    finally:

        client.close()





def objectid_to_str(

    df: pd.DataFrame,

) -> pd.DataFrame:



    if "_id" in df.columns:

        df["_id"] = df["_id"].astype(str)



    return df





def filter_by_date(

    df: pd.DataFrame,

    start_date: date,

    end_date: date,

) -> pd.DataFrame:

    """

    한국시간 날짜 기준으로 조회 기간 필터링.

    """



    if "date" not in df.columns:

        return df.iloc[0:0].copy()



    dates = (

        df["date"]

        .fillna("")

        .astype(str)

    )



    return df.loc[

        dates.between(

            start_date.isoformat(),

            end_date.isoformat(),

        )

    ].copy()





# ============================================================

# 4. MongoDB 데이터 로드

# ============================================================



try:

    events = load_events()



except Exception as exc:

    st.error(

        f"MongoDB 발화 데이터 조회 실패: {exc}"

    )

    st.stop()





if not events:

    st.warning(

        "아직 dementia_db.events에 "

        "저장된 이벤트가 없습니다."

    )

    st.stop()





events_df = pd.DataFrame(events)



events_df = objectid_to_str(

    events_df

)





# ------------------------------------------------------------

# Timestamp 처리

# ------------------------------------------------------------



if "timestamp" in events_df.columns:



    events_df["timestamp_dt"] = (

        events_df["timestamp"].apply(

            parse_timestamp

        )

    )



else:

    events_df["timestamp_dt"] = None





events_df["date"] = events_df.apply(

    lambda row: (

        get_date_string(row.get("date"))

        or get_date_string(row.get("timestamp"))

    ),

    axis=1,

)





# ============================================================

# 5. Sidebar

# ============================================================



st.sidebar.header("Merry AI")





# ------------------------------------------------------------

# 5-1. 사용자 선택

# ------------------------------------------------------------



user_options = []



if "user_id" in events_df.columns:



    user_options = sorted(

        events_df[

            "user_id"

        ].dropna().unique().tolist()

    )





if not user_options:

    st.warning(

        "user_id가 있는 이벤트가 없습니다."

    )

    st.stop()





selected_user = st.sidebar.selectbox(

    "사용자 선택",

    user_options,

    index=0,

)





# ============================================================

# 6. 화면 선택

# ============================================================



page = st.sidebar.radio(

    "화면",

    [

        "분석 대시보드",

        "월간 리포트",

    ],

    index=0,

)





# ============================================================

# 7. 월간 리포트 화면

# ============================================================



if page == "월간 리포트":



    st.subheader(

        "월간 인지·감정 분석 리포트"

    )

# --------------------------------------------------------
# 7-1. 실제 데이터가 존재하는 연월 선택
# --------------------------------------------------------

    report_user_df = events_df[
        events_df["user_id"] == selected_user
    ].copy()


    if report_user_df.empty:

        st.warning(
            f"{selected_user} 사용자의 발화 데이터가 없습니다."
        )

        st.stop()


    # date 컬럼을 날짜 형식으로 변환
    report_dates = pd.to_datetime(
        report_user_df["date"],
        errors="coerce",
    )


    # 날짜가 정상인 데이터만 사용
    valid_report_dates = (
        report_dates
        .dropna()
    )


    if valid_report_dates.empty:

        st.warning(
            f"{selected_user} 사용자의 유효한 날짜 데이터가 없습니다."
        )

        st.stop()


    # 실제 데이터가 존재하는 (연도, 월) 추출
    available_periods = sorted(
        {
            (
                int(dt.year),
                int(dt.month),
            )
            for dt in valid_report_dates
        },
        reverse=True,
    )


    if not available_periods:

        st.warning(
            "월간 리포트를 생성할 수 있는 기간이 없습니다."
        )

        st.stop()


    # 가장 최근 월이 기본 선택
    selected_period = st.selectbox(
        "리포트 기간",
        available_periods,
        index=0,
        format_func=lambda item: (
            f"{item[0]}년 {item[1]}월"
        ),
    )


    report_year, report_month = selected_period
    # --------------------------------------------------------
    # 7-2. 리포트 URL 생성
    # --------------------------------------------------------

    safe_id = quote(
        str(selected_user),
        safe="",
    )

    report_base_url = (
        get_report_public_url()
    )

    report_url = (
        f"{report_base_url}/report/"
        f"{safe_id}/"
        f"{int(report_year)}/"
        f"{int(report_month)}"
    )

    pdf_url = (
        f"{report_base_url}/report/"
        f"{safe_id}/"
        f"{int(report_year)}/"
        f"{int(report_month)}/pdf"
    )


    # --------------------------------------------------------
    # 7-3. HTML 리포트 / PDF 다운로드
    # --------------------------------------------------------

    open_col, pdf_col = st.columns(2)

    with open_col:

        st.link_button(
            "월간 리포트 열기 ↗",
            report_url,
            type="primary",
            use_container_width=True,
        )

    with pdf_col:

        st.link_button(
            "PDF 다운로드 ↓",
            pdf_url,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # 7-4. 대시보드 내부 미리보기
    # --------------------------------------------------------

    if st.toggle(
        "대시보드 안에서 리포트 미리보기",
        value=False,
    ):

        st.components.v1.iframe(
            report_url,
            height=980,
            scrolling=True,
        )

    # 월간 리포트 화면에서는
    # 아래 기존 분석 대시보드를 실행하지 않는다.
    st.stop()


# ============================================================

# 8. 기존 분석 대시보드

# ============================================================
st.sidebar.header("분석 필터")

user_df = events_df[
    events_df["user_id"] == selected_user
].copy()

# ------------------------------------------------------------
# 8-1. 날짜 범위
# ------------------------------------------------------------
valid_dates = (
    user_df["date"]
    .dropna()
    .sort_values()
)
if not valid_dates.empty:
    min_date = date.fromisoformat(
        valid_dates.iloc[0]
    )
    max_date = date.fromisoformat(
        valid_dates.iloc[-1]
    )
else:
    min_date = datetime.now(KST).date()
    max_date = min_date

date_range = st.sidebar.date_input(
    "조회 날짜 범위",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
if (
    isinstance(date_range, tuple)
    and len(date_range) == 2
):
    start_date, end_date = date_range
else:

    start_date = min_date

    end_date = max_date





# ------------------------------------------------------------

# 8-2. 발화 검색

# ------------------------------------------------------------



keyword = st.sidebar.text_input(

    "텍스트 검색"

)





# ============================================================

# 9. 분석 데이터 필터링

# ============================================================



# 일별 통계는 전체 사용자 발화 기준.

# 텍스트 검색은 일별 평균에 영향을 주지 않는다.



period_df = filter_by_date(

    user_df,

    start_date,

    end_date,

)





daily_df = build_daily_summaries(

    period_df.to_dict("records")

)





if not daily_df.empty:



    daily_df = (

        daily_df

        .sort_values(

            "date",

            ascending=False,

        )

        .reset_index(drop=True)

    )





# ------------------------------------------------------------

# 9-1. 텍스트 검색

# ------------------------------------------------------------



filtered_df = period_df.copy()





if keyword and "text" in filtered_df.columns:



    filtered_df = filtered_df[

        filtered_df["text"]

        .astype(str)

        .str.contains(

            keyword,

            case=False,

            na=False,

            regex=False,

        )

    ]





if filtered_df.empty:



    st.warning(

        "조건에 맞는 발화가 없습니다. "

        "검색어 또는 조회 기간을 조정해 주세요."

    )



    st.stop()





# ============================================================

# 10. 패턴 변화 감지 결과

# ============================================================



st.subheader(

    "패턴 변화 감지 결과"

)





latest_anomaly = load_latest_anomaly_result(

    selected_user,

    start_date,

    end_date,

)





risk_col1, risk_col2, risk_col3, risk_col4 = (

    st.columns(4)

)





if latest_anomaly:



    alert_level = latest_anomaly.get(

        "alert_level", "-"

    )



    trend_direction = latest_anomaly.get(

        "trend_direction", "-"

    )



    daily_risk_score = latest_anomaly.get(

        "daily_risk_score", "-"

    )



    ewma_risk_score = latest_anomaly.get(

        "ewma_risk_score", "-"

    )



    risk_col1.metric(

        "알림 단계",

        alert_level,

    )



    risk_col2.metric(

        "추세 방향",

        trend_direction,

    )



    risk_col3.metric(

        "당일 위험 점수",

        format_float(daily_risk_score),

    )



    risk_col4.metric(

        "EWMA 위험 점수",

        format_float(ewma_risk_score),

    )



    reasons = (

        latest_anomaly.get("main_reasons")

        or []

    )



    if reasons:



        st.info(

            " / ".join(

                str(x)

                for x in reasons[:3]

            )

        )



    with st.expander(

        "최신 패턴 변화 감지 결과 상세"

    ):



        display_result = dict(

            latest_anomaly

        )



        if "_id" in display_result:



            display_result["_id"] = str(

                display_result["_id"]

            )



        st.json(

            display_result

        )



else:



    risk_col1.metric(

        "알림 단계", "-"

    )



    risk_col2.metric(

        "추세 방향",

        "분석 결과 없음",

    )



    risk_col3.metric(

        "당일 위험 점수", "-"

    )



    risk_col4.metric(

        "EWMA 위험 점수", "-"

    )



    st.caption(

        "inference_trend.py 실행 후 "

        "anomaly_results에 결과가 저장되면 "

        "표시됩니다."

    )





st.markdown("---")





# ============================================================

# ============================================================
# 11. 핵심 분석 지표
# ============================================================

st.subheader("핵심 분석 지표")


# 가장 최근 발화 이벤트
latest_row = (
    filtered_df
    .sort_values(
        "timestamp_dt",
        ascending=False,
    )
    .iloc[0]
)


# ============================================================
# 11-1. STT 발화 내용
# ============================================================

st.markdown("### STT 발화 내용")

st.info(
    latest_row.get(
        "text",
        "발화 내용 없음",
    )
)


# ============================================================
# 11-2. CNN 음성 분석
# ============================================================

st.markdown("### CNN 음성 분석")

cnn_result = latest_row.get("cnn_result")

cnn_label = "-"
normal_prob = None
mci_prob = None

if isinstance(cnn_result, dict):
    cnn_label = (
        cnn_result.get("pred_label")
        or "-"
    )
    normal_prob = cnn_result.get("normal_prob")
    mci_prob = cnn_result.get("mci_prob")


cnn1, cnn2, cnn3 = st.columns(3)

cnn1.metric(
    "CNN 판단 (Normal/MCI)",
    cnn_label,
)

cnn2.metric(
    "Normal 가능성",
    (
        f"{float(normal_prob) * 100:.2f}%"
        if normal_prob is not None
        else "-"
    ),
)

cnn3.metric(
    "MCI 가능성",
    (
        f"{float(mci_prob) * 100:.2f}%"
        if mci_prob is not None
        else "-"
    ),
)


# ============================================================
# 11-3. 언어·인지 분석 / Rolling Z-score
# ============================================================

st.markdown(
    "### 언어·인지 분석 / Rolling Z-score"
)

pronoun_count = latest_row.get(
    "demonstrative_pronoun_count"
)

ttr_value = latest_row.get("ttr")

repeated_ratio = latest_row.get(
    "repeated_ratio"
)

complexity_score = latest_row.get(
    "sentence_complexity_score"
)

# sentence_complexity_score는 원래 0~1 범위가 아니므로,
# 기존 분석 기준상 '복잡' 구간이 시작되는 20점을 1.0으로 보고
# 대시보드 표시용으로만 0~1 범위로 정규화한다.
complexity_normalized = (
    normalize_complexity_score(
        complexity_score
    )
)

ewma_risk_score = None

if latest_anomaly:
    ewma_risk_score = latest_anomaly.get(
        "ewma_risk_score"
    )


l1, l2, l3, l4, l5 = st.columns(5)


try:
    pronoun_display = (
        f"{int(pronoun_count)}회"
        if pronoun_count is not None
        and not pd.isna(pronoun_count)
        else "-"
    )
except (
    TypeError,
    ValueError,
    OverflowError,
):
    pronoun_display = "-"


l1.metric(
    "지시대명사 (회)",
    pronoun_display,
)

l2.metric(
    "어휘 다양성 (0~1)",
    format_float(
        ttr_value,
        3,
    ),
)

l3.metric(
    "반복 발화 (0~1)",
    format_float(
        repeated_ratio,
        3,
    ),
)

l4.metric(
    "문장 복잡도 (0~1)",
    (
        f"{complexity_normalized:.3f}"
        if complexity_normalized is not None
        else "-"
    ),
)

l5.metric(
    "EWMA 위험도",
    format_float(
        ewma_risk_score,
        3,
    ),
)


# ============================================================
# 11-4. 감정 분석
# ============================================================

st.markdown("### 감정 분석")
st.caption("각 감정 점수는 0~100 범위입니다.")

e1, e2, e3, e4, e5 = st.columns(5)

e1.metric(
    "기쁨",
    format_float(
        latest_row.get("emotion_joy"),
        1,
    ),
)

e2.metric(
    "평온",
    format_float(
        latest_row.get("emotion_calm"),
        1,
    ),
)

e3.metric(
    "두려움",
    format_float(
        latest_row.get("emotion_fear"),
        1,
    ),
)

e4.metric(
    "화남",
    format_float(
        latest_row.get("emotion_anger"),
        1,
    ),
)

e5.metric(
    "슬픔",
    format_float(
        latest_row.get("emotion_sadness"),
        1,
    ),
)


# 12. 일별 추세 차트

# ============================================================



st.subheader(

    "일별 추세"

)





if not daily_df.empty:



    chart_df = (

        daily_df

        .sort_values("date")

        .copy()

    )



    chart_df = chart_df.set_index(

        "date"

    )



    chart_options = {



        "감정 총합": [

            "emotion_joy_sum",

            "emotion_calm_sum",

            "emotion_anger_sum",

            "emotion_fear_sum",

            "emotion_sadness_sum",

        ],



        "감정 지표": [

            "emotion_gap_mean",

            "positive_emotion_ratio",

            "negative_emotion_ratio",

            "dominant_emotion_ratio",

            "emotion_mixed_score",

        ],



        "언어/인지 평균": [

            "demonstrative_pronoun_count_mean",

            "ttr_mean",

            "repeated_ratio_mean",

            "sentence_complexity_score_mean",

        ],

    }



    selected_chart = st.selectbox(

        "차트 선택",

        list(chart_options.keys()),

    )



    chart_columns = [

        column

        for column in chart_options[

            selected_chart

        ]

        if column in chart_df.columns

    ]



    if chart_columns:



        if selected_chart == "감정 총합":



            st.caption(

                "감정 총합은 발화량의 영향을 받습니다. "

                "감정 상태 비교에는 감정 비율 지표를 "

                "함께 확인하세요."

            )



        st.line_chart(

            chart_df[chart_columns]

        )



    else:



        st.info(

            "표시할 차트 컬럼이 없습니다."

        )





# ============================================================

# 13. 발화 단위 이벤트 테이블

# ============================================================



st.subheader(

    "발화 단위 이벤트 테이블"

)





preferred_event_columns = [



    "timestamp",

    "date",

    "user_id",

    "session_id",

    "source_file",

    "text",



    "emotion_source",

    "emotion_scores_available",



    "emotion_joy",

    "emotion_calm",

    "emotion_anger",

    "emotion_fear",

    "emotion_sadness",



    "demonstrative_pronoun_count",

    "ttr",

    "repeated_ratio",

    "repetition_ratio",



    "sentence_complexity_score",

    "sentence_complexity_label",



    "risk_label",

    "_id",

]





visible_event_columns = [

    column

    for column in preferred_event_columns

    if column in filtered_df.columns

]





event_table = filtered_df[

    visible_event_columns

].copy()





st.dataframe(

    event_table.sort_values(

        "timestamp",

        ascending=False,

    ),

    use_container_width=True,

)





# ============================================================

# 14. 이벤트 상세 보기

# ============================================================



st.subheader(

    "이벤트 상세 보기"

)





row_options = (

    filtered_df

    .sort_values(

        "timestamp",

        ascending=False,

    )

    .index

    .tolist()

)





selected_index = st.selectbox(

    "확인할 이벤트 행 선택",

    row_options,

)





selected_row = filtered_df.loc[

    selected_index

].to_dict()





detail_data = {}





for key, value in selected_row.items():



    if key == "timestamp_dt":

        continue



    if key == "_id":



        detail_data[key] = str(value)



    elif isinstance(value, datetime):



        detail_data[key] = value.isoformat()



    else:



        detail_data[key] = value





st.json(

    detail_data

)
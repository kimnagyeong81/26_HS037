# Merry AI Server
# Python 3.11 + KoNLPy JVM + CNN PyTorch Runtime

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Asia/Seoul \
    CNN_MODEL_DIR=/Server/models

WORKDIR /app


# =============================================================================
# OS 패키지
# =============================================================================
# - openjdk-17-jre-headless : KoNLPy Okt
# - ca-certificates         : HTTPS 인증서
# - libsndfile1             : torchaudio WAV/FLAC 처리
# =============================================================================

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        ca-certificates \
        libsndfile1 \
        chromium \
        fonts-noto-cjk && \
    rm -rf /var/lib/apt/lists/*


# =============================================================================
# Python 기본 라이브러리
# =============================================================================

COPY requirements.txt /app/requirements.txt

RUN python -m pip install \
    --no-cache-dir \
    --upgrade pip


# =============================================================================
# CNN Runtime
# =============================================================================
# 현재 merry-consumer에서는 CNN을 CPU로 실행
# LLM이 RTX 3060 Ti GPU를 사용하므로 CNN은 우선 CPU로 분리
# =============================================================================

RUN python -m pip install \
    --no-cache-dir \
    torch==2.11.0 \
    torchvision==0.26.0 \
    torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cpu


# =============================================================================
# Merry AI Python dependencies
# =============================================================================

RUN python -m pip install \
    --no-cache-dir \
    -r /app/requirements.txt


# =============================================================================
# Merry AI 서버 코드
# =============================================================================
# 모델(.pth)은 Docker image에 포함하지 않음.
#
# Host:
#   /Server/models
#
# Container:
#   /Server/models
#
# K8s 또는 Docker volume mount로 연결 예정.
# =============================================================================

COPY \
    api_kafka_receiver.py \
    analysis_consumer.py \
    kafka_k8s_runtime.py \
    dementia_module.py \
    cnn_module.py \
    event_builder.py \
    pronoun_analyzer.py \
    lexical_diversity_analyzer.py \
    repetition_analyzer.py \
    sentence_complexity_analyzer.py \
    inference_trend.py \
    trend_monitor.py \
    daily_feature_builder.py \
    emotion_features.py \
    risk_feature_scoring.py \
    trend_calculator.py \
    risk_alerts.py \
    trend_settings.py \
    trend_mongodb.py \
    trend_cli.py \
    dashboard.py \
    monthly_report.py \
    /app/




# =============================================================================
# 월간 리포트 Template
# =============================================================================

COPY templates /app/templates

# =============================================================================
# CNN 모델 마운트 지점
# =============================================================================

RUN mkdir -p /Server/models


# =============================================================================
# 빌드 시 Python 문법 검사
# =============================================================================

RUN python -m compileall -q /app


# =============================================================================
# 비관리자 계정
# =============================================================================

RUN mkdir -p /data/utterances && \
    useradd -m -u 10001 merry && \
    chown -R merry:merry /app && \
    chown -R merry:merry /Server && \
    chown -R merry:merry /data

USER merry


# =============================================================================
# Port
# =============================================================================

EXPOSE 8000 8501 8502


# =============================================================================
# 기본 실행 명령어
# =============================================================================
# Consumer와 Dashboard는 Kubernetes Deployment에서 command를 덮어씀.
# =============================================================================

CMD ["python", "-m", "uvicorn", "api_kafka_receiver:app", "--host", "0.0.0.0", "--port", "8000"]
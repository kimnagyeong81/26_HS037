# -*- coding: utf-8 -*-
"""
kafka_k8s_runtime.py

Kubernetes Pod 안에서 사용할 Kafka 준비 유틸.
- Kafka broker 자동 실행 안 함
- /capstone/kafka-4.2.0-src 필요 없음
- kafka-python AdminClient로 topic 생성/확인
"""

from __future__ import annotations

import socket
import time

from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError


def _parse_host_port(bootstrap_server: str):
    host, port = bootstrap_server.split(":", 1)
    return host, int(port)


def wait_for_kafka(bootstrap_server: str, timeout_sec: int = 60) -> None:
    host, port = _parse_host_port(bootstrap_server)
    deadline = time.time() + timeout_sec

    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                print(f"[Kafka] broker 연결 확인: {bootstrap_server}")
                return
        except OSError:
            time.sleep(1)

    raise TimeoutError(f"Kafka broker 연결 실패: {bootstrap_server}")


def ensure_kafka_topic(
    bootstrap_server: str,
    topic: str,
    timeout_sec: int = 60,
    num_partitions: int = 1,
    replication_factor: int = 1,
) -> None:
    wait_for_kafka(bootstrap_server, timeout_sec=timeout_sec)

    admin = KafkaAdminClient(
        bootstrap_servers=bootstrap_server,
        client_id="merry-ai-k8s-admin",
    )

    try:
        topics = admin.list_topics()

        if topic in topics:
            print(f"[Kafka] topic 이미 존재: {topic}")
            return

        admin.create_topics(
            new_topics=[
                NewTopic(
                    name=topic,
                    num_partitions=num_partitions,
                    replication_factor=replication_factor,
                )
            ],
            validate_only=False,
        )

        print(f"[Kafka] topic 생성 완료: {topic}")

    except TopicAlreadyExistsError:
        print(f"[Kafka] topic 이미 존재: {topic}")

    finally:
        admin.close()
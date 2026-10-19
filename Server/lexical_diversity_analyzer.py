# lexical_diversity_analyzer.py

import re
from collections import Counter
from typing import List, Dict, Union


def tokenize_korean_text(text: str) -> List[str]:
    """
    한국어 텍스트를 토큰화하는 함수
    1순위: 형태소 분석기(Okt) 사용
    2순위: 없으면 정규식 기반 토큰화

    반환:
        의미 있는 토큰 리스트
    """
    text = str(text).strip()
    if not text:
        return []

    # 공백 정리
    text = re.sub(r'\s+', ' ', text)

    try:
        from konlpy.tag import Okt
        okt = Okt()

        # 명사, 동사, 형용사만 추출 / 동사·형용사는 원형(stem=True)
        pos_tags = okt.pos(text, stem=True)

        tokens = [
            word for word, pos in pos_tags
            if pos in ['Noun', 'Verb', 'Adjective'] and len(word) >= 2
        ]
        return tokens

    except Exception:
        # fallback: 한글 2글자 이상만 추출
        tokens = re.findall(r'[가-힣]{2,}', text)
        return tokens


def get_length_reliability(total_tokens: int) -> Dict[str, Union[str, float]]:
    """
    전체 토큰 수에 따라 어휘다양성 해석의 신뢰도를 반환

    기준 예시:
    - 10 미만: 매우 낮음
    - 10~29: 낮음
    - 30~49: 보통
    - 50 이상: 높음
    """
    if total_tokens < 10:
        return {
            "reliability_label": "매우 낮음",
            "reliability_score": 0.25,
            "interpretation": "발화량이 너무 적어 어휘다양성 해석이 어렵습니다."
        }
    elif total_tokens < 30:
        return {
            "reliability_label": "낮음",
            "reliability_score": 0.5,
            "interpretation": "어휘다양성은 계산 가능하지만 참고용으로만 해석해야 합니다."
        }
    elif total_tokens < 50:
        return {
            "reliability_label": "보통",
            "reliability_score": 0.75,
            "interpretation": "기본적인 해석은 가능하지만 충분히 안정적이지는 않을 수 있습니다."
        }
    else:
        return {
            "reliability_label": "높음",
            "reliability_score": 1.0,
            "interpretation": "발화량이 충분하여 비교적 안정적으로 해석할 수 있습니다."
        }


def interpret_ttr(ttr: float) -> str:
    """
    TTR 값 해석
    """
    if ttr == 0:
        return "분석 불가"
    elif ttr < 0.35:
        return "어휘다양성이 낮은 편"
    elif ttr < 0.55:
        return "어휘다양성이 보통 수준"
    else:
        return "어휘다양성이 높은 편"


def lexical_diversity_analysis(text: str) -> Dict:
    """
    단일 텍스트에 대한 기본 어휘다양성 분석

    반환값:
    - total_tokens: 전체 토큰 수
    - unique_tokens: 고유 토큰 수
    - ttr: Type-Token Ratio
    - repeated_ratio: 반복 비율
    - top_repeated_words: 자주 반복된 단어
    """
    tokens = tokenize_korean_text(text)

    total_tokens = len(tokens)
    unique_tokens = len(set(tokens))

    if total_tokens == 0:
        return {
            "total_tokens": 0,
            "unique_tokens": 0,
            "ttr": 0.0,
            "repeated_ratio": 0.0,
            "top_repeated_words": []
        }

    ttr = unique_tokens / total_tokens

    counter = Counter(tokens)
    repeated_count = sum(freq for _, freq in counter.items() if freq > 1)
    repeated_ratio = repeated_count / total_tokens

    top_repeated_words = counter.most_common(10)

    return {
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "ttr": round(ttr, 4),
        "repeated_ratio": round(repeated_ratio, 4),
        "top_repeated_words": top_repeated_words
    }


def lexical_diversity_with_reliability(text: str) -> Dict:
    """
    단일 텍스트에 대해
    - 어휘다양성
    - 반복 비율
    - 길이 기반 신뢰도
    - 해석 문구
    를 함께 반환
    """
    result = lexical_diversity_analysis(text)
    reliability = get_length_reliability(result["total_tokens"])
    ttr_interpretation = interpret_ttr(result["ttr"])

    return {
        **result,
        "reliability_label": reliability["reliability_label"],
        "reliability_score": reliability["reliability_score"],
        "length_interpretation": reliability["interpretation"],
        "ttr_interpretation": ttr_interpretation
    }


def lexical_diversity_from_dialogues(dialogues: List[str]) -> Dict:
    """
    여러 턴의 대화를 합쳐서 어휘다양성을 분석

    매개변수:
        dialogues: 사용자 발화 리스트
        예) ["안녕하세요", "오늘은 병원에 다녀왔어요", "점심은 김치찌개를 먹었어요"]

    반환:
        누적 대화 기준 어휘다양성 분석 결과
    """
    if not dialogues:
        return {
            "num_utterances": 0,
            "combined_text": "",
            "total_tokens": 0,
            "unique_tokens": 0,
            "ttr": 0.0,
            "repeated_ratio": 0.0,
            "top_repeated_words": [],
            "reliability_label": "매우 낮음",
            "reliability_score": 0.0,
            "length_interpretation": "대화 데이터가 없습니다.",
            "ttr_interpretation": "분석 불가"
        }

    cleaned_dialogues = [str(d).strip() for d in dialogues if str(d).strip()]
    combined_text = " ".join(cleaned_dialogues)

    result = lexical_diversity_with_reliability(combined_text)
    result["num_utterances"] = len(cleaned_dialogues)
    result["combined_text"] = combined_text

    return result


def analyze_single_utterance(text: str) -> Dict:
    """
    짧은 단일 발화 분석용
    """
    return lexical_diversity_with_reliability(text)


def analyze_conversation_window(dialogues: List[str], window_size: int = 5) -> Dict:
    """
    최근 N개 발화를 기준으로 어휘다양성 분석

    예:
        최근 5개 발화만 합쳐서 분석
    """
    if not dialogues:
        return lexical_diversity_from_dialogues([])

    recent_dialogues = dialogues[-window_size:]
    result = lexical_diversity_from_dialogues(recent_dialogues)
    result["window_size"] = window_size

    return result


def build_lexical_diversity_event_payload(
    result: Dict,
    user_id: str,
    session_id: str,
    timestamp: str,
    window_size: int = None
) -> Dict:
    return {
        "user_id": user_id,
        "session_id": session_id,
        "timestamp": timestamp,
        "analysis_type": "lexical_diversity",
        "window_size": result.get("window_size", window_size),
        "num_utterances": result.get("num_utterances"),
        "total_tokens": result.get("total_tokens", 0),
        "unique_tokens": result.get("unique_tokens", 0),
        "ttr": result.get("ttr", 0.0),
        "repeated_ratio": result.get("repeated_ratio", 0.0),
        "reliability_score": result.get("reliability_score", 0.0),
        "reliability_label": result.get("reliability_label", "unknown")
    }
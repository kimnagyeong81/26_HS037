# repetition_analyzer.py

import re
from collections import Counter


# 반복어 분석에서 제외할 단어
STOPWORDS = {
    # 조사
    "은", "는", "이", "가", "을", "를", "에", "에서", "에게",
    "의", "도", "로", "으로", "와", "과", "하고", "랑", "한테",

    # 너무 일반적인 표현
    "좀", "그", "저", "거", "것", "수", "게",

    # 간투사 / 머뭇거림 표현
    # 반복어 분석에서는 제외하고, 아래 hesitation 지표에서 따로 계산
    "음", "어", "아", "으음", "음음", "어어"
}


# 머뭇거림/회상 지연으로 볼 수 있는 표현
HESITATION_WORDS = {
    "음", "어", "아", "으음", "음음", "어어",
    "뭐더라", "뭐였지", "기억이", "생각이", "까먹었", "깜빡"
}


def tokenize(text: str) -> list:
    """
    문장을 단어 단위로 나누는 함수
    - 한글, 영어, 숫자만 추출
    - 소문자 변환
    """
    text = text.lower()
    tokens = re.findall(r"[가-힣a-zA-Z0-9]+", text)
    return tokens


def analyze_hesitation(tokens: list) -> dict:
    """
    머뭇거림/회상 지연 표현을 따로 분석
    - '음', '어', '아' 같은 짧은 표현은 정확히 일치할 때만 카운트
    - '뭐더라', '깜빡' 같은 표현은 포함 여부로 카운트
    """
    hesitation_counts = Counter()

    exact_words = {"음", "어", "아", "으음", "음음", "어어"}
    phrase_words = {"뭐더라", "뭐였지", "기억이", "생각이", "까먹었", "깜빡"}

    for token in tokens:
        # 짧은 간투사는 정확히 일치할 때만 카운트
        if token in exact_words:
            hesitation_counts[token] += 1

        # 회상 지연 표현은 단어 안에 포함돼도 카운트
        for phrase in phrase_words:
            if phrase in token:
                hesitation_counts[phrase] += 1

    hesitation_total = sum(hesitation_counts.values())

    return {
        "hesitation_exists": hesitation_total > 0,
        "hesitation_count": hesitation_total,
        "hesitation_words": dict(hesitation_counts)
    }


def analyze_repetition(text: str, min_repeat: int = 2) -> dict:
    """
    한 문장 내에서 같은 단어 반복을 분석하는 함수

    Args:
        text (str): STT로 변환된 문장
        min_repeat (int): 몇 번 이상 나오면 반복으로 볼지 기준

    Returns:
        dict: 반복 분석 결과
    """
    raw_tokens = tokenize(text)

    if not raw_tokens:
        return {
            "repetition_exists": False,
            "total_tokens": 0,
            "unique_tokens": 0,
            "repetition_ratio": 0.0,
            "repeated_words": {},
            "most_repeated_word": None,
            "most_repeated_count": 0,
            "hesitation_exists": False,
            "hesitation_count": 0,
            "hesitation_words": {}
        }

    # 머뭇거림 표현은 따로 계산
    hesitation_result = analyze_hesitation(raw_tokens)

    # 반복어 분석용 토큰: 조사, 간투사, 한 글자 단어 제거
    tokens = [
        token for token in raw_tokens
        if token not in STOPWORDS and len(token) > 1
    ]

    if not tokens:
        return {
            "repetition_exists": False,
            "total_tokens": len(raw_tokens),
            "unique_tokens": 0,
            "repetition_ratio": 0.0,
            "repeated_words": {},
            "most_repeated_word": None,
            "most_repeated_count": 0,
            **hesitation_result
        }

    word_counts = Counter(tokens)

    repeated_words = {
        word: count
        for word, count in word_counts.items()
        if count >= min_repeat
    }

    repetition_exists = len(repeated_words) > 0
    total_tokens = len(tokens)
    unique_tokens = len(set(tokens))

    # 반복 비율: 의미 있는 단어 기준으로 계산
    repetition_ratio = round((total_tokens - unique_tokens) / total_tokens, 3)

    most_repeated_word = None
    most_repeated_count = 0

    if repeated_words:
        most_repeated_word = max(repeated_words, key=repeated_words.get)
        most_repeated_count = repeated_words[most_repeated_word]

    return {
        "repetition_exists": repetition_exists,

        # 전체 토큰 수는 원문 기준
        "raw_total_tokens": len(raw_tokens),

        # 반복어 분석 토큰 수는 불용어 제거 후 기준
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,

        "repetition_ratio": repetition_ratio,
        "repeated_words": repeated_words,
        "most_repeated_word": most_repeated_word,
        "most_repeated_count": most_repeated_count,

        # 머뭇거림 지표
        **hesitation_result
    }
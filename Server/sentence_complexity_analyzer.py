# sentence_complexity_analyzer.py

# sentence_complexity_analyzer.py

import re
from statistics import mean

# 1. 문장 분리
def split_sentences(text):
    """
    텍스트를 문장 단위로 분리
    마침표, 물음표, 느낌표, 줄바꿈 기준
    """
    if not text or not text.strip():
        return []

    sentences = re.split(r'[.!?\n]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


# 2. 단어 분리
def tokenize_words(sentence):
    """
    문장을 공백 기준으로 단순 분리
    특수문자는 제거
    """
    cleaned = re.sub(r'[^\w\s가-힣]', '', sentence)
    words = cleaned.split()
    return [w for w in words if w]


# 3. 접속어 개수 세기
def count_connectives(sentence):
    """
    문장 내 접속어 개수 계산
    """
    connectives = [
        "왜냐하면", "그럼에도", "그러므로", "그리고", "그래서",
        "하지만", "그러나", "그런데", "따라서", "또한",
        "반면", "한편", "게다가", "아니면", "즉", "및", "또"
    ]

    count = 0
    found = []

    for conn in connectives:
        matches = re.findall(re.escape(conn), sentence)
        if matches:
            count += len(matches)
            found.extend([conn] * len(matches))

    return count, found


# 4. 절(Clause) 개수 추정
def count_clauses(sentence):
    """
    연결 어미/접속 표현을 기준으로 절 개수 대략 추정
    """
    clause_markers = [
        "지만", "는데", "아서", "어서", "니까", "므로",
        "면서", "다가", "자마자", "도록", "려고",
        "더니", "나서", "한 뒤", "때문에"
    ]

    count = 1
    found = []

    for marker in clause_markers:
        matches = re.findall(re.escape(marker), sentence)
        if matches:
            count += len(matches)
            found.extend([marker] * len(matches))

    return count, found

# 5. 문장 길이 계산
def sentence_length(sentence):
    """
    문장 길이: 글자 수 기준 (공백 제외)
    """
    return len(sentence.replace(" ", ""))


# 6. 문장별 복잡도 점수 계산
def analyze_sentence(sentence):
    """
    한 문장에 대한 복잡도 분석
    """
    words = tokenize_words(sentence)
    word_count = len(words)

    char_len = sentence_length(sentence)

    connective_count, connectives_found = count_connectives(sentence)
    clause_count, clauses_found = count_clauses(sentence)

    # 간단한 복잡도 점수
    # 길이 + 접속어 + 절 수를 종합 반영
    complexity_score = (
        char_len * 0.2 +
        word_count * 0.5 +
        connective_count * 2 +
        clause_count * 2
    )

    # 난이도 라벨
    if complexity_score < 10:
        level = "단순"
    elif complexity_score < 20:
        level = "보통"
    else:
        level = "복잡"

    return {
        "sentence": sentence,
        "char_length": char_len,
        "word_count": word_count,
        "connective_count": connective_count,
        "connectives_found": connectives_found,
        "clause_count_estimated": clause_count,
        "clause_markers_found": clauses_found,
        "complexity_score": round(complexity_score, 2),
        "complexity_level": level
    }


# 7. 전체 텍스트 분석
def analyze_sentence_complexity(text):
    """
    전체 텍스트의 문장 복잡도 분석
    """
    sentences = split_sentences(text)

    if not sentences:
        return {
            "sentence_count": 0,
            "average_sentence_length": 0,
            "average_word_count": 0,
            "average_connective_count": 0,
            "average_clause_count": 0,
            "overall_complexity_score": 0,
            "overall_complexity_level": "분석 불가",
            "sentence_details": []
        }

    results = [analyze_sentence(sentence) for sentence in sentences]

    avg_sentence_length = mean([r["char_length"] for r in results])
    avg_word_count = mean([r["word_count"] for r in results])
    avg_connective_count = mean([r["connective_count"] for r in results])
    avg_clause_count = mean([r["clause_count_estimated"] for r in results])
    avg_complexity_score = mean([r["complexity_score"] for r in results])

    if avg_complexity_score < 10:
        overall_level = "단순"
    elif avg_complexity_score < 20:
        overall_level = "보통"
    else:
        overall_level = "복잡"

    return {
        "sentence_count": len(sentences),
        "average_sentence_length": round(avg_sentence_length, 2),
        "average_word_count": round(avg_word_count, 2),
        "average_connective_count": round(avg_connective_count, 2),
        "average_clause_count": round(avg_clause_count, 2),
        "overall_complexity_score": round(avg_complexity_score, 2),
        "overall_complexity_level": overall_level,
        "sentence_details": results
    }


# 8. 테스트용 실행
if __name__ == "__main__":
    sample_text_1 = "밥 먹었다. 집 갔다."
    sample_text_2 = "나는 밥을 먹고 나서 집으로 돌아갔다. 그리고 잠시 쉬었다."

    print("=== 예시 1 ===")
    result1 = analyze_sentence_complexity(sample_text_1)
    print(result1)

    print("\n=== 예시 2 ===")
    result2 = analyze_sentence_complexity(sample_text_2)
    print(result2)
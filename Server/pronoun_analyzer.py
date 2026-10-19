# pronoun_analyzer.py

import re

def analyze_demonstrative_pronouns(text):
    patterns = [
        r"이거", r"그거", r"저거",
        r"이것", r"그것", r"저것",
        r"여기", r"거기", r"저기",
        r"이렇게", r"그렇게", r"저렇게",
        r"이런", r"그런", r"저런",
        r"얘", r"걔", r"쟤",
        r"이 사람", r"그 사람", r"저 사람",
        r"그거 있잖아", r"그거 뭐지", r"그 뭐냐",
        r"그거 말이야", r"하는 거", r"이런 거", r"저런 거", r"그런 거"
    ]

    found = []
    word_counts = {}

    patterns = sorted(patterns, key=len, reverse=True)

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            found.extend(matches)
            word_counts[pattern] = len(matches)

    return {
        "count": len(found),
        "words": found,
        "word_counts": word_counts,
        "exists": len(found) > 0
    }

def build_demonstrative_pronoun_event_payload(
    result: dict,
    user_id: str,
    session_id: str,
    timestamp: str
) -> dict:
    return {
        "user_id": user_id,
        "session_id": session_id,
        "timestamp": timestamp,
        "analysis_type": "demonstrative_pronoun",
        "demonstrative_pronoun_count": result.get("count", 0),
        "demonstrative_pronoun_exists": result.get("exists", False)
    }
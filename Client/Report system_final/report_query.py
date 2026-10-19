"""MongoDB 없이 동작하는 월간 리포트 시연 데이터.
실제 환자 데이터/의학적 정상 범위가 아닙니다. DEMO_SEED 변경 시 재생성.
동일 사용자/연월은 같은 값. 원본 report_server.py의 반환 구조와 호환됩니다.
언어 지표 6개는 templates/cognition_cards.html을 사용하세요.
"""
from __future__ import annotations
import hashlib
import json
import math
import random
from calendar import monthrange
from datetime import datetime, timezone
from statistics import mean
from typing import Any

DEMO_SEED = 2
EMOTIONS = ['joy', 'calm', 'anger', 'fear', 'sadness']
EMOTION_KO = dict(zip(EMOTIONS, ['기쁨', '평온', '분노', '두려움', '슬픔']))
FEATURE_KO = {
    'demonstrative_pronoun_count_mean': '지시대명사',
    'ttr_mean': '어휘다양성',
    'repeated_ratio_mean': '반복 단어 분석',
    'sentence_complexity_score_mean': '문장 복잡도 분석',
    'negative_emotion_ratio': '부정 감정분석',
    'positive_emotion_ratio': '긍정감정분석',
}
RISK_Z_THRESHOLD = 1.6
# 개인 기준선의 시연값 (의학적 판정 기준이 아님).
BASELINE = dict(zip(FEATURE_KO, [.058, .64, .045, .62, .13, .32]))
BASE_STD = dict(zip(FEATURE_KO, [.014, .040, .016, .045, .065, .065]))
DIRECTION = dict(zip(FEATURE_KO, [1, -1, 1, -1, 1, -1]))
KEYS = dict(zip(FEATURE_KO, ['demonstrative_pronoun_rate', 'ttr', 'repetition',
                            'sentence_complexity', 'negative_emotion_ratio', 'positive_emotion_ratio']))
TEXTS = {
    "joy": [
        "손주 녀석이 전화해서 할머니 밥 먹었냐고 묻더라고. "
        "그 한마디가 어찌나 반갑던지.",
        "아침에 슬슬 걸어 나갔더니 꽃이 벌써 피었네. "
        "구경하다 보니 기분이 참 좋아.",
    ],
    "calm": [
        "밥 먹고 창가에 좀 앉아 있었지. "
        "햇볕도 따뜻하고 마음이 편안하네.",
        "오늘은 별일 없어. "
        "차 한 잔 마시고 라디오 들으면서 쉬고 있지.",
    ],
    "anger": [
        "오라던 시간에 맞춰 갔는데 한참을 기다렸어. "
        "언제까지 기다리라는 건지 답답하더라고.",
        "설명을 좀 천천히 해주면 좋겠는데 "
        "자꾸 빨리 말하니까 못 알아듣겠어. 몇 번을 물었지.",
    ],
    "fear": [
        "내일 병원 가는 날인데, "
        "검사 결과가 어떨지 자꾸 마음이 쓰이네.",
        "밤에 밖에서 덜컥 소리가 나더라고. "
        "혼자 있으니까 무슨 일인가 싶어서 겁이 났지.",
    ],
    "sadness": [
        "애들도 다 바쁘지 뭐. "
        "이해는 하는데, 오늘은 얼굴이 좀 보고 싶네.",
        "친구 만나서 밥 먹기로 했는데 몸이 안 좋다네. "
        "못 만난다니까 좀 서운하고 허전하지.",
    ],
}


def _rng(user_id, year, month):
    seed = hashlib.sha256(f'{DEMO_SEED}:{user_id}:{year}:{month}'.encode()).digest()
    return random.Random(int.from_bytes(seed, 'big'))


def _week(day):
    return min((day - 1) // 7 + 1, 4)


def _change(value, baseline):
    return (value - baseline) / baseline if baseline else 0.0


def _grade(value, baseline, direction=1):
    adverse = _change(value, baseline) * direction
    return 'caution' if adverse >= .30 else 'watch' if adverse >= .15 else 'ok'


def _alert(score):
    # 그래프 색상용 시연 규칙. 실제 trend_monitor의 임상/운영 로직이 아님.
    return 'risk' if score >= 2.2 else 'warning' if score >= 1.6 else 'watch' if score >= 1.0 else 'normal'


def _month_rows(user_id, year, month):
    r = _rng(user_id, year, month)
    n = monthrange(year, month)[1]
    inactive = set(r.sample(range(1, n + 1), r.randint(2, 4)))
    # 연속된 생활 변화 구간과 작은 일별 잡음. 서로 독립된 점수 난수 대신 공유 흐름을 사용.
    center1 = n * r.uniform(.40, .48)
    center2 = n * r.uniform(.72, .80)
    amplitude = r.uniform(1.55, 1.85)
    rows = []
    ar_noise = 0.0
    for day in range(1, n + 1):
        ar_noise = .65 * ar_noise + r.gauss(0, .025)
        burden = (.24 + amplitude * math.exp(-.5 * ((day - center1) / 2.0) ** 2)
                  + .70 * math.exp(-.5 * ((day - center2) / 1.8) ** 2) + ar_noise)
        active = day not in inactive
        sessions = r.choices([1, 2, 3], [70, 27, 3])[0] if active else 0
        utterances = sum(r.randint(18, 32) for _ in range(sessions))
        features = {}
        for f in FEATURE_KO:
            z = max(0.0, burden + r.gauss(0, .07))
            features[f] = round(max(.001, min(.99, BASELINE[f] + DIRECTION[f] * BASE_STD[f] * z)), 5)
        # 감정 분포는 합계 100. 평온/기쁨이 우세, 변화 구간에서 두려움/슬픔 증가.
        weights = {
            'joy': 30 - 4 * burden + r.gauss(0, 1.2),
            'calm': 46 - 5 * burden + r.gauss(0, 1.1),
            'anger': 6 + 1.5 * burden + r.gauss(0, .5),
            'fear': 7 + 3 * burden + r.gauss(0, .6),
            'sadness': 11 + 4.5 * burden + r.gauss(0, .7),
        }
        denom = sum(weights.values())
        emotions = {e: round(100 * weights[e] / denom, 1) for e in EMOTIONS}
        emotions['calm'] = round(100 - sum(v for e, v in emotions.items() if e != 'calm'), 1)
        rows.append({
            'day': day, 'active': active, 'sessions': sessions, 'utterances': utterances,
            'minutes': round(sum(r.uniform(6, 12) for _ in range(sessions)), 1),
            'features': features, 'burden': burden, 'emotions': emotions,
            'stds': {e: round(1.2 + burden * factor + r.uniform(0, .7), 1)
                     for e, factor in [('anger', 1.1), ('fear', 1.8), ('sadness', 1.5)]},
            'marbles': {e: 0 for e in EMOTIONS},
        })
    # 월 10~18개, 배출일당 1개. 평온/기쁨이 대부분이며 위험 감정은 소수.
    eligible = [d for d in rows if d['active']]
    total = r.randint(10, 18)
    fear, sadness, anger = r.randint(0, 1), r.randint(1, 2), r.randint(0, 1)
    remaining = total - fear - sadness - anger
    calm = round(remaining * r.uniform(.55, .65))
    allocation = {'joy': remaining - calm, 'calm': calm, 'anger': anger, 'fear': fear, 'sadness': sadness}
    available = eligible[:]
    for e in ['fear', 'sadness', 'anger', 'joy', 'calm']:
        for _ in range(allocation[e]):
            weights = [1 + 3 * d['burden'] if e in ['fear', 'sadness', 'anger'] else 1 for d in available]
            d = r.choices(available, weights)[0]
            d['marbles'][e] = 1
            available.remove(d)
    return rows


def _mean(rows, field):
    return mean(d[field] for d in rows) if rows else 0.0


def _metric(value, baseline, direction=1, decimals=3):
    return {'value': round(value, decimals), 'baseline': round(baseline, decimals),
            'change': round(_change(value, baseline), 3), 'grade': _grade(value, baseline, direction)}


def build_report_data(user_id: str, year: int, month: int) -> dict[str, Any]:
    if not 2020 <= year <= 2099 or not 1 <= month <= 12:
        raise ValueError('year는 2020~2099, month는 1~12여야 합니다.')
    r = _rng(user_id, year, month)
    py, pm = (year - 1, 12) if month == 1 else (year, month - 1)
    rows, prev = _month_rows(user_id, year, month), _month_rows(user_id, py, pm)
    active, prev_active = [d for d in rows if d['active']], [d for d in prev if d['active']]
    longest, gap = [], []
    for d in rows:
        gap = gap + [d['day']] if not d['active'] else []
        if len(gap) > len(longest):
            longest = gap[:]
    avg, prev_avg = round(_mean(active, 'minutes'), 1), round(_mean(prev_active, 'minutes'), 1)
    activity = {
        'active_days': len(active), 'total_days': len(rows),
        'session_count': sum(d['sessions'] for d in rows), 'avg_daily_minutes': avg,
        'total_minutes': round(sum(d['minutes'] for d in rows), 1),
        'max_gap_days': len(longest), 'gap_start': f'{month}/{longest[0]}' if longest else None,
        'gap_end': f'{month}/{longest[-1]}' if longest else None, 'prev_avg_minutes': prev_avg,
        'prev_session_count': sum(d['sessions'] for d in prev), 'avg_minutes_change': _change(avg, prev_avg),
    }
    total_utterances = sum(d['utterances'] for d in active)
    feature_mean = {f: sum(d['features'][f] * d['utterances'] for d in active) / total_utterances for f in FEATURE_KO}
    cognition = {}
    for f, key in KEYS.items():
        scale = 100 if key in ['demonstrative_pronoun_rate', 'negative_emotion_ratio', 'positive_emotion_ratio'] else 1
        cognition[key] = _metric(feature_mean[f] * scale, BASELINE[f] * scale, DIRECTION[f], 1 if scale == 100 else 3)
    rep = cognition['repetition']
    rep_count = round(feature_mean['repeated_ratio_mean'] * total_utterances)
    prev_total_utterances = sum(d['utterances'] for d in prev_active)
    prev_rep_count = round(BASELINE['repeated_ratio_mean'] * prev_total_utterances)
    # 반복 단어는 반복 구간 횟수와 별도: 같은 단어가 여러 구간에 포함될 수 있음.
    rep.update(_metric(rep_count / total_utterances, BASELINE['repeated_ratio_mean'], 1, 4))
    rep.update({'count': rep_count, 'prev_count': prev_rep_count,
                'ratio': round(rep_count / total_utterances * 100, 1), 'baseline_ratio': 4.5,
                'total_utterances': total_utterances,
                'top_words': [{'word': word, 'count': max(1, round(rep_count * rate))}
                              for word, rate in [('그거', .30), ('맞아', .22), ('오늘', .15)]]})
    cognition['sentence_complexity']['label'] = '보통'
    # 기존 HTML이 참조하더라도 오류가 나지 않도록 유지. 새 카드에서는 표시하지 않음.
    cognition['response_delay_sec'] = _metric(2.3, 2.2, 1, 1)
    cognition['wpm'] = _metric(94, 96, -1, 1)
    cards = []
    descriptions = [
        '전체 어절 중 “이것·그것·저것” 등 지시대명사가 차지하는 비율',
        '서로 다른 단어 수 ÷ 전체 단어 수(TTR). 같은 길이의 발화끼리 비교한 시연값',
        '반복 표현이 포함된 발화 비율. 같은 단어의 단순 재등장과 구분',
        '문장 길이와 절 구성의 복잡도를 0~1로 요약한 시연 점수',
        '분석 발화 중 부정 감정으로 분류된 발화의 비율',
        '분석 발화 중 긍정 감정으로 분류된 발화의 비율',
    ]
    for (f, key), desc in zip(KEYS.items(), descriptions):
        m = cognition[key]
        value, base = (rep['ratio'], rep['baseline_ratio']) if key == 'repetition' else (m['value'], m['baseline'])
        unit = '%' if key in ['demonstrative_pronoun_rate', 'repetition', 'negative_emotion_ratio', 'positive_emotion_ratio'] else ''
        digits = 1 if unit else 2
        cards.append({'key': key, 'title': FEATURE_KO[f], 'value': value, 'baseline': base,
                      'display_value': f'{value:.{digits}f}', 'display_baseline': f'{base:.{digits}f}',
                      'unit': unit, 'change': m['change'], 'grade': m['grade'], 'description': desc,
                      'detail': f'전체 {total_utterances}개 발화 중 반복 표현 {rep_count}회 · 주요 단어: 그거, 맞아, 오늘'
                                if key == 'repetition' else ''})
    cognition['language_cards'] = cards

    counts = {e: sum(d['marbles'][e] for d in rows) for e in EMOTIONS}
    prev_counts = {e: sum(d['marbles'][e] for d in prev) for e in EMOTIONS}
    total, prev_total = sum(counts.values()), sum(prev_counts.values())
    danger = counts['fear'] + counts['sadness']
    dp, pdp = round(danger / total, 3), round((prev_counts['fear'] + prev_counts['sadness']) / prev_total, 3)
    marbles = {'counts': counts, 'total': total, 'danger_count': danger, 'danger_pct': dp,
               'prev_counts': prev_counts, 'prev_total': prev_total, 'prev_danger_pct': pdp,
               'danger_pct_change': round(dp - pdp, 3)}
    emotions = {}
    for e in EMOTIONS:
        value = round(mean(d['emotions'][e] for d in active), 1)
        baseline = round(mean(d['emotions'][e] for d in prev_active), 1)
        emotions[e] = {'mean': value, 'change': round(_change(value, baseline), 3),
                       'grade': _grade(value, baseline, -1 if e in ['joy', 'calm'] else 1)}
    weeks = []
    for w in range(1, 5):
        bucket = [d for d in active if _week(d['day']) == w]
        weeks.append({'week': w, **{e: round(mean(d['emotions'][e] for d in bucket), 1) if bucket else 0.0 for e in EMOTIONS}})
    calendar = {str(d['day']): round(sum(d['stds'].values()), 1) if d['active'] else 0.0 for d in rows}
    top_days = sorted(active, key=lambda d: calendar[str(d['day'])], reverse=True)[:3]
    top3 = [{'day': d['day'], 'date_str': f"{month}/{d['day']}", 'std_score': calendar[str(d['day'])],
             'dominant_emotion': EMOTION_KO[max(d['stds'], key=d['stds'].get)]} for d in top_days]
    cases, rest = [], {}
    for e in EMOTIONS:
        events = [d for d in active for _ in range(d['marbles'][e])]
        # 카드는 배출 이벤트에 연결된 가상의 발화. 전체 하루 평균 감정과 구분.
        case_rows = []
        for d in events:
            scores = {em: r.uniform(2, 10) for em in EMOTIONS if em != e}
            scores[e] = 100 - sum(scores.values())
            scores = {em: round(v, 1) for em, v in scores.items()}
            scores[e] = round(100 - sum(v for em, v in scores.items() if em != e), 1)
            case_rows.append((d, scores))
        case_rows.sort(key=lambda pair: pair[1][e], reverse=True)
        for i, (d, scores) in enumerate(case_rows[:2]):
            cases.append({'emotion': e, 'emotion_ko': EMOTION_KO[e],
                          'timestamp': f"{month}월 {d['day']}일 오후 {r.randint(1, 6)}:{r.randint(0, 59):02d}",
                          'text': TEXTS[e][i], 'scores': scores,
                          'marble_id': f'demo-{year}-{month:02d}-{e}-{i + 1}', 'is_demo': True})
        rest[e] = {'count': max(0, len(events) - 2),
                   'sample_dates': [f"{month}/{d['day']}" for d, _ in case_rows[2:7]]}

    daily, feature_z = {}, {f: [] for f in FEATURE_KO}
    distribution = {lvl: 0 for lvl in ['normal', 'watch', 'warning', 'risk']}
    ewma = None
    for d in active:
        values = {f: max(0, (d['features'][f] - BASELINE[f]) * DIRECTION[f] / BASE_STD[f]) for f in FEATURE_KO}
        for f, z in values.items():
            feature_z[f].append(z)
        score = round(mean(values.values()), 4)
        previous = ewma
        # 첫 관측값으로 시작하여 0에서 인위적으로 상승하는 구간을 없앱니다.
        ewma = score if ewma is None else round(.3 * score + .7 * ewma, 4)
        level = _alert(score)
        reasons = []
        for f, z in values.items():
            if z >= RISK_Z_THRESHOLD:
                v, b = d['features'][f], BASELINE[f]
                direction = '증가' if DIRECTION[f] == 1 else '감소'
                reasons.append(f"{FEATURE_KO[f]} 지표가 개인 기준선 대비 {abs(_change(v, b)) * 100:.1f}% {direction}")
        # 보고서의 근거는 아래 월간 수치에서 생성. 일별 근거도 해당 일의 값에서 계산.
        daily[f"{year}-{month:02d}-{d['day']:02d}"] = {
            'daily_risk_score': score, 'ewma_risk_score': ewma, 'alert_level': level,
            'risk_feature_count': sum(z >= RISK_Z_THRESHOLD for z in values.values()),
            'trend_direction': 'up' if previous is not None and ewma > previous + .03 else 'flat_or_down',
            'main_reasons': reasons,
        }
        distribution[level] += 1
    weekly_risk = []
    for w in range(1, 5):
        bucket = [v for date, v in daily.items() if _week(int(date[-2:])) == w]
        weekly_risk.append({'week': w, 'daily': round(mean(v['daily_risk_score'] for v in bucket), 4) if bucket else 0.0,
                            'ewma': round(mean(v['ewma_risk_score'] for v in bucket), 4) if bucket else 0.0})
    feature_avg = [{'feature': f, 'name_ko': FEATURE_KO[f], 'avg_risk_z': round(mean(zs), 4),
                    'risk_days': sum(z >= RISK_Z_THRESHOLD for z in zs), 'is_risk': mean(zs) >= RISK_Z_THRESHOLD}
                   for f, zs in feature_z.items()]
    feature_avg.sort(key=lambda v: v['avg_risk_z'], reverse=True)
    # 설명은 표시값·발화 수·관찰 일수에서 생성하며 진단을 단정하지 않습니다.
    p = cognition['demonstrative_pronoun_rate']
    t = cognition['ttr']
    c = cognition['sentence_complexity']
    neg, pos = cognition['negative_emotion_ratio'], cognition['positive_emotion_ratio']
    ndays = {f: sum(z >= RISK_Z_THRESHOLD for z in zs) for f, zs in feature_z.items()}
    reasons = [
        f"지시대명사 사용 비율이 개인 기준선 {p['baseline']:.1f}%에서 {p['value']:.1f}%로 증가했습니다. 명시적인 대상 대신 ‘그것·이것’ 등을 사용하는 표현을 추가 관찰합니다.",
        f"어휘다양성(TTR)이 {t['baseline']:.2f}에서 {t['value']:.2f}로 감소했습니다. 같은 길이의 발화 구간에서 사용하는 단어 종류가 줄어드는 양상입니다.",
        f"전체 {total_utterances}개 발화 중 반복 표현이 포함된 발화는 {rep_count}개({rep['ratio']:.1f}%)입니다. 기준선 {rep['baseline_ratio']:.1f}%보다 높으며 ‘그거·맞아·오늘’이 반복 표현에 자주 포함됐습니다.",
        f"문장 복잡도 점수가 {c['baseline']:.2f}에서 {c['value']:.2f}으로 낮아졌습니다. 복잡도 감소가 기준 변동폭을 넘은 날은 {ndays['sentence_complexity_score_mean']}일입니다.",
        f"부정 감정 발화 비율이 {neg['baseline']:.1f}%에서 {neg['value']:.1f}%로 증가하고, 긍정 감정 발화 비율은 {pos['baseline']:.1f}%에서 {pos['value']:.1f}%로 감소했습니다. 일시적인 생활 사건의 영향도 함께 확인할 필요가 있습니다.",
    ]
    severity = dict(zip(distribution, range(4)))
    final_level = max((v['alert_level'] for v in daily.values()), key=severity.get, default='normal')
    anomaly = {
        'daily_scores': daily, 'monthly_avg_risk': round(mean(v['daily_risk_score'] for v in daily.values()), 4),
        'monthly_max_risk': max(v['daily_risk_score'] for v in daily.values()),
        'peak_date': max(daily, key=lambda date: daily[date]['daily_risk_score']),
        'alert_distribution': distribution, 'final_alert_level': final_level,
        'weekly_risk': weekly_risk, 'top_reasons': reasons, 'monthly_feature_avg': feature_avg, 'has_data': True,
    }
    return {
        'meta': {'user_id': user_id, 'year': year, 'month': month,
                 'overall_grade': {'normal': '정상', 'watch': '관찰', 'warning': '주의', 'risk': '위험'}[final_level],
                 'generated_at': datetime.now(timezone.utc).isoformat(), 'is_demo': True,
                 'data_source': 'synthetic_demo', 'demo_notice': '시연용 합성 데이터이며 실제 측정·진단 결과가 아닙니다.',
                 'baseline_description': '가상의 개인 기준선', 'demo_seed': DEMO_SEED},
        'activity': activity, 'cognition': cognition, 'marbles': marbles,
        'emotion_summary': {'monthly_avg': emotions}, 'weekly_trend': {'weeks': weeks},
        'volatility_calendar': {'calendar': calendar, 'top3_volatile_days': top3},
        'xai_utterances': {'top_cases': cases, 'rest_summary': rest}, 'anomaly': anomaly,
    }


if __name__ == '__main__':
    print(json.dumps(build_report_data('user_001', 2026, 5), ensure_ascii=False, indent=2, allow_nan=False))

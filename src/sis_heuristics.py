import numpy as np


REQUEST_PATTERNS = (
    "can you",
    "could you",
    "would you",
    "will you",
    "please",
    "help me",
    "hear me out",
    "listen to me",
    "talk to me",
)

CELEBRATION_PATTERNS = (
    "perfect",
    "thanks",
    "thank you",
    "proud",
    "love this",
    "awesome",
    "great",
    "goal",
    "so happy",
    "finally",
    "cackled",
    "omg",
    "haha",
    "lol",
)

COMPANIONSHIP_PATTERNS = (
    "stay with me",
    "be here with me",
    "just stay with me",
    "some company",
    "not alone",
    "be here for me",
    "keep me company",
)

INTENSE_AFFECT_PATTERNS = (
    "panic",
    "panicking",
    "terrified",
    "afraid",
    "angry",
    "frustrated",
    "furious",
    "irritated",
    "upset",
    "shitty",
    "sorry",
    "astounding",
    "failing",
    "numb",
    "confused",
    "sociopath",
    "manipulative",
    "hate that",
)

PROFANITY_PATTERNS = (
    "fuck",
    "fucked",
    "shit",
    "damn",
    "hell of a time",
)

WITHDRAWAL_PATTERNS = (
    "leave me alone",
    "don't want to talk",
    "do not want to talk",
    "rather keep to myself",
    "not up for conversation",
    "need some distance",
    "need space",
    "keep your distance",
)

BOUNDARY_STRONG_PATTERNS = (
    "back off",
    "stay back",
    "give me space",
    "keep your distance",
    "don't come closer",
    "do not come closer",
    "too close",
    "crowd me",
    "stop insisting",
    "don't push",
    "do not push",
    "don't press",
    "do not press",
    "not comfortable",
    "unsafe",
    "don't stand so close",
)

BOUNDARY_MILD_PATTERNS = (
    "please stop",
    "don't want help",
    "do not want help",
    "don't want any more questions",
    "do not want any more questions",
    "i said i'm fine",
    "i said i am fine",
    "need room",
    "more room",
    "stressed",
    "overwhelmed",
    "tense",
    "guarded",
)

ANGER_RISK_TERMS = (
    "angry",
    "frustrated",
    "furious",
    "irritated",
    "snapping",
    "upset",
    "temper",
    "tense",
    "guarded",
    "uncomfortable",
)


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _has_any(text: str, patterns) -> bool:
    return any(p in text for p in patterns)


def _score_request(text: str) -> float:
    score = 0.0
    if "?" in text:
        score += 0.35
    if _has_any(text, REQUEST_PATTERNS):
        score += 0.45
    return float(min(score, 1.0))


def _score_companionship(text: str) -> float:
    return 1.0 if _has_any(text, COMPANIONSHIP_PATTERNS) else 0.0


def _score_celebration(text: str) -> float:
    return 1.0 if _has_any(text, CELEBRATION_PATTERNS) else 0.0


def _score_withdrawal(text: str) -> float:
    return 1.0 if _has_any(text, WITHDRAWAL_PATTERNS) else 0.0


def _score_expressivity(text: str) -> float:
    score = 0.0
    exclam = text.count("!")
    if exclam >= 2:
        score += 0.45
    elif exclam == 1:
        score += 0.25
    if "?" in text:
        score += 0.10
    if any(token.isupper() and len(token) >= 3 for token in text.split()):
        score += 0.20
    if _has_any(text, (" omg", " lol", " haha", "!!!", "??")):
        score += 0.20
    return float(min(score, 1.0))


def _score_affect_intensity(text: str) -> float:
    score = 0.0
    if _has_any(text, INTENSE_AFFECT_PATTERNS):
        score += 0.60
    if _has_any(text, PROFANITY_PATTERNS):
        score += 0.25
    if _has_any(text, CELEBRATION_PATTERNS):
        score += 0.25
    return float(min(score, 1.0))


def _score_boundary(text: str) -> float:
    if _has_any(text, BOUNDARY_STRONG_PATTERNS):
        return 1.0
    if _has_any(text, BOUNDARY_MILD_PATTERNS):
        return 0.6
    return 0.0


def _score_disclosure(text: str) -> float:
    disclosure_patterns = ("i feel", "i am", "i'm", "i've", "my ", " me ")
    return 1.0 if _has_any(text, disclosure_patterns) else 0.0


def _score_risk_language(text: str) -> float:
    score = _score_boundary(text)
    if _has_any(text, ANGER_RISK_TERMS):
        score = min(1.0, score + 0.35)
    return float(score)


def weak_sis_from_text(v, a, b_probs, dist, text, behaviors):
    """Build weak SIS targets with lightweight text-aware heuristics."""
    b = int(np.argmax(b_probs))
    text_n = _normalize_text(text)

    if b == behaviors.index("CONGRATULATE"):
        intent = 2
    elif v < -0.3 and b in [behaviors.index("LISTEN"), behaviors.index("VERBAL_COMFORT")]:
        intent = 1
    elif _score_boundary(text_n) >= 0.6 or (v < -0.4 and a > 0.4):
        intent = 3
    else:
        intent = 0

    listen_bias = float(b_probs[behaviors.index("LISTEN")])
    request_score = _score_request(text_n)
    companionship_score = _score_companionship(text_n)
    celebration_score = _score_celebration(text_n)
    withdrawal_score = _score_withdrawal(text_n)
    boundary_score = _score_boundary(text_n)
    disclosure_score = _score_disclosure(text_n)
    expressivity_score = _score_expressivity(text)
    affect_intensity_score = _score_affect_intensity(text_n)

    # Engagement here is intended to reflect interactional intensity / involvement
    # rather than friendliness alone, so expressive celebration and strong affect
    # should both be able to push the target high.
    engagement = (
        0.10
        + 0.18 * ((a + 1.0) / 2.0)
        + 0.10 * abs(float(a))
        + 0.12 * listen_bias
        + 0.12 * request_score
        + 0.10 * companionship_score
        + 0.10 * celebration_score
        + 0.14 * expressivity_score
        + 0.16 * affect_intensity_score
        + 0.08 * disclosure_score
        + 0.06 * boundary_score
        + 0.04 * withdrawal_score
    )
    engagement = float(np.clip(engagement, 0.0, 1.0))

    closeness = float(np.clip(1.0 - (dist - 0.3) / (2.5 - 0.3), 0.0, 1.0))

    emotion_risk = 1.0 if (v < -0.5 and a > 0.3) else (0.6 if v < -0.35 else 0.25)
    close_risk = 1.0 if dist < 0.8 else (0.6 if dist < 1.2 and boundary_score > 0.0 else 0.2)
    language_risk = _score_risk_language(text_n)
    withdrawal_risk = 0.6 if (withdrawal_score > 0.0 and boundary_score > 0.0) else 0.0

    risk = (
        0.35 * emotion_risk
        + 0.20 * close_risk
        + 0.35 * language_risk
        + 0.10 * withdrawal_risk
    )
    risk = float(np.clip(risk, 0.0, 1.0))

    return [intent, engagement, closeness, risk]

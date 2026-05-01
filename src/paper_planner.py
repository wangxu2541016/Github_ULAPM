import numpy as np


PAPER_BEHAVIORS = ["HUG", "VERBAL_COMFORT", "LISTEN", "CONGRATULATE", "NEUTRAL"]

PAPER_SIS_BEHAVIOR_SETS = {
    "support": ["VERBAL_COMFORT", "LISTEN", "HUG"],
    "celebration": ["CONGRATULATE", "NEUTRAL"],
    "companionship": ["LISTEN", "NEUTRAL"],
    "deescalation": ["LISTEN", "NEUTRAL"],
}

PAPER_SIS_DISTANCE_RANGES = {
    "support": (0.7, 1.5),
    "celebration": (0.9, 1.6),
    "companionship": (1.0, 1.8),
    "deescalation": (1.3, 2.3),
}

PAPER_INTENT_TO_STATE = {
    0: "companionship",
    1: "support",
    2: "celebration",
    3: "deescalation",
}


def sis_state_from_prediction(sis_vec, risk_override_threshold: float = 0.6) -> str:
    sis = np.asarray(sis_vec, dtype=np.float32).reshape(-1)
    intent = float(sis[0])
    risk = float(sis[3])
    if risk > risk_override_threshold:
        return "deescalation"
    intent_idx = int(np.clip(np.rint(intent), 0, 3))
    return PAPER_INTENT_TO_STATE[intent_idx]


def project_behavior_from_logits(
    behavior_logits,
    sis_vec,
    behaviors=None,
    risk_override_threshold: float = 0.6,
):
    behaviors = list(behaviors or PAPER_BEHAVIORS)
    logits = np.asarray(behavior_logits, dtype=np.float32).reshape(-1)
    if len(logits) != len(behaviors):
        raise ValueError(f"logit length {len(logits)} does not match behaviors {len(behaviors)}")

    state = sis_state_from_prediction(sis_vec, risk_override_threshold=risk_override_threshold)
    feasible = PAPER_SIS_BEHAVIOR_SETS[state]
    index = {name: i for i, name in enumerate(behaviors)}
    feasible_idx = [index[name] for name in feasible if name in index]
    if not feasible_idx:
        best_idx = int(np.argmax(logits))
        return {
            "state": state,
            "feasible_behaviors": [],
            "behavior": behaviors[best_idx],
            "behavior_index": best_idx,
        }

    best_idx = max(feasible_idx, key=lambda i: float(logits[i]))
    return {
        "state": state,
        "feasible_behaviors": feasible,
        "behavior": behaviors[best_idx],
        "behavior_index": int(best_idx),
    }


def project_distance_from_sis(
    distance_m,
    sis_vec,
    risk_override_threshold: float = 0.6,
):
    d = float(distance_m)
    state = sis_state_from_prediction(sis_vec, risk_override_threshold=risk_override_threshold)
    lo, hi = PAPER_SIS_DISTANCE_RANGES[state]
    return {
        "state": state,
        "distance_m": float(np.clip(d, lo, hi)),
        "distance_range_m": [float(lo), float(hi)],
    }


def planner_project_prediction(
    behavior_logits,
    distance_m,
    sis_vec,
    behaviors=None,
    risk_override_threshold: float = 0.6,
):
    behavior_info = project_behavior_from_logits(
        behavior_logits=behavior_logits,
        sis_vec=sis_vec,
        behaviors=behaviors,
        risk_override_threshold=risk_override_threshold,
    )
    distance_info = project_distance_from_sis(
        distance_m=distance_m,
        sis_vec=sis_vec,
        risk_override_threshold=risk_override_threshold,
    )
    return {
        "planner_state": behavior_info["state"],
        "feasible_behaviors": behavior_info["feasible_behaviors"],
        "behavior": behavior_info["behavior"],
        "behavior_index": behavior_info["behavior_index"],
        "distance_m": distance_info["distance_m"],
        "distance_range_m": distance_info["distance_range_m"],
    }

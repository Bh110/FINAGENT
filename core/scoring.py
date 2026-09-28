"""Explainable confidence scoring. The LLM's self-reported confidence is only ONE of four
components, so the final number is decomposable and not a black-box guess."""
from __future__ import annotations

import config


def _clip(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


def signal_strength(change: float, volume_ratio=None) -> float:
    price_part = min(100.0, abs(change) * 20.0)                       # 5% move -> 100
    vol_part = 50.0 if volume_ratio is None else min(100.0, volume_ratio * 50.0)
    return round(0.7 * price_part + 0.3 * vol_part, 1)


def agreement_score(sentiment: str, votes: dict):
    """How many independent votes back the sentiment. Returns (score, agreeing, conflicting)."""
    if not votes:
        return 50.0, [], []
    target = {"bullish": 1, "bearish": -1, "neutral": 0}.get(sentiment, 0)
    agree, conflict = [], []
    for name, v in votes.items():
        if v == target:
            agree.append(name)
        elif (target == 0 and v != 0) or (target != 0 and v == -target):
            conflict.append(name)
    score = 50.0 + 50.0 * (len(agree) - len(conflict)) / len(votes)
    return round(_clip(score), 1), agree, conflict


def evidence_quality(docs) -> float:
    if not docs:
        return 0.0
    top = docs[: config.TOP_K]
    rel = sum(d.get("relevance", 0.0) for d in top) / len(top)
    q = _clip((rel - 0.15) / 0.40) * 100.0                             # cos 0.15 -> 0, 0.55 -> 100
    if not any(d.get("asset_match") for d in top):
        q *= 0.5                                                       # docs about someone else
    return round(q, 1)


def blend(llm_conf, sig, agr, evi, n_conflicts, has_docs) -> dict:
    w = config.SCORE_WEIGHTS
    raw = w["llm"] * llm_conf + w["signal"] * sig + w["agreement"] * agr + w["evidence"] * evi
    penalty = min(config.MAX_CONFLICT_PENALTY, config.CONFLICT_PENALTY * n_conflicts)
    final = raw - penalty
    capped = False
    if not has_docs and final > config.NO_EVIDENCE_CAP:
        final, capped = config.NO_EVIDENCE_CAP, True
    return {
        "final": int(round(_clip(final))),
        "components": {"llm": round(llm_conf, 1), "signal": sig,
                       "agreement": agr, "evidence": evi},
        "weights": dict(w),
        "conflict_penalty": penalty,
        "capped_no_evidence": capped,
        "capped_rule_based": False,
    }

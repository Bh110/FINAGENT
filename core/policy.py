"""Deterministic decision policy. The LLM proposes; these rules decide."""
from __future__ import annotations

import config


def confidence_level(conf: int) -> str:
    return "High" if conf > config.HIGH_CONF else "Medium" if conf >= config.LOW_CONF else "Low"


def decide_action(sentiment: str, conf: int) -> str:
    if conf < config.LOW_CONF:
        return "WATCH"
    if sentiment == "neutral":
        return "HOLD"
    if conf > config.HIGH_CONF:
        return "BUY" if sentiment == "bullish" else "SELL"
    return "WATCH"


def is_alert(conf: int, threshold: int) -> bool:
    """PS wording: 'confidence ABOVE a configurable threshold'. Same boundary as High (>70)."""
    return conf > threshold


def build_evidence(norm, indicators, headlines, docs, news_score) -> list:
    """2-3 bullets built from real data (never LLM-written, so they can't be hallucinated)."""
    out = [f"Price {norm['price']:.2f} ({norm['change_percent']:+.1f}%) on volume "
           f"{norm['volume']:,}."]

    parts = []
    if indicators:
        keys = ("rsi14", "sma5", "volume_ratio", "momentum_5")
        parts.append(", ".join(f"{k}={indicators[k]}" for k in keys if k in indicators))
    if headlines:
        parts.append(f"news tone {news_score:+.2f} ('{headlines[0][:90]}')")
    if parts:
        out.append("Technicals/news: " + "; ".join(p for p in parts if p))

    for d in docs[:2]:
        snippet = " ".join(d["content"].split())[:200]
        out.append(f"{d['source']} p.{d['page']} (relevance {d['relevance']:.2f}): {snippet}...")
    return out[:3]


def build_risk_flags(norm, analysis, indicators, docs, kb_available, prior_action, decision_action):
    flags = list(analysis.get("risk_flags", []))
    s = analysis["sentiment"]

    for name in analysis.get("conflicts", []):
        flags.append(f"Contradictory signal: {name.replace('_', ' ')} disagrees with the {s} view")

    rsi = indicators.get("rsi14")
    if rsi is not None:
        if s == "bullish" and rsi > 70:
            flags.append(f"RSI {rsi} is overbought - bullish move may be extended")
        if s == "bearish" and rsi < 30:
            flags.append(f"RSI {rsi} is oversold - bearish move may be exhausted")
    vr = indicators.get("volume_ratio")
    if vr is not None and vr < 0.7 and s != "neutral":
        flags.append(f"Weak volume confirmation ({vr}x recent average)")
    if not indicators:
        flags.append("Insufficient price history for technical indicators")

    if not kb_available:
        flags.append("Knowledge base unavailable - no document grounding")
    elif not docs:
        flags.append("No sufficiently relevant document evidence retrieved (data gap)")
    elif not any(d.get("asset_match") for d in docs):
        flags.append(f"Retrieved documents do not mention {norm['asset']}")

    if analysis.get("engine", "llm") != "llm":
        flags.append("LLM not used for this analysis - " + analysis["engine"])

    opposite = {"BUY": "SELL", "SELL": "BUY"}
    if prior_action and opposite.get(prior_action) == decision_action:
        flags.append(f"Reverses the previous {prior_action} recommendation for this asset")

    return list(dict.fromkeys(flags))  # de-duplicate, keep order

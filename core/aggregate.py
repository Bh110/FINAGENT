"""Cross-signal aggregation: per-asset consensus over all generated artefacts."""
from __future__ import annotations

_SIGN = {"bullish": 1, "bearish": -1, "neutral": 0}


def portfolio_summary(artefacts) -> list:
    """One row per asset. Consensus is confidence-weighted so weak calls count less."""
    by_asset: dict = {}
    for a in artefacts:  # artefacts are newest-first
        by_asset.setdefault(a["decision"]["asset"], []).append(a)

    rows = []
    for asset, arts in by_asset.items():
        weights = [a["decision"]["confidence"] for a in arts]
        signs = [_SIGN.get(a["analysis"].get("sentiment", "neutral"), 0) for a in arts]
        total = sum(weights) or 1
        score = sum(s * w for s, w in zip(signs, weights)) / total
        consensus = "bullish" if score > 0.2 else "bearish" if score < -0.2 else "mixed/neutral"
        rows.append({
            "asset": asset,
            "signals_analysed": len(arts),
            "latest_action": arts[0]["decision"]["action"],
            "latest_confidence": arts[0]["decision"]["confidence"],
            "avg_confidence": round(sum(weights) / len(weights), 1),
            "bullish": signs.count(1), "bearish": signs.count(-1), "neutral": signs.count(0),
            "consensus": consensus,
            "consensus_score": round(score, 2),
        })
    return sorted(rows, key=lambda r: -r["avg_confidence"])

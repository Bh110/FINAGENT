"""Backtest: compare each directional hypothesis with the asset's subsequent price move."""
from __future__ import annotations

import pandas as pd

from core.signals import parse_ts


def evaluate(artefacts, market_df: pd.DataFrame, horizon: int = 3):
    """For each bullish/bearish artefact, look `horizon` ticks ahead in the same asset's series."""
    df = market_df.copy()
    df["_ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["_ts"]).sort_values("_ts")

    rows = []
    for a in artefacts:
        sentiment = a["analysis"].get("sentiment", "neutral")
        if sentiment not in ("bullish", "bearish"):
            continue
        ts = parse_ts(a["signal"].get("timestamp"))
        asset = a["decision"]["asset"]
        if ts is None:
            continue
        future = df[(df["asset"] == asset) & (df["_ts"] > ts)].head(horizon)
        if len(future) < horizon:
            continue                                   # not enough future data yet
        entry = float(a["signal"]["price"])
        exit_price = float(future.iloc[-1]["price"])
        fwd = (exit_price / entry - 1) * 100
        direction = 1 if sentiment == "bullish" else -1
        rows.append({
            "timestamp": a["signal"]["timestamp"], "asset": asset, "sentiment": sentiment,
            "action": a["decision"]["action"], "confidence": a["decision"]["confidence"],
            "entry": entry, "exit": exit_price, "forward_return_%": round(fwd, 2),
            "signed_return_%": round(fwd * direction, 2), "hit": fwd * direction > 0,
        })

    res = pd.DataFrame(rows)
    if res.empty:
        return res, {"evaluated": 0}

    summary = {
        "evaluated": len(res),
        "hit_rate_%": round(res["hit"].mean() * 100, 1),
        "avg_signed_return_%": round(res["signed_return_%"].mean(), 2),
    }
    high = res[res["confidence"] > 70]
    low = res[res["confidence"] <= 70]
    summary["hit_rate_high_conf_%"] = round(high["hit"].mean() * 100, 1) if len(high) else None
    summary["hit_rate_other_%"] = round(low["hit"].mean() * 100, 1) if len(low) else None
    return res, summary

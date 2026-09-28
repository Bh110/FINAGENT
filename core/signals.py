"""Pure-Python signal processing: validation, indicators, news sentiment, asset matching.
No LangChain imports here, so everything in this module is unit-testable."""
from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------- timestamps


def parse_ts(value):
    """Best-effort timestamp parse -> naive datetime, or None."""
    if value is None or value == "":
        return None
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        return ts.to_pydatetime()
    except Exception:
        return None


# ----------------------------------------------------------------- validation


def normalize_signal(raw: dict) -> dict:
    """Validate and normalise a raw signal. Raises ValueError with a readable message."""
    if not isinstance(raw, dict):
        raise ValueError("signal must be a dict")
    try:
        asset = str(raw["asset"]).strip()
        price = float(raw["price"])
        change = float(raw["change_percent"])
        volume = int(float(raw["volume"]))
    except KeyError as e:
        raise ValueError(f"missing field {e}") from None
    except (TypeError, ValueError):
        raise ValueError("price, change_percent and volume must be numeric") from None

    if not asset:
        raise ValueError("asset name is empty")
    if not all(math.isfinite(x) for x in (price, change)):
        raise ValueError("price/change must be finite numbers")
    if price <= 0:
        raise ValueError("price must be > 0")
    if volume < 0:
        raise ValueError("volume must be >= 0")
    if change <= -100:
        raise ValueError("change_percent must be > -100")

    ts = parse_ts(raw.get("timestamp")) or datetime.now()
    return {
        "asset": asset,
        "price": round(price, 4),
        "change_percent": round(change, 4),
        "volume": volume,
        "timestamp": ts.isoformat(timespec="seconds"),
        "source": str(raw.get("source", "unknown")),
        "history": [float(p) for p in raw.get("history", [])],
        "volume_history": [float(v) for v in raw.get("volume_history", [])],
        "headlines": [str(h) for h in raw.get("headlines", [])],
    }


# ----------------------------------------------------------------- indicators


def rsi_wilder(prices, period: int = 14):
    """Wilder-smoothed RSI. Needs at least period+1 prices."""
    if len(prices) < period + 1:
        return None
    deltas = [b - a for a, b in zip(prices[:-1], prices[1:])]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for g, l in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return round(100 - 100 / (1 + avg_gain / avg_loss), 1)


def compute_indicators(history, price, volume, volume_history=None) -> dict:
    """`history` = prior prices (oldest first, excluding the current tick)."""
    series = list(history) + [price]
    ind: dict = {}
    if len(series) >= 5:
        sma5 = sum(series[-5:]) / 5
        ind["sma5"] = round(sma5, 2)
        ind["price_vs_sma5_pct"] = round((price / sma5 - 1) * 100, 2)
    if len(series) >= 10:
        ind["sma10"] = round(sum(series[-10:]) / 10, 2)
    if len(series) >= 6:
        ind["momentum_5"] = round((series[-1] / series[-6] - 1) * 100, 2)
    rsi = rsi_wilder(series)
    if rsi is not None:
        ind["rsi14"] = rsi
    vh = [v for v in (volume_history or []) if v > 0][-20:]
    if len(vh) >= 3:
        ind["volume_ratio"] = round(volume / (sum(vh) / len(vh)), 2)
    return ind


# ----------------------------------------------------------------- news sentiment

_POS = {
    "beat", "beats", "surge", "surges", "record", "growth", "upgrade", "upgraded", "raises",
    "strong", "secures", "wins", "profit", "expands", "approval", "approved", "rally",
    "gains", "outperform", "boosts", "soars", "partnership", "breakthrough", "exceeds",
}
_NEG = {
    "miss", "misses", "cut", "cuts", "trims", "probe", "inquiry", "lawsuit", "delay",
    "delayed", "downgrade", "downgraded", "plunge", "plunges", "falls", "weak", "recall",
    "investigation", "fraud", "breach", "warning", "layoffs", "bankruptcy", "slump", "loss",
    "losses", "fine", "fined", "halts", "concerns", "decline", "declines", "shortfall",
}


def headline_sentiment(headlines):
    """Lexicon score in [-1, 1] (mean over headlines). Independent of the LLM on purpose."""
    if not headlines:
        return 0.0, []
    scores = []
    for h in headlines:
        words = re.findall(r"[a-z]+", h.lower())
        pos = sum(w in _POS for w in words)
        neg = sum(w in _NEG for w in words)
        scores.append((pos - neg) / max(1, pos + neg))
    return round(sum(scores) / len(scores), 2), scores


def direction_votes(change, price, indicators, headlines) -> dict:
    """Independent directional votes: +1 bullish, -1 bearish, 0 neutral."""
    votes = {"price_change": 1 if change >= 0.5 else -1 if change <= -0.5 else 0}
    pv = indicators.get("price_vs_sma5_pct")
    if pv is not None:
        votes["trend_vs_sma5"] = 1 if pv > 0.3 else -1 if pv < -0.3 else 0
    if headlines:
        score, _ = headline_sentiment(headlines)
        votes["news_sentiment"] = 1 if score >= 0.25 else -1 if score <= -0.25 else 0
    return votes


# ----------------------------------------------------------------- asset matching

_SUFFIX = {"ltd", "limited", "inc", "corp", "corporation", "co", "plc", "llc", "the",
           "company", "group", "holdings"}


def asset_key(name: str) -> str:
    return " ".join(t for t in re.findall(r"[a-z0-9]+", (name or "").lower()) if t not in _SUFFIX)


def matches_asset(asset: str, *texts) -> bool:
    key = asset_key(asset)
    if not key:
        return False
    for t in texts:
        norm = " ".join(re.findall(r"[a-z0-9]+", (t or "").lower()))
        if re.search(r"\b" + re.escape(key) + r"\b", norm):
            return True
    return False


# ----------------------------------------------------------------- news loading


def load_headlines(asset: str, as_of=None, limit: int = 3, path: Path | None = None):
    """Latest headlines for `asset`, never newer than `as_of` (prevents look-ahead)."""
    import config
    path = Path(path or config.NEWS_FILE)
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
        df = df[df["asset"].map(asset_key) == asset_key(asset)].copy()
        df["_ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
        cutoff = parse_ts(as_of)
        if cutoff is not None:
            df = df[df["_ts"].isna() | (df["_ts"] <= cutoff)]
        return df.sort_values("_ts", ascending=False)["headline"].astype(str).head(limit).tolist()
    except Exception:
        return []

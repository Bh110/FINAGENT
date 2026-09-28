import os
import tempfile

import pandas as pd

from tests import _fakes  # noqa: F401  (sets sys.path)
import config
from core import aggregate, backtest, policy, scoring, signals as S


# ---------------------------------------------------------------- signals
def test_normalize_ok():
    n = S.normalize_signal({"asset": " Apex ", "price": "150", "change_percent": 2.4,
                            "volume": 1000, "timestamp": "2026-09-24 22:55:10"})
    assert n["asset"] == "Apex" and n["price"] == 150.0
    assert n["timestamp"] == "2026-09-24T22:55:10"


def test_normalize_rejects_bad_input():
    good = {"asset": "A", "price": 1, "change_percent": 0, "volume": 1}
    for bad in ({**good, "price": 0}, {**good, "price": "abc"}, {**good, "asset": " "},
                {**good, "volume": -1}, {**good, "change_percent": float("nan")},
                {"asset": "A"}):
        try:
            S.normalize_signal(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted bad signal {bad}")


def test_rsi():
    assert S.rsi_wilder([1, 2, 3]) is None
    assert S.rsi_wilder(list(range(1, 30))) == 100.0
    assert S.rsi_wilder(list(range(30, 1, -1))) == 0.0
    mixed = [10, 11, 10.5, 11.5, 11, 12, 11.5, 12.5, 12, 13, 12.5, 13.5, 13, 14, 13.5, 14.5]
    assert 50 < S.rsi_wilder(mixed) < 100


def test_indicators():
    ind = S.compute_indicators([100, 101, 102, 103], 104, 2000, [1000, 1000, 1000, 1000])
    assert ind["sma5"] == 102.0 and ind["price_vs_sma5_pct"] > 0
    assert ind["volume_ratio"] == 2.0
    assert "rsi14" not in ind
    assert S.compute_indicators([], 10, 5) == {}


def test_headline_sentiment_and_votes():
    pos, _ = S.headline_sentiment(["Company beats estimates on record growth"])
    neg, _ = S.headline_sentiment(["Regulators open inquiry, analysts downgrade"])
    mixed, _ = S.headline_sentiment(["Beats revenue but cuts guidance"])
    assert pos > 0.5 and neg < -0.5 and mixed == 0.0
    assert S.headline_sentiment([]) == (0.0, [])
    v = S.direction_votes(2.0, 100, {"price_vs_sma5_pct": 1.0}, ["record growth"])
    assert v == {"price_change": 1, "trend_vs_sma5": 1, "news_sentiment": 1}
    assert S.direction_votes(0.1, 100, {}, [])["price_change"] == 0


def test_asset_matching():
    assert S.matches_asset("Apex Technologies Ltd.", "Apex Technologies Ltd.")
    assert S.matches_asset("Apex Technologies Ltd.", "x", "about apex technologies today")
    assert not S.matches_asset("Apex Technologies Ltd.", "Nova Energy Corp.")
    assert not S.matches_asset("", "anything")
    assert S.asset_key("Nova Energy Corp.") == "nova energy"


def test_load_headlines_no_lookahead():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "news.csv")
        pd.DataFrame({"timestamp": ["2026-09-24 10:00:00", "2026-09-24 12:00:00"],
                      "asset": ["Apex Technologies Ltd."] * 2,
                      "headline": ["early", "late"]}).to_csv(p, index=False)
        assert S.load_headlines("Apex Technologies Ltd.", "2026-09-24 11:00:00", path=p) == ["early"]
        assert S.load_headlines("Apex Technologies Ltd.", None, path=p) == ["late", "early"]
        assert S.load_headlines("Other", None, path=p) == []
        assert S.load_headlines("x", None, path=os.path.join(d, "missing.csv")) == []


# ---------------------------------------------------------------- scoring
def test_signal_strength():
    assert scoring.signal_strength(0) == 15.0
    assert scoring.signal_strength(5, 2) == 100.0
    assert scoring.signal_strength(-10, None) > scoring.signal_strength(1, None)


def test_agreement():
    assert scoring.agreement_score("bullish", {"a": 1, "b": 1})[0] == 100.0
    assert scoring.agreement_score("bullish", {"a": -1, "b": -1})[0] == 0.0
    s, agree, conflict = scoring.agreement_score("bullish", {"price": 1, "news": -1, "t": 0})
    assert agree == ["price"] and conflict == ["news"] and s == 50.0
    assert scoring.agreement_score("neutral", {"a": 1})[2] == ["a"]
    assert scoring.agreement_score("bullish", {})[0] == 50.0


def test_evidence_quality():
    assert scoring.evidence_quality([]) == 0.0
    good = scoring.evidence_quality([{"relevance": 0.55, "asset_match": True}])
    other = scoring.evidence_quality([{"relevance": 0.55, "asset_match": False}])
    assert good == 100.0 and other == 50.0


def test_blend_caps_and_penalty():
    r = scoring.blend(100, 100, 100, 100, 0, True)
    assert r["final"] == 100
    r = scoring.blend(100, 100, 100, 0, 0, False)
    assert r["final"] == config.NO_EVIDENCE_CAP and r["capped_no_evidence"]
    a = scoring.blend(80, 60, 60, 60, 0, True)["final"]
    b = scoring.blend(80, 60, 60, 60, 2, True)["final"]
    assert a - b == 16
    assert abs(sum(config.SCORE_WEIGHTS.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------- policy
def test_levels_and_actions():
    assert [policy.confidence_level(c) for c in (39, 40, 70, 71)] == ["Low", "Medium", "Medium", "High"]
    assert policy.decide_action("bullish", 71) == "BUY"
    assert policy.decide_action("bearish", 71) == "SELL"
    assert policy.decide_action("bullish", 70) == "WATCH"
    assert policy.decide_action("neutral", 55) == "HOLD"
    assert policy.decide_action("bullish", 39) == "WATCH"


def test_alert_boundary_matches_high_level():
    assert not policy.is_alert(70, 70) and policy.is_alert(71, 70)
    assert policy.confidence_level(70) != "High"     # consistent: no alert on a 'Medium' card


def test_evidence_and_flags():
    norm = {"price": 150.0, "change_percent": 2.4, "volume": 1250000, "asset": "Apex"}
    ev = policy.build_evidence(norm, {"rsi14": 60.0}, ["Apex beats"], [
        {"content": "text " * 100, "source": "a.pdf", "page": "2", "relevance": 0.5}], 0.5)
    assert 2 <= len(ev) <= 3 and "a.pdf p.2" in ev[-1]
    analysis = {"sentiment": "bullish", "risk_flags": ["x"], "conflicts": ["news_sentiment"],
                "engine": "llm"}
    flags = policy.build_risk_flags(norm, analysis, {"rsi14": 75.0, "volume_ratio": 0.5}, [],
                                    True, "SELL", "BUY")
    text = " | ".join(flags)
    for needle in ("news sentiment", "overbought", "Weak volume", "data gap", "Reverses"):
        assert needle in text, needle
    assert "Knowledge base unavailable" in " ".join(policy.build_risk_flags(
        norm, analysis, {}, [], False, "", "WATCH"))


# ---------------------------------------------------------------- aggregate / backtest
def _art(asset, sentiment, conf, action, ts="2026-09-24T10:00:00", price=100.0):
    return {"decision": {"asset": asset, "confidence": conf, "action": action},
            "analysis": {"sentiment": sentiment},
            "signal": {"timestamp": ts, "price": price}}


def test_portfolio_summary():
    rows = aggregate.portfolio_summary([_art("A", "bullish", 80, "BUY"),
                                        _art("A", "bearish", 30, "WATCH"),
                                        _art("B", "bearish", 90, "SELL")])
    a = next(r for r in rows if r["asset"] == "A")
    assert a["consensus"] == "bullish" and a["bullish"] == 1 and a["bearish"] == 1
    assert next(r for r in rows if r["asset"] == "B")["consensus"] == "bearish"


def test_backtest():
    market = pd.DataFrame({
        "timestamp": [f"2026-09-24 10:0{i}:00" for i in range(6)],
        "asset": ["A"] * 6, "price": [100, 101, 102, 103, 99, 98]})
    arts = [_art("A", "bullish", 80, "BUY", "2026-09-24T10:00:00", 100.0),   # +3% -> hit
            _art("A", "bearish", 60, "WATCH", "2026-09-24T10:01:00", 101.0),  # 99/101 -> hit
            _art("A", "neutral", 60, "HOLD", "2026-09-24T10:02:00", 102.0),   # skipped
            _art("A", "bullish", 60, "WATCH", "2026-09-24T10:04:00", 99.0)]   # no future -> skipped
    res, summary = backtest.evaluate(arts, market, horizon=3)
    assert summary["evaluated"] == 2 and summary["hit_rate_%"] == 100.0
    assert summary["hit_rate_high_conf_%"] == 100.0
    assert backtest.evaluate([], market)[1] == {"evaluated": 0}

import json

from tests import _fakes
from tests._fakes import FakeSearch, doc

GOOD = json.dumps({"sentiment": "bullish", "rationale": "Strong momentum.",
                   "hypothesis": "Cloud demand drives near-term upside.",
                   "time_horizon": "short-term", "confidence": 85,
                   "risk_flags": ["Regulatory inquiry"]})

SIGNAL = {"asset": "Apex Technologies Ltd.", "price": 150.0, "change_percent": 3.2,
          "volume": 2000000, "timestamp": "2026-09-24 23:00:00",
          "history": [140, 141, 143, 144, 146, 147, 148, 149, 148, 149, 150, 149, 150, 149.5, 150],
          "volume_history": [1000000] * 15,
          "headlines": ["Apex beats estimates on record growth"]}


def run(signal=SIGNAL, llm_reply=GOOD, results=None, use_llm=True, threshold=70, prior=""):
    llm, search = _fakes.install(llm_reply, FakeSearch(results if results is not None else [[doc()]]))
    import importlib
    import agents.nodes, agents.graph
    importlib.reload(agents.nodes)
    importlib.reload(agents.graph)
    graph = agents.graph.build_finagent_graph()
    art, errors = agents.graph.analyze_signal(graph, signal, threshold, use_llm, prior)
    return art, errors, llm, search


def test_happy_path():
    art, errors, llm, search = run()
    d, a = art["decision"], art["analysis"]
    assert d["action"] in {"BUY", "SELL", "HOLD", "WATCH"}
    assert a["engine"] == "llm" and a["sentiment"] == "bullish"
    assert 0 <= d["confidence"] <= 100 and d["confidence"] == a["confidence"]
    assert 2 <= len(d["evidence"]) <= 3
    order = [t.split("]")[0] for t in art["trace"]]
    assert order[0] == "[Ingestion Agent" and order[-1] == "[Decision Agent"
    assert order.index("[Retrieval Agent") < order.index("[Analysis Agent") < order.index("[Decision Agent")
    assert any("Retrieval Agent" in t for t in art["trace"])
    assert len(search.calls) == 1 and search.calls[0]["asset"] == SIGNAL["asset"]
    assert art["retrieved_documents"][0]["source"] == "apex.pdf"
    assert errors == []


def test_alert_only_above_threshold():
    art, *_ = run(threshold=10)
    assert art["alert"] is True
    art, *_ = run(threshold=100)
    assert art["alert"] is False


def test_invalid_signal_stops_pipeline():
    art, errors, llm, search = run(signal={"asset": "X", "price": "abc",
                                           "change_percent": 1, "volume": 1})
    assert art is None and errors and "rejected" in errors[0]
    assert llm.calls == 0 and search.calls == []       # nothing downstream ran


def test_llm_garbage_falls_back():
    art, errors, *_ = run(llm_reply="I think it goes up!!")
    assert art["analysis"]["engine"] == "rule-based (LLM fallback)"
    assert any("LLM call failed" in e for e in errors)
    assert any("LLM not used" in f for f in art["decision"]["risk_flags"])
    assert art["decision"]["confidence"] <= 70 and art["alert"] is False   # rule-based never 'High'
    assert art["decision"]["action"] in ("WATCH", "HOLD")


def test_llm_exception_falls_back():
    art, errors, *_ = run(llm_reply=TimeoutError("boom"))
    assert art is not None and art["analysis"]["engine"].startswith("rule-based")
    assert any("TimeoutError" in e for e in errors)


def test_quant_only_mode_skips_llm():
    art, errors, llm, _ = run(use_llm=False)
    assert llm.calls == 0 and art["analysis"]["engine"] == "rule-based (quant-only mode)"
    assert errors == []


def test_missing_kb_degrades_and_does_not_retry():
    art, errors, _, search = run(results=[FileNotFoundError("Knowledge base not built")])
    assert art is not None and len(search.calls) == 1
    assert any("Retrieval fallback" in e for e in errors)
    flags = " ".join(art["decision"]["risk_flags"])
    assert "Knowledge base unavailable" in flags
    assert art["analysis"]["score"]["capped_no_evidence"] or art["decision"]["confidence"] <= 55


def test_retry_loop_when_evidence_weak_and_confidence_low():
    weak = json.dumps({"sentiment": "bullish", "hypothesis": "Maybe up.", "confidence": 10,
                       "time_horizon": "intraday"})
    quiet = {**SIGNAL, "change_percent": 0.1, "history": [], "volume_history": [],
             "headlines": ["Nothing new"]}
    art, _, _, search = run(signal=quiet, llm_reply=weak, results=[[], [doc(rel=0.4)]])
    assert len(search.calls) == 2                        # retried exactly once
    assert "RETRY" in " ".join(art["trace"])
    assert search.calls[1]["min_relevance"] < search.calls[0]["min_relevance"]
    assert len(art["retrieved_documents"]) == 1          # second pass found evidence


def test_retry_is_bounded():
    weak = json.dumps({"sentiment": "bullish", "hypothesis": "Maybe.", "confidence": 5})
    quiet = {**SIGNAL, "change_percent": 0.1, "history": [], "volume_history": [], "headlines": []}
    _, _, _, search = run(signal=quiet, llm_reply=weak, results=[[]])
    assert len(search.calls) == 2                        # never loops forever


def test_prior_action_reversal_flagged():
    art, *_ = run(threshold=100, prior="SELL")
    if art["decision"]["action"] == "BUY":
        assert any("Reverses" in f for f in art["decision"]["risk_flags"])


def test_llm_output_validation():
    import agents.nodes as N
    assert N.parse_llm_json('noise {"a": 1} noise') == {"a": 1}
    assert N.parse_llm_json("nope") is None
    v = N.validate_llm_output({"sentiment": "Bullish!", "hypothesis": "h", "confidence": "85%",
                               "time_horizon": "Long-term"})
    assert v["sentiment"] == "bullish" and v["llm_confidence"] == 85
    assert v["time_horizon"].startswith("medium")
    assert N.validate_llm_output({"sentiment": "bullish"}) is None
    assert N.validate_llm_output({"hypothesis": "h", "confidence": 900})["llm_confidence"] == 100

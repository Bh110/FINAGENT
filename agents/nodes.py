"""The four agents. Each is a LangGraph node: (state) -> partial state update."""
import json
import logging
import re

import config
from core import policy, scoring
from core import signals as sig

log = logging.getLogger("finagent")
alert_log = logging.getLogger("finagent.alert")


# =====================================================================================
# 1. INGESTION AGENT
# =====================================================================================
def ingestion_node(state):
    raw = state.get("signal", {})
    try:
        n = sig.normalize_signal(raw)
    except ValueError as e:
        return {
            "valid": False,
            "errors": [f"Ingestion rejected the signal: {e}. Pipeline stopped."],
            "trace": [f"[Ingestion Agent] REJECTED signal: {e}"],
        }

    if not n["headlines"]:  # only headlines published up to this tick (no look-ahead)
        n["headlines"] = sig.load_headlines(n["asset"], as_of=n["timestamp"])

    indicators = sig.compute_indicators(n["history"], n["price"], n["volume"], n["volume_history"])
    news_score, _ = sig.headline_sentiment(n["headlines"])
    votes = sig.direction_votes(n["change_percent"], n["price"], indicators, n["headlines"])
    history_len = len(n.pop("history"))
    n.pop("volume_history")
    n.update(indicators=indicators, news_score=news_score, votes=votes, history_len=history_len)

    query = f"{n['asset']} earnings outlook risks competition regulation"
    if n["headlines"]:
        query += f" {n['headlines'][0]}"

    return {
        "normalized": n, "valid": True, "query": query, "retries": 0, "kb_available": True,
        "trace": [
            f"[Ingestion Agent] normalised {n['asset']} @ {n['timestamp']} | "
            f"{n['change_percent']:+.1f}% | vol {n['volume']:,} | history={history_len} ticks",
            f"[Ingestion Agent] indicators={indicators or 'n/a (not enough history)'} | "
            f"headlines={len(n['headlines'])} (tone {news_score:+.2f}) | votes={votes}",
        ],
    }


# =====================================================================================
# 2. RETRIEVAL AGENT
# =====================================================================================
def retrieval_node(state):
    from rag.rag_engine import search_knowledge_base

    n = state["normalized"]
    retries = state.get("retries", 0)
    query, min_rel = state["query"], config.MIN_RELEVANCE
    tag = ""
    if retries > 0:  # retry pass: refine query with the hypothesis, relax the threshold
        query = f"{query} {state.get('analysis', {}).get('hypothesis', '')}".strip()
        min_rel, tag = config.RETRY_MIN_RELEVANCE, " [RETRY: refined query, relaxed threshold]"

    trace = [f"[Retrieval Agent] tool=search_knowledge_base(query='{query[:80]}...', "
             f"asset='{n['asset']}', min_relevance={min_rel}){tag}"]
    try:
        docs = search_knowledge_base.invoke(
            {"query": query, "asset": n["asset"], "k": config.TOP_K, "min_relevance": min_rel})
    except FileNotFoundError as e:
        return {"retrieved_documents": [], "retries": retries + 1, "kb_available": False,
                "errors": [f"Retrieval fallback: {e} Continuing without document context."],
                "trace": trace + ["[Retrieval Agent] knowledge base unavailable -> no context"]}
    except Exception as e:
        log.exception("retrieval failed")
        return {"retrieved_documents": [], "retries": retries + 1, "kb_available": False,
                "errors": [f"Retrieval fallback: {type(e).__name__}: {e}. "
                           "Continuing without document context."],
                "trace": trace + ["[Retrieval Agent] error -> no context"]}

    if docs:
        summary = ", ".join(f"{d['source']} p.{d['page']} (rel {d['relevance']:.2f})" for d in docs)
        trace.append(f"[Retrieval Agent] returned {len(docs)} passage(s): {summary}")
    else:
        trace.append("[Retrieval Agent] no passage above the relevance threshold")
    return {"retrieved_documents": docs, "retries": retries + 1, "kb_available": True,
            "trace": trace}


# =====================================================================================
# 3. ANALYSIS AGENT
# =====================================================================================
SYSTEM_PROMPT = (
    "You are the Analysis Agent in a multi-agent financial system. Analyse the market signal "
    "using ONLY the data provided. Do not invent facts. Respond with a single JSON object "
    "with exactly these keys: sentiment (one of bullish, bearish, neutral); rationale (1-2 "
    "sentences); hypothesis (one concise statement of the opportunity or risk); time_horizon "
    "(one of intraday, short-term, medium-term); confidence (integer 0-100, lower it when "
    "evidence is weak or contradictory); risk_flags (list of up to 3 short strings)."
)
HUMAN_PROMPT = (
    "SIGNAL: asset={asset}, price={price}, change={change}%, volume={volume}\n"
    "TECHNICAL INDICATORS: {indicators}\n"
    "INDEPENDENT VOTES (+1 bullish, -1 bearish): {votes}\n"
    "NEWS (tone {news_score}):\n{news}\n\n"
    "RETRIEVED DOCUMENT EVIDENCE:\n{context}"
)


def parse_llm_json(text):
    """Tolerant JSON extraction: plain JSON, or the first {...} block in the text."""
    for candidate in (text, (re.search(r"\{.*\}", text or "", re.S) or [None])[0]):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except (ValueError, TypeError):
            continue
    return None


def normalize_horizon(text) -> str:
    t = str(text).lower()
    if "intraday" in t or "day trade" in t:
        return config.HORIZONS[0]
    if "short" in t or "1-5" in t:
        return config.HORIZONS[1]
    return config.HORIZONS[2]  # medium / long / unknown -> medium


def validate_llm_output(d):
    if not isinstance(d, dict) or not d.get("hypothesis"):
        return None
    s = str(d.get("sentiment", "")).lower().strip()
    sentiment = "bullish" if s.startswith("bull") else "bearish" if s.startswith("bear") else "neutral"
    try:
        conf = int(round(float(re.sub(r"[^\d.]", "", str(d.get("confidence", 40))) or 40)))
    except ValueError:
        conf = 40
    flags = d.get("risk_flags", [])
    return {
        "sentiment": sentiment,
        "rationale": str(d.get("rationale", "")).strip(),
        "hypothesis": str(d["hypothesis"]).strip(),
        "time_horizon": normalize_horizon(d.get("time_horizon", "")),
        "llm_confidence": max(0, min(100, conf)),
        "risk_flags": [str(f) for f in flags][:3] if isinstance(flags, list) else [],
    }


def rule_based_analysis(n):
    """Quant-only analysis from the independent votes (used in fallback / quant-only mode)."""
    total = sum(n["votes"].values())
    sentiment = "bullish" if total > 0 else "bearish" if total < 0 else "neutral"
    return {
        "sentiment": sentiment,
        "rationale": f"Rule-based: independent votes {n['votes']} sum to {total:+d}.",
        "hypothesis": f"{n['asset']} shows a {sentiment} short-term bias "
                      f"({n['change_percent']:+.1f}% on volume {n['volume']:,}).",
        "time_horizon": config.HORIZONS[1],
        "llm_confidence": min(70, 40 + 10 * abs(total)),
        "risk_flags": [],
    }


def _call_llm(n, docs):
    from langchain_core.prompts import ChatPromptTemplate
    from llm.model import json_llm

    context = "\n\n".join(
        f"[{i}] {d['source']} p.{d['page']} (relevance {d['relevance']:.2f}): {d['content'][:600]}"
        for i, d in enumerate(docs, 1)) or "No document evidence available."
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])
    return (prompt | json_llm).invoke({
        "asset": n["asset"], "price": n["price"], "change": n["change_percent"],
        "volume": n["volume"], "indicators": n["indicators"] or "not enough history",
        "votes": n["votes"], "news_score": n["news_score"],
        "news": "\n".join(f"- {h}" for h in n["headlines"]) or "- none",
        "context": context,
    }).content


def analysis_node(state):
    n = state["normalized"]
    docs = state.get("retrieved_documents", [])
    errors, raw, result = [], "", None

    if state.get("use_llm", True):
        engine = "llm"
        try:
            raw = _call_llm(n, docs)
            result = validate_llm_output(parse_llm_json(raw))
            if result is None:
                raise ValueError("model returned unusable JSON")
        except Exception as e:
            engine = "rule-based (LLM fallback)"
            errors.append(f"Analysis fallback: LLM call failed ({type(e).__name__}: {e}). "
                          "Used rule-based analysis instead.")
    else:
        engine = "rule-based (quant-only mode)"

    if result is None:
        result = rule_based_analysis(n)

    # ---- explainable confidence: LLM opinion is only one of four components
    sig_strength = scoring.signal_strength(n["change_percent"], n["indicators"].get("volume_ratio"))
    agr, agree, conflicts = scoring.agreement_score(result["sentiment"], n["votes"])
    evidence_q = scoring.evidence_quality(docs)
    score = scoring.blend(result["llm_confidence"], sig_strength, agr, evidence_q,
                          len(conflicts), bool(docs))

    if engine != "llm" and score["final"] > config.RULE_BASED_CAP:
        score["final"], score["capped_rule_based"] = config.RULE_BASED_CAP, True

    analysis = {**result, "confidence": score["final"], "score": score, "engine": engine,
                "agreeing": agree, "conflicts": conflicts,
                "raw_analysis": raw or json.dumps(result)}
    c = score["components"]
    return {
        "analysis": analysis, "errors": errors,
        "trace": [
            f"[Analysis Agent] engine={engine} | sentiment={result['sentiment']} | "
            f"horizon={result['time_horizon']}",
            f"[Analysis Agent] confidence {result['llm_confidence']} (LLM) -> {score['final']} "
            f"(signal {c['signal']}, agreement {c['agreement']}, evidence {c['evidence']}, "
            f"conflict penalty -{score['conflict_penalty']}"
            f"{', capped: no evidence' if score['capped_no_evidence'] else ''}"
            f"{', capped: rule-based' if score['capped_rule_based'] else ''})",
        ],
    }


# =====================================================================================
# 4. DECISION AGENT
# =====================================================================================
def decision_node(state):
    n, a = state["normalized"], state["analysis"]
    docs = state.get("retrieved_documents", [])
    conf = a["confidence"]
    threshold = state.get("threshold", config.DEFAULT_ALERT_THRESHOLD)
    action = policy.decide_action(a["sentiment"], conf)

    decision = {
        "asset": n["asset"],
        "action": action,
        "confidence": conf,
        "confidence_level": policy.confidence_level(conf),
        "evidence": policy.build_evidence(n, n["indicators"], n["headlines"], docs, n["news_score"]),
        "risk_flags": policy.build_risk_flags(
            n, a, n["indicators"], docs, state.get("kb_available", True),
            state.get("prior_action", ""), action),
    }
    decision["raw_decision"] = json.dumps(decision, indent=2)

    alert = policy.is_alert(conf, threshold)
    trace = [f"[Decision Agent] {action} {n['asset']} ({decision['confidence_level']}, {conf}%) | "
             f"{len(decision['risk_flags'])} risk flag(s)"]
    if alert:
        msg = f"HIGH-CONFIDENCE ALERT: {action} {n['asset']} at {conf}% (> {threshold}%)"
        alert_log.warning(msg)
        print(f"🚨 {msg}")
        trace.append(f"[Decision Agent] alert emitted (threshold {threshold})")
    return {"decision": decision, "alert": alert, "trace": trace}

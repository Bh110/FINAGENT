"""Follow-up Q&A about a specific artefact (bonus feature). Answers are grounded in the
artefact's own data; if the LLM is down, a deterministic explanation is returned."""
import json

SYSTEM = (
    "You are FinAgent's explainer. Answer the user's question about ONE decision artefact using "
    "ONLY the artefact data below. Explain reasoning plainly, refer to the confidence "
    "breakdown when asked about confidence, and say so if the data does not contain the answer. "
    "Never give personal financial advice; this is a simulation.\n\nARTEFACT:\n{artefact}"
)


def compact(artefact: dict) -> str:
    a, d = artefact["analysis"], artefact["decision"]
    view = {
        "signal": {k: v for k, v in artefact["signal"].items() if k != "votes"},
        "votes": artefact["signal"].get("votes"),
        "decision": {k: d[k] for k in ("asset", "action", "confidence", "confidence_level",
                                       "evidence", "risk_flags")},
        "analysis": {k: a[k] for k in ("sentiment", "rationale", "hypothesis", "time_horizon",
                                       "engine", "score", "agreeing", "conflicts")},
        "retrieved": [{"source": r["source"], "page": r["page"], "relevance": r["relevance"],
                       "text": r["content"][:300]} for r in artefact["retrieved_documents"]],
        "errors": artefact["errors"],
    }
    return json.dumps(view, indent=1)[:5000]


def explain_confidence(artefact: dict) -> str:
    """Deterministic fallback: spells out exactly how the confidence number was built."""
    s = artefact["analysis"]["score"]
    c, w = s["components"], s["weights"]
    lines = [f"Confidence {artefact['decision']['confidence']}% was computed as:"]
    for key, label in (("llm", "LLM's own confidence"), ("signal", "Signal strength"),
                       ("agreement", "Agreement of independent indicators"),
                       ("evidence", "Quality of retrieved document evidence")):
        lines.append(f"- {label}: {c[key]} x weight {w[key]}")
    if s["conflict_penalty"]:
        lines.append(f"- Penalty for contradicting indicators: -{s['conflict_penalty']}")
    if s["capped_no_evidence"]:
        lines.append("- Capped because no document evidence was retrieved.")
    if s.get("capped_rule_based"):
        lines.append("- Capped because the LLM was not used (rule-based analysis cannot be High).")
    return "\n".join(lines)


def answer_followup(artefact: dict, history: list, question: str) -> str:
    try:
        from llm.model import llm
        messages = [("system", SYSTEM.format(artefact=compact(artefact)))]
        for role, text in history[-6:]:
            messages.append(("human" if role == "user" else "ai", text))
        messages.append(("human", question))
        return llm.invoke(messages).content.strip()
    except Exception as e:
        return (f"(LLM unavailable: {type(e).__name__}. Deterministic explanation instead.)\n\n"
                + explain_confidence(artefact))

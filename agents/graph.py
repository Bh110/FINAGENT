"""LangGraph orchestration with typed, conditional transitions.

  START -> ingestion --valid?--> retrieval -> analysis --evidence weak AND low confidence
                 |                   ^                     AND retry left AND KB up?--+
                 +--invalid--> END   +-------------------------------- yes ------------+
                                                       no -> decision -> END
"""
import time
import uuid
from datetime import datetime

from langgraph.graph import END, START, StateGraph

import config
from agents.nodes import analysis_node, decision_node, ingestion_node, retrieval_node
from agents.state import FinAgentState


def route_after_ingestion(state) -> str:
    return "retrieval" if state.get("valid") else "end"


def route_after_analysis(state) -> str:
    """Loop back to retrieval only when better evidence could plausibly help."""
    a = state["analysis"]
    weak_evidence = a["score"]["components"]["evidence"] < 50
    can_retry = state.get("retries", 0) <= config.MAX_RETRIEVAL_RETRIES
    if can_retry and state.get("kb_available", True) and weak_evidence \
            and a["confidence"] < config.LOW_CONF:
        return "retrieval"
    return "decision"


def build_finagent_graph():
    g = StateGraph(FinAgentState)
    g.add_node("ingestion", ingestion_node)
    g.add_node("retrieval", retrieval_node)
    g.add_node("analysis", analysis_node)
    g.add_node("decision", decision_node)

    g.add_edge(START, "ingestion")
    g.add_conditional_edges("ingestion", route_after_ingestion,
                            {"retrieval": "retrieval", "end": END})
    g.add_edge("retrieval", "analysis")
    g.add_conditional_edges("analysis", route_after_analysis,
                            {"retrieval": "retrieval", "decision": "decision"})
    g.add_edge("decision", END)
    return g.compile()


def analyze_signal(graph, signal, threshold=config.DEFAULT_ALERT_THRESHOLD,
                   use_llm=True, prior_action=""):
    """Run one signal through the graph. Returns (artefact | None, errors)."""
    t0 = time.perf_counter()
    result = graph.invoke({
        "signal": signal, "threshold": threshold, "use_llm": use_llm,
        "prior_action": prior_action or "", "trace": [], "errors": [],
    })
    errors = result.get("errors", [])
    if "decision" not in result:
        return None, errors
    artefact = {
        "id": uuid.uuid4().hex[:8],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "signal": result["normalized"],
        "analysis": result["analysis"],
        "decision": result["decision"],
        "retrieved_documents": result.get("retrieved_documents", []),
        "alert": result.get("alert", False),
        "trace": result.get("trace", []),
        "errors": errors,
    }
    return artefact, errors

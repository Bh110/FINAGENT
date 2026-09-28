import operator
from typing import Annotated, Any, Dict, List, TypedDict


class FinAgentState(TypedDict, total=False):
    """Shared LangGraph state. Agents communicate ONLY through this object."""
    # ---- inputs
    signal: Dict[str, Any]
    threshold: int
    use_llm: bool
    prior_action: str
    # ---- ingestion
    normalized: Dict[str, Any]
    valid: bool
    query: str
    # ---- retrieval
    retrieved_documents: List[Dict[str, Any]]
    retries: int
    kb_available: bool
    # ---- analysis / decision
    analysis: Dict[str, Any]
    decision: Dict[str, Any]
    alert: bool
    # ---- observability: append-only, agents can never overwrite each other's logs
    trace: Annotated[List[str], operator.add]
    errors: Annotated[List[str], operator.add]

"""Integration check: builds the index and runs a query. python -m scripts.smoke_rag"""
from rag.rag_engine import build_vectorstore, search_with_scores

store, n = build_vectorstore()
if store is None:
    raise SystemExit("No documents found in rag/documents")
print(f"Indexed {n} chunks")
for r in search_with_scores("What are the major risks?", asset="Apex Technologies Ltd."):
    print(f"\n[{r['source']} p.{r['page']}] relevance={r['relevance']} asset_match={r['asset_match']}")
    print(r["content"][:300])

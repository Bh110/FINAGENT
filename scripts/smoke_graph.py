"""End-to-end run (needs Ollama + built index): python -m scripts.smoke_graph"""
from agents.graph import analyze_signal, build_finagent_graph

art, errors = analyze_signal(build_finagent_graph(), {
    "asset": "Apex Technologies Ltd.", "price": 150.0, "change_percent": 2.4,
    "volume": 1250000, "source": "manual"})
print("ERRORS:", errors)
if art:
    print("DECISION:", art["decision"]["action"], art["decision"]["confidence"])
    print("\n".join(art["trace"]))

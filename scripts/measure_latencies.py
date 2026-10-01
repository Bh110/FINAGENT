"""Measure per-node and end-to-end latencies with the real stack.

Run from the project root:
    .venv/Scripts/python.exe -m scripts.measure_latencies
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when the script is run directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from agents.graph import build_finagent_graph, analyze_signal
import config

print("=" * 60)
print("FINAGENT REAL LATENCY & SYSTEM MEASUREMENT")
print("=" * 60)

# Environment specs
print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
print(f"Machine: {platform.machine()} | Processor: {platform.processor()}")
print(f"Python: {platform.python_version()} ({sys.executable})")
print(f"CPU Cores (Logical): {os.cpu_count()}")
print(f"LLM Model: {config.LLM_MODEL} (Ollama)")
print(f"Embedding Model: {config.EMBED_MODEL}")
print("=" * 60)

graph = build_finagent_graph()

# Load real signals from data/market_signals.csv
df = pd.read_csv(config.DEFAULT_SIGNAL_FILE)
df["_ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)

test_rows = [
    df[df["asset"] == "Apex Technologies Ltd."].iloc[0],
    df[df["asset"] == "Nova Energy Corp."].iloc[0],
    df[df["asset"] == "Helix Pharma Inc."].iloc[0],
]

results = []

for idx, row in enumerate(test_rows, 1):
    asset = row["asset"]
    print(f"\n[{idx}/3] Benchmarking signal for {asset}...")
    
    # Build signal with history
    h = df[(df["asset"] == asset) & (df["_ts"] < row["_ts"])].tail(60)
    prices = h["price"].astype(float).tolist()
    vols = h["volume"].astype(float).tolist()
    
    signal = {
        "asset": asset,
        "price": float(row["price"]),
        "change_percent": float(row["change_percent"]),
        "volume": int(row["volume"]),
        "timestamp": str(row["timestamp"]),
        "source": "benchmark",
        "history": prices,
        "volume_history": vols,
    }
    
    t0 = time.perf_counter()
    art, errors = analyze_signal(graph, signal, threshold=70, use_llm=True)
    total_time = round(time.perf_counter() - t0, 3)
    
    if art:
        dec = art["decision"]
        ana = art["analysis"]
        ret = art.get("retrieved_documents", [])
        print(f"   Action: {dec['action']} | Confidence: {dec['confidence']}% ({dec['confidence_level']})")
        print(f"   Sentiment: {ana['sentiment']} | Horizon: {ana['time_horizon']}")
        print(f"   Retrieved Passages: {len(ret)} (Top relevance: {ret[0]['relevance'] if ret else 'N/A'})")
        print(f"   Total Latency: {total_time:.2f}s")
        print(f"   Risk Flags: {len(dec['risk_flags'])}")
        
        results.append({
            "asset": asset,
            "action": dec["action"],
            "confidence": dec["confidence"],
            "sentiment": ana["sentiment"],
            "retrieved_count": len(ret),
            "top_relevance": ret[0]["relevance"] if ret else 0.0,
            "total_latency_s": total_time,
            "errors": errors,
        })
    else:
        print(f"   FAILED: {errors}")

# Also benchmark quant-only mode (rule-based, no LLM)
print("\n[Quant-Only] Benchmarking rule-based mode (use_llm=False)...")
t0 = time.perf_counter()
art_quant, _ = analyze_signal(graph, signal, threshold=70, use_llm=False)
quant_time = round(time.perf_counter() - t0, 4)
print(f"   Quant-Only Total Latency: {quant_time:.4f}s")
print(f"   Action: {art_quant['decision']['action']} | Confidence: {art_quant['decision']['confidence']}%")

summary = {
    "system": {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu_cores": os.cpu_count(),
        "llm": config.LLM_MODEL,
        "embeddings": config.EMBED_MODEL,
    },
    "runs": results,
    "avg_llm_latency_s": round(sum(r["total_latency_s"] for r in results) / len(results), 2) if results else 0,
    "quant_only_latency_s": quant_time,
}

output_path = Path("docs") / "benchmark_baseline.json"
output_path.write_text(json.dumps(summary, indent=2))
print(f"\nSaved benchmark results to {output_path}")

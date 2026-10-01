"""Dependency-free runner: `python -m tests.run_all` (pytest also works: `pytest tests`).

Pass --integration to also run tests that need the real FAISS index and embeddings.
"""
import importlib
import sys
import traceback

HERMETIC = ("tests.test_core", "tests.test_graph")
INTEGRATION = ("tests.test_rag_math",)  # needs real vectorstore + sentence-transformers

run_integration = "--integration" in sys.argv

failed = 0
suites = HERMETIC + (INTEGRATION if run_integration else ())
for suite_idx, mod in enumerate(suites):
    # Before running integration tests, evict any fake module stubs that test_graph
    # injected via tests._fakes.install() so the real rag.rag_engine is importable.
    if run_integration and mod in INTEGRATION:
        for fake_key in ("rag.rag_engine", "llm.model"):
            sys.modules.pop(fake_key, None)

    m = importlib.import_module(mod)
    for name in sorted(n for n in dir(m) if n.startswith("test_")):
        try:
            getattr(m, name)()
            print(f"PASS {mod}.{name}")
        except Exception:
            failed += 1
            print(f"FAIL {mod}.{name}")
            traceback.print_exc()

if not run_integration:
    skipped = sum(
        len([n for n in dir(importlib.import_module(m)) if n.startswith("test_")])
        for m in INTEGRATION
    )
    print(f"\n(Skipped {skipped} integration test(s)."
          " Re-run with --integration to include them.)")

print("\nFAILED:", failed)
sys.exit(1 if failed else 0)

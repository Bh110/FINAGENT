"""Dependency-free runner: `python -m tests.run_all` (pytest also works: `pytest tests`)."""
import importlib
import sys
import traceback

failed = 0
for mod in ("tests.test_core", "tests.test_graph"):
    m = importlib.import_module(mod)
    for name in sorted(n for n in dir(m) if n.startswith("test_")):
        try:
            getattr(m, name)()
            print(f"PASS {mod}.{name}")
        except Exception:
            failed += 1
            print(f"FAIL {mod}.{name}")
            traceback.print_exc()
print("\nFAILED:", failed)
sys.exit(1 if failed else 0)

"""Central configuration. Every tunable lives here (override via environment variables)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ------------------------------------------------------------------ LLM
# Open-source LLM served locally by Ollama. Declare this in your report.
LLM_MODEL = os.getenv("FINAGENT_LLM", "llama3.2:3b")
LLM_TEMPERATURE = 0
LLM_TIMEOUT_S = 90          # seconds before an LLM call is abandoned (-> fallback)
LLM_NUM_CTX = 4096

# ------------------------------------------------------------------ RAG
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DOCS_DIR = BASE_DIR / "rag" / "documents"
INDEX_DIR = BASE_DIR / "rag" / "vectorstore"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3
FETCH_K = 12                 # candidates fetched before asset-aware re-ranking
MIN_RELEVANCE = 0.25         # cosine similarity below this is discarded
RETRY_MIN_RELEVANCE = 0.15   # relaxed threshold on the retry pass

# ------------------------------------------------------------------ Data
DATA_DIR = BASE_DIR / "data"
DEFAULT_SIGNAL_FILE = DATA_DIR / "market_signals.csv"
NEWS_FILE = DATA_DIR / "news.csv"
ARTEFACT_FILE = DATA_DIR / "artefacts.json"
MAX_STORED_ARTEFACTS = 200

# ------------------------------------------------------------------ Decision policy
# PS: High > 70, Medium 40-70, Low < 40. Alert fires when confidence is ABOVE threshold.
HIGH_CONF = 70
LOW_CONF = 40
DEFAULT_ALERT_THRESHOLD = 70
MAX_RETRIEVAL_RETRIES = 1

# ------------------------------------------------------------------ Confidence blend
# final = w_llm*LLM + w_signal*signal_strength + w_agreement*agreement + w_evidence*evidence
SCORE_WEIGHTS = {"llm": 0.30, "signal": 0.25, "agreement": 0.25, "evidence": 0.20}
CONFLICT_PENALTY = 8         # points removed per contradicting indicator
MAX_CONFLICT_PENALTY = 16
NO_EVIDENCE_CAP = 55         # confidence cannot exceed this without document evidence
RULE_BASED_CAP = 70          # rule-based (no LLM) analysis can never reach 'High' -> no BUY/SELL

HORIZONS = ["intraday", "short-term (1-5 days)", "medium-term (1-4 weeks)"]

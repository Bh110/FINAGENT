# FinAgent — Real-Time AI Investment Decision System

Multi-agent pipeline (LangGraph + LangChain) that turns market signals into structured,
explainable decision artefacts, grounded in a FAISS knowledge base of financial documents.
**Simulation only. No trading. Not financial advice.**

## Run
```bash
ollama pull llama3.2:3b          # LLM (open-source, local)
pip install -r requirements.txt
streamlit run app.py             # first launch auto-builds the knowledge base
python -m tests.run_all          # 27 hermetic tests (no Ollama / GPU needed)
```
Override the model with `FINAGENT_LLM=mistral streamlit run app.py`. All tunables are in `config.py`.

## Architecture
```
START -> INGESTION --valid?--> RETRIEVAL -> ANALYSIS --(weak evidence AND conf < 40
              |                    ^                    AND retry left AND KB up)--+
              +-- invalid -> END   +----------- yes ------------------------------+
                                                    no -> DECISION -> END
```
| Agent | Role |
|---|---|
| Ingestion | validate + timestamp + normalise; SMA/RSI/volume-ratio/momentum; point-in-time news (no look-ahead); independent directional votes |
| Retrieval | `search_knowledge_base` LangChain tool: FAISS + MiniLM, cosine relevance scores, relevance cut-off, asset-aware re-ranking, source + page attribution |
| Analysis | LLM (JSON mode) -> sentiment, hypothesis, horizon; **confidence = blend of 4 components** (see below); rule-based fallback |
| Decision | deterministic policy -> BUY/SELL/HOLD/WATCH, evidence bullets (built from data, not LLM text), risk flags, alert |

Agents communicate only through the shared typed `FinAgentState`; `trace` and `errors` use append-only reducers.

### Confidence (explainable, not just "what the LLM said")
`final = 0.30·LLM + 0.25·signal strength + 0.25·indicator agreement + 0.20·evidence quality`
− 8 per contradicting indicator (max 16); capped at 55 with no document evidence; capped at 70
when the LLM was not used (rule-based analysis can never produce BUY/SELL). Weights live in
`config.SCORE_WEIGHTS`. This is a transparent heuristic, **not** a statistically calibrated
probability — the Backtest tab is how you sanity-check it.

### Policy (matches the problem statement)
High > 70, Medium 40-70, Low < 40. BUY/SELL require High; HOLD = neutral view at ≥ 40; WATCH otherwise.
Alert fires when confidence is **above** the threshold (same boundary as "High").

## Features
Live replay stream (auto-analyses each tick) · per-artefact trace, retrieved passages, error/fallback
events · confidence breakdown · portfolio consensus · backtesting vs later price moves · follow-up
chat about an artefact · JSON + Markdown export · artefacts persist across restarts.

## Known limitations (be upfront about these)
- Data is simulated (CSV replay), not a live feed; sample prices are a random walk.
- Confidence is a heuristic; hit rates on small simulated samples are not statistically meaningful.
- A 3B local model reasons weakly; the guardrails (validation, caps, deterministic policy) exist for that reason.
- Headline sentiment is a small lexicon, not a trained model.
- Single-user, single-process (Streamlit session); FAISS index is rebuilt in full on each build.
- FAISS is loaded with `allow_dangerous_deserialization=True` — only load indexes you built yourself.

## Layout
`app.py` UI · `config.py` · `agents/` (state, nodes, graph, chat) · `core/` (pure-Python signals,
scoring, policy, aggregate, backtest) · `rag/` · `llm/` · `data/` · `tests/` · `scripts/` (integration smoke tests)

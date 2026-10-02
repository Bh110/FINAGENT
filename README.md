FinAgent

Real-Time AI Investment Decision System — Multi-Agent · RAG · LangGraph

FinAgent is a local, multi-agent decision-support system that connects numerical market
signals with the qualitative context buried in financial documents. A price move alone
rarely explains itself; a 5% jump means something different next to "beats estimates and
raises guidance" than next to "regulatory inquiry opened." FinAgent brings both kinds of
evidence into one auditable pipeline instead of one opaque prompt.

**This is a simulation and research prototype.** It does not place trades, does not connect
to a brokerage, and its output is not financial advice.

> **Documentation status** This README documents the actual, verified
> state of the project. Section 0 below lists exactly what has been run and confirmed, and
> what is designed but not yet executed. Nothing in this document reports a number, metric,
> or test result that wasn't actually produced by the system. Where a planned capability
> (deeper evaluation, hybrid retrieval, a generated graph diagram) has not been built yet,
> it is marked **Not done — planned**, not described as if it exists.

---

0. What's actually verified vs. what's designed but not yet run

Being explicit about this is the point of this README, since there is no video or deck to
fall back on.

| Area Status                                                                             |                                                                 |
| --------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| Core architecture (4 agents, shared LangGraph state, conditional routing)               |  Built and running                                             |
| Hermetic test suite                                                                     |  Passing (`FAILED: 0` — see §15)                               |
| Local environment verified end-to-end (venv, Ollama, Streamlit launch)                  |  Done (Phase 0)                                                |
| RAG pipeline (FAISS + Sentence Transformers) running against real documents             |  Done — 8 indexed chunks in the demo environment               |
| One real manual analysis run, inspected                                                 |  Done — WATCH, 33% confidence, \~15s                           |
| One real backtest run                                                                   |  Done — 5 calls, 80% hit rate, +0.72% avg signed return        |
| Mermaid diagram generated from the *actual* compiled graph                              |  Not done — the diagram in §3 is hand-drawn, not auto-exported |
| Statistical evaluation (baselines, Wilson confidence intervals, ablations, calibration) |  Not done — only the single 5-call backtest above exists       |
| Retrieval benchmark (Recall\@k, MRR)                                                    |  Not done                                                      |
| Hybrid retrieval (BM25 + dense fusion)                                                  |  Not implemented — current retrieval is dense-only             |
| Structured trace schema / rotating file logs                                            |  Not done                                                      |
| Prompt-injection hardening and tests                                                    |  Not done                                                      |

If you are evaluating this project, §16 (Evaluation Results) is the one section with
numbers from an actual run. Everything beyond that in "evaluation science" is future work,
honestly labeled as such throughout this document.
1. Problem Definition

Financial information arrives at different levels of structure: numerical ticks (price,
change, volume), unstructured text (headlines), and semi-structured documents (reports,
filings). A useful decision-support system has to answer three questions:

1. What is happening in the incoming signal?
2. What does the available financial context say about the asset?
3. How strongly does the combined evidence support a decision?

FinAgent answers these through a staged pipeline rather than one language-model call. The
signal is normalised first; retrieval then locates relevant context; analysis combines the
signal with that context into a hypothesis; decision synthesis converts the hypothesis into
a structured, inspectable artefact.

This follows the FinAgent mock problem statement's requirements: LangChain, LangGraph, an
open-source vector store, an open-source embedding model, a dedicated retrieval agent,
structured decision artefacts, and observable agent execution.
2. System Overview

| Layer Technology Role  |                               |                                                                                                     |
| ---------------------- | ----------------------------- | --------------------------------------------------------------------------------------------------- |
| Interface              | Streamlit                     | Signal replay, manual analysis, artefact inspection, portfolio view, backtesting, Q&A               |
| Orchestration          | LangGraph                     | Multi-stage workflow; agents share one typed state object instead of hiding logic in one model call |
| Evidence               | FAISS + Sentence Transformers | Converts documents to vectors; local similarity search with source/page attribution                 |
| Reasoning              | Llama 3.2 3B (Ollama, local)  | Generates the sentiment/hypothesis proposal — not the final decision                                |
| Evaluation             | Backtest, portfolio, Ask tabs | Lets a reviewer question and check artefacts after the fact                                         |

3. Architecture
```
                         FinAgent

                  +---------------------+
                  |    Streamlit UI     |
                  +----------+----------+
                             |
                             v
                  +---------------------+
                  |   Signal Ingestion  |
                  |   and Normalisation |
                  +----------+----------+
                             |
                             v
                  +---------------------+
                  |     LangGraph       |
                  |  Shared Graph State |
                  +----------+----------+
                             |
                +------------+-------------+
                |                          |
                v                          v
       +----------------+        +----------------------+
       | Signal Analysis|        | Knowledge Base       |
       | and Hypothesis |<-------| Retrieval            |
       +--------+-------+        +----------+-----------+
                |                           |
                |                           v
                |                 +----------------------+
                |                 | Sentence Transformers|
                |                 +----------+-----------+
                |                            |
                |                            v
                |                 +----------------------+
                |                 | FAISS Vector Store   |
                |                 +----------------------+
                |
                v
       +----------------------+
       | Decision Synthesis   |
       +----------+-----------+
                  |
                  v
       +----------------------+
       | Decision Artefact    |
       +----------+-----------+
                  |
          +-------+--------+----------+
          |                |          |
          v                v          v
      Portfolio        Backtest     Ask

```

> **Note on this diagram:** this is hand-drawn for readability, not generated from the
> compiled graph object. Auto-generating it via `graph.get_graph().draw_mermaid()` is listed
> as not-done work in §0 and §18.

The important property is that the model is **one component**, not the whole system.
Retrieval, state management, confidence scoring, and the decision policy are explicit code,
not something the LLM is trusted to do correctly on its own.

3.1 Shared state

Agents communicate **only** through one typed state object — never by calling each other's
functions directly. This is what makes the trace and the "why did it do that" story
possible.

| Key Set by Read by Purpose                       |                          |                     |                                                               |
| ------------------------------------------------ | ------------------------ | ------------------- | ------------------------------------------------------------- |
| `signal`                                         | caller                   | Ingestion           | raw input before normalisation                                |
| `threshold`, `use_llm`, `prior_action`           | caller                   | Decision, Analysis  | run configuration                                             |
| `normalized`, `valid`, `query`                   | Ingestion                | Retrieval, Analysis | the cleaned signal and the query to search with               |
| `retrieved_documents`, `retries`, `kb_available` | Retrieval                | Analysis, Decision  | evidence passages, retry count, whether the KB responded      |
| `analysis`                                       | Analysis                 | Decision            | sentiment, hypothesis, horizon, confidence components         |
| `decision`, `alert`                              | Decision                 | UI                  | the final artefact and whether it crosses the alert threshold |
| `trace`, `errors`                                | every node (append-only) | UI                  | full execution log; no agent can overwrite another's entries  |

 3.2 Routing (the conditional edges)

**Ingestion → Retrieval**, only if the signal passes validation (non-empty asset, positive price, non-negative volume). An invalid signal routes straight to `END` with a logged reason — the pipeline never guesses at bad input.
**Analysis → Retrieval (retry, bounded to one pass)**, only when the retrieved evidence was weak *and* confidence came out low *and* the knowledge base was actually reachable. The retry refines the query using the hypothesis just produced and relaxes the relevance threshold slightly — it is a second, smarter attempt, not a blind repeat.
**Analysis → Decision**, the default path once there's nothing more useful retrieval can add.

This means the Analysis Agent only "calls" the Retrieval Agent indirectly, through this one
retry edge — the first pass always retrieves before analysing. We're flagging this
explicitly in §7's traceability matrix as **Partial (by design)** rather than claiming full
compliance with a stricter reading of the problem statement.

---

4. The Four Agents

Ingestion Agent

**Input:** raw signal dict. **Output:** `normalized`, `valid`, `query`.
Validates and timestamps the signal, builds price/volume history context for replayed data,
and constructs the initial retrieval query. On invalid input, it fails loudly and routes to
`END` rather than letting a bad signal flow downstream.

Retrieval Agent

**Input:** `query`, `normalized.asset`. **Output:** `retrieved_documents`, `kb_available`.
Embeds the query with Sentence Transformers (`all-MiniLM-L6-v2`), searches FAISS, and
returns passages with source file, page, and relevance score. If the knowledge base is
missing or errors, this is caught and logged as a fallback event — the pipeline continues
with empty evidence rather than crashing.

Analysis Agent

**Input:** `normalized`, `retrieved_documents`. **Output:** `analysis` (sentiment,
hypothesis, time horizon, confidence, risk flags).
Calls the local LLM with the signal and retrieved context, asking for structured output. If
the LLM is unreachable or returns unusable text, a rule-based fallback runs instead — the
pipeline never halts because the model failed.

Decision Agent

**Input:** `analysis`, `retrieved_documents`. **Output:** `decision`, `alert`.
Converts the hypothesis into one of **BUY / SELL / HOLD / WATCH**, assembles 2–3 evidence
bullets from real data (not LLM-written text, so they can't be hallucinated), computes risk
flags (contradicting indicators, missing evidence, LLM not used), and decides whether to
fire an alert.

5. Retrieval-Augmented Generation

Financial document
        |
        v
Document ingestion
        |
        v
Text extraction and chunking   (chunk size 1000, overlap 200 — see §17 config reference)
        |
        v
Sentence Transformer embedding (all-MiniLM-L6-v2)
        |
        v
FAISS vector index (local, persisted to disk)
        |
        v
Semantic similarity search, cosine relevance score
        |
        v
Ranked passages with source + page metadata
        |
        v
Analysis workflow

The demonstration knowledge base contains financial PDF material, including a synthetic
Apex Technologies Ltd. report. A query about major risks returns passages covering
competition, cybersecurity, regulation, customer concentration, supply chain, and
macroeconomic exposure — these are retrievable and verifiable directly through the
Knowledge Base tab, independent of whether the LLM is working.

**Why FAISS + Sentence Transformers:** both are open-source and run fully locally, which
keeps the retrieval stack self-contained and avoids a hosted vector database dependency
that a judge would need credentials for.

**On the relevance math:** cosine similarity between normalised embeddings relates to
squared L2 distance by `cos(θ) = 1 − ‖u − v‖² / 2`. Passages below a relevance floor are
discarded rather than always returning the top-k regardless of quality — a low-quality
match is treated as "no evidence," not forced evidence.

**Not done — planned:** a labelled retrieval benchmark (Recall\@k, MRR) and a hybrid
BM25-plus-dense comparison. The current corpus (a handful of PDFs, \~8 chunks) is also too
small for such a benchmark to be meaningful yet; this needs a larger, legitimately sourced
document set first.
6. Confidence: how the number is actually built

Confidence is treated as a separate analytical quantity, not a copy of whatever number the
LLM states. Four independent signals are blended:

| Component What it measures  |                                                                                                                        |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| LLM confidence              | The model's own stated confidence in its hypothesis                                                                    |
| Signal strength             | How large the price move is, and how it compares to volume                                                             |
| Agreement                   | How many independent indicators (price direction, trend vs. moving average, news tone) agree with the stated sentiment |
| Evidence quality            | How relevant the retrieved passages are, and whether they're actually about the asset in question                      |

Contradicting indicators subtract points. Confidence is **capped** when there is no document
evidence at all, and separately capped when the LLM was not used (so a rule-based fallback
can never produce a "High" confidence BUY or SELL — only the LLM path can reach that tier).

**Why this design:** an action and a confidence number answer two different questions — the
action says what the evidence currently points to, the confidence says how strongly it
points there. A result like the demonstrated **WATCH at 33%** is a feature, not a weak
answer: it tells a reviewer the evidence was present but didn't strongly support a
directional call, rather than forcing every signal into a confident-sounding BUY or SELL.

> **Exact weights and caps:** these live in `config.py` (`SCORE_WEIGHTS`,
> `CONFLICT_PENALTY`, `NO_EVIDENCE_CAP`, `RULE_BASED_CAP`, `HIGH_CONF`/`LOW_CONF`). Please
> read them directly from your `config.py` before quoting exact numbers in a presentation —
> this README intentionally doesn't repeat specific figures here that weren't re-verified
> after the Phase 0 pass, to avoid stating something that may have drifted from the running
> code.

**Policy:** High confidence is above 70, Medium is 40–70, Low is below 40 (per the problem
statement). BUY/SELL require High confidence; a neutral sentiment at Medium-or-above
produces HOLD; everything else is WATCH. An alert fires when confidence crosses the
configurable threshold, using the same boundary as "High" so the alert banner and the
confidence label never disagree.

**Honest caveat:** this is a transparent, explainable heuristic — not a statistically
calibrated probability. It has not been validated against enough outcomes to claim
calibration (see §0 and §16). Treat it as a ranking signal, not a probability of being
right.

7. Requirements Traceability

| Problem statement requirement Implementation Verified how Status                        |                                                 |                                                              |                                                            |
| --------------------------------------------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------- |
| Multi-agent architecture via LangGraph                                                  | `agents/graph.py`, `agents/nodes.py`            | Graph compiles and runs; test suite                          |  Done                                                     |
| LangChain for agent/model tooling                                                       | `llm/model.py`, prompt templates                | Smoke tests                                                  |  Done                                                     |
| Communication only through shared state                                                 | `agents/state.py` (`FinAgentState`)             | Code review; no direct node-to-node calls                    |  Done                                                     |
| Typed conditional transitions                                                           | `route_after_ingestion`, `route_after_analysis` | Test suite covers both branches                              |  Done                                                     |
| Ingestion agent: parse/normalise/timestamp                                              | Ingestion node                                  | Unit + integration tests                                     |  Done                                                     |
| Retrieval agent: open-source vector store + embeddings                                  | FAISS + `all-MiniLM-L6-v2`                      | Smoke test, live KB search                                   |  Done                                                     |
| Source attribution on retrieved passages                                                | passage dict includes source + page             | Verified via Knowledge Base tab                              |  Done                                                     |
| Analysis agent: sentiment, hypothesis, confidence, time horizon                         | Analysis node                                   | Test suite + one real run                                    |  Done                                                     |
| Analysis agent queries retrieval agent before finalising                                | Retry edge only                                 | Code review                                                  |  Partial (by design — see §3.2)                          |
| Decision agent: action, confidence level, evidence, risk flags                          | Decision node (deterministic policy)            | Test suite + one real run                                    |  Done                                                     |
| Alert on high-confidence results                                                        | `is_alert()`, UI banner/toast                   | One real run crossed no alert in the demo; logic unit-tested |  Done (logic verified; alert-firing run not yet captured) |
| Observability: agent trace, retrieved passages, errors                                  | `trace`/`errors` state keys, UI expanders       | One real run inspected end to end                            |  Done                                                     |
| No prohibited components (single-LLM bypass, code-exec agents, live trading, paid data) | N/A by design                                   | Code review                                                  |  Done                                                     |
| Statistical evaluation (baselines, calibration, ablations)                              | —                                               | —                                                            |  Not done                                                 |
| Hybrid retrieval, retrieval benchmark                                                   | —                                               | —                                                            |  Not done                                                 |
8. One Real Run, Annotated

This is the one full, real walkthrough that has actually been captured. It is deliberately
left exactly as it ran rather than padded out with a more impressive invented example.

**Input:** one manually selected market signal, run through "Run FinAgent analysis."
**Knowledge base:** 8 indexed chunks (demo environment, synthetic Apex Technologies document).
**Result:** `WATCH`, 33% confidence.
**Elapsed time:** \~15 seconds end to end (signal normalisation → retrieval → local LLM call → decision synthesis).
**Artefact contents inspected:** hypothesis, evidence bullets, risk flags including at least one contradictory-signal flag, full agent trace, and the retrieved passages with source/page metadata.

**Reading this result:** a WATCH at 33% means the system found some relevant evidence and
produced a hypothesis, but the independent signals (price direction, indicator trend, news
tone) didn't line up strongly enough, or the evidence quality wasn't high enough, to push
confidence into BUY/SELL territory. That's the system working as designed — see §6.

**Not done — planned:** capturing and publishing the full raw JSON of this artefact (and a
second one that actually crosses the alert threshold) directly in this document, so a
reader can see the exact trace lines and retrieved text without re-running the app
themselves.
 9. Design Decisions

Short decision records — context, what was considered, what was chosen, and the trade-off
accepted.

**Local inference (Ollama) vs. a hosted LLM API**
Local keeps the system self-contained and free to run, at the cost of requiring the
evaluator to install Ollama and pull a model first, and accepting that a 3B model reasons
less reliably than a hosted frontier model. The validation and fallback layers in the
Analysis Agent exist specifically to absorb that weakness.

**FAISS vs. a hosted vector database**
FAISS runs locally with no infrastructure or credentials needed, matching the
locally-runnable requirement. The cost is that it doesn't scale past a single-machine,
single-process demo — acceptable for this project's scope.

**Deterministic decision policy vs. letting the LLM decide the action**
The Decision Agent's action, evidence bullets, and most risk flags are rule-based, not
LLM-written. The LLM proposes an analysis; it does not get the final word on an action label
that implies financial risk. This is slower to extend than "just ask the LLM for
everything," but it means the BUY/SELL boundary is auditable code, not a prompt that could
silently drift.

**Heuristic, explainable confidence vs. the LLM's self-reported confidence**
A blended score (§6) is more work to build and tune than simply trusting the model's number,
but it is decomposable — a reviewer can see exactly which component moved the needle. The
trade-off is that it is still a heuristic, not a calibrated probability, and that's stated
plainly rather than implied otherwise.

**Simulated data vs. live market feeds / brokerage integration**
Simulated or supplied CSV data keeps the system compliant with the problem statement's
explicit prohibition on live trading, and keeps evaluation deterministic. It means the
backtest results (§16) describe pipeline behaviour, not market-beating performance.
10. Application Capabilities

**Signals** — simulated replay and manual signal entry; edit price/change/volume before
running; analyse the latest signal per asset.

**Artefacts** — the primary inspection surface: action, confidence, hypothesis, evidence,
risk flags, retrieved passages, and full agent trace, filterable by asset/action.

**Portfolio** — aggregates generated artefacts per asset: count, latest action, confidence,
and an overall consensus view, without re-running anything.

**Backtesting** — replays historical/simulated signals and compares the generated decision
to the subsequent price move. The one real run so far: 5 calls, 80% hit rate, +0.72% average
signed return. **This is a small software-behaviour check, not a trading study** — see §16.

**Ask** — question a specific artefact ("why low confidence?", "what are the biggest
risks?", "what evidence influenced this?"), grounded in that artefact's own data.

**Knowledge Base** — query the retrieval layer directly, independent of the decision
workflow, so retrieval quality can be judged separately from LLM reasoning.
11. Technology Stack

| Layer Technology Purpose  |                                            |                                                       |
| ------------------------- | ------------------------------------------ | ----------------------------------------------------- |
| UI                        | Streamlit                                  | Interactive web interface                             |
| Orchestration             | LangGraph                                  | Graph-based state management and conditional workflow |
| Agent/model tooling       | LangChain                                  | Model integration, prompt templates, tool wrapping    |
| Language model            | Llama 3.2 3B via Ollama                    | Local reasoning — declared, open-source               |
| Embeddings                | Sentence Transformers (`all-MiniLM-L6-v2`) | Semantic representation of financial text             |
| Vector search             | FAISS                                      | Local similarity search                               |
| Data processing           | Pandas                                     | Market data loading and transformation                |
| Language                  | Python                                     | Implementation                                        |
12. Project Structure
```
FINAGENT/
├── app.py
├── config.py
├── README.md
├── requirements.txt
├── agents/
│   ├── graph.py
│   ├── nodes.py
│   └── chat.py
├── core/
│   ├── aggregate.py
│   └── backtest.py
├── rag/
│   ├── rag_engine.py
│   └── vectorstore/
├── llm/
│   └── model.py
├── data/
├── scripts/
│   └── measure_latencies.py
├── tests/
│   ├── run_all.py
│   └── test_rag_math.py
└── docs/

```

`app.py` — Streamlit interface and session coordination.
`agents/` — graph workflow, node logic, artefact Q&A.
`rag/` — document indexing and retrieval.
`llm/` — local model integration.
`core/` — portfolio aggregation and backtesting.
`tests/` — test suite, including retrieval-math checks.

---

## 13. Local Installation

### Requirements

Python, Git, Ollama, and the local `llama3.2:3b` model.

### Windows

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama --version
ollama list
ollama pull llama3.2:3b          # if not already pulled
.\.venv\Scripts\streamlit.exe run app.py

```

macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama --version
ollama list
ollama pull llama3.2:3b          # if not already pulled
streamlit run app.py

```

Open the address Streamlit prints, normally `http://localhost:8501` (the port may shift if
that one is already in use).

Troubleshooting

| Symptom Likely cause Fix                                    |                                                    |                                                                    |
| ----------------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------ |
| `ollama: command not found`                                 | Ollama not installed, or not on PATH               | Install from ollama.com, restart the terminal                      |
| Streamlit launches but analysis hangs                       | Ollama not running                                 | Start Ollama, confirm with `ollama list`                           |
| Hugging Face "unauthenticated request" warning during tests | Rate-limit notice on model download, not a failure | Safe to ignore; does not fail tests (see §15)                      |
| `requirements.txt` fails to parse / garbled characters      | File saved as UTF-16 from PowerShell `pip freeze`  | Re-save as UTF-8                                                   |
| Port 8501 already in use                                    | Another Streamlit instance running                 | Streamlit auto-picks the next free port; check the terminal output |

---

14. Configuration Reference

All tunables live in `config.py`. Read the current values there directly — they're not
repeated here with fixed numbers, to avoid this document drifting out of sync with the code
(see §6's note). The categories to know about:

| Category What it controls  |                                                                                                       |
| -------------------------- | ----------------------------------------------------------------------------------------------------- |
| LLM                        | which model, temperature, context window, timeout                                                     |
| RAG                        | chunk size/overlap, embedding model, relevance thresholds, how many passages are fetched vs. returned |
| Decision policy            | High/Medium/Low confidence boundaries, default alert threshold, max retrieval retries                 |
| Confidence scoring         | the weight of each of the four components (§6), conflict penalty, evidence/rule-based caps            |
| Data                       | where market signal and news CSVs are read from, artefact persistence path                            |

---

15. Testing

```bash
python -m tests.run_all                 # core suite
python -m tests.run_all --integration   # includes integration tests (needs Ollama + a built index)

```

**Last verified run:** `FAILED: 0`.

Covered: core application behaviour, graph routing (both the valid/invalid branch and the
retry branch), embedding normalisation, the L2-to-cosine relevance relationship, relevance
threshold behaviour, and integration tests against the real LLM/vector store.

A Hugging Face "unauthenticated request" rate-limit warning appears during testing; it is
informational and does not fail any test.

---

16. Evaluation Results (the one real run)

This is the complete, actual evaluation output — nothing here is extrapolated or rounded up
from a larger claimed run.

Market data, the local Llama model, and the FAISS knowledge base all loaded successfully.
One manual analysis completed in **\~15 seconds**, producing a **WATCH** decision at **33% confidence**, with a full hypothesis, evidence, risk flags (including a contradictory- signal flag), agent trace, and retrieved passages.
The knowledge base held **8 indexed chunks** in this environment.
A backtest run evaluated **5 calls**, reporting an **80% hit rate** and **+0.72% average signed return**.

**What this does and doesn't show:** it shows the pipeline runs end to end, produces
internally consistent artefacts, and that the backtest mechanism itself works correctly on
real output. Five calls cannot support any claim about predictive skill — there's no
baseline comparison, no confidence interval, and no calibration check yet (see §0). Treat
this section as "the plumbing works," not "the system is accurate."

**Not done — planned:** baseline comparisons (random, always-bullish, momentum-sign),
Wilson confidence intervals on hit rate, calibration analysis (does a higher confidence
bucket actually hit more often), and ablations (LLM vs. quant-only, with/without RAG,
with/without news).
17. Limitations

Market data is simulated or supplied, not a continuously validated live feed.
The local 3B model's reasoning quality depends heavily on what's fed to it; the fallback layers exist precisely because it is not reliable on its own.
Retrieval quality is bounded by the knowledge base's coverage — if the answer isn't in the indexed documents, retrieval cannot invent it.
The backtest sample (5 calls) validates software behaviour, not predictive performance.
Confidence is an internal, explainable heuristic, not a calibrated probability.
No live trading, brokerage connection, or order placement exists or is planned.
No statistical evaluation (baselines, calibration, ablations) has been run yet — this is the single biggest gap versus a fully rigorous evaluation.
Retrieval is dense-only; no hybrid BM25 fusion yet, and the document corpus is too small for a meaningful retrieval benchmark.
No structured trace schema or persistent rotating logs yet — the trace exists in the UI and in exported JSON, but isn't yet schema-validated.

Future work, roughly in priority order

1. Run the statistical evaluation (baselines + Wilson intervals + calibration) on a larger, still-simulated dataset, and report the honest result either way.
2. Expand the document corpus and add a retrieval benchmark (Recall\@k, MRR).
3. Auto-generate the Mermaid diagram from the compiled graph so the architecture diagram can never drift from the real code.
4. Add a structured trace schema and rotating file logs.
5. Prompt-injection hardening and tests against the retrieved document text.
18. Compliance and Safety

Not connected to any brokerage; places no orders.
Not a source of financial advice; outputs are decision-support artefacts for inspection, not recommendations to act on.
The demonstration Apex Technologies document is explicitly synthetic, created for retrieval testing, and is not a real company filing.
No prohibited components from the problem statement are present: no single-LLM bypass of the multi-agent structure, no code-generation/execution agents, no live trading, no paid data dependency as a primary source.
19. FAQ

**Is this really real-time?**
No — it's a simulated CSV replay that the user (or a timed replay loop) triggers. "Real-time
simulation," not a live feed.

**Is the confidence score calibrated?**
No, and §6 says so directly. It's an explainable blend of four signals with caps and
penalties, not a statistically validated probability. Calibration analysis is listed as
not-done work in §0 and §16.

**What happens if Ollama isn't running?**
The Analysis Agent catches the failure and falls back to a rule-based analysis. The run
doesn't crash; it's logged as an error/fallback event in the trace, and the rule-based path
is capped so it can never produce a "High" confidence BUY/SELL (see §6).

**What happens if the knowledge base is empty or missing?**
Retrieval catches it, logs a fallback event, and the pipeline continues with no document
evidence. Confidence is capped lower as a result, and a risk flag notes the missing
evidence.

**Why isn't the decision made by the LLM directly?**
So the action label is auditable code, not something a prompt could silently change. See
the design decision in §9.

**Does the Analysis Agent really call the Retrieval Agent, as the problem statement asks?**
Only through the bounded retry edge, not on every run — flagged honestly as "Partial (by
design)" in §7, with the reasoning in §3.2.

**How was the backtest evaluated?**
By replaying a signal through the full pipeline and comparing the decision's direction to
the asset's subsequent price move. One real run exists (5 calls, 80% hit rate) — too small
to generalise from; see §16.

**Why llama3.2:3b and not a bigger or hosted model?**
Keeps the system locally runnable with no API key or hosting cost, at the cost of weaker
raw reasoning — which is why validation and fallback exist at every LLM call site.

**Is retrieval hybrid (keyword + semantic)?**
No, dense-only right now. Hybrid BM25 fusion is listed as future work in §17.

**What stops a malicious PDF from injecting instructions into the LLM?**
Nothing robust yet — this is explicitly listed as not-done in §0 and §17. Treat uploaded
documents as trusted for now.

**Can this place real trades?**
No, by design and by the problem statement's constraints. See §18.

**What's the single biggest gap right now?**
Rigorous evaluation. The architecture, fallbacks, and observability are solid; the
statistical evidence for "does this actually work well" doesn't exist yet beyond one small
backtest run.

20. Reproducible Evaluation Flow

```text
Install dependencies → Start Ollama → Launch Streamlit → Open Signals
   → Run a manual analysis → Inspect the artefact (evidence, risks, trace)
   → Query the Knowledge Base directly → Ask a follow-up question
   → Review Portfolio aggregation → Run Backtest → Run the test suite



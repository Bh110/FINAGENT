import html
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from agents.chat import answer_followup, explain_confidence
from agents.graph import analyze_signal, build_finagent_graph
from core.aggregate import portfolio_summary
from core.backtest import evaluate
from llm.model import llm_status
from rag.rag_engine import (build_vectorstore, index_info, list_documents,
                            search_with_scores, vectorstore_ready)

st.set_page_config(page_title="FinAgent", page_icon="📈", layout="wide")

ACTION_COLORS = {"BUY": "#16a34a", "SELL": "#dc2626", "HOLD": "#ca8a04", "WATCH": "#2563eb"}
esc = html.escape

st.markdown("""
<style>
.hero-title {font-size: 38px; font-weight: 800; margin-bottom: 0;}
.hero-sub {font-size: 15px; color: #6b7280; margin-bottom: .5rem;}
.art-card {border: 1px solid rgba(128,128,128,.35); border-left-width: 6px; border-radius: 12px;
           padding: .9rem 1.2rem; margin: .6rem 0 .3rem 0;}
.badge {display:inline-block; padding: 2px 12px; border-radius: 999px; color: white;
        font-weight: 700; font-size: 14px;}
.trace-item {padding: 5px 10px; margin: 3px 0; border-radius: 6px; font-family: monospace;
             font-size: 12.5px; background: rgba(128,128,128,.12);}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">📈 FinAgent</div>'
            '<div class="hero-sub">Real-time AI investment decision system · '
            'Multi-agent · RAG · LangGraph — simulation only, not financial advice</div>',
            unsafe_allow_html=True)

ss = st.session_state


# =============================================================================== state
def _load_saved():
    try:
        saved = json.loads(config.ARTEFACT_FILE.read_text())
        return [a for a in saved if "score" in a.get("analysis", {})]   # skip old formats
    except Exception:
        return []


def _persist():
    try:
        config.ARTEFACT_FILE.write_text(
            json.dumps(ss.artefacts[: config.MAX_STORED_ARTEFACTS], default=str))
    except Exception:
        pass


if "artefacts" not in ss:
    ss.artefacts = _load_saved()          # newest first; survives restarts
ss.setdefault("errors", [])
ss.setdefault("chat", {})
ss.setdefault("bt_artefacts", [])
ss.setdefault("streaming", False)
ss.setdefault("stream_pos", 0)
ss.setdefault("last_tick", 0.0)
ss.setdefault("ingested", [])


@st.cache_resource
def get_graph():
    return build_finagent_graph()


@st.cache_data(ttl=30)
def cached_llm_status():
    return llm_status()


@st.cache_data
def load_market(path: str, mtime: float):
    df = pd.read_csv(path)
    df["_ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.dropna(subset=["_ts"]).sort_values("_ts", kind="stable").reset_index(drop=True)


def history_before(df, asset, ts, n=60):
    h = df[(df["asset"] == asset) & (df["_ts"] < ts)].tail(n)
    return h["price"].astype(float).tolist(), h["volume"].astype(float).tolist()


def build_signal(row, df, source="csv_replay"):
    prices, vols = history_before(df, row["asset"], row["_ts"])
    return {"asset": row["asset"], "price": float(row["price"]),
            "change_percent": float(row["change_percent"]), "volume": int(row["volume"]),
            "timestamp": str(row["timestamp"]), "source": source,
            "history": prices, "volume_history": vols}


def run_and_store(signal, threshold, use_llm):
    prior = next((a["decision"]["action"] for a in ss.artefacts
                  if a["decision"]["asset"] == signal["asset"]), "")
    art, errors = analyze_signal(get_graph(), signal, threshold, use_llm, prior)
    stamp = datetime.now().strftime("%H:%M:%S")
    for e in errors:
        ss.errors.insert(0, f"{stamp} · {e}")
    if art:
        ss.artefacts.insert(0, art)
        _persist()
    return art


def report_markdown(artefacts):
    out = [f"# FinAgent report\nGenerated {datetime.now():%Y-%m-%d %H:%M}\n"]
    for a in artefacts:
        d, an = a["decision"], a["analysis"]
        out.append(f"## {d['action']} — {d['asset']} ({d['confidence']}% {d['confidence_level']})")
        out.append(f"*{a['generated_at']} · {an['sentiment']} · {an['time_horizon']} · "
                   f"engine: {an['engine']}*\n\n**Hypothesis:** {an['hypothesis']}\n")
        out += ["**Evidence**"] + [f"- {e}" for e in d["evidence"]]
        out += ["", "**Risk flags**"] + ([f"- {r}" for r in d["risk_flags"]] or ["- none"])
        out += ["", "**Agent trace**"] + [f"- {t}" for t in a["trace"]] + [""]
    return "\n".join(out)


# =============================================================================== sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    threshold = st.slider("Alert threshold (confidence above)", 50, 100,
                          config.DEFAULT_ALERT_THRESHOLD, 5)
    use_llm = st.toggle("Use LLM in Analysis Agent", value=True,
                        help="Off = quant-only rules (fast, never 'High' confidence).")
    ok, msg = cached_llm_status()
    st.caption(("✅ " if ok else "⚠️ ") + msg)
    st.caption(f"Embeddings: all-MiniLM-L6-v2 · Vector store: FAISS")

    st.divider()
    st.header("📚 Knowledge Base")
    docs_on_disk = list_documents()
    if docs_on_disk:
        st.caption("Documents: " + ", ".join(docs_on_disk))
    files = st.file_uploader("Add documents", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if st.button("🔨 Build / Rebuild Knowledge Base", width="stretch"):
        config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
        for f in files or []:
            safe = "".join(c if c.isalnum() or c in "._- " else "_" for c in Path(f.name).name)
            (config.DOCS_DIR / safe).write_bytes(f.getbuffer())
        with st.spinner("Chunking → embedding → FAISS..."):
            try:
                _, n_chunks = build_vectorstore()
                if n_chunks:
                    st.success(f"✓ {len(list_documents())} document(s) · {n_chunks} chunks embedded "
                               "and indexed.")
                else:
                    st.warning("No documents found to index.")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")
    elif not vectorstore_ready() and docs_on_disk and not ss.get("auto_built"):
        ss.auto_built = True
        with st.spinner("First run: building knowledge base..."):
            try:
                _, n_chunks = build_vectorstore()
                st.success(f"✓ Built automatically · {n_chunks} chunks")
            except Exception as e:
                st.error(f"Auto-build failed: {e}")
    info = index_info()
    st.caption("Index: " + (f"✅ {info.get('chunks', '?')} chunks" if vectorstore_ready()
                            else "⚠️ not built"))

    st.divider()
    if ss.artefacts:
        st.download_button("⬇️ Export JSON (artefacts + traces)",
                           json.dumps(ss.artefacts, indent=2, default=str),
                           "finagent_report.json", "application/json", width="stretch")
        st.download_button("⬇️ Export Markdown report", report_markdown(ss.artefacts),
                           "finagent_report.md", "text/markdown", width="stretch")
        if st.button("🗑️ Clear artefacts", width="stretch"):
            ss.artefacts, ss.errors = [], []
            _persist()
            st.rerun()

tab_run, tab_art, tab_port, tab_bt, tab_chat, tab_kb = st.tabs(
    ["📡 Signals", "🤖 Artefacts", "📊 Portfolio", "🧪 Backtest", "💬 Ask", "🔎 Knowledge Base"])

# =============================================================================== signals tab
with tab_run:
    path = st.text_input("Market data source (CSV: timestamp, asset, price, change_percent, volume)",
                         str(config.DEFAULT_SIGNAL_FILE))
    market = None
    if Path(path).exists():
        try:
            market = load_market(path, Path(path).stat().st_mtime)
        except Exception as e:
            st.error(f"Could not read the signal file: {e}")
    else:
        st.info("No signal file at that path.")

    if market is not None and not market.empty:
        # ---------------------------------------------------------- live feed
        st.subheader("Live Signal Feed")
        n_feed = st.number_input("Signals to show", 5, 50, 10, key="n_feed")
        if ss.ingested:
            feed = pd.DataFrame(ss.ingested[-int(n_feed):][::-1])
            st.caption(f"Replay: {ss.stream_pos}/{len(market)} signals ingested")
        else:
            feed = market.tail(int(n_feed)).iloc[::-1][
                ["timestamp", "asset", "price", "change_percent", "volume"]].copy()
            st.caption("Latest signals in the file (start the replay to stream them live).")
        feed["change_percent"] = feed["change_percent"].map(lambda x: f"{x:+.1f}%")
        feed["volume"] = feed["volume"].map(lambda x: f"{int(x):,}")
        st.dataframe(feed.rename(columns=str.title), width="stretch", hide_index=True)

        # ---------------------------------------------------------- streaming replay
        st.subheader("Real-time replay")
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        interval = c1.slider("Seconds between signals", 5, 60, 15)
        if c2.button("▶️ Start", width="stretch"):
            ss.streaming, ss.last_tick = True, 0.0
        if c3.button("⏸️ Stop", width="stretch"):
            ss.streaming = False
        if c4.button("↺ Reset", width="stretch"):
            ss.streaming, ss.stream_pos, ss.ingested = False, 0, []
            st.rerun()

        def _stream_body():
            if not ss.streaming:
                return
            if ss.stream_pos >= len(market):
                ss.streaming = False
                st.success("Replay finished.")
                st.rerun()
            wait = interval - (time.time() - ss.last_tick)
            if wait > 1:            # a full rerun just happened; wait for the next timer tick
                st.caption(f"⏱️ next signal in ~{int(wait)}s")
                return
            row = market.iloc[ss.stream_pos]
            ss.stream_pos += 1
            ss.last_tick = time.time()
            ss.ingested.append({"timestamp": row["timestamp"], "asset": row["asset"],
                                "price": row["price"], "change_percent": row["change_percent"],
                                "volume": row["volume"]})
            with st.spinner(f"Analysing {row['asset']}..."):
                art = run_and_store(build_signal(row, market), threshold, use_llm)
            if art and art["alert"]:
                st.toast(f"🚨 {art['decision']['action']} {art['decision']['asset']} "
                         f"({art['decision']['confidence']}%)")
            st.rerun()          # refresh every tab with the new artefact

        runner = (st.fragment(run_every=f"{interval}s")(_stream_body) if ss.streaming
                  else st.fragment(_stream_body))
        runner()

        # ---------------------------------------------------------- manual analysis
        st.subheader("Manual analysis")
        a_col, s_col = st.columns(2)
        assets = sorted(market["asset"].unique())
        pick_asset = a_col.selectbox("Asset", assets)
        rows = market[market["asset"] == pick_asset].iloc[::-1]
        pick = s_col.selectbox(
            "Signal", range(len(rows)),
            format_func=lambda i: f"{rows.iloc[i]['timestamp']} | {rows.iloc[i]['change_percent']:+.1f}%")
        row = rows.iloc[pick]
        k = f"{pick_asset}|{row['timestamp']}"
        p1, p2, p3 = st.columns(3)
        price = p1.number_input("Price", value=float(row["price"]), key=f"p_{k}")
        change = p2.number_input("Change (%)", value=float(row["change_percent"]), key=f"c_{k}")
        volume = p3.number_input("Volume", value=int(row["volume"]), step=1000, key=f"v_{k}")

        b1, b2 = st.columns(2)
        if b1.button("🚀 Run FinAgent analysis", type="primary", width="stretch"):
            sig = build_signal(row, market, "manual")
            sig.update(price=price, change_percent=change, volume=int(volume))
            with st.spinner("Running multi-agent analysis..."):
                art = run_and_store(sig, threshold, use_llm)
            if art is None:
                st.error("Signal rejected - see the error events below.")
            else:
                st.success(f"Done in {art['elapsed_s']}s - see the Artefacts tab.")
                if art["alert"]:
                    st.toast(f"🚨 {art['decision']['action']} {art['decision']['asset']}")
        if b2.button("⚡ Analyse the latest signal of every asset", width="stretch"):
            bar = st.progress(0.0)
            for i, asset in enumerate(assets, 1):
                latest = market[market["asset"] == asset].iloc[-1]
                run_and_store(build_signal(latest, market), threshold, use_llm)
                bar.progress(i / len(assets))
            st.success(f"Analysed {len(assets)} assets - see the Artefacts tab.")

        hist = market[market["asset"] == pick_asset].set_index("_ts")["price"]
        st.caption(f"Price history: {pick_asset}")
        st.line_chart(hist)

# =============================================================================== artefacts tab
with tab_art:
    if not ss.artefacts:
        st.info("No artefacts yet. Run an analysis or start the replay.")
    else:
        top = ss.artefacts[0]
        if top["alert"]:
            st.success(f"🚨 HIGH-CONFIDENCE ALERT — {top['decision']['action']} "
                       f"{top['decision']['asset']} at {top['decision']['confidence']}% "
                       f"(threshold {threshold}%).")
        f1, f2 = st.columns(2)
        f_assets = f1.multiselect("Filter by asset", sorted({a["decision"]["asset"] for a in ss.artefacts}))
        f_actions = f2.multiselect("Filter by action", list(ACTION_COLORS))
        shown = [a for a in ss.artefacts
                 if (not f_assets or a["decision"]["asset"] in f_assets)
                 and (not f_actions or a["decision"]["action"] in f_actions)]
        st.caption(f"{len(shown)} artefact(s), newest first")

        for art in shown[:50]:
            d, an, sg = art["decision"], art["analysis"], art["signal"]
            color = ACTION_COLORS.get(d["action"], "#6b7280")
            st.markdown(
                f'<div class="art-card" style="border-left-color:{color}">'
                f'<span class="badge" style="background:{color}">{esc(d["action"])}</span> '
                f'&nbsp;<b>{esc(d["asset"])}</b> &nbsp;·&nbsp; {esc(d["confidence_level"])} '
                f'confidence ({d["confidence"]}%){" &nbsp;🚨" if art["alert"] else ""}<br>'
                f'<small>signal {esc(str(sg["timestamp"]))} · {esc(an["sentiment"])} · '
                f'{esc(an["time_horizon"])} · engine: {esc(an["engine"])} · {art["elapsed_s"]}s</small>'
                f'</div>', unsafe_allow_html=True)
            st.progress(min(d["confidence"] / 100, 1.0))
            left, right = st.columns(2)
            with left:
                st.markdown("**Hypothesis**")
                st.write(an["hypothesis"])
                if an.get("rationale"):
                    st.caption(an["rationale"])
                st.markdown("**Key evidence**")
                for e in d["evidence"]:
                    st.markdown(f"- {e}")
            with right:
                st.markdown("**Risk flags**")
                if d["risk_flags"]:
                    for r in d["risk_flags"]:
                        st.warning(r)
                else:
                    st.success("No major risks identified.")

            with st.expander("🧮 Why this confidence?"):
                s = an["score"]
                comp = pd.DataFrame({
                    "score (0-100)": s["components"], "weight": s["weights"]})
                comp["contribution"] = (comp["score (0-100)"] * comp["weight"]).round(1)
                st.dataframe(comp, width="stretch")
                st.text(explain_confidence(art))
                if an.get("agreeing") or an.get("conflicts"):
                    st.caption(f"Agreeing indicators: {an.get('agreeing') or '-'} · "
                               f"Conflicting: {an.get('conflicts') or '-'}")
            with st.expander("🔎 Agent trace, retrieved passages, errors, raw output"):
                for ev in art["trace"]:
                    st.markdown(f'<div class="trace-item">→ {esc(ev)}</div>', unsafe_allow_html=True)
                for e in art["errors"]:
                    st.error(e)
                st.markdown("**Retrieved passages**")
                if art["retrieved_documents"]:
                    for i, p in enumerate(art["retrieved_documents"], 1):
                        st.caption(f"{i}. {p['source']} · page {p['page']} · "
                                   f"relevance {p['relevance']:.2f}")
                        st.write(p["content"][:700])
                else:
                    st.caption("None (no relevant evidence or knowledge base unavailable).")
                st.markdown("**Raw analysis output**")
                st.code(an.get("raw_analysis", ""), language="json")
            st.divider()

    if ss.errors:
        with st.expander(f"⚠️ Error & fallback events ({len(ss.errors)})"):
            for e in ss.errors[:50]:
                st.warning(e)

# =============================================================================== portfolio tab
with tab_port:
    st.subheader("Cross-signal consensus")
    st.caption("Aggregates every artefact per asset; consensus is confidence-weighted.")
    if ss.artefacts:
        st.dataframe(pd.DataFrame(portfolio_summary(ss.artefacts)), width="stretch", hide_index=True)
    else:
        st.info("Nothing to aggregate yet.")

# =============================================================================== backtest tab
with tab_bt:
    st.subheader("Backtest hypotheses against subsequent price moves")
    st.caption("Replays historical signals through the full pipeline, then checks whether the "
               "bullish/bearish call matched the price move N ticks later. Simulated data, small "
               "samples: illustrative, not statistically significant.")
    if market is None or market.empty:
        st.info("Load a signal file first.")
    else:
        source = st.radio("Evaluate", ["New backtest run", "My live artefacts"], horizontal=True)
        horizon = st.slider("Look-ahead (ticks)", 1, 10, 3)
        if source == "New backtest run":
            b1, b2, b3 = st.columns(3)
            bt_asset = b1.selectbox("Asset", sorted(market["asset"].unique()), key="bt_asset")
            n_sig = b2.number_input("Signals to replay", 3, 40, 10)
            bt_llm = b3.toggle("Use LLM (slow)", value=False, key="bt_llm")
            if st.button("▶️ Run backtest"):
                sub = market[market["asset"] == bt_asset].reset_index(drop=True)
                start = min(14, max(0, len(sub) - int(n_sig) - horizon))  # warm-up for RSI
                picked = sub.iloc[start: start + int(n_sig)]
                ss.bt_artefacts, prior, bar = [], "", st.progress(0.0)
                for i, (_, r) in enumerate(picked.iterrows(), 1):
                    art, _ = analyze_signal(get_graph(), build_signal(r, market, "backtest"),
                                            101, bt_llm, prior)   # 101: never alert in backtests
                    if art:
                        ss.bt_artefacts.append(art)
                        prior = art["decision"]["action"]
                    bar.progress(i / len(picked))
            arts = ss.bt_artefacts
        else:
            arts = ss.artefacts
        if arts:
            res, summary = evaluate(arts, market, horizon)
            if summary["evaluated"] == 0:
                st.warning("No directional calls with enough future ticks to evaluate.")
            else:
                cols = st.columns(4)
                cols[0].metric("Calls evaluated", summary["evaluated"])
                cols[1].metric("Hit rate", f"{summary['hit_rate_%']}%")
                cols[2].metric("Avg signed return", f"{summary['avg_signed_return_%']}%")
                hc = summary.get("hit_rate_high_conf_%")
                cols[3].metric("Hit rate (High conf.)", "n/a" if hc is None else f"{hc}%")
                st.dataframe(res, width="stretch", hide_index=True)

# =============================================================================== ask tab
with tab_chat:
    st.subheader("Ask about an artefact")
    if not ss.artefacts:
        st.info("Generate an artefact first.")
    else:
        labels = {a["id"]: f"{a['generated_at']} · {a['decision']['action']} "
                           f"{a['decision']['asset']} ({a['decision']['confidence']}%)"
                  for a in ss.artefacts[:30]}
        chosen = st.selectbox("Artefact", list(labels), format_func=labels.get)
        art = next(a for a in ss.artefacts if a["id"] == chosen)
        hist = ss.chat.setdefault(chosen, [])
        st.caption("Try: “Why did you rate this High confidence?” · “What are the biggest risks?”")
        for role, text in hist:
            with st.chat_message("user" if role == "user" else "assistant"):
                st.write(text)
        q = st.chat_input("Ask a follow-up question")
        if q:
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    ans = answer_followup(art, hist, q)
                st.write(ans)
            hist += [("user", q), ("assistant", ans)]

# =============================================================================== KB tab
with tab_kb:
    st.subheader("Search the knowledge base")
    if not vectorstore_ready():
        st.warning("Knowledge base not built yet. Upload documents in the sidebar.")
    q = st.text_input("Query", placeholder="What are the major risks?")
    asset_f = st.text_input("Prefer passages about (optional)", placeholder="Apex Technologies Ltd.")
    if q:
        try:
            res = search_with_scores(q, k=5, asset=asset_f or None, min_relevance=0.1)
            if not res:
                st.info("No passages above the relevance threshold.")
            for i, r in enumerate(res, 1):
                with st.expander(f"Result {i} — {r['source']} p.{r['page']} · "
                                 f"relevance {r['relevance']:.2f}", expanded=(i == 1)):
                    st.write(r["content"])
        except FileNotFoundError:
            st.warning("Build the knowledge base first.")
        except Exception as e:
            st.error(f"Search failed: {e}")

st.divider()
st.caption(f"FinAgent · LangGraph + LangChain + FAISS + Sentence Transformers + {config.LLM_MODEL} "
           "· Simulation/demo — no live trading.")

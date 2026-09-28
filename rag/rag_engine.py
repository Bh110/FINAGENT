"""RAG pipeline: load -> chunk -> embed (Sentence Transformers) -> FAISS, with
relevance scores, asset-aware re-ranking and source attribution."""
import json

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from core.signals import matches_asset

_embeddings = None   # loaded once, reused (model load is slow)
_store = None
_META = config.INDEX_DIR / "meta.json"
SUPPORTED = (".pdf", ".txt", ".md")


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBED_MODEL,
            encode_kwargs={"normalize_embeddings": True},  # cosine = 1 - d^2/2
        )
    return _embeddings


def list_documents():
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(p.name for p in config.DOCS_DIR.iterdir() if p.suffix.lower() in SUPPORTED)


def _load_documents():
    docs = []
    for name in list_documents():
        path = config.DOCS_DIR / name
        loaded = (PyPDFLoader(str(path)).load() if path.suffix.lower() == ".pdf"
                  else TextLoader(str(path), encoding="utf-8").load())
        if not loaded:
            continue
        # Document title = first non-empty line; lets us tie chunks back to a company.
        first = next((ln.strip() for ln in loaded[0].page_content.splitlines() if ln.strip()), name)
        for d in loaded:
            d.metadata["source"] = name
            d.metadata["doc_title"] = first
            label = d.metadata.get("page_label")
            if not label:
                page = d.metadata.get("page")
                label = str(page + 1) if isinstance(page, int) else "1"
            d.metadata["page_label"] = str(label)
        docs.extend(loaded)
    return docs


def build_vectorstore():
    """Rebuild the FAISS index from rag/documents. Returns (store, chunk_count) or (None, 0)."""
    global _store
    docs = _load_documents()
    if not docs:
        return None, 0
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    ).split_documents(docs)
    chunks = [c for c in chunks if c.page_content.strip()]
    if not chunks:
        return None, 0

    store = FAISS.from_documents(chunks, get_embeddings())
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(config.INDEX_DIR))
    _META.write_text(json.dumps({
        "embed_model": config.EMBED_MODEL, "normalized": True,
        "chunks": len(chunks), "files": list_documents(),
    }))
    _store = store
    return store, len(chunks)


def vectorstore_ready() -> bool:
    """True only if an index exists AND was built with the current embedding settings."""
    try:
        meta = json.loads(_META.read_text())
        return (config.INDEX_DIR / "index.faiss").exists() and \
            meta.get("embed_model") == config.EMBED_MODEL and meta.get("normalized") is True
    except Exception:
        return False


def index_info() -> dict:
    try:
        return json.loads(_META.read_text())
    except Exception:
        return {}


def load_vectorstore():
    global _store
    if _store is None and vectorstore_ready():
        _store = FAISS.load_local(
            str(config.INDEX_DIR), get_embeddings(),
            allow_dangerous_deserialization=True,  # index is produced locally by build_vectorstore
        )
    return _store


def search_with_scores(query, k=config.TOP_K, asset=None, min_relevance=config.MIN_RELEVANCE):
    """Top-k passages as dicts {content, source, page, relevance, asset_match}.
    Passages below `min_relevance` are dropped. If any passage is about `asset`,
    passages about other companies are dropped too."""
    store = load_vectorstore()
    if store is None:
        raise FileNotFoundError("Knowledge base not built (or built with old settings). "
                                "Upload documents and click 'Build Knowledge Base'.")
    out = []
    for doc, dist in store.similarity_search_with_score(query, k=config.FETCH_K):
        rel = max(0.0, min(1.0, 1.0 - float(dist) / 2.0))
        if rel < min_relevance:
            continue
        out.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page_label", "?"),
            "relevance": round(rel, 3),
            "asset_match": bool(asset) and matches_asset(
                asset, doc.metadata.get("doc_title", ""), doc.page_content),
        })
    if asset and any(r["asset_match"] for r in out):
        out = [r for r in out if r["asset_match"]]
    out.sort(key=lambda r: -r["relevance"])
    return out[:k]


def search_vectorstore(query, k=3):
    """Backward-compatible helper returning LangChain Documents."""
    store = load_vectorstore()
    if store is None:
        raise FileNotFoundError("Knowledge base not built yet.")
    return store.similarity_search(query, k=k)


@tool
def search_knowledge_base(query: str, asset: str = "", k: int = config.TOP_K,
                          min_relevance: float = config.MIN_RELEVANCE) -> list:
    """Search the financial knowledge base. Returns passages with source document,
    page number, relevance score and whether the passage is about the given asset."""
    return search_with_scores(query, k=k, asset=asset or None, min_relevance=min_relevance)

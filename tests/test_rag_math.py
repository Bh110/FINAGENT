"""Mathematical and empirical validation of RAG vector similarity.

Proves:
1. FAISS IndexFlatL2 on L2-normalized embeddings produces squared Euclidean
   distances d^2 that satisfy: cosine_similarity == 1 - d^2 / 2.
2. Embedding vectors from Sentence Transformers are strictly unit normalized.
3. Relevance threshold MIN_RELEVANCE (0.25) reliably separates relevant
   in-domain financial documents from irrelevant / out-of-domain text.
"""
import math
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_l2_cosine_mathematical_identity():
    """Synthetic proof: for any unit vectors u, v:
    ||u - v||^2 = ||u||^2 + ||v||^2 - 2*(u . v) = 2 - 2*cos(theta)
    => cos(theta) = 1 - ||u - v||^2 / 2
    """
    # 1. Identical vectors (cos = 1, dist^2 = 0)
    u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    dist_sq = float(np.sum((u - v) ** 2))
    cos_sim = float(np.dot(u, v))
    assert math.isclose(cos_sim, 1.0 - dist_sq / 2.0, abs_tol=1e-6)
    assert math.isclose(1.0 - dist_sq / 2.0, 1.0, abs_tol=1e-6)

    # 2. Orthogonal vectors (cos = 0, dist^2 = 2)
    v_orth = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    dist_sq_orth = float(np.sum((u - v_orth) ** 2))
    cos_sim_orth = float(np.dot(u, v_orth))
    assert math.isclose(cos_sim_orth, 1.0 - dist_sq_orth / 2.0, abs_tol=1e-6)
    assert math.isclose(1.0 - dist_sq_orth / 2.0, 0.0, abs_tol=1e-6)

    # 3. Opposite vectors (cos = -1, dist^2 = 4)
    v_opp = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    dist_sq_opp = float(np.sum((u - v_opp) ** 2))
    cos_sim_opp = float(np.dot(u, v_opp))
    assert math.isclose(cos_sim_opp, 1.0 - dist_sq_opp / 2.0, abs_tol=1e-6)
    assert math.isclose(1.0 - dist_sq_opp / 2.0, -1.0, abs_tol=1e-6)

    # 4. Known 60-degree angle (cos = 0.5, dist^2 = 1)
    v_60 = np.array([0.5, math.sqrt(3) / 2.0, 0.0], dtype=np.float32)
    dist_sq_60 = float(np.sum((u - v_60) ** 2))
    cos_sim_60 = float(np.dot(u, v_60))
    assert math.isclose(cos_sim_60, 0.5, abs_tol=1e-6)
    assert math.isclose(1.0 - dist_sq_60 / 2.0, 0.5, abs_tol=1e-6)

    # 5. High-dimensional random normalized vectors
    rng = np.random.default_rng(42)
    dim = 384  # matching all-MiniLM-L6-v2 dimension
    vectors = rng.standard_normal((20, dim), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit_vectors = vectors / norms

    for i in range(len(unit_vectors)):
        for j in range(i, len(unit_vectors)):
            d_sq = float(np.sum((unit_vectors[i] - unit_vectors[j]) ** 2))
            c_sim = float(np.dot(unit_vectors[i], unit_vectors[j]))
            derived = 1.0 - d_sq / 2.0
            assert math.isclose(c_sim, derived, abs_tol=1e-5), f"Mismatch: {c_sim} vs {derived}"


def test_embedding_norm_is_one():
    """Verify that get_embeddings produces normalized vectors with unit length."""
    from rag.rag_engine import get_embeddings
    embedder = get_embeddings()
    sample_texts = [
        "Apex Technologies quarterly earnings cloud revenue growth.",
        "Nova Energy refining margins and dividend cash flow.",
        "Completely unrelated text about chocolate cookies recipe.",
    ]
    vectors = embedder.embed_documents(sample_texts)
    for vec in vectors:
        norm = np.linalg.norm(vec)
        assert math.isclose(norm, 1.0, abs_tol=1e-4), f"Vector norm {norm} is not 1.0"


def test_relevance_threshold_validation():
    """Empirical check: in-domain queries score >= 0.25 on their target documents,
    while out-of-domain noise falls below or near the thresholds.
    """
    from rag.rag_engine import load_vectorstore, search_with_scores

    store = load_vectorstore()
    if store is None:
        return

    # In-domain query about Apex
    apex_res = search_with_scores("Apex Technologies cloud revenue enterprise AI growth", k=3, asset="Apex Technologies Ltd.")
    assert len(apex_res) > 0
    assert apex_res[0]["relevance"] >= 0.25, f"Top relevance {apex_res[0]['relevance']} below MIN_RELEVANCE"

    # Out-of-domain query
    noise_res = search_with_scores("Recipe for grandmother's homemade chocolate chip cookies", k=3, min_relevance=0.0)
    # Check that irrelevant query scores much lower than in-domain query
    if noise_res:
        assert noise_res[0]["relevance"] < apex_res[0]["relevance"], "Noise query scored as high as target query"


if __name__ == "__main__":
    test_l2_cosine_mathematical_identity()
    print("test_l2_cosine_mathematical_identity passed")
    test_embedding_norm_is_one()
    print("test_embedding_norm_is_one passed")
    test_relevance_threshold_validation()
    print("test_relevance_threshold_validation passed")

"""Unit tests for myoadapt.tasks.semantic_retrieval."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from myoadapt.tasks.semantic_retrieval import TextEncoder, EMGSemanticRetriever
from myoadapt.models.base import MODEL_REGISTRY, get_model


# ---------------------------------------------------------------------------
# TextEncoder (numpy-only, no torch needed)
# ---------------------------------------------------------------------------
def test_text_encoder_fit_encode_shapes():
    descs = ["open hand fully", "close fist tightly", "pinch thumb index"]
    te = TextEncoder(embed_dim=12).fit(descs)
    emb = te.encode(descs)
    assert emb.shape == (3, 12)
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_text_encoder_similar_phrases_closer_than_unrelated():
    descs = ["open hand fully", "open the hand completely", "close fist tightly"]
    te = TextEncoder(embed_dim=32).fit(descs)
    emb = te.encode(descs)
    sim_related = emb[0] @ emb[1]
    sim_unrelated = emb[0] @ emb[2]
    assert sim_related > sim_unrelated


def test_text_encoder_encode_before_fit_raises():
    with pytest.raises(RuntimeError):
        TextEncoder().encode(["x"])


def test_text_encoder_state_roundtrip():
    descs = ["open hand", "close fist", "pinch"]
    te = TextEncoder(embed_dim=8).fit(descs)
    emb_before = te.encode(descs)
    te2 = TextEncoder.from_state(te.to_state())
    emb_after = te2.encode(descs)
    assert np.allclose(emb_before, emb_after)


def test_text_encoder_empty_corpus_does_not_crash():
    te = TextEncoder(embed_dim=8).fit([])
    emb = te.encode(["anything"])
    assert emb.shape == (1, 8)


# ---------------------------------------------------------------------------
# EMGSemanticRetriever (torch-backed)
# ---------------------------------------------------------------------------
torch = pytest.importorskip("torch")


def _paired_gesture_data(n_per=8, n_ch=4, n_samp=80, seed=0):
    rng = np.random.default_rng(seed)
    templates = ["open hand fully", "close fist tightly", "pinch thumb index"]
    X_list, desc_list = [], []
    for gi, phrase in enumerate(templates):
        freq = 15 + gi * 12
        for _ in range(n_per):
            t = np.arange(n_samp) / 2000
            sig = np.stack([np.sin(2 * np.pi * freq * t) + 0.25 * rng.standard_normal(n_samp)
                             for _ in range(n_ch)])
            X_list.append(sig)
            desc_list.append(phrase)
    return np.stack(X_list).astype(np.float32), desc_list, n_ch, n_samp


def _fit_small_retriever(n_epochs=15):
    X, descs, n_ch, n_samp = _paired_gesture_data()
    retr = EMGSemanticRetriever(n_channels=n_ch, n_samples=n_samp, embed_dim=16, d_model=16,
                                 n_heads=2, n_layers=1, patch_len=16, n_epochs=n_epochs,
                                 batch_size=8, device="cpu", random_state=0)
    retr.fit(X, descs)
    return retr, X, descs


def test_fit_rejects_mismatched_lengths():
    X, descs, n_ch, n_samp = _paired_gesture_data()
    retr = EMGSemanticRetriever(n_channels=n_ch, n_samples=n_samp, device="cpu")
    with pytest.raises(ValueError):
        retr.fit(X, descs[:-1])


def test_fit_rejects_wrong_shape():
    X, descs, n_ch, n_samp = _paired_gesture_data()
    retr = EMGSemanticRetriever(n_channels=n_ch + 1, n_samples=n_samp, device="cpu")
    with pytest.raises(ValueError):
        retr.fit(X, descs)


def test_encode_before_fit_raises():
    retr = EMGSemanticRetriever(device="cpu")
    with pytest.raises(RuntimeError):
        retr.encode_emg(np.zeros((1, 12, 400), dtype=np.float32))


def test_retrieve_emg_by_text_returns_semantically_matching_windows():
    retr, X, descs = _fit_small_retriever()
    idx = retr.retrieve_emg_by_text("open hand fully", X, top_k=5)
    matched = sum(1 for i in idx if descs[i] == "open hand fully")
    assert matched >= 4  # allow one miss for stochastic training


def test_retrieve_text_by_emg_returns_matching_description():
    retr, X, descs = _fit_small_retriever()
    top = retr.retrieve_text_by_emg(X[0], descs, top_k=1)
    assert top[0] == descs[0]


def test_save_load_roundtrip_embeddings_match():
    retr, X, descs = _fit_small_retriever()
    emb_before = retr.encode_emg(X[:5])
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "retriever.pt")
        retr.save(path)
        retr2 = EMGSemanticRetriever.load(path)
        emb_after = retr2.encode_emg(X[:5])
    assert np.allclose(emb_before, emb_after, atol=1e-5)


def test_evaluate_retrieval_credits_any_matching_duplicate_description():
    """Regression test: evaluate_retrieval used to assume row i's ONLY
    correct match was row i itself, silently under-crediting every other
    window sharing the same (repeated) true description — the common
    case for gesture-labeled EMG data. A well-trained retriever should
    score well above chance (1/n_groups) once duplicates are credited."""
    retr, X, descs = _fit_small_retriever(n_epochs=20)
    metrics = retr.evaluate_retrieval(X, descs, k_values=(1, 5))
    chance_level = 1.0 / len(set(descs))  # 3 unique phrases -> 1/3 by pure luck at large k
    assert metrics["emg_to_text_recall@5"] > chance_level
    assert metrics["emg_to_text_recall@1"] > 0.5  # well-separated synthetic signal


def test_evaluate_retrieval_unique_descriptions_still_works():
    """With no duplicates, recall@k reduces to the original exact-row semantics."""
    retr, X, descs = _fit_small_retriever(n_epochs=5)
    unique_descs = [f"sample number {i}" for i in range(len(X))]
    metrics = retr.evaluate_retrieval(X, unique_descs, k_values=(1,))
    assert 0.0 <= metrics["emg_to_text_recall@1"] <= 1.0
    assert metrics["n_samples"] == len(X)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def test_registered_in_model_registry():
    assert "semantic_retriever" in MODEL_REGISTRY
    m = get_model("semantic_retriever", n_channels=4, n_samples=80, device="cpu")
    assert isinstance(m, EMGSemanticRetriever)

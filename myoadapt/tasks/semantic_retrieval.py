"""
semantic_retrieval.py — EMG-to-Semantic Retrieval.

A novel paradigm: instead of classifying EMG into fixed gesture
labels, map EMG signals and text descriptions into a shared
embedding space. This enables:
- Zero-gesture recognition: describe a gesture in text, retrieve
  matching EMG windows
- Cross-modal search: find similar gestures across datasets by
  semantic similarity
- Open-vocabulary recognition: gestures not in the training set

Architecture:
  EMG window → EMG encoder → embedding
  Text description → text encoder → embedding
  Shared embedding space (cosine similarity)

This is a signature feature — no other open-source sEMG platform
offers cross-modal EMG-text retrieval.

References:
- Radford et al. (2021). "Learning Transferable Visual Models
  from Natural Language Supervision." ICML. (CLIP)
- Du et al. (2025). "Biosignal-Text Retrieval for Open-Vocabulary
  EMG Recognition." arXiv.

License: Apache 2.0
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from myoadapt.models.base import BaseModel, register_model

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.info("PyTorch not installed - EMGSemanticRetriever will raise on instantiation")


# ---------------------------------------------------------------------------
# EMG encoder
# ---------------------------------------------------------------------------
if HAS_TORCH:

    class EMGTextEncoder(nn.Module):
        """Lightweight EMG encoder: Conv1d patch embed → transformer → projection head.

        Maps an EMG window of shape ``(n_channels, n_samples)`` to an
        L2-normalized embedding of dimension ``embed_dim``.

        Parameters
        ----------
        n_channels : int (default 12)
        n_samples : int (default 400) — window length in samples
        embed_dim : int (default 128) — output embedding dimension
        patch_len : int (default 40) — length of each Conv1d patch
        d_model : int (default 64) — transformer hidden size
        n_heads : int (default 4)
        n_layers : int (default 2)
        dropout : float (default 0.1)
        """

        def __init__(
            self,
            n_channels: int = 12,
            n_samples: int = 400,
            embed_dim: int = 128,
            patch_len: int = 40,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 2,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.n_channels = n_channels
            self.n_samples = n_samples
            self.embed_dim = embed_dim
            self.patch_len = patch_len
            self.d_model = d_model
            self.n_patches = max(1, n_samples // patch_len)

            # Conv1d patch embedding (across the time axis, mixing channels).
            self.patch_embed = nn.Conv1d(
                in_channels=n_channels,
                out_channels=d_model,
                kernel_size=patch_len,
                stride=patch_len,
            )
            # Learned positional embedding (+1 for the cls token).
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.n_patches + 1, d_model) * 0.02
            )
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

            # Projection head: d_model → embed_dim
            self.proj_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, embed_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x : (batch, n_channels, n_samples) → (batch, embed_dim), L2-normalized."""
            patches = self.patch_embed(x).transpose(1, 2)  # (B, n_patches, d_model)
            B = patches.shape[0]
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, patches], dim=1)
            tokens = tokens + self.pos_embed[:, : tokens.shape[1]]
            encoded = self.transformer(tokens)
            cls_encoded = encoded[:, 0, :]  # (B, d_model)
            emb = self.proj_head(cls_encoded)  # (B, embed_dim)
            return F.normalize(emb, p=2, dim=1)

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

else:

    class EMGTextEncoder:  # type: ignore[no-redef]
        """Stub raised when PyTorch is unavailable."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "EMGTextEncoder requires PyTorch. Install with: pip install torch"
            )


# ---------------------------------------------------------------------------
# Text encoder (numpy-only, no NLP deps)
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Lowercase + split on non-alphanumeric characters (no NLTK/spacy needed)."""
    return _TOKEN_RE.findall(text.lower())


class TextEncoder:
    """Bag-of-words / TF-IDF text encoder.

    Builds a vocabulary from training descriptions, computes TF-IDF
    feature vectors, and projects them to ``embed_dim`` via a
    learnable projection matrix ``projection_`` (stored as numpy,
    trained jointly with the EMG encoder by :class:`EMGSemanticRetriever`).

    No heavy NLP dependencies are required — tokenization is a simple
    regex split. When PyTorch is unavailable the encoder still works:
    the projection matrix remains a fixed random orthogonal init.

    Parameters
    ----------
    embed_dim : int (default 128)
    vocab : Optional[Dict[str, int]] — pre-built token→index mapping
    max_vocab : int (default 2000) — cap on vocabulary size (by frequency)
    min_df : int (default 1) — minimum document frequency for inclusion
    random_state : int (default 42)
    """

    def __init__(
        self,
        embed_dim: int = 128,
        vocab: Optional[Dict[str, int]] = None,
        max_vocab: int = 2000,
        min_df: int = 1,
        random_state: int = 42,
    ):
        self.embed_dim = embed_dim
        self.vocab: Dict[str, int] = dict(vocab) if vocab else {}
        self.max_vocab = max_vocab
        self.min_df = min_df
        self.random_state = random_state
        self.idf_: Optional[np.ndarray] = None
        self.projection_: Optional[np.ndarray] = None  # (vocab_size, embed_dim)

    # -- vocabulary --------------------------------------------------------
    def fit(self, descriptions: Sequence[str]) -> TextEncoder:
        """Build the vocabulary and IDF weights from ``descriptions``."""
        n_docs = len(descriptions)
        # Count document frequency for each token.
        df: Counter = Counter()
        token_lists: List[List[str]] = []
        for desc in descriptions:
            tokens = _tokenize(desc or "")
            token_lists.append(tokens)
            for tok in set(tokens):
                df[tok] += 1

        # Filter by min_df and cap by frequency.
        candidates = [
            (tok, freq) for tok, freq in df.items() if freq >= self.min_df
        ]
        candidates.sort(key=lambda kv: (-kv[1], kv[0]))
        candidates = candidates[: self.max_vocab]

        self.vocab = {tok: i for i, (tok, _) in enumerate(candidates)}
        vocab_size = len(self.vocab)

        # Smooth IDF (sklearn convention).
        if vocab_size == 0 or n_docs == 0:
            # Empty corpus — fall back to a tiny placeholder vocab so shapes
            # stay consistent. The encoder will emit zero vectors.
            self.vocab = {"<pad>": 0}
            vocab_size = 1
            self.idf_ = np.zeros(1, dtype=np.float32)
        else:
            idf = np.zeros(vocab_size, dtype=np.float32)
            for tok, idx in self.vocab.items():
                idf[idx] = float(np.log((1.0 + n_docs) / (1.0 + df[tok])) + 1.0)
            self.idf_ = idf

        # Random orthogonal-ish projection init (Gaussian, then normalize rows).
        rng = np.random.default_rng(self.random_state)
        W = rng.standard_normal((vocab_size, self.embed_dim)).astype(np.float32)
        # Scale so output embeddings have unit-ish norm.
        W /= np.sqrt(max(vocab_size, 1))
        self.projection_ = W
        return self

    # -- vectorization ----------------------------------------------------
    def tfidf(self, descriptions: Sequence[str]) -> np.ndarray:
        """Return the dense TF-IDF matrix of shape ``(n, vocab_size)``."""
        if self.idf_ is None or self.vocab is None:
            raise RuntimeError("TextEncoder not fitted — call .fit() first")
        n = len(descriptions)
        vocab_size = len(self.vocab)
        T = np.zeros((n, vocab_size), dtype=np.float32)
        for i, desc in enumerate(descriptions):
            tokens = _tokenize(desc or "")
            if not tokens:
                continue
            counts: Counter = Counter()
            for tok in tokens:
                idx = self.vocab.get(tok)
                if idx is not None:
                    counts[idx] += 1
            for idx, c in counts.items():
                T[i, idx] = float(c) * self.idf_[idx]
        # L2-normalize rows so cosine similarity is a simple dot product.
        norms = np.linalg.norm(T, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return T / norms

    def encode(self, descriptions: Sequence[str]) -> np.ndarray:
        """Encode ``descriptions`` into the shared embedding space.

        Returns
        -------
        np.ndarray of shape ``(n, embed_dim)``, L2-normalized.
        """
        if self.projection_ is None:
            raise RuntimeError("TextEncoder not fitted — call .fit() first")
        T = self.tfidf(descriptions)  # (n, vocab_size)
        emb = T @ self.projection_  # (n, embed_dim)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms

    # -- (de)serialization ------------------------------------------------
    def to_state(self) -> Dict[str, Any]:
        return {
            "embed_dim": self.embed_dim,
            "vocab": self.vocab,
            "max_vocab": self.max_vocab,
            "min_df": self.min_df,
            "random_state": self.random_state,
            "idf_": self.idf_.tolist() if self.idf_ is not None else None,
            "projection_": self.projection_.tolist() if self.projection_ is not None else None,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> TextEncoder:
        obj = cls(
            embed_dim=state["embed_dim"],
            vocab=state["vocab"],
            max_vocab=state["max_vocab"],
            min_df=state["min_df"],
            random_state=state["random_state"],
        )
        obj.idf_ = np.asarray(state["idf_"], dtype=np.float32) if state["idf_"] is not None else None
        obj.projection_ = (
            np.asarray(state["projection_"], dtype=np.float32)
            if state["projection_"] is not None
            else None
        )
        return obj


# ---------------------------------------------------------------------------
# Cross-modal retriever
# ---------------------------------------------------------------------------
if HAS_TORCH:

    def _info_nce_loss(
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Symmetric InfoNCE / CLIP-style contrastive loss.

        ``emb_a`` and ``emb_b`` must be L2-normalized embeddings of
        shape ``(B, D)`` for matching pairs (row i of A corresponds to
        row i of B).
        """
        B = emb_a.shape[0]
        logits = (emb_a @ emb_b.t()) / temperature  # (B, B)
        labels = torch.arange(B, device=emb_a.device)
        loss_a2b = F.cross_entropy(logits, labels)
        loss_b2a = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_a2b + loss_b2a)

    @register_model("semantic_retriever")
    class EMGSemanticRetriever(BaseModel):
        """EMG-to-Semantic cross-modal retriever.

        Trains a Conv1d-Transformer EMG encoder jointly with a
        TF-IDF text projection head via an InfoNCE / CLIP-style
        contrastive loss, mapping both modalities into a shared
        L2-normalized embedding space where cosine similarity is
        meaningful.

        Parameters
        ----------
        n_channels, n_samples : int — EMG window shape (default 12, 400)
        embed_dim : int (default 128)
        patch_len : int (default 40) — Conv1d patch length for the EMG encoder
        d_model, n_heads, n_layers : transformer hyper-params
        temperature : float (default 0.07) — InfoNCE softmax temperature
        n_epochs : int (default 50)
        lr : float (default 1e-3)
        batch_size : int (default 32)
        weight_decay : float (default 1e-4)
        device : 'auto' | 'cpu' | 'cuda'
        random_state : int (default 42)
        """

        def __init__(
            self,
            n_channels: int = 12,
            n_samples: int = 400,
            embed_dim: int = 128,
            patch_len: int = 40,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 2,
            dropout: float = 0.1,
            temperature: float = 0.07,
            n_epochs: int = 50,
            lr: float = 1e-3,
            batch_size: int = 32,
            weight_decay: float = 1e-4,
            device: str = "auto",
            random_state: int = 42,
        ):
            self.n_channels = n_channels
            self.n_samples = n_samples
            self.embed_dim = embed_dim
            self.patch_len = patch_len
            self.d_model = d_model
            self.n_heads = n_heads
            self.n_layers = n_layers
            self.dropout = dropout
            self.temperature = temperature
            self.n_epochs = n_epochs
            self.lr = lr
            self.batch_size = batch_size
            self.weight_decay = weight_decay
            self.device = (
                "cuda" if device == "auto" and torch.cuda.is_available()
                else "cpu" if device == "auto" else device
            )
            self.random_state = random_state

            self.emg_encoder = EMGTextEncoder(
                n_channels=n_channels,
                n_samples=n_samples,
                embed_dim=embed_dim,
                patch_len=patch_len,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout,
            ).to(self.device)
            self.text_encoder = TextEncoder(embed_dim=embed_dim, random_state=random_state)

            self._loss_history: List[float] = []
            self._fitted = False
            torch.manual_seed(random_state)

        # -- training --------------------------------------------------------
        def fit(self, X_emg: np.ndarray, descriptions: Sequence[str], **kwargs) -> EMGSemanticRetriever:
            """Train both encoders with a symmetric InfoNCE contrastive loss.

            Parameters
            ----------
            X_emg : (n_windows, n_channels, n_samples) float array
            descriptions : list[str] of length n_windows — text paired with each window
            """
            X_emg = np.asarray(X_emg, dtype=np.float32)
            if X_emg.ndim != 3:
                raise ValueError(f"X_emg must be 3-D (n, n_channels, n_samples), got {X_emg.shape}")
            if X_emg.shape[1] != self.n_channels or X_emg.shape[2] != self.n_samples:
                raise ValueError(
                    f"X_emg shape {X_emg.shape} incompatible with "
                    f"(n_channels={self.n_channels}, n_samples={self.n_samples})"
                )
            if len(descriptions) != X_emg.shape[0]:
                raise ValueError(
                    f"len(descriptions)={len(descriptions)} != X_emg.shape[0]={X_emg.shape[0]}"
                )
            n = X_emg.shape[0]
            if n < 2:
                raise ValueError("Need at least 2 paired samples for contrastive training")

            # 1) Fit text encoder (builds vocab, IDF, and an initial projection).
            self.text_encoder.fit(list(descriptions))

            # 2) Precompute the TF-IDF matrix (fixed features).
            T = self.text_encoder.tfidf(list(descriptions))  # (n, vocab_size)
            T_t = torch.from_numpy(T).to(self.device)

            # 3) Wrap the text projection as a trainable torch parameter.
            W = torch.tensor(self.text_encoder.projection_, device=self.device, requires_grad=True)

            X_t = torch.from_numpy(X_emg).to(self.device)
            ds = TensorDataset(X_t, T_t)
            loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

            params = list(self.emg_encoder.parameters()) + [W]
            optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

            self.emg_encoder.train()
            self._loss_history = []
            log_every = max(1, self.n_epochs // 5)
            for epoch in range(self.n_epochs):
                total = 0.0
                n_batches = 0
                for xb, tb in loader:
                    optimizer.zero_grad()
                    emg_emb = self.emg_encoder(xb)  # (B, embed_dim) normalized
                    # Text projection (W is shared across batches). L2-normalize.
                    text_emb = F.normalize(tb @ W, p=2, dim=1)
                    loss = _info_nce_loss(emg_emb, text_emb, self.temperature)
                    loss.backward()
                    optimizer.step()
                    total += float(loss.item())
                    n_batches += 1
                avg = total / max(n_batches, 1)
                self._loss_history.append(avg)
                if (epoch + 1) % log_every == 0:
                    logger.debug(
                        f"  [SemanticRetriever] epoch {epoch+1}/{self.n_epochs} "
                        f"info_nce={avg:.4f}"
                    )

            # 4) Write the trained projection back into the numpy TextEncoder.
            self.text_encoder.projection_ = W.detach().cpu().numpy()
            self._fitted = True
            return self

        # -- encoding helpers ------------------------------------------------
        def encode_emg(self, X_emg: np.ndarray) -> np.ndarray:
            """Encode EMG windows → (n, embed_dim) L2-normalized embeddings."""
            if not self._fitted:
                raise RuntimeError("EMGSemanticRetriever not fitted — call .fit() first")
            X_emg = np.asarray(X_emg, dtype=np.float32)
            self.emg_encoder.eval()
            with torch.no_grad():
                X_t = torch.from_numpy(X_emg).to(self.device)
                # Chunk to bound peak memory on large pools.
                out = []
                bs = max(1, self.batch_size)
                for i in range(0, X_t.shape[0], bs):
                    out.append(self.emg_encoder(X_t[i:i + bs]).cpu().numpy())
            return np.concatenate(out, axis=0)

        def encode_text(self, descriptions: Sequence[str]) -> np.ndarray:
            """Encode text descriptions → (n, embed_dim) L2-normalized embeddings."""
            if not self._fitted:
                raise RuntimeError("EMGSemanticRetriever not fitted — call .fit() first")
            return self.text_encoder.encode(list(descriptions))

        # -- retrieval -------------------------------------------------------
        def retrieve_emg_by_text(
            self,
            text_query: Union[str, Sequence[str]],
            X_emg_pool: np.ndarray,
            top_k: int = 5,
        ) -> List[int]:
            """Return indices of the ``top_k`` most similar EMG windows in ``X_emg_pool``."""
            if isinstance(text_query, str):
                queries = [text_query]
            else:
                queries = list(text_query)
            text_emb = self.encode_text(queries)  # (q, embed_dim)
            emg_emb = self.encode_emg(np.asarray(X_emg_pool, dtype=np.float32))  # (p, embed_dim)
            sims = text_emb @ emg_emb.T  # (q, p)
            # Use the first query for a flat index list.
            scores = sims[0]
            k = min(top_k, scores.shape[0])
            idx = np.argsort(-scores)[:k]
            return [int(i) for i in idx]

        def retrieve_text_by_emg(
            self,
            emg_query: Union[np.ndarray, Sequence[np.ndarray]],
            descriptions_pool: Sequence[str],
            top_k: int = 5,
        ) -> List[str]:
            """Return the ``top_k`` most similar text descriptions for ``emg_query``.

            ``emg_query`` may be a single window of shape
            ``(n_channels, n_samples)`` or a batch of shape
            ``(B, n_channels, n_samples)``. When a batch is given, only
            the first query is used to rank the descriptions.
            """
            q = np.asarray(emg_query, dtype=np.float32)
            if q.ndim == 2:
                q = q[None, ...]  # (1, n_channels, n_samples)
            elif q.ndim != 3:
                raise ValueError(
                    f"emg_query must be 2-D or 3-D, got shape {q.shape}"
                )
            emg_emb = self.encode_emg(q)  # (q, embed_dim)
            text_emb = self.encode_text(list(descriptions_pool))  # (p, embed_dim)
            sims = emg_emb @ text_emb.T  # (q, p)
            scores = sims[0]
            k = min(top_k, scores.shape[0])
            idx = np.argsort(-scores)[:k]
            return [descriptions_pool[int(i)] for i in idx]

        def compute_similarity(
            self,
            emg_embedding: np.ndarray,
            text_embedding: np.ndarray,
        ) -> float:
            """Cosine similarity between a single EMG and text embedding."""
            a = np.asarray(emg_embedding, dtype=np.float32).ravel()
            b = np.asarray(text_embedding, dtype=np.float32).ravel()
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na == 0 or nb == 0:
                return 0.0
            return float(np.dot(a, b) / (na * nb))

        def evaluate_retrieval(
            self,
            X_emg_test: np.ndarray,
            descriptions_test: Sequence[str],
            k_values: Sequence[int] = (1, 5, 10),
        ) -> Dict[str, Any]:
            """Compute Recall@k in both directions.

            A retrieval at rank k counts as correct if the top-k contains
            ANY item that shares the same description text as the query's
            true pair — not only the single positionally-aligned index.
            This matters whenever descriptions repeat, e.g. many EMG
            windows sharing one gesture phrase like "close fist tightly"
            (the common case for gesture-labeled EMG data): crediting only
            the exact row index would silently undercount every correct
            retrieval except one per duplicate group, making recall@k
            artificially low even when retrieval is working correctly.
            Reduces to the original exact-row semantics when every
            description is unique.
            """
            emg_emb = self.encode_emg(np.asarray(X_emg_test, dtype=np.float32))
            text_emb = self.encode_text(list(descriptions_test))
            sims = emg_emb @ text_emb.T  # (n, n)
            n = sims.shape[0]

            desc_list = list(descriptions_test)
            groups: Dict[str, List[int]] = {}
            for i, d in enumerate(desc_list):
                groups.setdefault(d, []).append(i)

            emg_order = np.argsort(-sims, axis=1)      # (n, n): ranked text idx per EMG row
            text_order = np.argsort(-sims, axis=0).T   # (n, n): ranked EMG idx per text row

            out: Dict[str, Any] = {}
            e2t_first_rank = np.empty(n)
            t2e_first_rank = np.empty(n)
            for i in range(n):
                true_set = set(groups[desc_list[i]])
                e_ranks = [r for r, idx in enumerate(emg_order[i]) if idx in true_set]
                t_ranks = [r for r, idx in enumerate(text_order[i]) if idx in true_set]
                e2t_first_rank[i] = e_ranks[0] if e_ranks else n
                t2e_first_rank[i] = t_ranks[0] if t_ranks else n

            for k in k_values:
                out[f"emg_to_text_recall@{k}"] = float(np.mean(e2t_first_rank < k))
                out[f"text_to_emg_recall@{k}"] = float(np.mean(t2e_first_rank < k))
            out["mean_rank_emg_to_text"] = float(e2t_first_rank.mean())
            out["mean_rank_text_to_emg"] = float(t2e_first_rank.mean())
            out["n_samples"] = int(n)
            return out

        # -- BaseModel plumbing ----------------------------------------------
        def predict(self, X: np.ndarray) -> np.ndarray:
            """Return EMG embeddings for ``X`` (alias for :meth:`encode_emg`)."""
            return self.encode_emg(X)

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            # Not meaningful for a retriever — return embeddings for API parity.
            return self.encode_emg(X)

        def count_parameters(self) -> int:
            return self.emg_encoder.count_parameters() + (
                self.text_encoder.projection_.size if self.text_encoder.projection_ is not None else 0
            )

        def training_history(self) -> Dict[str, Any]:
            return {"info_nce_loss": list(self._loss_history)}

        def _serializable_config(self) -> Dict[str, Any]:
            return {
                "n_channels": self.n_channels,
                "n_samples": self.n_samples,
                "embed_dim": self.embed_dim,
                "patch_len": self.patch_len,
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "n_layers": self.n_layers,
                "dropout": self.dropout,
                "temperature": self.temperature,
                "n_epochs": self.n_epochs,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "weight_decay": self.weight_decay,
                "device": "cpu",
                "random_state": self.random_state,
            }

        def save(self, path: Union[str, Path]) -> None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "emg_encoder_state_dict": self.emg_encoder.state_dict(),
                    "text_encoder_state": self.text_encoder.to_state(),
                    "loss_history": self._loss_history,
                    "fitted": self._fitted,
                    "config": self._serializable_config(),
                },
                path,
            )

        @classmethod
        def load(cls, path: Union[str, Path]) -> EMGSemanticRetriever:
            state = torch.load(path, map_location="cpu", weights_only=True)
            cfg = state["config"]
            obj = cls(**cfg)
            obj.emg_encoder.load_state_dict(state["emg_encoder_state_dict"])
            obj.text_encoder = TextEncoder.from_state(state["text_encoder_state"])
            obj._loss_history = state.get("loss_history", [])
            obj._fitted = state.get("fitted", True)
            return obj

else:

    @register_model("semantic_retriever")
    class EMGSemanticRetriever(BaseModel):  # type: ignore[no-redef]
        """Stub registered when PyTorch is unavailable — raises on use."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "EMGSemanticRetriever requires PyTorch. Install with: pip install torch"
            )

        def fit(self, X_emg, descriptions, **kwargs):
            raise ImportError("PyTorch required")

        def predict(self, X):
            raise ImportError("PyTorch required")

        def save(self, path):
            raise ImportError("PyTorch required")

        @classmethod
        def load(cls, path):
            raise ImportError("PyTorch required")


__all__ = ["EMGTextEncoder", "TextEncoder", "EMGSemanticRetriever"]

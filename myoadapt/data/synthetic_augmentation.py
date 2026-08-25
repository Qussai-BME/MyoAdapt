"""
synthetic_augmentation.py — Synthetic EMG data augmentation.

Generates realistic synthetic sEMG signals to expand small datasets,
addressing the data scarcity problem in EMG research. Uses two
approaches:
- GAN-based: a lightweight generator/discriminator pair
- Diffusion-based: a simple DDPM (denoising diffusion probabilistic model)

Both approaches are CPU-native and produce EMG windows of the same
shape as the training data. The augmented data can be used to:
- Balance underrepresented classes
- Simulate new subjects (for LOSO robustness)
- Pretrain foundation models

References:
- Goodfellow et al. (2014). "Generative Adversarial Networks."
- Ho et al. (2020). "Denoising Diffusion Probabilistic Models."
- Soriano et al. (2024). "Data augmentation for sEMG gesture
  recognition." JNER.

License: Apache 2.0
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

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
    logger.info("PyTorch not installed - synthetic augmenters will raise on instantiation")


# Conditional base class so that ``class EMGGenerator(nn.Module if HAS_TORCH
# else object)`` works without a NameError on ``nn`` when torch is missing.
if HAS_TORCH:
    _ModuleBase = nn.Module
else:
    _ModuleBase = object


# ===========================================================================
# GAN primitives
# ===========================================================================
class EMGGenerator(_ModuleBase):
    """Lightweight EMG generator: Linear → ReLU → Linear → Tanh.

    Maps a latent vector of shape ``(batch, latent_dim)`` to a synthetic
    EMG window of shape ``(batch, n_channels, n_samples)``. The Tanh
    non-linearity bounds the output to ``[-1, 1]`` — the augmenter
    handles per-channel normalization / denormalization around training.

    Parameters
    ----------
    latent_dim : int (default 64)
    n_channels : int (default 12)
    n_samples : int (default 400) — window length in samples
    hidden_dim : int (default 256)
    """

    def __init__(
        self,
        latent_dim: int = 64,
        n_channels: int = 12,
        n_samples: int = 400,
        hidden_dim: int = 256,
    ):
        if not HAS_TORCH:
            raise ImportError("EMGGenerator requires PyTorch. Install with: pip install torch")
        super().__init__()
        self.latent_dim = latent_dim
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.hidden_dim = hidden_dim

        out_dim = n_channels * n_samples
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z : (batch, latent_dim) → (batch, n_channels, n_samples)."""
        h = self.net(z)
        return h.view(-1, self.n_channels, self.n_samples)

    def count_parameters(self) -> int:
        if not HAS_TORCH:
            return 0
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EMGDiscriminator(_ModuleBase):
    """Lightweight EMG discriminator: flatten → Linear → LeakyReLU → Linear → Sigmoid.

    Parameters
    ----------
    n_channels : int (default 12)
    n_samples : int (default 400)
    hidden_dim : int (default 256)
    """

    def __init__(
        self,
        n_channels: int = 12,
        n_samples: int = 400,
        hidden_dim: int = 256,
    ):
        if not HAS_TORCH:
            raise ImportError("EMGDiscriminator requires PyTorch. Install with: pip install torch")
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.hidden_dim = hidden_dim

        in_dim = n_channels * n_samples
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, n_channels, n_samples) → (batch, 1) probability."""
        h = x.view(x.shape[0], -1)
        return self.net(h)

    def count_parameters(self) -> int:
        if not HAS_TORCH:
            return 0
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# Quality evaluation utilities (shared by GAN and diffusion augmenters)
# ===========================================================================
def _window_stats(X: np.ndarray) -> np.ndarray:
    """Compute per-window summary statistics: [mean, std, rms, zcr] per channel.

    X : (n, C, T) → returns (n, 4*C) float32 array.
    """
    X = np.asarray(X, dtype=np.float32)
    means = X.mean(axis=2)
    stds = X.std(axis=2)
    rms = np.sqrt((X ** 2).mean(axis=2))
    # Zero-crossing rate (per window per channel).
    sign_changes = np.diff(np.sign(X), axis=2) != 0
    zcr = sign_changes.mean(axis=2)
    feats = np.concatenate([means, stds, rms, zcr], axis=1)
    return feats.astype(np.float32)


def _mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: Optional[float] = None) -> float:
    """Unbiased Gaussian-kernel Maximum Mean Discrepancy between two sample sets.

    Returns a float in [0, ~2]. Lower is better (0 ≡ identical distributions).
    """
    from scipy.spatial.distance import cdist
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    if X.shape[0] < 2 or Y.shape[0] < 2:
        return float("nan")
    if gamma is None:
        # Median heuristic on the combined set.
        Z = np.vstack([X, Y])
        d2 = cdist(Z, Z, "sqeuclidean")
        med = np.median(d2[d2 > 0])
        gamma = 1.0 / (med + 1e-12)
    Kxx = np.exp(-gamma * cdist(X, X, "sqeuclidean"))
    Kyy = np.exp(-gamma * cdist(Y, Y, "sqeuclidean"))
    Kxy = np.exp(-gamma * cdist(X, Y, "sqeuclidean"))
    return float(Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean())


def _mean_psd(X: np.ndarray, fs: int = 2000, nperseg: int = 256) -> np.ndarray:
    """Mean power spectral density, averaged over windows and channels."""
    from scipy.signal import welch
    X = np.asarray(X, dtype=np.float32)
    n, C, T = X.shape
    nperseg = min(nperseg, T)
    psds = []
    for ch in range(C):
        _, p = welch(X[:, ch, :], fs=fs, axis=-1, nperseg=nperseg)
        psds.append(p.mean(axis=0))  # average over windows
    return np.array(psds, dtype=np.float32).mean(axis=0)  # average over channels


def _evaluate_augmentation_quality(
    X_real: np.ndarray,
    X_synthetic: np.ndarray,
    fs: int = 2000,
) -> Dict[str, Any]:
    """Shared quality-evaluation routine used by both augmenters."""
    X_real = np.asarray(X_real, dtype=np.float32)
    X_synthetic = np.asarray(X_synthetic, dtype=np.float32)

    def _stats(X):
        return {
            "mean": float(X.mean()),
            "std": float(X.std()),
            "rms": float(np.sqrt((X ** 2).mean())),
            "max_abs": float(np.abs(X).max()),
            "min": float(X.min()),
            "max": float(X.max()),
        }

    real_stats = _stats(X_real)
    syn_stats = _stats(X_synthetic)

    # Spectral similarity (Pearson correlation of mean PSDs, clamped to [-1, 1]).
    try:
        psd_r = _mean_psd(X_real, fs=fs)
        psd_s = _mean_psd(X_synthetic, fs=fs)
        if psd_r.std() > 0 and psd_s.std() > 0:
            spec_sim = float(np.corrcoef(psd_r, psd_s)[0, 1])
        else:
            spec_sim = 0.0
    except Exception as exc:  # pragma: no cover — welch can fail on tiny arrays
        logger.warning(f"PSD computation failed: {exc}")
        spec_sim = float("nan")

    # MMD on window summary statistics.
    try:
        feat_r = _window_stats(X_real)
        feat_s = _window_stats(X_synthetic)
        mmd = _mmd_rbf(feat_r, feat_s)
    except Exception as exc:  # pragma: no cover
        logger.warning(f"MMD computation failed: {exc}")
        mmd = float("nan")

    return {
        "real_signal_stats": real_stats,
        "synthetic_signal_stats": syn_stats,
        "spectral_similarity": spec_sim,
        "mmd": mmd,
    }


# ===========================================================================
# GAN augmenter
# ===========================================================================
if HAS_TORCH:

    class EMGGANAugmenter:
        """GAN-based synthetic EMG augmenter.

        Trains a lightweight generator/discriminator pair with the
        standard non-saturating GAN objective, then samples synthetic
        windows from the learned generator distribution. Inputs are
        normalized per channel to ``[-1, 1]`` (matching the generator's
        Tanh output) and denormalized on the way out.

        Parameters
        ----------
        n_channels, n_samples : int — EMG window shape (default 12, 400)
        latent_dim : int (default 64)
        n_epochs : int (default 100)
        lr : float (default 2e-4)
        batch_size : int (default 32)
        device : 'auto' | 'cpu' | 'cuda'
        random_state : int (default 42)
        fs : int (default 2000) — sampling rate, used only for quality eval
        """

        def __init__(
            self,
            n_channels: int = 12,
            n_samples: int = 400,
            latent_dim: int = 64,
            hidden_dim: int = 256,
            n_epochs: int = 100,
            lr: float = 2e-4,
            batch_size: int = 32,
            device: str = "auto",
            random_state: int = 42,
            fs: int = 2000,
        ):
            self.n_channels = n_channels
            self.n_samples = n_samples
            self.latent_dim = latent_dim
            self.hidden_dim = hidden_dim
            self.n_epochs = n_epochs
            self.lr = lr
            self.batch_size = batch_size
            self.device = (
                "cuda" if device == "auto" and torch.cuda.is_available()
                else "cpu" if device == "auto" else device
            )
            self.random_state = random_state
            self.fs = fs

            self.generator: Optional[EMGGenerator] = None
            self.discriminator: Optional[EMGDiscriminator] = None
            self.data_min_: Optional[np.ndarray] = None  # (C, 1)
            self.data_max_: Optional[np.ndarray] = None  # (C, 1)
            self._d_loss_history: list = []
            self._g_loss_history: list = []
            self._fitted = False
            torch.manual_seed(random_state)

        # -- normalization --------------------------------------------------
        def _normalize(self, X: np.ndarray) -> np.ndarray:
            """Per-channel min-max normalization to ``[-1, 1]``."""
            X = X.astype(np.float32, copy=False)
            mn = self.data_min_  # (C, 1)
            mx = self.data_max_
            span = mx - mn
            span[span == 0] = 1.0
            return 2.0 * (X - mn) / span - 1.0

        def _denormalize(self, X: np.ndarray) -> np.ndarray:
            X = X.astype(np.float32, copy=False)
            mn = self.data_min_
            mx = self.data_max_
            return (X + 1.0) / 2.0 * (mx - mn) + mn

        # -- training -------------------------------------------------------
        def fit(self, X_real: np.ndarray) -> EMGGANAugmenter:
            """Train the GAN on real EMG windows.

            Parameters
            ----------
            X_real : (n_windows, n_channels, n_samples_raw) array
            """
            X = np.asarray(X_real, dtype=np.float32)
            if X.ndim != 3:
                raise ValueError(f"X_real must be 3-D (n, n_channels, n_samples), got {X.shape}")
            if X.shape[1] != self.n_channels or X.shape[2] != self.n_samples:
                raise ValueError(
                    f"X_real shape {X.shape} incompatible with "
                    f"(n_channels={self.n_channels}, n_samples={self.n_samples})"
                )
            if X.shape[0] < 2:
                raise ValueError("Need at least 2 windows to train a GAN")

            # Per-channel normalization stats.
            self.data_min_ = X.min(axis=(0, 2), keepdims=False).reshape(self.n_channels, 1)
            self.data_max_ = X.max(axis=(0, 2), keepdims=False).reshape(self.n_channels, 1)

            Xn = self._normalize(X)
            X_t = torch.from_numpy(Xn).to(self.device)

            self.generator = EMGGenerator(
                latent_dim=self.latent_dim,
                n_channels=self.n_channels,
                n_samples=self.n_samples,
                hidden_dim=self.hidden_dim,
            ).to(self.device)
            self.discriminator = EMGDiscriminator(
                n_channels=self.n_channels,
                n_samples=self.n_samples,
                hidden_dim=self.hidden_dim,
            ).to(self.device)

            opt_G = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
            opt_D = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
            bce = nn.BCELoss()

            ds = TensorDataset(X_t)
            loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

            self._d_loss_history = []
            self._g_loss_history = []
            log_every = max(1, self.n_epochs // 5)
            self.generator.train()
            self.discriminator.train()
            for epoch in range(self.n_epochs):
                d_tot, g_tot, n_batches = 0.0, 0.0, 0
                for (xb,) in loader:
                    bs = xb.shape[0]
                    real_label = torch.ones(bs, 1, device=self.device)
                    fake_label = torch.zeros(bs, 1, device=self.device)

                    # --- Discriminator step ---
                    opt_D.zero_grad()
                    z = torch.randn(bs, self.latent_dim, device=self.device)
                    fake = self.generator(z)
                    d_real = self.discriminator(xb)
                    d_fake_det = self.discriminator(fake.detach())
                    loss_D = bce(d_real, real_label) + bce(d_fake_det, fake_label)
                    loss_D.backward()
                    opt_D.step()

                    # --- Generator step (non-saturating) ---
                    opt_G.zero_grad()
                    d_fake = self.discriminator(fake)
                    loss_G = bce(d_fake, real_label)
                    loss_G.backward()
                    opt_G.step()

                    d_tot += float(loss_D.item())
                    g_tot += float(loss_G.item())
                    n_batches += 1
                self._d_loss_history.append(d_tot / max(n_batches, 1))
                self._g_loss_history.append(g_tot / max(n_batches, 1))
                if (epoch + 1) % log_every == 0:
                    logger.debug(
                        f"  [EMGGANAugmenter] epoch {epoch+1}/{self.n_epochs} "
                        f"D={self._d_loss_history[-1]:.4f} "
                        f"G={self._g_loss_history[-1]:.4f}"
                    )
            self._fitted = True
            return self

        # -- generation -----------------------------------------------------
        def generate(self, n_samples: int) -> np.ndarray:
            """Generate ``n_samples`` synthetic EMG windows.

            ``n_samples`` here is the *number of windows* (count), not the
            window length — the window length is :attr:`self.n_samples`.
            """
            if not self._fitted or self.generator is None:
                raise RuntimeError("EMGGANAugmenter not fitted — call .fit() first")
            self.generator.eval()
            out_chunks = []
            bs = max(1, self.batch_size)
            remaining = int(n_samples)
            with torch.no_grad():
                while remaining > 0:
                    k = min(bs, remaining)
                    z = torch.randn(k, self.latent_dim, device=self.device)
                    fake = self.generator(z).cpu().numpy()
                    out_chunks.append(fake)
                    remaining -= k
            Xn = np.concatenate(out_chunks, axis=0)
            return self._denormalize(Xn)

        def augment(self, X_real: np.ndarray, augmentation_factor: int = 2) -> np.ndarray:
            """Return real data concatenated with synthetic data.

            ``augmentation_factor=2`` means the returned array is twice the
            size of ``X_real`` (i.e. as many synthetic as real windows are
            generated and concatenated).
            """
            X_real = np.asarray(X_real, dtype=np.float32)
            n = X_real.shape[0]
            n_syn = max(0, int(augmentation_factor - 1)) * n
            if n_syn == 0:
                return X_real.copy()
            X_syn = self.generate(n_syn)
            return np.concatenate([X_real, X_syn], axis=0)

        # -- quality --------------------------------------------------------
        def evaluate_quality(self, X_real: np.ndarray, X_synthetic: np.ndarray) -> Dict[str, Any]:
            """Compare real vs. synthetic distributions.

            Returns mean signal statistics, spectral similarity (Pearson
            correlation of mean PSDs), and MMD on per-window summary
            statistics.
            """
            return _evaluate_augmentation_quality(
                X_real, X_synthetic, fs=self.fs,
            )

        # -- (de)serialization ----------------------------------------------
        def training_history(self) -> Dict[str, Any]:
            return {"d_loss": list(self._d_loss_history), "g_loss": list(self._g_loss_history)}

        def _serializable_config(self) -> Dict[str, Any]:
            return {
                "n_channels": self.n_channels,
                "n_samples": self.n_samples,
                "latent_dim": self.latent_dim,
                "hidden_dim": self.hidden_dim,
                "n_epochs": self.n_epochs,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "device": "cpu",
                "random_state": self.random_state,
                "fs": self.fs,
            }

        def save(self, path: Union[str, Path]) -> None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not self._fitted or self.generator is None or self.discriminator is None:
                raise RuntimeError("Cannot save an unfitted EMGGANAugmenter")
            torch.save(
                {
                    "generator_state_dict": self.generator.state_dict(),
                    "discriminator_state_dict": self.discriminator.state_dict(),
                    "data_min_": torch.as_tensor(self.data_min_, dtype=torch.float32),
                    "data_max_": torch.as_tensor(self.data_max_, dtype=torch.float32),
                    "d_loss_history": self._d_loss_history,
                    "g_loss_history": self._g_loss_history,
                    "config": self._serializable_config(),
                },
                path,
            )

        @classmethod
        def load(cls, path: Union[str, Path]) -> EMGGANAugmenter:
            state = torch.load(path, map_location="cpu", weights_only=True)
            cfg = state["config"]
            obj = cls(**cfg)
            obj.generator = EMGGenerator(
                latent_dim=cfg["latent_dim"],
                n_channels=cfg["n_channels"],
                n_samples=cfg["n_samples"],
                hidden_dim=cfg["hidden_dim"],
            )
            obj.discriminator = EMGDiscriminator(
                n_channels=cfg["n_channels"],
                n_samples=cfg["n_samples"],
                hidden_dim=cfg["hidden_dim"],
            )
            obj.generator.load_state_dict(state["generator_state_dict"])
            obj.discriminator.load_state_dict(state["discriminator_state_dict"])
            obj.data_min_ = state["data_min_"].cpu().numpy()
            obj.data_max_ = state["data_max_"].cpu().numpy()
            obj._d_loss_history = state.get("d_loss_history", [])
            obj._g_loss_history = state.get("g_loss_history", [])
            obj._fitted = True
            return obj

else:

    class EMGGANAugmenter:  # type: ignore[no-redef]
        """Stub registered when PyTorch is unavailable — raises on use."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "EMGGANAugmenter requires PyTorch. Install with: pip install torch"
            )


# ===========================================================================
# Diffusion augmenter (DDPM)
# ===========================================================================
if HAS_TORCH:

    class _DenoiseNet(nn.Module):
        """Simple 1-D denoising network for DDPM.

        Architecture: Conv1d patch + sinusoidal time embedding +
        N residual Conv1d blocks with FiLM-style time modulation +
        output Conv1d. No downsampling/upsampling (window length is
        preserved throughout).
        """

        def __init__(
            self,
            n_channels: int = 12,
            n_samples: int = 400,
            hidden_dim: int = 64,
            n_blocks: int = 3,
            time_dim: int = 64,
        ):
            super().__init__()
            self.n_channels = n_channels
            self.n_samples = n_samples
            self.hidden_dim = hidden_dim
            self.time_dim = time_dim

            # Time embedding: sinusoidal → MLP.
            self.time_mlp = nn.Sequential(
                nn.Linear(time_dim, time_dim * 2),
                nn.SiLU(),
                nn.Linear(time_dim * 2, time_dim),
            )

            # Input projection (channel mixing).
            self.in_conv = nn.Conv1d(n_channels, hidden_dim, kernel_size=7, padding=3)
            # Output projection back to signal space.
            self.out_conv = nn.Conv1d(hidden_dim, n_channels, kernel_size=7, padding=3)

            # Residual blocks.
            self.blocks = nn.ModuleList()
            for _ in range(n_blocks):
                self.blocks.append(_ResBlock(hidden_dim, time_dim))

        def _sinusoidal_embedding(self, t: torch.Tensor) -> torch.Tensor:
            """t : (B,) long → (B, time_dim)."""
            half = self.time_dim // 2
            freqs = torch.exp(
                -torch.arange(half, device=t.device, dtype=torch.float32)
                * (np.log(10000.0) / max(half, 1))
            )
            args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
            emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
            if emb.shape[1] < self.time_dim:
                emb = F.pad(emb, (0, self.time_dim - emb.shape[1]))
            return emb  # (B, time_dim)

        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """x : (B, C, T), t : (B,) long → (B, C, T) predicted noise."""
            temb = self.time_mlp(self._sinusoidal_embedding(t))  # (B, time_dim)
            h = self.in_conv(x)  # (B, hidden, T)
            for block in self.blocks:
                h = block(h, temb)
            return self.out_conv(h)

        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    class _ResBlock(nn.Module):
        """Residual Conv1d block with FiLM-style time conditioning."""

        def __init__(self, hidden_dim: int, time_dim: int):
            super().__init__()
            self.norm1 = nn.GroupNorm(min(8, hidden_dim), hidden_dim)
            self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
            self.time_proj = nn.Linear(time_dim, hidden_dim)
            self.norm2 = nn.GroupNorm(min(8, hidden_dim), hidden_dim)
            self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)

        def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
            h = self.conv1(F.silu(self.norm1(x)))
            # FiLM: add time embedding broadcast over time.
            h = h + self.time_proj(temb).unsqueeze(-1)  # (B, hidden, 1)
            h = self.conv2(F.silu(self.norm2(h)))
            return x + h

    class EMGDiffusionAugmenter:
        """Simple DDPM for 1-D EMG signals.

        Implements the discrete-time denoising diffusion probabilistic
        model from Ho et al. (2020): a fixed forward noise schedule
        (linear beta) and a learned reverse process parameterized by a
        Conv1d denoising network conditioned on the diffusion timestep.

        Inputs are normalized per channel to zero-mean unit-variance for
        training and denormalized on the way out.

        Parameters
        ----------
        n_channels, n_samples : int — EMG window shape (default 12, 400)
        n_timesteps : int (default 100) — number of diffusion steps T
        beta_start, beta_end : float — linear schedule endpoints
        hidden_dim : int (default 64) — denoising net hidden width
        n_blocks : int (default 3) — number of residual blocks
        n_epochs : int (default 100)
        lr : float (default 1e-3)
        batch_size : int (default 32)
        device : 'auto' | 'cpu' | 'cuda'
        random_state : int (default 42)
        fs : int (default 2000) — sampling rate for quality eval only
        """

        def __init__(
            self,
            n_channels: int = 12,
            n_samples: int = 400,
            n_timesteps: int = 100,
            beta_start: float = 1e-4,
            beta_end: float = 0.02,
            hidden_dim: int = 64,
            n_blocks: int = 3,
            time_dim: int = 64,
            n_epochs: int = 100,
            lr: float = 1e-3,
            batch_size: int = 32,
            device: str = "auto",
            random_state: int = 42,
            fs: int = 2000,
        ):
            self.n_channels = n_channels
            self.n_samples = n_samples
            self.n_timesteps = n_timesteps
            self.beta_start = beta_start
            self.beta_end = beta_end
            self.hidden_dim = hidden_dim
            self.n_blocks = n_blocks
            self.time_dim = time_dim
            self.n_epochs = n_epochs
            self.lr = lr
            self.batch_size = batch_size
            self.device = (
                "cuda" if device == "auto" and torch.cuda.is_available()
                else "cpu" if device == "auto" else device
            )
            self.random_state = random_state
            self.fs = fs

            self.net: Optional[_DenoiseNet] = None
            self.data_mean_: Optional[np.ndarray] = None  # (C, 1)
            self.data_std_: Optional[np.ndarray] = None  # (C, 1)
            self._loss_history: list = []
            self._fitted = False
            # Precompute noise schedule (lazy, after device is known).
            self._betas: Optional[np.ndarray] = None
            self._alphas: Optional[np.ndarray] = None
            self._alpha_bars: Optional[np.ndarray] = None
            torch.manual_seed(random_state)

        # -- noise schedule -------------------------------------------------
        def _build_schedule(self) -> None:
            betas = np.linspace(self.beta_start, self.beta_end, self.n_timesteps, dtype=np.float32)
            alphas = 1.0 - betas
            alpha_bars = np.cumprod(alphas)
            self._betas = betas
            self._alphas = alphas
            self._alpha_bars = alpha_bars
            # Torch buffers on the right device.
            self._betas_t = torch.from_numpy(betas).to(self.device)
            self._alphas_t = torch.from_numpy(alphas).to(self.device)
            self._alpha_bars_t = torch.from_numpy(alpha_bars).to(self.device)
            self._sqrt_alpha_bars_t = torch.sqrt(self._alpha_bars_t)
            self._sqrt_one_minus_alpha_bars_t = torch.sqrt(1.0 - self._alpha_bars_t)
            self._sqrt_alphas_t = torch.sqrt(self._alphas_t)

        # -- normalization --------------------------------------------------
        def _normalize(self, X: np.ndarray) -> np.ndarray:
            X = X.astype(np.float32, copy=False)
            std = self.data_std_
            std_safe = np.where(std == 0, 1.0, std)
            return (X - self.data_mean_) / std_safe

        def _denormalize(self, X: np.ndarray) -> np.ndarray:
            X = X.astype(np.float32, copy=False)
            return X * self.data_std_ + self.data_mean_

        # -- training -------------------------------------------------------
        def fit(self, X_real: np.ndarray) -> EMGDiffusionAugmenter:
            """Train the denoising network on real EMG windows."""
            X = np.asarray(X_real, dtype=np.float32)
            if X.ndim != 3:
                raise ValueError(f"X_real must be 3-D (n, n_channels, n_samples), got {X.shape}")
            if X.shape[1] != self.n_channels or X.shape[2] != self.n_samples:
                raise ValueError(
                    f"X_real shape {X.shape} incompatible with "
                    f"(n_channels={self.n_channels}, n_samples={self.n_samples})"
                )
            if X.shape[0] < 2:
                raise ValueError("Need at least 2 windows to train the diffusion model")

            # Per-channel mean/std normalization (target ~ N(0, 1)).
            self.data_mean_ = X.mean(axis=(0, 2), keepdims=False).reshape(self.n_channels, 1)
            self.data_std_ = X.std(axis=(0, 2), keepdims=False).reshape(self.n_channels, 1)

            self._build_schedule()
            Xn = self._normalize(X)
            X_t = torch.from_numpy(Xn).to(self.device)

            self.net = _DenoiseNet(
                n_channels=self.n_channels,
                n_samples=self.n_samples,
                hidden_dim=self.hidden_dim,
                n_blocks=self.n_blocks,
                time_dim=self.time_dim,
            ).to(self.device)

            optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
            mse = nn.MSELoss()

            ds = TensorDataset(X_t)
            loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

            self._loss_history = []
            log_every = max(1, self.n_epochs // 5)
            self.net.train()
            for epoch in range(self.n_epochs):
                tot, n_batches = 0.0, 0
                for (xb,) in loader:
                    bs = xb.shape[0]
                    # Sample t uniformly and noise.
                    t = torch.randint(0, self.n_timesteps, (bs,), device=self.device)
                    eps = torch.randn_like(xb)
                    ab = self._alpha_bars_t[t].view(bs, 1, 1)
                    sqrt_ab = self._sqrt_alpha_bars_t[t].view(bs, 1, 1)
                    sqrt_omab = self._sqrt_one_minus_alpha_bars_t[t].view(bs, 1, 1)
                    x_t = sqrt_ab * xb + sqrt_omab * eps
                    eps_pred = self.net(x_t, t)
                    loss = mse(eps_pred, eps)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    tot += float(loss.item())
                    n_batches += 1
                self._loss_history.append(tot / max(n_batches, 1))
                if (epoch + 1) % log_every == 0:
                    logger.debug(
                        f"  [EMGDiffusionAugmenter] epoch {epoch+1}/{self.n_epochs} "
                        f"mse={self._loss_history[-1]:.6f}"
                    )
            self._fitted = True
            return self

        # -- generation (reverse process) -----------------------------------
        @torch.no_grad()
        def generate(self, n_samples: int) -> np.ndarray:
            """Run the reverse diffusion process.

            ``n_samples`` is the number of windows to generate; the window
            length is :attr:`self.n_samples`.
            """
            if not self._fitted or self.net is None:
                raise RuntimeError("EMGDiffusionAugmenter not fitted — call .fit() first")
            if self._betas_t is None:
                self._build_schedule()
            self.net.eval()
            out_chunks = []
            bs = max(1, self.batch_size)
            remaining = int(n_samples)
            while remaining > 0:
                k = min(bs, remaining)
                x = torch.randn(k, self.n_channels, self.n_samples, device=self.device)
                for t_idx in reversed(range(self.n_timesteps)):
                    t = torch.full((k,), t_idx, device=self.device, dtype=torch.long)
                    eps_pred = self.net(x, t)
                    beta_t = self._betas_t[t_idx]
                    alpha_t = self._alphas_t[t_idx]
                    alpha_bar_t = self._alpha_bars_t[t_idx]
                    mean = (1.0 / torch.sqrt(alpha_t)) * (
                        x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_pred
                    )
                    if t_idx > 0:
                        z = torch.randn_like(x)
                        x = mean + torch.sqrt(beta_t) * z
                    else:
                        x = mean
                out_chunks.append(x.cpu().numpy())
                remaining -= k
            Xn = np.concatenate(out_chunks, axis=0)
            return self._denormalize(Xn)

        def augment(self, X_real: np.ndarray, augmentation_factor: int = 2) -> np.ndarray:
            """Return real data concatenated with synthetic data."""
            X_real = np.asarray(X_real, dtype=np.float32)
            n = X_real.shape[0]
            n_syn = max(0, int(augmentation_factor - 1)) * n
            if n_syn == 0:
                return X_real.copy()
            X_syn = self.generate(n_syn)
            return np.concatenate([X_real, X_syn], axis=0)

        def evaluate_quality(self, X_real: np.ndarray, X_synthetic: np.ndarray) -> Dict[str, Any]:
            """Compare real vs. synthetic distributions."""
            return _evaluate_augmentation_quality(
                X_real, X_synthetic, fs=self.fs,
            )

        # -- (de)serialization ----------------------------------------------
        def training_history(self) -> Dict[str, Any]:
            return {"mse_loss": list(self._loss_history)}

        def _serializable_config(self) -> Dict[str, Any]:
            return {
                "n_channels": self.n_channels,
                "n_samples": self.n_samples,
                "n_timesteps": self.n_timesteps,
                "beta_start": self.beta_start,
                "beta_end": self.beta_end,
                "hidden_dim": self.hidden_dim,
                "n_blocks": self.n_blocks,
                "time_dim": self.time_dim,
                "n_epochs": self.n_epochs,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "device": "cpu",
                "random_state": self.random_state,
                "fs": self.fs,
            }

        def save(self, path: Union[str, Path]) -> None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not self._fitted or self.net is None:
                raise RuntimeError("Cannot save an unfitted EMGDiffusionAugmenter")
            torch.save(
                {
                    "net_state_dict": self.net.state_dict(),
                    "data_mean_": torch.as_tensor(self.data_mean_, dtype=torch.float32),
                    "data_std_": torch.as_tensor(self.data_std_, dtype=torch.float32),
                    "loss_history": self._loss_history,
                    "config": self._serializable_config(),
                },
                path,
            )

        @classmethod
        def load(cls, path: Union[str, Path]) -> EMGDiffusionAugmenter:
            state = torch.load(path, map_location="cpu", weights_only=True)
            cfg = state["config"]
            obj = cls(**cfg)
            obj._build_schedule()
            obj.net = _DenoiseNet(
                n_channels=cfg["n_channels"],
                n_samples=cfg["n_samples"],
                hidden_dim=cfg["hidden_dim"],
                n_blocks=cfg["n_blocks"],
                time_dim=cfg["time_dim"],
            )
            obj.net.load_state_dict(state["net_state_dict"])
            obj.data_mean_ = state["data_mean_"].cpu().numpy()
            obj.data_std_ = state["data_std_"].cpu().numpy()
            obj._loss_history = state.get("loss_history", [])
            obj._fitted = True
            return obj

else:

    class EMGDiffusionAugmenter:  # type: ignore[no-redef]
        """Stub registered when PyTorch is unavailable — raises on use."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "EMGDiffusionAugmenter requires PyTorch. Install with: pip install torch"
            )


__all__ = [
    "EMGGenerator",
    "EMGDiscriminator",
    "EMGGANAugmenter",
    "EMGDiffusionAugmenter",
]

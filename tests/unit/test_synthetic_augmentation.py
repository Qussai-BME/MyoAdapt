"""Unit tests for myoadapt.data.synthetic_augmentation (GAN + diffusion)."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from myoadapt.data.synthetic_augmentation import EMGGANAugmenter, EMGDiffusionAugmenter


def _real_emg(n=30, n_ch=4, n_samp=64, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_samp) / 2000
    return np.stack([
        np.stack([np.sin(2 * np.pi * 30 * t) * rng.uniform(0.5, 1.5)
                  + 0.2 * rng.standard_normal(n_samp) for _ in range(n_ch)])
        for _ in range(n)
    ]).astype(np.float32)


AUGMENTER_PARAMS = [
    (EMGGANAugmenter, dict(n_epochs=8, batch_size=8)),
    (EMGDiffusionAugmenter, dict(n_epochs=8, batch_size=8, n_timesteps=15)),
]


@pytest.mark.parametrize("cls,kwargs", AUGMENTER_PARAMS)
def test_generate_produces_correct_shape_no_nans(cls, kwargs):
    X_real = _real_emg()
    aug = cls(n_channels=4, n_samples=64, device="cpu", random_state=0, **kwargs)
    aug.fit(X_real)
    gen = aug.generate(12)
    assert gen.shape == (12, 4, 64)
    assert not np.isnan(gen).any()
    assert not np.isinf(gen).any()


@pytest.mark.parametrize("cls,kwargs", AUGMENTER_PARAMS)
def test_generated_amplitude_same_order_of_magnitude_as_real(cls, kwargs):
    """Not a tight statistical test — just a sanity guard against the
    generator collapsing to ~0 or exploding, which a short smoke-training
    run can silently do if something is mis-wired."""
    X_real = _real_emg(n=40)
    aug = cls(n_channels=4, n_samples=64, device="cpu", random_state=0, **kwargs)
    aug.fit(X_real)
    gen = aug.generate(20)
    real_scale = np.abs(X_real).mean()
    gen_scale = np.abs(gen).mean()
    assert 0.05 * real_scale < gen_scale < 20 * real_scale


@pytest.mark.parametrize("cls,kwargs", AUGMENTER_PARAMS)
def test_augment_includes_original_data(cls, kwargs):
    X_real = _real_emg(n=20)
    aug = cls(n_channels=4, n_samples=64, device="cpu", random_state=0, **kwargs)
    aug.fit(X_real)
    combined = aug.augment(X_real, augmentation_factor=2)
    assert combined.shape[1:] == X_real.shape[1:]
    assert combined.shape[0] >= X_real.shape[0]


@pytest.mark.parametrize("cls,kwargs", AUGMENTER_PARAMS)
def test_generate_before_fit_raises(cls, kwargs):
    aug = cls(n_channels=4, n_samples=64, device="cpu", **kwargs)
    with pytest.raises(RuntimeError):
        aug.generate(5)


@pytest.mark.parametrize("cls,kwargs", AUGMENTER_PARAMS)
def test_save_load_roundtrip_generates_without_error(cls, kwargs):
    X_real = _real_emg(n=20)
    aug = cls(n_channels=4, n_samples=64, device="cpu", random_state=0, **kwargs)
    aug.fit(X_real)
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "aug.pt")
        aug.save(path)
        aug2 = cls.load(path)
        gen = aug2.generate(5)
    assert gen.shape == (5, 4, 64)
    assert not np.isnan(gen).any()

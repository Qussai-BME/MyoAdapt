"""
Unit tests for the EMG fatigue tracking module.

Previously shipped with zero test coverage — added as part of the
v2.0 reliability pass. Includes a regression test for a real bug:
``FatigueTracker.analyze`` called ``np.atleast_2d`` *before* checking
``ndim == 1``, which silently turned a genuine (n_samples,) single-
channel trace into a (1, n_samples) "one sample, n_samples channels"
array instead of the documented (n_samples, 1) shape.
"""
from __future__ import annotations

import numpy as np
import pytest

from myoadapt.evaluation.fatigue import FatigueTracker


def _fatiguing_signal(fs=2000, n_windows=40, win_sec=0.2, seed=0):
    """MNF decreasing + RMS increasing over time -> classic fatigue signature."""
    rng = np.random.default_rng(seed)
    win_n = int(fs * win_sec)
    chunks = []
    for i in range(n_windows):
        t = np.arange(win_n) / fs
        freq = 120 - i * 1.5          # mean frequency drifts down
        amp = 1.0 + i * 0.02          # amplitude drifts up
        chunks.append(amp * np.sin(2 * np.pi * freq * t) + 0.1 * rng.standard_normal(win_n))
    return np.concatenate(chunks)


def _stable_signal(fs=2000, n_windows=40, win_sec=0.2, seed=0):
    """Constant frequency/amplitude -> no fatigue trend."""
    rng = np.random.default_rng(seed)
    win_n = int(fs * win_sec)
    t = np.arange(win_n) / fs
    chunk = np.sin(2 * np.pi * 80 * t)
    return np.concatenate([chunk + 0.1 * rng.standard_normal(win_n) for _ in range(n_windows)])


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [
    {"fs": 0}, {"fs": -100}, {"window_ms": 0}, {"increment_ms": -5},
])
def test_invalid_params_raise(kwargs):
    with pytest.raises(ValueError):
        FatigueTracker(**kwargs)


# ---------------------------------------------------------------------------
# Shape handling (regression test for the atleast_2d-before-ndim-check bug)
# ---------------------------------------------------------------------------
def test_analyze_accepts_1d_single_channel_signal():
    """A plain (n_samples,) trace must be treated as ONE channel, not
    one sample with n_samples channels."""
    signal = _fatiguing_signal()
    tracker = FatigueTracker(fs=2000, window_ms=200, increment_ms=200)
    result = tracker.analyze(signal)
    assert result["n_channels"] == 1
    assert result["n_windows"] == 40


def test_analyze_accepts_2d_multichannel_signal():
    rng = np.random.default_rng(1)
    signal = rng.standard_normal((8000, 4))  # (n_samples, n_channels)
    tracker = FatigueTracker(fs=2000, window_ms=200, increment_ms=200)
    result = tracker.analyze(signal)
    assert result["n_channels"] == 4
    assert result["n_windows"] == 20


def test_signal_too_short_raises_with_correct_message():
    tracker = FatigueTracker(fs=2000, window_ms=200, increment_ms=200)
    with pytest.raises(ValueError, match="Signal too short"):
        tracker.analyze(np.zeros(50))


# ---------------------------------------------------------------------------
# Correctness: fatigue index should discriminate fatiguing vs stable signals
# ---------------------------------------------------------------------------
def test_fatigue_index_higher_for_fatiguing_than_stable_signal():
    tracker = FatigueTracker(fs=2000, window_ms=200, increment_ms=200)
    fatiguing = tracker.analyze(_fatiguing_signal())
    stable = tracker.analyze(_stable_signal())
    assert fatiguing["fatigue_index"] > stable["fatigue_index"]
    assert fatiguing["mnf_slope"] < 0  # mean frequency decreasing


def test_fatigue_level_classification_bounds():
    tracker = FatigueTracker(fs=2000, window_ms=200, increment_ms=200)
    result = tracker.analyze(_fatiguing_signal())
    assert 0.0 <= result["fatigue_index"] <= 1.0
    assert result["fatigue_level"] in ("low", "moderate", "high")


def test_narrative_mentions_fatigue_level():
    tracker = FatigueTracker(fs=2000, window_ms=200, increment_ms=200)
    result = tracker.analyze(_fatiguing_signal())
    text = tracker.fatigue_narrative(result)
    assert isinstance(text, str) and len(text) > 0
    assert result["fatigue_level"].upper() in text

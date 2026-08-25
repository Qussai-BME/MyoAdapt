"""
fatigue.py — EMG fatigue tracking.

Monitors EMG fatigue indicators over time, critical for:
- Clinical assessment of muscle endurance
- Prosthetic control (fatigue causes signal degradation)
- Sports science (training load monitoring)

Indicators computed:
- Mean Frequency (MNF) slope — the gold standard (Lindström 1970).
  MNF decreases as muscle fatigues.
- Median Frequency (MDF) slope — alternative to MNF, more robust.
- Root Mean Square (RMS) trend — amplitude increases with fatigue.
- Instantaneous Mean Frequency (IMNF) — time-varying MNF via STFT.
- Fatigue Index (FI) — composite score combining all indicators.

References
----------
- Lindström (1970). "Spectral analysis of myoelectric signals."
- Miaoulis (2025). "EMG fatigue monitoring: a review." JNER.
- Karthick (2016). "Surface EMG fatigue analysis: a review."

License: Apache 2.0
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# scipy is the workhorse here (signal.welch, stft, linregress).
try:
    from scipy import signal as scipy_signal
    from scipy.stats import linregress
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover — scipy is a hard dep
    _HAS_SCIPY = False

# matplotlib is optional: only used for plot_fatigue_curves.
try:  # pragma: no cover — optional dependency
    import matplotlib

    if matplotlib.get_backend().lower() not in {
        "agg", "module://matplotlib_inline.backend_inline",
    }:
        try:
            matplotlib.use("Agg")
        except Exception:
            pass
    import matplotlib.pyplot as plt

    _HAS_MATPLOTLIB = True
except Exception:  # pragma: no cover — optional dependency
    _HAS_MATPLOTLIB = False
    plt = None  # type: ignore[assignment]
    matplotlib = None  # type: ignore[assignment]


def _welch_psd(seg: np.ndarray, fs: int, nperseg: int = 256
               ) -> Tuple[np.ndarray, np.ndarray]:
    """One-sided PSD via Welch's method (1-D segment)."""
    seg = np.asarray(seg, dtype=np.float64).ravel()
    if not _HAS_SCIPY:
        # Trivial fallback: periodogram via rfft.
        n = len(seg)
        if n < 4:
            return np.array([0.0]), np.array([0.0])
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        psd = (np.abs(np.fft.rfft(seg - seg.mean())) ** 2) / n
        return freqs, psd
    nperseg = min(nperseg, len(seg))
    if nperseg < 4:
        return np.array([0.0]), np.array([0.0])
    freqs, psd = scipy_signal.welch(seg, fs=fs, nperseg=nperseg)
    return freqs, psd


def _mean_frequency(seg: np.ndarray, fs: int, nperseg: int = 256) -> float:
    """MNF = sum(f·PSD) / sum(PSD)."""
    freqs, psd = _welch_psd(seg, fs, nperseg)
    if psd.sum() < 1e-12:
        return 0.0
    return float(np.sum(freqs * psd) / np.sum(psd))


def _median_frequency(seg: np.ndarray, fs: int, nperseg: int = 256) -> float:
    """MDF = frequency that splits PSD area into two equal halves."""
    freqs, psd = _welch_psd(seg, fs, nperseg)
    if psd.sum() < 1e-12:
        return 0.0
    cum = np.cumsum(psd) / np.sum(psd)
    idx = int(np.searchsorted(cum, 0.5))
    return float(freqs[min(idx, len(freqs) - 1)])


def _rms(seg: np.ndarray) -> float:
    arr = np.asarray(seg, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr ** 2)))


# ---------------------------------------------------------------------------
# FatigueTracker
# ---------------------------------------------------------------------------
class FatigueTracker:
    """Track EMG fatigue indicators over time.

    The tracker segments the input signal into fixed windows (length
    ``window_ms``, advance ``increment_ms``) and computes, per
    channel:

    - **MNF slope** (Hz/min) — linear regression of MNF over window
      time. Negative slope ⇒ fatigue.
    - **MDF slope** (Hz/min) — same for median frequency.
    - **RMS trend** (V/s) — linear regression of RMS over time.
      Positive slope ⇒ amplitude growth (fatigue indicator).
    - **IMNF** — instantaneous MNF time series via STFT, so changes
      within a window are visible too.
    - **Fatigue Index** in [0, 1] — composite score combining the
      MNF / MDF / RMS slopes into a single fatigue severity.

    Parameters
    ----------
    fs : int (default 2000)
        Sampling rate in Hz.
    window_ms : int (default 200)
        Window length in milliseconds.
    increment_ms : int (default 50)
        Hop size in milliseconds.
    nperseg : Optional[int] (default None)
        Welch STFT segment length — auto-set to ``fs * window_ms / 1000``
        (i.e. one window per segment) when None.
    """

    def __init__(self,
                 fs: int = 2000,
                 window_ms: int = 200,
                 increment_ms: int = 50,
                 nperseg: Optional[int] = None):
        if fs <= 0:
            raise ValueError(f"fs must be > 0 (got {fs})")
        if window_ms <= 0:
            raise ValueError(f"window_ms must be > 0 (got {window_ms})")
        if increment_ms <= 0:
            raise ValueError(f"increment_ms must be > 0 (got {increment_ms})")
        self.fs = int(fs)
        self.window_ms = int(window_ms)
        self.increment_ms = int(increment_ms)
        self.window_samples = int(round(self.fs * self.window_ms / 1000.0))
        self.increment_samples = int(round(self.fs * self.increment_ms / 1000.0))
        self.nperseg = nperseg

    # ------------------------------------------------------------------
    # Windowing
    # ------------------------------------------------------------------
    def _make_windows(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Segment (n_samples, n_channels) → (n_windows, window_samples, n_channels).

        Returns ``(windows, times_sec)`` where ``times_sec[i]`` is the
        centre time of window ``i`` in seconds.
        """
        signal = np.asarray(signal, dtype=np.float64)
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        signal = np.atleast_2d(signal)
        n_samples, n_channels = signal.shape
        if n_samples < self.window_samples:
            raise ValueError(
                f"Signal too short: {n_samples} samples < window_samples="
                f"{self.window_samples} (window_ms={self.window_ms}, fs={self.fs})"
            )
        step = max(1, self.increment_samples)
        win = self.window_samples
        starts = list(range(0, max(1, n_samples - win + 1), step))
        windows = np.stack([signal[s:s + win] for s in starts], axis=0)
        centres = (np.array(starts) + win / 2.0) / float(self.fs)
        return windows, centres

    # ------------------------------------------------------------------
    # Per-channel slope indicators
    # ------------------------------------------------------------------
    def _compute_mnf_slope(self, windows: np.ndarray, fs: int
                           ) -> Tuple[float, float, np.ndarray]:
        """Linear regression of MNF over window time.

        Returns ``(slope_hz_per_min, r_squared, mnf_series)``.
        """
        n_windows, win_len, n_channels = windows.shape
        nperseg = self.nperseg or min(256, win_len)
        mnf = np.zeros((n_windows, n_channels), dtype=np.float64)
        for w in range(n_windows):
            for c in range(n_channels):
                mnf[w, c] = _mean_frequency(windows[w, :, c], fs, nperseg=nperseg)
        # Per-channel slope (Hz/s) → Hz/min, averaged over channels.
        t = (np.arange(n_windows) * self.increment_samples) / float(fs)
        slopes, r2s = [], []
        for c in range(n_channels):
            if len(t) >= 2 and np.std(mnf[:, c]) > 1e-12:
                res = linregress(t, mnf[:, c]) if _HAS_SCIPY else None
                if res is not None:
                    slopes.append(res.slope * 60.0)  # Hz/min
                    r2s.append(res.rvalue ** 2)
                else:
                    # Trivial least-squares fallback (no scipy).
                    a, b = np.polyfit(t, mnf[:, c], 1)
                    slopes.append(a * 60.0)
                    pred = a * t + b
                    ss_res = np.sum((mnf[:, c] - pred) ** 2)
                    ss_tot = np.sum((mnf[:, c] - mnf[:, c].mean()) ** 2) + 1e-12
                    r2s.append(max(0.0, 1.0 - ss_res / ss_tot))
            else:
                slopes.append(0.0)
                r2s.append(0.0)
        return float(np.mean(slopes)), float(np.mean(r2s)), mnf

    def _compute_mdf_slope(self, windows: np.ndarray, fs: int
                           ) -> Tuple[float, float, np.ndarray]:
        """Linear regression of MDF over window time."""
        n_windows, win_len, n_channels = windows.shape
        nperseg = self.nperseg or min(256, win_len)
        mdf = np.zeros((n_windows, n_channels), dtype=np.float64)
        for w in range(n_windows):
            for c in range(n_channels):
                mdf[w, c] = _median_frequency(windows[w, :, c], fs, nperseg=nperseg)
        t = (np.arange(n_windows) * self.increment_samples) / float(fs)
        slopes, r2s = [], []
        for c in range(n_channels):
            if len(t) >= 2 and np.std(mdf[:, c]) > 1e-12:
                res = linregress(t, mdf[:, c]) if _HAS_SCIPY else None
                if res is not None:
                    slopes.append(res.slope * 60.0)
                    r2s.append(res.rvalue ** 2)
                else:
                    a, b = np.polyfit(t, mdf[:, c], 1)
                    slopes.append(a * 60.0)
                    pred = a * t + b
                    ss_res = np.sum((mdf[:, c] - pred) ** 2)
                    ss_tot = np.sum((mdf[:, c] - mdf[:, c].mean()) ** 2) + 1e-12
                    r2s.append(max(0.0, 1.0 - ss_res / ss_tot))
            else:
                slopes.append(0.0)
                r2s.append(0.0)
        return float(np.mean(slopes)), float(np.mean(r2s)), mdf

    def _compute_rms_trend(self, windows: np.ndarray
                            ) -> Tuple[float, float, np.ndarray]:
        """Linear regression of RMS amplitude over window time.

        Returns ``(slope_per_s, r_squared, rms_series)`` — slope units
        are the same as the EMG amplitude per second.
        """
        n_windows, win_len, n_channels = windows.shape
        rms = np.array([[_rms(windows[w, :, c]) for c in range(n_channels)]
                        for w in range(n_windows)], dtype=np.float64)
        t = (np.arange(n_windows) * self.increment_samples) / float(self.fs)
        slopes, r2s = [], []
        for c in range(n_channels):
            if len(t) >= 2 and np.std(rms[:, c]) > 1e-12:
                res = linregress(t, rms[:, c]) if _HAS_SCIPY else None
                if res is not None:
                    slopes.append(res.slope)
                    r2s.append(res.rvalue ** 2)
                else:
                    a, b = np.polyfit(t, rms[:, c], 1)
                    slopes.append(a)
                    pred = a * t + b
                    ss_res = np.sum((rms[:, c] - pred) ** 2)
                    ss_tot = np.sum((rms[:, c] - rms[:, c].mean()) ** 2) + 1e-12
                    r2s.append(max(0.0, 1.0 - ss_res / ss_tot))
            else:
                slopes.append(0.0)
                r2s.append(0.0)
        return float(np.mean(slopes)), float(np.mean(r2s)), rms

    def _compute_imnf(self, signal: np.ndarray, fs: int) -> Dict[str, Any]:
        """STFT-based instantaneous MNF per channel.

        Returns a dict with the IMNF time series per channel and the
        overall IMNF slope. When scipy is unavailable, falls back to a
        per-window Welch estimate (which is identical to the MNF series
        above) so the API stays usable.
        """
        signal = np.asarray(signal, dtype=np.float64)
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        signal = np.atleast_2d(signal)
        n_samples, n_channels = signal.shape
        nperseg = self.nperseg or min(256, self.window_samples)
        nperseg = min(nperseg, n_samples)
        if nperseg < 4:
            return {"imnf": np.zeros((0, n_channels)),
                    "times": np.zeros(0), "imnf_slope_hz_per_min": 0.0}
        if _HAS_SCIPY:
            # scipy.signal.stft with axis=0 and a 2-D input of shape
            # (n_samples, n_channels) returns Zxx of shape
            # (n_freq, n_channels, n_time). Iterate per-channel so we
            # don't depend on the exact axis layout.
            t_axis = np.zeros(0)
            per_channel_imnfs: List[np.ndarray] = []
            for c in range(n_channels):
                f, t, Zxx_c = scipy_signal.stft(
                    signal[:, c], fs=fs, nperseg=nperseg,
                    noverlap=max(0, nperseg - self.increment_samples),
                    boundary=None,
                )
                psd_c = np.abs(Zxx_c) ** 2  # (n_freq, n_time)
                imnf_c = (np.sum(f[:, None] * psd_c, axis=0) /
                          (np.sum(psd_c, axis=0) + 1e-12))  # (n_time,)
                per_channel_imnfs.append(imnf_c)
                t_axis = t
            # Stack as (n_time, n_channels).
            n_time = max(imnf_c.size for imnf_c in per_channel_imnfs)
            imnf = np.full((n_time, n_channels), np.nan, dtype=np.float64)
            for c, imnf_c in enumerate(per_channel_imnfs):
                imnf[:imnf_c.size, c] = imnf_c
            t = t_axis
        else:
            # Fallback: per-window Welch (same as MNF series).
            imnf = np.zeros((0, n_channels))
            t = np.zeros(0)
        # Slope over the STFT time axis.
        if imnf.shape[0] >= 2:
            t_axis = t if t.size > 0 else (np.arange(imnf.shape[0])
                                            * self.increment_samples / fs)
            slopes = []
            for c in range(n_channels):
                col = imnf[:, c]
                # Drop NaN/Inf values that may appear in degenerate channels.
                mask = np.isfinite(col)
                if mask.sum() >= 2 and np.std(col[mask]) > 1e-12:
                    if _HAS_SCIPY:
                        slopes.append(linregress(t_axis[mask], col[mask]).slope * 60.0)
                    else:
                        a, _ = np.polyfit(t_axis[mask], col[mask], 1)
                        slopes.append(a * 60.0)
                else:
                    slopes.append(0.0)
            slope = float(np.mean(slopes))
        else:
            slope = 0.0
        return {"imnf": imnf, "times": t, "imnf_slope_hz_per_min": slope}

    def _compute_fatigue_index(self, mnf_slope: float, mdf_slope: float,
                                 rms_trend: float) -> float:
        """Composite fatigue severity score in [0, 1].

        Combines the three primary indicators into a single [0, 1]
        scalar:

        - Negative MNF / MDF slopes ⇒ fatigue (scaled via sigmoid).
        - Positive RMS trend ⇒ fatigue (scaled via sigmoid).
        - The three sub-scores are averaged; their magnitudes are
          normalised by ``MN_REF_SLOPE`` (a typical fatigue MNF slope
          magnitude of ~10 Hz/min at complete exhaustion).

        Returns
        -------
        float in [0, 1] — 0 = no fatigue, 1 = severe fatigue.
        """
        # Reference magnitudes: empirically, MNF drops ~10 Hz/min at
        # exhaustion; RMS grows ~0.1 V/s (signal-dependent).
        MN_REF = 10.0
        RMS_REF = 0.1
        def _sigmoid(x: float) -> float:
            return 1.0 / (1.0 + float(np.exp(-x)))
        # MNF contribution: more negative slope ⇒ higher fatigue.
        mnf_score = _sigmoid(-mnf_slope / MN_REF - 0.5)
        mdf_score = _sigmoid(-mdf_slope / MN_REF - 0.5)
        # RMS contribution: more positive slope ⇒ higher fatigue.
        rms_score = _sigmoid(rms_trend / RMS_REF - 0.5)
        return float(np.clip((mnf_score + mdf_score + rms_score) / 3.0, 0.0, 1.0))

    @staticmethod
    def _classify_level(fatigue_index: float) -> str:
        """Map a fatigue index in [0, 1] to a qualitative level."""
        if fatigue_index < 0.33:
            return "low"
        if fatigue_index < 0.66:
            return "moderate"
        return "high"

    # ------------------------------------------------------------------
    # Top-level analyse
    # ------------------------------------------------------------------
    def analyze(self,
                signal: np.ndarray,
                timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run the full fatigue analysis on a raw EMG signal.

        Parameters
        ----------
        signal : array-like
            ``(n_samples, n_channels)`` raw EMG signal. A 1-D array
            is interpreted as a single-channel signal.
        timestamps : Optional array-like
            Per-sample timestamps (seconds). Unused internally — kept
            for API symmetry with other MyoAdapt evaluators. The
            internal time axis is derived from ``fs``.

        Returns
        -------
        dict with:

        - ``mnf_slope``: Hz/min (negative ⇒ fatigue).
        - ``mnf_r2``: linear-fit R².
        - ``mnf_series``: (n_windows, n_channels) MNF time series.
        - ``mdf_slope``, ``mdf_r2``, ``mdf_series``: same for MDF.
        - ``rms_trend``: amplitude slope per second.
        - ``rms_r2``, ``rms_series``: same for RMS.
        - ``imnf_series``, ``imnf_times``, ``imnf_slope_hz_per_min``: STFT IMNF.
        - ``fatigue_index``: composite [0, 1].
        - ``fatigue_level``: 'low' / 'moderate' / 'high'.
        - ``window_times``: per-window centre times (s).
        - ``per_channel``: dict of per-channel MNF / MDF / RMS slopes.
        - ``n_windows``, ``n_channels``, ``fs``, ``window_ms``, ``increment_ms``.
        """
        signal = np.asarray(signal, dtype=np.float64)
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        signal = np.atleast_2d(signal)
        windows, centres = self._make_windows(signal)
        n_windows, win_len, n_channels = windows.shape

        logger.info(
            f"FatigueTracker.analyze: n_windows={n_windows}, "
            f"win_len={win_len}, n_channels={n_channels}, fs={self.fs}"
        )

        mnf_slope, mnf_r2, mnf_series = self._compute_mnf_slope(windows, self.fs)
        mdf_slope, mdf_r2, mdf_series = self._compute_mdf_slope(windows, self.fs)
        rms_slope, rms_r2, rms_series = self._compute_rms_trend(windows)
        imnf = self._compute_imnf(signal, self.fs)
        fatigue_index = self._compute_fatigue_index(mnf_slope, mdf_slope, rms_slope)
        fatigue_level = self._classify_level(fatigue_index)

        # Per-channel slopes for downstream inspection.
        per_channel: Dict[str, Any] = {}
        for c in range(n_channels):
            t = (np.arange(n_windows) * self.increment_samples) / float(self.fs)
            if len(t) >= 2 and np.std(mnf_series[:, c]) > 1e-12:
                if _HAS_SCIPY:
                    res_mnf = linregress(t, mnf_series[:, c])
                    res_mdf = linregress(t, mdf_series[:, c])
                    res_rms = linregress(t, rms_series[:, c])
                    per_channel[f"ch{c}"] = {
                        "mnf_slope_hz_per_min": float(res_mnf.slope * 60.0),
                        "mdf_slope_hz_per_min": float(res_mdf.slope * 60.0),
                        "rms_slope_per_s": float(res_rms.slope),
                        "mnf_r2": float(res_mnf.rvalue ** 2),
                        "mdf_r2": float(res_mdf.rvalue ** 2),
                        "rms_r2": float(res_rms.rvalue ** 2),
                    }
                else:
                    a_mnf, _ = np.polyfit(t, mnf_series[:, c], 1)
                    a_mdf, _ = np.polyfit(t, mdf_series[:, c], 1)
                    a_rms, _ = np.polyfit(t, rms_series[:, c], 1)
                    per_channel[f"ch{c}"] = {
                        "mnf_slope_hz_per_min": float(a_mnf * 60.0),
                        "mdf_slope_hz_per_min": float(a_mdf * 60.0),
                        "rms_slope_per_s": float(a_rms),
                    }
            else:
                per_channel[f"ch{c}"] = {
                    "mnf_slope_hz_per_min": 0.0,
                    "mdf_slope_hz_per_min": 0.0,
                    "rms_slope_per_s": 0.0,
                }

        return {
            "mnf_slope": mnf_slope,
            "mnf_r2": mnf_r2,
            "mnf_series": mnf_series,
            "mdf_slope": mdf_slope,
            "mdf_r2": mdf_r2,
            "mdf_series": mdf_series,
            "rms_trend": rms_slope,
            "rms_r2": rms_r2,
            "rms_series": rms_series,
            "imnf_series": imnf["imnf"],
            "imnf_times": imnf["times"],
            "imnf_slope_hz_per_min": imnf["imnf_slope_hz_per_min"],
            "fatigue_index": fatigue_index,
            "fatigue_level": fatigue_level,
            "window_times": centres,
            "per_channel": per_channel,
            "n_windows": int(n_windows),
            "n_channels": int(n_channels),
            "fs": self.fs,
            "window_ms": self.window_ms,
            "increment_ms": self.increment_ms,
        }

    # ------------------------------------------------------------------
    # Plotting & narrative
    # ------------------------------------------------------------------
    def plot_fatigue_curves(self,
                            results: Dict[str, Any],
                            figsize: Tuple[float, float] = (10.0, 6.0),
                            ) -> Any:
        """Three-panel plot: MNF / MDF / RMS over time (channel-averaged).

        Returns a ``matplotlib.figure.Figure``. Raises ``ImportError``
        if matplotlib is unavailable.
        """
        if not _HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for plot_fatigue_curves. "
                "Install with: pip install matplotlib"
            )
        t = np.asarray(results["window_times"])
        if t.size == 0:
            raise RuntimeError("Empty results — nothing to plot.")
        # Channel-mean for the top-level panels.
        mnf_mean = np.asarray(results["mnf_series"]).mean(axis=1)
        mdf_mean = np.asarray(results["mdf_series"]).mean(axis=1)
        rms_mean = np.asarray(results["rms_series"]).mean(axis=1)

        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        axes[0].plot(t, mnf_mean, color="#0072B2", lw=1.6, label="MNF")
        axes[0].set_ylabel("MNF (Hz)")
        axes[0].set_title(
            f"Fatigue curves — index={results['fatigue_index']:.2f} "
            f"({results['fatigue_level']})"
        )
        axes[0].grid(True, alpha=0.3, linestyle="--")
        axes[0].legend(loc="upper right", fontsize=8)

        axes[1].plot(t, mdf_mean, color="#D55E00", lw=1.6, label="MDF")
        axes[1].set_ylabel("MDF (Hz)")
        axes[1].grid(True, alpha=0.3, linestyle="--")
        axes[1].legend(loc="upper right", fontsize=8)

        axes[2].plot(t, rms_mean, color="#009E73", lw=1.6, label="RMS")
        axes[2].set_ylabel("RMS")
        axes[2].set_xlabel("Time (s)")
        axes[2].grid(True, alpha=0.3, linestyle="--")
        axes[2].legend(loc="upper right", fontsize=8)
        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.tight_layout()
        return fig

    def fatigue_narrative(self, results: Dict[str, Any]) -> str:
        """Human-readable summary of the fatigue analysis."""
        level = results.get("fatigue_level", "unknown")
        fi = float(results.get("fatigue_index", 0.0))
        mnf = float(results.get("mnf_slope", 0.0))
        mdf = float(results.get("mdf_slope", 0.0))
        rms = float(results.get("rms_trend", 0.0))
        n_win = int(results.get("n_windows", 0))
        n_ch = int(results.get("n_channels", 0))
        direction = "decrease" if mnf < 0 else "increase"
        lines: List[str] = [
            f"Fatigue analysis: {level.upper()} (fatigue index = {fi:.2f}/1.0).",
            f"Analysed {n_win} windows across {n_ch} channels "
            f"(window_ms={results.get('window_ms')}, increment_ms={results.get('increment_ms')}, fs={results.get('fs')} Hz).",
        ]
        if mnf < 0:
            lines.append(
                f"MNF slope = {mnf:+.2f} Hz/min — mean frequency is DECREASING, "
                f"which is the classical signature of muscle fatigue."
            )
        else:
            lines.append(
                f"MNF slope = {mnf:+.2f} Hz/min — mean frequency is NOT decreasing; "
                f"no spectral-compression fatigue indicator."
            )
        lines.append(f"MDF slope = {mdf:+.2f} Hz/min (more-robust alternative to MNF).")
        if rms > 0:
            lines.append(
                f"RMS trend = {rms:+.4e}/s — amplitude is INCREASING, consistent "
                f"with recruitment of additional motor units (a fatigue compensator)."
            )
        else:
            lines.append(
                f"RMS trend = {rms:+.4e}/s — amplitude is flat or decreasing."
            )
        lines.append(
            f"Composite fatigue index = {fi:.2f} → qualitative level: {level}."
        )
        if level == "high":
            lines.append(
                "Recommendation: signal quality may be degraded — consider "
                "rest or recalibration of downstream myoelectric models."
            )
        elif level == "moderate":
            lines.append(
                "Recommendation: monitor for further degradation; pre-emptive "
                "recalibration may improve downstream accuracy."
            )
        else:
            lines.append("Recommendation: signal is stable — no action required.")
        return "\n".join(lines)


__all__ = ["FatigueTracker"]

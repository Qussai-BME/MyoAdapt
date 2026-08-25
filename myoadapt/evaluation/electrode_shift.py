"""
electrode_shift.py — Electrode-Shift Robustness evaluation.

One of the biggest practical challenges in real-world sEMG systems:
electrodes shift between donning sessions, changing the channel-to-
muscle mapping. This module simulates shifts and measures how
accuracy degrades — a critical metric for clinical deployment.

Simulated shifts:
- Circular shift: channels rotate by k positions (simulates armband
  rotation)
- Channel dropout: k channels set to zero (simulates loose contact)
- Channel swap: pairs of adjacent channels swapped (simulates
  misplacement)
- Gaussian noise injection: simulates impedance change

References:
- Boschmann et al. (2023). "Electrode shift robustness in sEMG-
  based gesture recognition." JNER.
- Campbell et al. (2024). "Robust myoelectric control under
  real-world conditions."
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Matplotlib is optional: only required by ``plot_degradation_curve``.
# All other functionality works without it.
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

logger = logging.getLogger(__name__)


class ElectrodeShiftRobustness:
    """
    Electrode-shift robustness evaluator.

    Wraps an *already trained* model and exposes four families of
    perturbation (circular shift, channel dropout, channel swap,
    Gaussian noise) along with sweep utilities that quantify how
    accuracy degrades as the perturbation magnitude grows.

    The model is expected to expose a ``predict(X) -> labels`` API
    (any sklearn-compatible estimator, ONNX-runtime wrapper, or a
    custom callable will work).

    X conventions
    -------------
    The shift operators assume that the *channel axis is axis=1*.
    Both 2-D feature matrices of shape ``(n_samples, n_channels)``
    and 3-D raw windows of shape ``(n_samples, n_channels,
    n_timepoints)`` are supported transparently — the perturbations
    operate along axis=1 in either case.

    Parameters
    ----------
    model : fitted estimator
        Must expose ``predict``. The evaluator never retrains.
    n_channels : int
        Number of channels in the underlying armband (e.g. 12 for
        the Delsys Trigno, 8 for the Myo armband). Used to bound the
        maximum circular-shift magnitude and to choose channel
        indices for the swap / dropout operations.
    random_state : int
        Seed for the stochastic perturbations (channel dropout /
        noise injection) so sweeps are reproducible.
    """

    SHIFT_TYPES: Tuple[str, ...] = ("circular", "dropout", "swap", "noise")

    def __init__(self,
                 model: Any,
                 n_channels: int = 12,
                 random_state: int = 42):
        if not hasattr(model, "predict"):
            raise TypeError(
                "model must expose a .predict() method "
                f"(got {type(model).__name__})"
            )
        if n_channels < 2:
            raise ValueError(
                f"n_channels must be ≥ 2 (got {n_channels})"
            )
        self.model = model
        self.n_channels = int(n_channels)
        self.random_state = int(random_state)
        self.results_: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Perturbation primitives
    # ------------------------------------------------------------------
    def circular_shift(self, X: np.ndarray, k: int) -> np.ndarray:
        """Rotate channels by ``k`` positions along axis=1.

        Simulates armband rotation. ``k`` is taken modulo
        ``n_channels`` so values larger than the channel count wrap
        around cleanly.
        """
        X = np.asarray(X)
        if X.ndim < 2:
            raise ValueError(
                f"circular_shift requires X.ndim >= 2 (got {X.ndim})"
            )
        if X.shape[1] != self.n_channels:
            logger.warning(
                f"circular_shift: X has {X.shape[1]} channels but "
                f"n_channels={self.n_channels}; using X.shape[1]."
            )
        k = int(k) % max(1, X.shape[1])
        return np.roll(X, shift=k, axis=1).copy()

    def channel_dropout(self, X: np.ndarray, k: int,
                        prob: float = 0.5) -> np.ndarray:
        """Zero out ``k`` random channels.

        ``prob`` controls whether the dropout is applied to each
        sample independently (per-sample intermittent dropout, e.g.
        simulating transient loose contact). ``prob=1.0`` drops the
        same ``k`` channels for every sample; ``prob=0.5`` drops them
        for roughly half of the samples.

        Parameters
        ----------
        X : (n_samples, n_channels[, n_timepoints]) array.
        k : int — number of channels to drop.
        prob : float in [0, 1] — per-sample dropout probability.
        """
        X = np.asarray(X)
        if X.ndim < 2:
            raise ValueError(
                f"channel_dropout requires X.ndim >= 2 (got {X.ndim})"
            )
        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"prob must be in [0, 1] (got {prob})")
        n_ch = X.shape[1]
        k = int(min(max(k, 0), n_ch))
        if k == 0 or prob == 0.0:
            return X.copy()

        rng = np.random.default_rng(self.random_state)
        X_out = X.copy()
        # Choose the channels to drop ONCE — they are the same for
        # every sample (a real electrode fault is spatially fixed).
        drop_idx = rng.choice(n_ch, size=k, replace=False)
        # Per-sample application mask.
        n_samples = X.shape[0]
        apply_mask = rng.random(n_samples) < prob
        if not apply_mask.any():
            return X_out
        # Set the chosen channels to zero for the selected samples.
        # ``np.ix_`` builds an outer-product (broadcasting) index that
        # works uniformly for 2-D ``(n_samples, n_channels)`` and 3-D
        # ``(n_samples, n_channels, n_timepoints)`` arrays — the
        # remaining axes (time, ...) are implicitly fully sliced.
        X_out[np.ix_(apply_mask, drop_idx)] = 0.0
        return X_out

    def channel_swap(self, X: np.ndarray, n_swaps: int) -> np.ndarray:
        """Swap ``n_swaps`` adjacent channel pairs.

        For each swap, picks a uniformly-random starting index ``i``
        in ``[0, n_channels-1]`` and swaps channels ``i`` and
        ``(i + 1) % n_channels`` (cyclic adjacency). Multiple swaps
        are applied sequentially so the net permutation can be more
        complex than a single transposition.
        """
        X = np.asarray(X)
        if X.ndim < 2:
            raise ValueError(
                f"channel_swap requires X.ndim >= 2 (got {X.ndim})"
            )
        n_ch = X.shape[1]
        n_swaps = int(max(0, n_swaps))
        if n_swaps == 0:
            return X.copy()

        rng = np.random.default_rng(self.random_state)
        X_out = X.copy()
        for _ in range(n_swaps):
            i = int(rng.integers(0, n_ch))
            j = (i + 1) % n_ch
            # Swap along axis=1.
            tmp = X_out[:, i].copy()
            X_out[:, i] = X_out[:, j]
            X_out[:, j] = tmp
        return X_out

    def noise_injection(self, X: np.ndarray,
                        sigma: float = 0.1) -> np.ndarray:
        """Add Gaussian noise scaled by the signal's std.

        ``sigma`` is interpreted as a *fraction* of the global signal
        standard deviation — this keeps the perturbation magnitude
        comparable across databases that use different amplitude
        scales (raw mV, normalised, log-variance, …).

        Parameters
        ----------
        X : (n_samples, n_channels[, n_timepoints]) array.
        sigma : float — noise std as a fraction of ``X.std()``.
        """
        X = np.asarray(X, dtype=float)
        if sigma <= 0.0:
            return X.copy()
        rng = np.random.default_rng(self.random_state)
        signal_std = float(X.std())
        if signal_std < 1e-12:
            # Degenerate (all-constant) array — fall back to absolute sigma.
            noise_scale = float(sigma)
        else:
            noise_scale = float(sigma) * signal_std
        noise = rng.normal(loc=0.0, scale=noise_scale, size=X.shape)
        return X + noise.astype(X.dtype)

    # ------------------------------------------------------------------
    # Sweep utilities
    # ------------------------------------------------------------------
    def evaluate_shift(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       shift_type: str = "circular",
                       max_shift: Union[int, float] = 6,
                       step: Union[int, float] = 1,
                       tolerance_pp: float = 5.0) -> Dict[str, Any]:
        """Sweep a single shift type and measure accuracy at each step.

        Parameters
        ----------
        X, y : test data + labels.
        shift_type : one of ``"circular"``, ``"dropout"``, ``"swap"``,
            ``"noise"``.
        max_shift : maximum perturbation magnitude (inclusive). For
            integer shift types (circular, dropout, swap) this is a
            channel count; for ``"noise"`` it is the maximum sigma
            (e.g. ``0.5``).
        step : step size (``range(0, max_shift+1, step)``). For the
            ``"noise"`` shift type this is interpreted as a sigma
            step (use ``step=0.1`` for a 0–0.5 sweep, etc.).
        tolerance_pp : accuracy-degradation threshold (in percentage
            points) below which a shift is considered "tolerated".

        Returns
        -------
        dict with:
            ``shift_type``
            ``per_shift`` : list of {shift, accuracy, macro_f1,
                                       degradation_pp}
            ``baseline_accuracy``
            ``max_degradation`` (fraction)
            ``max_degradation_pp``
            ``shift_tolerance_threshold`` — largest shift at which
                degradation stays ≤ ``tolerance_pp``.
        """
        if shift_type not in self.SHIFT_TYPES:
            raise ValueError(
                f"shift_type must be one of {self.SHIFT_TYPES} "
                f"(got {shift_type!r})"
            )
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim < 2:
            raise ValueError("evaluate_shift requires X.ndim >= 2")

        # Baseline accuracy on the unshifted data.
        y_pred_baseline = self.model.predict(X)
        baseline_acc = float(accuracy_score(y, y_pred_baseline))
        classes = sorted(np.unique(np.concatenate([y, y_pred_baseline])).tolist(),
                         key=lambda x: str(x))
        baseline_f1 = float(f1_score(y, y_pred_baseline, average="macro",
                                       zero_division=0, labels=classes))

        # Sweep magnitudes. For "noise" the magnitude is a sigma value
        # (float); for the others it is an integer count.
        shift_fn = self._resolve_shift_fn(shift_type)
        magnitudes = self._sweep_magnitudes(shift_type, max_shift, step)

        per_shift: List[Dict[str, Any]] = []
        tol_threshold = 0.0
        max_degradation = 0.0
        for mag in magnitudes:
            X_perturbed = shift_fn(X, mag)
            y_pred = self.model.predict(X_perturbed)
            acc = float(accuracy_score(y, y_pred))
            f1 = float(f1_score(y, y_pred, average="macro",
                                  zero_division=0, labels=classes))
            degradation = baseline_acc - acc
            degradation_pp = float(degradation * 100.0)
            per_shift.append({
                "shift": float(mag),
                "accuracy": acc,
                "macro_f1": f1,
                "degradation": float(degradation),
                "degradation_pp": degradation_pp,
            })
            max_degradation = max(max_degradation, degradation)
            if degradation_pp <= tolerance_pp:
                tol_threshold = float(mag)
            logger.debug(
                f"  {shift_type} shift={mag}: acc={acc:.4f} "
                f"deg={degradation_pp:.2f}pp"
            )

        result: Dict[str, Any] = {
            "shift_type": shift_type,
            "per_shift": per_shift,
            "baseline_accuracy": baseline_acc,
            "baseline_macro_f1": baseline_f1,
            "max_degradation": float(max_degradation),
            "max_degradation_pp": float(max_degradation * 100.0),
            "shift_tolerance_threshold": float(tol_threshold),
            "tolerance_pp": float(tolerance_pp),
            "n_channels": int(self.n_channels),
            "n_samples": int(X.shape[0]),
            "magnitudes": [float(m) for m in magnitudes],
        }
        return result

    def evaluate_all_shifts(self,
                            X: np.ndarray,
                            y: np.ndarray,
                            max_shift: int = 6,
                            step: int = 1,
                            noise_max_sigma: float = 0.5,
                            noise_step: float = 0.1,
                            tolerance_pp: float = 5.0) -> Dict[str, Any]:
        """Run sweeps for all four shift types.

        For circular / dropout / swap the sweep uses
        ``range(0, max_shift+1, step)`` (integer magnitudes). For
        noise the sweep uses ``np.arange(0, noise_max_sigma +
        noise_step, noise_step)`` (float sigmas).

        Returns
        -------
        dict keyed by shift type with the per-type result plus a
        ``summary`` block.
        """
        results: Dict[str, Any] = {}
        for st in ("circular", "dropout", "swap"):
            results[st] = self.evaluate_shift(
                X, y, shift_type=st, max_shift=max_shift, step=step,
                tolerance_pp=tolerance_pp,
            )
        # Noise sweep uses float sigmas — pass ``max_shift`` directly
        # as the maximum sigma value (evaluate_shift interprets it
        # correctly for ``shift_type="noise"``).
        results["noise"] = self.evaluate_shift(
            X, y, shift_type="noise",
            max_shift=float(noise_max_sigma),
            step=float(noise_step),
            tolerance_pp=tolerance_pp,
        )

        # Overall summary across shift types.
        summary = {
            "baseline_accuracy": results["circular"]["baseline_accuracy"],
            "per_type_max_degradation_pp": {
                st: r["max_degradation_pp"] for st, r in results.items()
            },
            "per_type_tolerance": {
                st: r["shift_tolerance_threshold"] for st, r in results.items()
            },
            "worst_shift_type": max(
                results.items(),
                key=lambda kv: kv[1]["max_degradation_pp"],
            )[0],
            "n_channels": int(self.n_channels),
        }
        results["summary"] = summary
        self.results_ = results
        return results

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_degradation_curve(self,
                               results: Optional[Dict[str, Any]] = None,
                               figsize: Tuple[float, float] = (7.0, 4.0),
                               title: Optional[str] = None,
                               ) -> matplotlib.figure.Figure:  # type: ignore[name-defined]
        """Plot accuracy vs shift magnitude for all shift types.

        Parameters
        ----------
        results : dict
            Output of :meth:`evaluate_all_shifts`. If ``None``,
            uses the cached ``self.results_`` (requires a prior call
            to ``evaluate_all_shifts``).

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not _HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for plot_degradation_curve. "
                "Install with `pip install matplotlib`."
            )
        if results is None:
            results = self.results_
        if results is None:
            raise RuntimeError(
                "No results to plot. Call evaluate_all_shifts() first "
                "or pass results explicitly."
            )

        # Apply a paper-ready style consistent with myoadapt.evaluation.diagrams.
        plt.rcParams.update({
            "figure.figsize": figsize,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
        })

        # Okabe-Ito colourblind-safe palette.
        colors = {
            "circular": "#0072B2",  # blue
            "dropout":  "#D55E00",  # vermillion
            "swap":     "#009E73",  # green
            "noise":    "#CC79A7",  # pink
        }
        markers = {
            "circular": "o",
            "dropout":  "s",
            "swap":     "^",
            "noise":    "D",
        }
        labels = {
            "circular": "Circular shift (channels)",
            "dropout":  "Channel dropout",
            "swap":     "Channel swap",
            "noise":    "Gaussian noise (σ)",
        }

        fig, ax = plt.subplots(figsize=figsize)
        baseline_acc: Optional[float] = None
        for st in ("circular", "dropout", "swap", "noise"):
            if st not in results:
                continue
            per_shift = results[st]["per_shift"]
            if not per_shift:
                continue
            xs = [e["shift"] for e in per_shift]
            ys = [e["accuracy"] for e in per_shift]
            if baseline_acc is None:
                baseline_acc = results[st]["baseline_accuracy"]
            ax.plot(xs, ys, marker=markers[st], color=colors[st],
                    label=labels[st], lw=1.5, markersize=5)

        if baseline_acc is not None:
            ax.axhline(baseline_acc, color="gray", ls=":",
                       lw=1.0, alpha=0.7, label=f"Baseline ({baseline_acc:.3f})")

        ax.set_xlabel("Shift magnitude")
        ax.set_ylabel("Accuracy")
        ax.set_title(title or "Electrode-shift robustness: accuracy vs perturbation")
        ax.set_ylim(bottom=0.0, top=1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_shift_fn(self, shift_type: str) -> Callable[..., np.ndarray]:
        if shift_type == "circular":
            return self.circular_shift
        if shift_type == "dropout":
            # prob=1.0 keeps the sweep deterministic (drop same k channels
            # for every sample), which is what a robustness sweep wants.
            return lambda X, k: self.channel_dropout(X, int(k), prob=1.0)
        if shift_type == "swap":
            return lambda X, k: self.channel_swap(X, int(k))
        if shift_type == "noise":
            return lambda X, k: self.noise_injection(X, float(k))
        raise ValueError(f"Unknown shift_type: {shift_type!r}")

    def _sweep_magnitudes(self, shift_type: str,
                          max_shift: float, step: float) -> List[float]:
        """Build the list of perturbation magnitudes to sweep."""
        if step <= 0:
            raise ValueError(f"step must be > 0 (got {step})")
        if shift_type == "noise":
            # Float sigmas. Use integer multiplication to avoid the
            # classic ``np.arange`` float round-off issues.
            n_steps = int(round(float(max_shift) / float(step) + 1e-9)) + 1
            magnitudes = [float(i) * float(step) for i in range(n_steps)]
            # Drop any magnitude that exceeds max_shift due to round-off.
            magnitudes = [m for m in magnitudes if m <= float(max_shift) + 1e-9]
            return magnitudes

        # Integer magnitudes for the other shift types.
        max_shift_int = int(max_shift)
        step_int = int(step) if step >= 1 else 1
        # Cap by n_channels for circular / swap / dropout — a larger
        # magnitude is redundant because of the cyclic structure.
        cap = self.n_channels if shift_type in ("circular", "swap") else max_shift_int
        upper = min(max_shift_int, cap)
        return [float(m) for m in range(0, upper + 1, step_int)]

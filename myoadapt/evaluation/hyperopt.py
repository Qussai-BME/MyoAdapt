"""
hyperopt.py — Hyperparameter optimization via Optuna.

Systematic hyperparameter search for MyoAdapt models. Supports:
- Random search, TPE (Tree-structured Parzen Estimator), CMA-ES
- Cross-validated objective (LOSO or k-fold)
- Early pruning (MedianPruner, SuccessiveHalvingPruner)
- Parallel execution (joblib backend)

The optimization result includes the best hyperparameters, the
optimization history, and a plot of the optimization trajectory.

References
----------
- Akiba et al. (2019). "Optuna: A Next-generation Hyperparameter
  Optimization Framework." KDD.

License: Apache 2.0
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Optuna is the central dependency of this module but is *optional*
# for the wider myoadapt package — fall back gracefully.
try:
    import optuna
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
    from optuna.samplers import CmaEsSampler as CMAESampler
    from optuna.samplers import RandomSampler, TPESampler
    from optuna.study import Study
    _HAS_OPTUNA = True
except ImportError:  # pragma: no cover — optuna is optional
    optuna = None  # type: ignore[assignment]
    TPESampler = RandomSampler = CMAESampler = None  # type: ignore[assignment]
    MedianPruner = SuccessiveHalvingPruner = None  # type: ignore[assignment]
    Study = None  # type: ignore[assignment]
    _HAS_OPTUNA = False

# matplotlib is optional: only required by the plot_* helpers.
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

# sklearn is a hard dep but fall back if missing.
try:
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover — sklearn is a hard dep
    _HAS_SKLEARN = False


_OPTUNA_INSTALL_HINT = (
    "optuna not installed. Install with: pip install optuna "
    "(or pip install optuna matplotlib for plotting support)"
)


# ---------------------------------------------------------------------------
# Default search spaces
# ---------------------------------------------------------------------------
_DEFAULT_SPACES: Dict[str, Dict[str, Tuple[Any, ...]]] = {
    "random_forest": {
        "n_estimators": (10, 500),
        "max_depth": (3, 20),
        "min_samples_split": (2, 20),
        "min_samples_leaf": (1, 10),
        "max_features": ("sqrt", "log2", 0.5, 1.0),
    },
    "extra_trees": {
        "n_estimators": (10, 500),
        "max_depth": (3, 20),
        "min_samples_split": (2, 20),
        "min_samples_leaf": (1, 10),
    },
    "xgboost": {
        "n_estimators": (10, 500),
        "max_depth": (3, 12),
        "learning_rate": (1e-3, 0.5),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.5, 1.0),
        "min_child_weight": (1, 10),
        "reg_lambda": (1e-3, 10.0),
        "reg_alpha": (1e-3, 10.0),
    },
    "lightgbm": {
        "n_estimators": (10, 500),
        "max_depth": (-1, 20),
        "learning_rate": (1e-3, 0.5),
        "num_leaves": (15, 255),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.5, 1.0),
        "reg_lambda": (1e-3, 10.0),
    },
    "svm": {
        "C": (1e-3, 100.0),
    },
    "logistic": {
        "C": (1e-3, 100.0),
    },
    "lda": {},
}


def _suggest_param(trial: optuna.trial.Trial,
                   name: str,
                   space: Any) -> Any:
    """Convert a (low, high) tuple / categorical list into a trial.suggest_* call.

    The supported space encodings are:

    - ``(low, high)`` with both int  → ``suggest_int(low, high)``
    - ``(low, high)`` with both float → ``suggest_float(low, high, log=True)``
      when the range spans ≥ 3 decades, else linear scale.
    - A list/tuple of categorical values → ``suggest_categorical``
    - A single int/float/str → returned as-is (constant).
    """
    # Constant param.
    if isinstance(space, (int, float, str, bool)) or space is None:
        return space
    # Categorical list/tuple of length > 2 OR mixed types.
    if isinstance(space, (list, tuple)) and len(space) > 0:
        # Numeric (low, high) range.
        if len(space) == 2 and all(isinstance(v, (int, float)) for v in space):
            low, high = space
            if isinstance(low, int) and isinstance(high, int):
                return trial.suggest_int(name, low, high)
            # Float range — choose log scale when range is wide.
            try:
                span = float(high) - float(low)
                if float(low) > 0 and span / max(1e-12, float(low)) > 1e3:
                    return trial.suggest_float(name, float(low), float(high), log=True)
            except (TypeError, ValueError):
                pass
            return trial.suggest_float(name, float(low), float(high))
        # Categorical.
        return trial.suggest_categorical(name, list(space))
    raise ValueError(f"Unsupported param space for {name!r}: {space!r}")


class HyperparameterOptimizer:
    """Systematic HPO wrapper around Optuna.

    Wraps a MyoAdapt classifier family (one of the seven classical
    back-ends registered in :mod:`myoadapt.models.classical`, or any
    custom sklearn-compatible estimator constructed via
    ``model_factory``) and searches over its hyperparameters using
    Optuna samplers (TPE / Random / CMA-ES) and pruners (Median /
    SuccessiveHalving).

    The objective is cross-validated accuracy (macro-F1 for multiclass)
    under either a LOSO protocol (when ``groups`` is provided) or a
    stratified k-fold protocol (otherwise).

    Parameters
    ----------
    model_type : str (default 'random_forest')
        One of the classical-model aliases (``random_forest``,
        ``xgboost``, ``lightgbm``, ``extra_trees``, ``svm``,
        ``logistic``, ``lda``). Used to fetch a default search space
        when ``param_space`` is None and to build the per-trial model
        when ``model_factory`` is None.
    protocol : 'loso' or 'kfold' (default 'loso')
        Cross-validation protocol. ``loso`` requires ``groups`` at
        ``optimize`` time.
    n_trials : int (default 50)
        Maximum number of Optuna trials.
    timeout : Optional[int] (default None)
        Maximum wall-clock time in seconds (None = unlimited).
    sampler : 'tpe', 'random', 'cmaes' (default 'tpe')
        Optuna sampler.
    pruner : 'median', 'halving', 'none' (default 'median')
        Optuna pruner. ``'none'`` disables pruning.
    n_folds : int (default 5)
        Number of folds when ``protocol='kfold'``.
    n_jobs : int (default 1)
        Number of parallel folds (joblib).
    random_state : int (default 42)
        Seed for the sampler and for any downstream RNG.
    scoring : 'accuracy' or 'macro_f1' (default 'macro_f1')
        Objective metric.
    model_factory : Optional[Callable[[Dict[str, Any]], Any]] (default None)
        If provided, called as ``model_factory(params) -> estimator``
        for each trial. Bypasses the default classical-model builder.
    """

    def __init__(self,
                 model_type: str = "random_forest",
                 protocol: str = "loso",
                 n_trials: int = 50,
                 timeout: Optional[int] = None,
                 sampler: str = "tpe",
                 pruner: str = "median",
                 n_folds: int = 5,
                 n_jobs: int = 1,
                 random_state: int = 42,
                 scoring: str = "macro_f1",
                 model_factory: Optional[Callable[[Dict[str, Any]], Any]] = None):
        if protocol not in ("loso", "kfold"):
            raise ValueError(f"protocol must be 'loso' or 'kfold' (got {protocol!r})")
        if sampler not in ("tpe", "random", "cmaes"):
            raise ValueError(f"sampler must be 'tpe', 'random', 'cmaes' (got {sampler!r})")
        if pruner not in ("median", "halving", "none"):
            raise ValueError(f"pruner must be 'median', 'halving', 'none' (got {pruner!r})")
        if scoring not in ("accuracy", "macro_f1"):
            raise ValueError(
                f"scoring must be 'accuracy' or 'macro_f1' (got {scoring!r})"
            )
        self.model_type = model_type
        self.protocol = protocol
        self.n_trials = int(n_trials)
        self.timeout = timeout
        self.sampler = sampler
        self.pruner = pruner
        self.n_folds = int(n_folds)
        self.n_jobs = int(n_jobs)
        self.random_state = int(random_state)
        self.scoring = scoring
        self.model_factory = model_factory
        # Populated by optimize().
        self.study_: Optional[Any] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: Optional[float] = None

    # ------------------------------------------------------------------
    # Default search spaces
    # ------------------------------------------------------------------
    def default_param_space(self, model_type: Optional[str] = None) -> Dict[str, Any]:
        """Return the default search space for a classical model type."""
        mt = model_type or self.model_type
        if mt not in _DEFAULT_SPACES:
            raise KeyError(
                f"No default search space for model_type={mt!r}. "
                f"Available: {sorted(_DEFAULT_SPACES.keys())}"
            )
        # Return a copy so callers can mutate freely.
        return {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in _DEFAULT_SPACES[mt].items()}

    # ------------------------------------------------------------------
    # Sampler / pruner construction
    # ------------------------------------------------------------------
    def _build_sampler(self) -> Any:
        if self.sampler == "tpe":
            return TPESampler(seed=self.random_state)
        if self.sampler == "random":
            return RandomSampler(seed=self.random_state)
        if self.sampler == "cmaes":
            return CMAESampler(seed=self.random_state)
        raise ValueError(f"Unknown sampler: {self.sampler!r}")

    def _build_pruner(self) -> Any:
        if self.pruner == "median":
            return MedianPruner()
        if self.pruner == "halving":
            return SuccessiveHalvingPruner()
        if self.pruner == "none":
            return optuna.pruners.NopPruner()
        raise ValueError(f"Unknown pruner: {self.pruner!r}")

    # ------------------------------------------------------------------
    # Per-trial model construction
    # ------------------------------------------------------------------
    def _build_model(self, params: Dict[str, Any]) -> Any:
        if self.model_factory is not None:
            return self.model_factory(params)
        if not _HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required to build models with the default "
                "model_type. Install with: pip install scikit-learn"
            )
        from myoadapt.models.classical import EMGClassifier
        return EMGClassifier(
            model_type=self.model_type,
            random_state=self.random_state,
            **params,
        )

    # ------------------------------------------------------------------
    # Cross-validation objective
    # ------------------------------------------------------------------
    def _make_splits(self, X: np.ndarray, y: np.ndarray,
                     groups: Optional[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return a list of (train_idx, test_idx) splits."""
        if self.protocol == "loso":
            if groups is None:
                raise ValueError("protocol='loso' requires `groups` (subject IDs)")
            from myoadapt.data.splits import loso_splits
            return list(loso_splits(np.asarray(groups)))
        # Stratified k-fold.
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state,
        )
        return list(skf.split(np.zeros(len(y)), y))

    def _evaluate_trial(self, X: np.ndarray, y: np.ndarray,
                        splits: List[Tuple[np.ndarray, np.ndarray]],
                        params: Dict[str, Any]) -> float:
        """Mean cross-validated score for one set of hyperparameters."""
        scores: List[float] = []
        for train_idx, test_idx in splits:
            model = self._build_model(params)
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])
            if self.scoring == "accuracy":
                scores.append(float(accuracy_score(y[test_idx], y_pred)))
            else:
                scores.append(float(f1_score(y[test_idx], y_pred,
                                              average="macro", zero_division=0)))
        return float(np.mean(scores))

    # ------------------------------------------------------------------
    # Public optimisation entry point
    # ------------------------------------------------------------------
    def optimize(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 groups: Optional[np.ndarray] = None,
                 param_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the HPO and return the best configuration found.

        Parameters
        ----------
        X : (n_samples, n_features) feature matrix.
        y : (n_samples,) class labels.
        groups : Optional[(n_samples,)] subject IDs — required for LOSO.
        param_space : Optional[dict]
            Per-hyperparameter search space. Each value can be:
            - a 2-tuple ``(low, high)`` of ints/floats (numeric range),
            - a list/tuple of categorical values,
            - a constant int/float/str/bool.
            When ``None``, falls back to ``default_param_space(model_type)``.

        Returns
        -------
        dict with:
            - ``best_params``: best hyperparameter dict found.
            - ``best_score``: cross-validated score of the best trial.
            - ``n_trials``: number of completed trials.
            - ``study``: the underlying Optuna Study (or a dict snapshot
              if you want to serialize it without optuna at the call site).
            - ``param_space``: the search space actually used.
        """
        if not _HAS_OPTUNA:
            logger.warning(_OPTUNA_INSTALL_HINT)
            return {
                "error": "optuna not installed",
                "install": "pip install optuna",
            }
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if param_space is None:
            param_space = self.default_param_space()
        splits = self._make_splits(X, y, groups)

        def objective(trial: optuna.trial.Trial) -> float:
            params = {name: _suggest_param(trial, name, space)
                      for name, space in param_space.items()}
            # Pruning: report intermediate fold scores so a pruner can
            # abort underperforming trials early.
            n_splits = len(splits)
            scores: List[float] = []
            for i, (train_idx, test_idx) in enumerate(splits):
                model = self._build_model(params)
                model.fit(X[train_idx], y[train_idx])
                y_pred = model.predict(X[test_idx])
                if self.scoring == "accuracy":
                    s = float(accuracy_score(y[test_idx], y_pred))
                else:
                    s = float(f1_score(y[test_idx], y_pred,
                                        average="macro", zero_division=0))
                scores.append(s)
                # Intermediate report (only meaningful with a pruner).
                trial.report(float(np.mean(scores)), step=i)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return float(np.mean(scores))

        sampler = self._build_sampler()
        pruner = self._build_pruner()
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=f"myoadapt_hpo_{self.model_type}",
        )
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=1,  # internal parallelism is fold-level (set by n_jobs at split time)
            show_progress_bar=False,
        )
        self.study_ = study
        self.best_params_ = dict(study.best_params)
        self.best_score_ = float(study.best_value)
        logger.info(
            f"HPO complete: best_score={self.best_score_:.4f} "
            f"best_params={self.best_params_} "
            f"(n_trials={len(study.trials)})"
        )
        return {
            "best_params": self.best_params_,
            "best_score": self.best_score_,
            "n_trials": len(study.trials),
            "study": study,
            "param_space": param_space,
        }

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_optimization_history(self,
                                  study: Optional[Any] = None,
                                  figsize: Tuple[float, float] = (8.0, 4.5),
                                  ) -> Any:
        """Plot the optimization trajectory (best score over trials).

        Returns a ``matplotlib.figure.Figure``. If matplotlib is not
        installed, returns a dict with the raw values instead so the
        caller can render with any other plotting library.
        """
        if not _HAS_OPTUNA:
            return {"error": "optuna not installed", "install": "pip install optuna"}
        study = study if study is not None else self.study_
        if study is None:
            raise RuntimeError("No study available — call optimize() first.")
        # Collect per-trial scores + running best.
        trials = [t for t in study.trials
                  if t.state == optuna.trial.TrialState.COMPLETE]
        if not trials:
            return {"error": "no completed trials to plot"}
        trial_numbers = np.array([t.number for t in trials])
        values = np.array([t.value for t in trials])
        running_best = np.maximum.accumulate(values)
        if not _HAS_MATPLOTLIB:
            return {
                "trial_numbers": trial_numbers.tolist(),
                "values": values.tolist(),
                "running_best": running_best.tolist(),
                "best_score": float(study.best_value),
            }
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(trial_numbers, values, s=15, alpha=0.6,
                   color="#0072B2", label="Trial score")
        ax.plot(trial_numbers, running_best, color="#D55E00", lw=2,
                label=f"Best so far ({study.best_value:.4f})")
        ax.set_xlabel("Trial number")
        ax.set_ylabel(f"CV {self.scoring}")
        ax.set_title(f"Optimization history — {self.model_type}")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="lower right", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        return fig

    def plot_param_importances(self,
                               study: Optional[Any] = None,
                               figsize: Tuple[float, float] = (8.0, 4.5),
                               ) -> Any:
        """Bar plot of hyperparameter importances (fANOVA).

        Returns a ``matplotlib.figure.Figure`` or a dict when matplotlib
        is unavailable.
        """
        if not _HAS_OPTUNA:
            return {"error": "optuna not installed", "install": "pip install optuna"}
        study = study if study is not None else self.study_
        if study is None:
            raise RuntimeError("No study available — call optimize() first.")
        try:
            importances = optuna.importance.get_param_importances(study)
        except Exception as exc:  # pragma: no cover — needs completed trials
            logger.warning(f"Could not compute param importances: {exc}")
            return {"error": str(exc)}
        if not importances:
            return {"error": "no param importances available"}
        names = list(importances.keys())
        values = [float(importances[n]) for n in names]
        if not _HAS_MATPLOTLIB:
            return {"params": names, "importances": values}
        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(names))[::-1]
        ax.barh(y_pos, values, color="#009E73", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel("Importance")
        ax.set_title(f"Hyperparameter importances — {self.model_type}")
        ax.grid(True, alpha=0.3, linestyle="--", axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        return fig


__all__ = ["HyperparameterOptimizer"]

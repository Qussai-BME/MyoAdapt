"""
registry.py — Feature plugin registry
=====================================

Lets users register custom feature extractors and dispatches the
extraction across all enabled modules.

"""
from __future__ import annotations

import inspect
import logging
from typing import Dict, List, Optional, Type, Union

import numpy as np

from myoadapt.features.correlation import CorrelationFeatures
from myoadapt.features.frequency_domain import FrequencyFeatures
from myoadapt.features.histograms import HistogramFeatures
from myoadapt.features.minirocket import MiniRocketFeatures
from myoadapt.features.time_domain import TimeDomainFeatures
from myoadapt.features.time_frequency import TimeFrequencyFeatures

logger = logging.getLogger(__name__)

FEATURE_REGISTRY: Dict[str, Type] = {
    "time_domain": TimeDomainFeatures,
    "frequency_domain": FrequencyFeatures,
    "time_frequency": TimeFrequencyFeatures,
    "histogram": HistogramFeatures,
    "correlation": CorrelationFeatures,
    "minirocket": MiniRocketFeatures,
}


def register_feature(name: str, cls: Type) -> None:
    """Register a custom feature extractor class."""
    if not hasattr(cls, "fit") or not hasattr(cls, "transform"):
        raise TypeError(f"{cls.__name__} must have fit() and transform() methods")
    FEATURE_REGISTRY[name] = cls
    logger.info(f"Registered feature extractor: {name} -> {cls.__name__}")


def list_features() -> List[str]:
    """Return the names of all registered feature extractors."""
    return sorted(FEATURE_REGISTRY.keys())


def _accepted_kwargs(cls: Type) -> set:
    """Return the set of kwarg names accepted by ``cls.__init__``.

    Falls back gracefully when ``__init__`` accepts ``**kwargs`` (in which
    case any name is accepted) and when introspection fails for any reason.
    """
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return set()
    accepted: set = set()
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            # **kwargs — accept anything.
            return None  # type: ignore[return-value]
        accepted.add(name)
    return accepted


def extract_features(windows: np.ndarray,
                     modules: Optional[List[str]] = None,
                     return_names: bool = False,
                     **kwargs) -> Union[np.ndarray, tuple]:
    """
    Extract features from a windowed signal.

    Parameters
    ----------
    windows : (n_windows, n_channels, n_samples) array
    modules : list of module names to use (default: all except minirocket
              and time_frequency, which are heavy)
    return_names : if True, also return feature names
    kwargs : per-module kwargs (e.g., ar_order=6, fs=2000)

    Returns
    -------
    features : (n_windows, n_features_total) array
    feature_names (optional) : list of strings
    """
    if modules is None:
        modules = ["time_domain", "frequency_domain", "histogram", "correlation"]

    feature_blocks: List[np.ndarray] = []
    feature_names: List[str] = []

    for mod_name in modules:
        if mod_name not in FEATURE_REGISTRY:
            logger.warning(f"Unknown feature module: {mod_name} — skipping")
            continue
        cls = FEATURE_REGISTRY[mod_name]
        accepted = _accepted_kwargs(cls)
        if accepted is None:
            extractor_kwargs = dict(kwargs)
        else:
            extractor_kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        try:
            extractor = cls(**extractor_kwargs)
        except TypeError:
            extractor = cls()
        extractor.fit(windows)
        feats = extractor.transform(windows)
        feature_blocks.append(feats)
        # Names
        if hasattr(extractor, "feature_names"):
            names = extractor.feature_names
            n_ch = windows.shape[1]
            if mod_name == "correlation":
                feature_names.extend(names[:feats.shape[1]])
            else:
                for ch in range(n_ch):
                    for n in names:
                        feature_names.append(f"ch{ch}_{n}")

    if not feature_blocks:
        raise RuntimeError("No feature blocks were computed — check `modules`")

    features = np.concatenate(feature_blocks, axis=1)

    if return_names:
        return features, feature_names
    return features

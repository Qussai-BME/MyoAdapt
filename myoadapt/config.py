"""
config.py — Central configuration for MyoAdapt v2.0
=====================================================

Single source of truth for filter / windowing / dataset parameters.
Used by data, features, evaluation, and CLI layers so that experiments
are reproducible from a single YAML or Python dataclass.

License: Apache 2.0
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


@dataclass
class FilterConfig:
    """Filter parameters following IEEE/ISEK biomedical standards."""
    sampling_rate: int = 2000
    cutoff_low: float = 20.0       # Hz (high-pass)
    cutoff_high: float = 450.0     # Hz (low-pass)
    filter_order: int = 4
    filter_type: str = "butterworth"  # butterworth, chebyshev, bessel, elliptic
    notch_freq: float = 50.0
    notch_quality: float = 30.0

    def validate(self) -> bool:
        if self.sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")
        if self.cutoff_high >= self.sampling_rate / 2:
            raise ValueError(
                f"cutoff_high ({self.cutoff_high}) must be < Nyquist "
                f"({self.sampling_rate / 2})"
            )
        if self.cutoff_low >= self.cutoff_high:
            raise ValueError("cutoff_low must be < cutoff_high")
        if not (1 <= self.filter_order <= 10):
            raise ValueError("filter_order must be 1-10")
        return True


@dataclass
class WindowConfig:
    """Windowing parameters."""
    window_ms: int = 200             # window length in ms
    increment_ms: int = 50           # step in ms
    overlap: float = field(init=False, default=0.0)

    def __post_init__(self):
        if self.window_ms <= 0 or self.increment_ms <= 0:
            raise ValueError("window_ms and increment_ms must be positive")
        if self.increment_ms > self.window_ms:
            raise ValueError("increment_ms cannot exceed window_ms")
        self.overlap = 1.0 - (self.increment_ms / self.window_ms)


@dataclass
class FeatureConfig:
    """Feature extraction options."""
    include_time_domain: bool = True
    include_frequency_domain: bool = True
    include_time_frequency: bool = False
    include_histogram: bool = True
    include_hjorth: bool = True
    include_inter_channel: bool = True
    include_ar: bool = True
    ar_order: int = 4
    histogram_bins: int = 10
    k_best: int = 420
    use_euclidean_alignment: bool = True
    use_minirocket: bool = False
    minirocket_num_kernels: int = 10_000

    def expected_dim(self, n_channels: int = 12) -> int:
        """Approximate raw feature dimensionality (before SelectKBest)."""
        per_ch = 0
        if self.include_time_domain:
            per_ch += 22
        if self.include_frequency_domain:
            per_ch += 8
        if self.include_histogram:
            per_ch += self.histogram_bins
        if self.include_hjorth:
            per_ch += 3
        total = per_ch * n_channels
        if self.include_inter_channel:
            total += n_channels * (n_channels - 1) // 2
        return total


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    model_type: str = "lite_dan"   # xgboost, random_forest, lda, svm, cnn1d, lite_dan
    n_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lambda_schedule: str = "gradual"  # gradual, fixed, none
    lambda_max: float = 1.0
    device: str = "auto"             # auto, cpu, cuda
    random_state: int = 42
    n_jobs: int = 4
    early_stopping_patience: int = 10


@dataclass
class EvaluationConfig:
    """Evaluation protocol options."""
    protocol: str = "loso"           # loso, lodo, kfold
    rest_class: str = "rest"
    metrics: tuple = ("accuracy", "macro_f1", "weighted_f1", "active_only_acc")
    statistical_test: str = "friedman_wilcoxon"  # friedman_wilcoxon, friedman_nemenyi
    significance_level: float = 0.05
    holm_correction: bool = True
    n_bootstrap: int = 1000


@dataclass
class DeploymentConfig:
    """Deployment options."""
    onnx_opset: int = 17
    onnx_optimize: bool = True
    onnx_dynamic_axes: bool = True
    quantize_int8: bool = False
    realtime_buffer_ms: int = 200
    realtime_overlap: float = 0.5
    trust_threshold: float = 0.5
    shap_max_display: int = 20


@dataclass
class MyoAdaptConfig:
    """Top-level configuration combining all sub-configs."""
    filter: FilterConfig = field(default_factory=FilterConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    data_root: Optional[Path] = None
    output_dir: Path = Path("./outputs")
    experiment_name: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["data_root"] = str(self.data_root) if self.data_root else None
        d["output_dir"] = str(self.output_dir)
        return d

    def to_yaml(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_json(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> MyoAdaptConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> MyoAdaptConfig:
        sub_keys = {
            "filter": FilterConfig,
            "window": WindowConfig,
            "features": FeatureConfig,
            "training": TrainingConfig,
            "evaluation": EvaluationConfig,
            "deployment": DeploymentConfig,
        }
        # Field names that should NOT be passed to sub-config constructors
        # (because they're computed in __post_init__)
        sub_init_blacklist = {
            "window": {"overlap"},  # overlap is computed from window_ms/increment_ms
        }
        kwargs: Dict[str, Any] = {}
        for k, v in d.items():
            if k in sub_keys and isinstance(v, dict):
                # Drop blacklisted keys
                clean_v = {kk: vv for kk, vv in v.items()
                           if kk not in sub_init_blacklist.get(k, set())}
                kwargs[k] = sub_keys[k](**clean_v)
            elif k == "data_root":
                kwargs[k] = Path(v) if v else None
            elif k == "output_dir":
                kwargs[k] = Path(v)
            else:
                kwargs[k] = v
        # Filter unknown keys
        valid = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in kwargs.items() if k in valid}
        return cls(**kwargs)

    def validate(self) -> bool:
        self.filter.validate()
        self.window.__post_init__()
        return True


# Default singleton for quick access
DEFAULT_CONFIG = MyoAdaptConfig()

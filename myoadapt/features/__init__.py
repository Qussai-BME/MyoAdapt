"""
myoadapt.features — Unified feature extraction
=================================================

Public API:
    extract_features    — main dispatcher
    list_features       — list registered extractor names
    FEATURE_REGISTRY    — plugin registry
    TimeDomainFeatures  — MAV, RMS, ZCR, WL, SSC, WAMP, MYOP, AR (Yule-Walker), Hjorth
    FrequencyFeatures   — MNF, MDF, PKF, PSR, SM
    TimeFrequencyFeatures — wavelet + STFT
    HistogramFeatures   — amplitude distribution
    CorrelationFeatures — inter-channel correlation
    MiniRocketFeatures  — PPV features
"""
from myoadapt.features.correlation import CorrelationFeatures
from myoadapt.features.frequency_domain import FrequencyFeatures
from myoadapt.features.histograms import HistogramFeatures
from myoadapt.features.minirocket import MiniRocketFeatures, MiniRocketVerifier
from myoadapt.features.registry import (
    FEATURE_REGISTRY,
    extract_features,
    list_features,
    register_feature,
)
from myoadapt.features.time_domain import AR_COEFFICIENTS_YULE_WALKER, TimeDomainFeatures
from myoadapt.features.time_frequency import TimeFrequencyFeatures

__all__ = [
    "TimeDomainFeatures", "AR_COEFFICIENTS_YULE_WALKER",
    "FrequencyFeatures", "TimeFrequencyFeatures",
    "HistogramFeatures", "CorrelationFeatures",
    "MiniRocketFeatures", "MiniRocketVerifier",
    "FEATURE_REGISTRY", "register_feature", "list_features", "extract_features",
]

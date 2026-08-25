"""Unit tests for feature extraction."""
import numpy as np
import pytest

from myoadapt.features.time_domain import (
    mav, rms, wl, zcr, ssc, wamp, myop, hjorth_parameters,
    extract_per_channel, TimeDomainFeatures,
)
from myoadapt.features.frequency_domain import (
    mean_frequency, median_frequency, FrequencyFeatures,
)
from myoadapt.features.histograms import histogram_features, HistogramFeatures
from myoadapt.features.correlation import inter_channel_correlation, CorrelationFeatures
from myoadapt.features.registry import extract_features, FEATURE_REGISTRY


def test_mav():
    assert abs(mav(np.array([-1, 2, -3, 4])) - 2.5) < 1e-6


def test_rms():
    assert abs(rms(np.array([3, 4])) - 3.53553) < 1e-3


def test_wl():
    assert abs(wl(np.array([1, 3, 2])) - 3.0) < 1e-6  # |3-1| + |2-3|


def test_zcr():
    seg = np.array([1, -1, 1, -1])
    assert zcr(seg) > 0.5


def test_ssc():
    seg = np.array([1, 2, 1, 2, 1])
    assert ssc(seg) > 0


def test_hjorth():
    seg = np.random.randn(500)
    h = hjorth_parameters(seg)
    assert "activity" in h
    assert "mobility" in h
    assert "complexity" in h
    assert h["activity"] > 0


def test_extract_per_channel_count():
    seg = np.random.randn(400)
    feats = extract_per_channel(seg, ar_order=4)
    # 7 classical + 3 Hjorth + 4 AR + 8 derived = 22
    assert len(feats) == 22


def test_time_domain_extractor_transform():
    windows = np.random.randn(10, 12, 400).astype(np.float32)
    ext = TimeDomainFeatures()
    ext.fit(windows)
    out = ext.transform(windows)
    assert out.shape == (10, 12 * 22)


def test_frequency_features():
    seg = np.random.randn(400)
    fs = 2000
    assert mean_frequency(seg, fs) > 0
    assert median_frequency(seg, fs) > 0


def test_histogram():
    seg = np.random.randn(400)
    h = histogram_features(seg, n_bins=10)
    assert len(h) == 10
    assert abs(sum(h.values()) - 1.0) < 1e-5  # normalized


def test_correlation():
    window = np.random.randn(12, 400)
    corr = inter_channel_correlation(window)
    assert corr.shape == (66,)  # C(12, 2)


def test_extract_features_dispatch():
    windows = np.random.randn(8, 12, 400).astype(np.float32)
    feats, names = extract_features(
        windows,
        modules=["time_domain", "histogram"],
        return_names=True,
        ar_order=4,
    )
    assert feats.shape[0] == 8
    assert feats.shape[1] == 12 * (22 + 10)
    assert len(names) == feats.shape[1]


def test_feature_registry_has_all():
    expected = {"time_domain", "frequency_domain", "time_frequency",
                "histogram", "correlation", "minirocket"}
    assert expected.issubset(set(FEATURE_REGISTRY.keys()))


def test_time_frequency_features_via_extract_features():
    """TimeFrequencyFeatures (wavelet/STFT) is registered in extract_features'
    module dispatch but excluded from the CLI's default module list (it's
    slower), so it had no coverage from any existing test path."""
    import numpy as np
    from myoadapt.features import extract_features
    rng = np.random.default_rng(0)
    windows = rng.standard_normal((5, 4, 400)).astype(np.float64)
    X = extract_features(windows, modules=["time_frequency"], fs=2000)
    assert X.shape[0] == 5
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()

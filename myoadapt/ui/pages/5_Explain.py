"""Page 5 — Explain: SHAP transparency report (fast)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from myoadapt.ui.theme import (
    apply_theme,
    footer,
    hero,
    section_header,
    sidebar_brand,
    spacer,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Explain — MyoAdapt", layout="wide", page_icon="🔍")
apply_theme(st)
sidebar_brand(st)

st.markdown(hero(
    eyebrow="Workflow · 05",
    title="SHAP transparency report",
    subtitle=(
        "Generate a per-prediction SHAP report with per-channel contribution, "
        "top features, and a calibration-aware trust score. Designed to "
        "support EU AI Act Article 13 documentation (research-grade)."
    ),
), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached model (same as Deploy page)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_trained_model():
    """Train a small RF model on cached synthetic data."""
    rng = np.random.default_rng(42)
    n_subjects, n_classes, n_channels, win_n, per = 4, 3, 4, 100, 4
    windows, labels = [], []
    for s in range(n_subjects):
        for c in range(n_classes):
            for _ in range(per):
                env = np.exp(-((np.arange(win_n) - win_n // 2) ** 2) /
                             (2 * (win_n / 6) ** 2))
                sig = (rng.standard_normal((n_channels, win_n))
                       * (0.05 + 0.15 * env)
                       + s * 0.05 + c * 0.3)
                windows.append(sig.astype(np.float32))
                labels.append(c)
    from myoadapt.features import extract_features
    feats = extract_features(np.stack(windows), modules=["time_domain"])
    from myoadapt.models.classical import EMGClassifier
    clf = EMGClassifier(model_type="random_forest", n_estimators=10,
                        k_features=20, random_state=42)
    clf.fit(feats, np.asarray(labels))
    return clf, feats


# ---------------------------------------------------------------------------
# Section: model + sample selection
# ---------------------------------------------------------------------------
st.markdown(section_header("01", "Select sample"), unsafe_allow_html=True)

clf, X = get_trained_model()

sample_idx = st.slider("Sample index", 0, len(X) - 1, 0,
                        help="Pick which test sample to explain.")
sample = X[sample_idx:sample_idx + 1]

c1, c2 = st.columns(2)
with c1:
    st.metric("Sample index", str(sample_idx))
with c2:
    pred = clf.predict(sample)[0]
    st.metric("Predicted class", str(pred))

# ---------------------------------------------------------------------------
# Section: generate report
# ---------------------------------------------------------------------------
st.markdown(spacer(), unsafe_allow_html=True)
st.markdown(section_header("02", "Generate SHAP report"), unsafe_allow_html=True)

explain_clicked = st.button("🔍 Generate report", type="primary", use_container_width=False)

if explain_clicked:
    progress = st.progress(0, text="Computing SHAP values...")
    try:
        from myoadapt.deployment.shap_reports import ShapReportGenerator
        gen = ShapReportGenerator(clf)

        progress.progress(50, text="Generating report...")
        report = gen.explain(sample)
        report_dict = report.to_dict() if hasattr(report, "to_dict") else report.__dict__

        progress.progress(100, text="Done ✅")
        st.success("Report generated")

        # Prediction summary
        st.markdown(spacer(), unsafe_allow_html=True)
        st.markdown(section_header("03", "Prediction summary"), unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Predicted class", str(report_dict.get("predicted_class", "N/A")))
        with r2:
            conf = report_dict.get("confidence", 0)
            st.metric("Confidence", f"{conf:.3f}" if isinstance(conf, (int, float)) else str(conf))
        with r3:
            ts = report_dict.get("trust_score", 0)
            st.metric("Trust score", f"{ts:.3f}" if isinstance(ts, (int, float)) else str(ts))

        # Top features
        st.markdown(spacer(), unsafe_allow_html=True)
        st.markdown(section_header("04", "Top contributing features"), unsafe_allow_html=True)
        top_feats = report_dict.get("top_features", [])
        if top_feats:
            feat_rows = []
            for i, feat in enumerate(top_feats[:10]):
                if isinstance(feat, dict):
                    feat_rows.append({
                        "Rank": i + 1,
                        "Feature": feat.get("name", f"f{i}"),
                        "SHAP value": f"{feat.get('shap_value', 0):.6f}",
                        "Contribution %": f"{feat.get('contribution_pct', 0):.2f}%",
                    })
                else:
                    feat_rows.append({"Rank": i + 1, "Feature": str(feat), "SHAP value": "N/A", "Contribution %": "N/A"})
            st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No top features available.")

        # Per-channel contribution
        st.markdown(spacer(), unsafe_allow_html=True)
        st.markdown(section_header("05", "Per-channel contribution"), unsafe_allow_html=True)
        per_ch = report_dict.get("per_channel_contribution", {})
        if per_ch:
            ch_rows = [{"Channel": k, "Contribution": f"{v:+.4f}"} for k, v in list(per_ch.items())[:12]]
            st.dataframe(pd.DataFrame(ch_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Per-channel breakdown not available.")

        # Explanation text
        st.markdown(spacer(), unsafe_allow_html=True)
        st.markdown(section_header("06", "Plain-language explanation"), unsafe_allow_html=True)
        explanation = report_dict.get("explanation", "N/A")
        st.info(explanation)

    except ImportError:
        st.error("SHAP not installed. Run: pip install shap")
    except Exception as e:
        st.error(f"Report generation failed: {e}")

# ---------------------------------------------------------------------------
# Section: CLI equivalent
# ---------------------------------------------------------------------------
st.markdown(spacer(), unsafe_allow_html=True)
st.markdown(section_header("07", "CLI equivalent"), unsafe_allow_html=True)
st.code(
    "myoadapt explain --model-path ./models/model.pkl --features ./data/sample_features.csv",
    language="bash",
)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(spacer(), unsafe_allow_html=True)
st.markdown(footer(), unsafe_allow_html=True)

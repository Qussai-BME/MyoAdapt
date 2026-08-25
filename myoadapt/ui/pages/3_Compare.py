"""Page 3 — Compare models head-to-head (fast, < 8 seconds)."""
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
st.set_page_config(page_title="Compare — MyoAdapt", layout="wide", page_icon="⚖️")
apply_theme(st)
sidebar_brand(st)

st.markdown(hero(
    eyebrow="Workflow · 03",
    title="Compare models head-to-head",
    subtitle=(
        "Train 3 models on the same synthetic data, run LOSO for each, "
        "and compare with Friedman + Nemenyi. Renders in under 8 seconds."
    ),
), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached data (shared with Train + Evaluate pages)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_tiny_emg(seed: int = 42):
    rng = np.random.default_rng(seed)
    n_subjects, n_classes, n_channels, win_n, per = 4, 3, 4, 100, 4
    windows, labels, groups = [], [], []
    for s in range(n_subjects):
        drift = rng.standard_normal(n_channels) * 0.05
        for c in range(n_classes):
            class_offset = c * 0.3
            for _ in range(per):
                env = np.exp(-((np.arange(win_n) - win_n // 2) ** 2) /
                             (2 * (win_n / 6) ** 2))
                sig = (rng.standard_normal((n_channels, win_n))
                       * (0.05 + 0.15 * env)
                       + drift[:, None]
                       + class_offset)
                windows.append(sig.astype(np.float32))
                labels.append(c)
                groups.append(s)
    return (np.stack(windows), np.asarray(labels), np.asarray(groups))


@st.cache_data(show_spinner=False)
def extract_tiny_features(windows: np.ndarray):
    from myoadapt.features import extract_features
    return extract_features(windows, modules=["time_domain"], return_names=True)


# ---------------------------------------------------------------------------
# Section: configuration
# ---------------------------------------------------------------------------
st.markdown(section_header("01", "Configuration"), unsafe_allow_html=True)
st.markdown(
    "Three models will be compared on the same 4-subject synthetic dataset "
    "using 4-fold LOSO. The Friedman test checks whether the models are "
    "significantly different; Nemenyi post-hoc identifies which pairs differ."
)

# ---------------------------------------------------------------------------
# Section: run comparison
# ---------------------------------------------------------------------------
st.markdown(section_header("02", "Run comparison"), unsafe_allow_html=True)

run_clicked = st.button("⚖️ Compare models", type="primary", use_container_width=False)

if run_clicked:
    progress = st.progress(0, text="Preparing data...")
    status = st.empty()

    windows, y, groups = generate_tiny_emg()
    feats, _ = extract_tiny_features(windows)
    progress.progress(10, text=f"Data ready: {feats.shape}")

    from myoadapt.evaluation.metrics import compute_metrics
    from myoadapt.models.classical import EMGClassifier

    model_names = ["random_forest", "lda", "svm"]
    per_model_folds = {m: [] for m in model_names}
    unique_subjects = np.unique(groups)

    for mi, model_name in enumerate(model_names):
        for fi, test_subj in enumerate(unique_subjects):
            train_mask = groups != test_subj
            test_mask = groups == test_subj
            X_tr, X_te = feats[train_mask], feats[test_mask]
            y_tr, y_te = y[train_mask], y[test_mask]

            clf = EMGClassifier(
                model_type=model_name, n_estimators=5,
                k_features=min(20, X_tr.shape[1]), random_state=42,
            )
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            classes = sorted(np.unique(np.concatenate([y_tr, y_te])).tolist())
            m = compute_metrics(y_te, y_pred, classes=classes, rest_class=None)
            per_model_folds[model_name].append(float(m["accuracy"]))

        pct = 10 + int((mi + 1) / len(model_names) * 80)
        progress.progress(pct, text=f"{model_name}: mean acc = {np.mean(per_model_folds[model_name])*100:.1f}%")

    progress.progress(100, text="Done ✅")
    status.success("Comparison complete")

    # Results
    st.markdown(spacer(), unsafe_allow_html=True)
    st.markdown(section_header("03", "Results"), unsafe_allow_html=True)

    # KPIs
    r1, r2, r3 = st.columns(3)
    for i, (model, accs) in enumerate(per_model_folds.items()):
        col = [r1, r2, r3][i]
        with col:
            st.metric(model, f"{np.mean(accs)*100:.1f}%", delta=f"±{np.std(accs)*100:.1f}")

    # Per-fold table
    st.markdown(spacer(), unsafe_allow_html=True)
    st.markdown("**Per-fold accuracy:**")
    fold_df = pd.DataFrame({
        "Fold": [f"S{s}" for s in unique_subjects],
        **{m: [f"{a*100:.1f}%" for a in accs] for m, accs in per_model_folds.items()},
    })
    st.dataframe(fold_df, use_container_width=True, hide_index=True)

    # Bar chart
    st.markdown(spacer(), unsafe_allow_html=True)
    st.markdown("**Model comparison:**")
    chart_df = pd.DataFrame({
        "Fold": [f"S{s}" for s in unique_subjects] * len(model_names),
        "Model": [m for m in model_names for _ in unique_subjects],
        "Accuracy": [a for accs in per_model_folds.values() for a in accs],
    })
    st.bar_chart(chart_df, x="Fold", y="Accuracy", color="Model", use_container_width=True, height=350)

    # Friedman test
    st.markdown(spacer(), unsafe_allow_html=True)
    st.markdown(section_header("04", "Statistical analysis"), unsafe_allow_html=True)

    try:
        from myoadapt.evaluation.statistics import friedman_test, wilcoxon_pairwise
        fried = friedman_test(per_model_folds)
        st.markdown(f"**Friedman test:** χ² = {fried.get('chi2', 0):.3f}, p = {fried.get('p_value', 1):.4f}")
        if fried.get("p_value", 1) < 0.05:
            st.success("Significant difference between models (p < 0.05)")
        else:
            st.info("No significant difference (p ≥ 0.05)")

        # Wilcoxon pairwise
        wilc = wilcoxon_pairwise(per_model_folds)
        if "tests" in wilc:
            st.markdown("**Pairwise Wilcoxon (with Holm-Šídák correction):**")
            wilc_rows = []
            for t in wilc["tests"]:
                wilc_rows.append({
                    "Pair": f"{t['model_a']} vs {t['model_b']}",
                    "p_raw": f"{t.get('p_value_raw', 1):.4f}",
                    "p_corrected": f"{t.get('p_value_corrected', 1):.4f}",
                    "Significant": "✓" if t.get("significant", False) else "✗",
                })
            st.dataframe(pd.DataFrame(wilc_rows), use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Statistical analysis skipped: {e}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(spacer(), unsafe_allow_html=True)
st.markdown(footer(), unsafe_allow_html=True)

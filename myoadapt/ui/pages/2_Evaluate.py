"""Page 2 — Evaluate with LOSO on tiny synthetic EMG (fast, < 5 seconds)."""
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
st.set_page_config(page_title="Evaluate — MyoAdapt", layout="wide", page_icon="📊")
apply_theme(st)
sidebar_brand(st)

st.markdown(hero(
    eyebrow="Workflow · 02",
    title="Leave-One-Subject-Out evaluation",
    subtitle=(
        "Run k-fold LOSO cross-validation on tiny synthetic data. Each fold "
        "trains on k-1 subjects and tests on the held-out subject. Capped at "
        "4 folds so the demo renders in under 5 seconds."
    ),
), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached tiny-EMG generator (same seed as Train page)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_tiny_emg(seed: int = 42):
    """Generate 40 windows × 4 channels × 100 samples × 3 classes × 4 subjects."""
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
    """Extract time-domain features (cached)."""
    from myoadapt.features import extract_features
    return extract_features(windows, modules=["time_domain"], return_names=True)


# ---------------------------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Evaluation configuration")

n_folds = st.sidebar.slider(
    "Number of folds (subjects)", 2, 4, 3,
    help="LOSO with k subjects. The dataset has 4 subjects total.",
)
model_type = st.sidebar.selectbox(
    "Model",
    ["random_forest", "lda", "svm", "logistic"],
    help="Classical models only. n_estimators=5 for speed.",
)

# ---------------------------------------------------------------------------
# Section: configuration
# ---------------------------------------------------------------------------
st.markdown(section_header("01", "Configuration"), unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Folds", n_folds)
with c2:
    st.metric("Model", model_type)
with c3:
    st.metric("Protocol", "LOSO")

# ---------------------------------------------------------------------------
# Section: run LOSO
# ---------------------------------------------------------------------------
st.markdown(section_header("02", "Run LOSO"), unsafe_allow_html=True)

run_clicked = st.button("📊 Run LOSO evaluation", type="primary", use_container_width=False)

if run_clicked:
    progress = st.progress(0, text="Preparing data...")
    status = st.empty()

    # Data
    windows, y, groups = generate_tiny_emg()
    feats, feat_names = extract_tiny_features(windows)
    progress.progress(20, text=f"Data: {feats.shape[0]} windows, {feats.shape[1]} features")

    # LOSO loop
    from myoadapt.evaluation.metrics import compute_metrics
    from myoadapt.models.classical import EMGClassifier

    unique_subjects = np.unique(groups)[:n_folds]
    fold_results = []
    all_true = []
    all_pred = []

    for i, test_subj in enumerate(unique_subjects):
        train_mask = groups != test_subj
        test_mask = groups == test_subj

        X_train, X_test = feats[train_mask], feats[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        clf = EMGClassifier(
            model_type=model_type,
            n_estimators=5,
            k_features=min(20, X_train.shape[1]),
            random_state=42,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        classes = sorted(np.unique(np.concatenate([y_train, y_test])).tolist())
        m = compute_metrics(y_test, y_pred, classes=classes, rest_class=None)

        fold_results.append({
            "Fold": f"S{test_subj}",
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "Accuracy": f"{m['accuracy']*100:.1f}%",
            "Macro-F1": f"{m['macro_f1']*100:.1f}%",
        })
        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())

        pct = 20 + int((i + 1) / len(unique_subjects) * 70)
        progress.progress(pct, text=f"Fold {i+1}/{len(unique_subjects)}: S{test_subj} acc={m['accuracy']*100:.1f}%")

    # Aggregate
    accs = [float(r["Accuracy"].rstrip("%")) for r in fold_results]
    f1s = [float(r["Macro-F1"].rstrip("%")) for r in fold_results]
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    mean_f1 = np.mean(f1s)
    std_f1 = np.std(f1s)

    progress.progress(100, text="Done ✅")
    status.success(f"LOSO complete — {len(unique_subjects)} folds")

    # Results KPIs
    st.markdown(spacer(), unsafe_allow_html=True)
    st.markdown(section_header("03", "Results"), unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("Mean accuracy", f"{mean_acc:.1f}%", delta=f"±{std_acc:.1f}")
    with r2:
        st.metric("Mean macro-F1", f"{mean_f1:.1f}%", delta=f"±{std_f1:.1f}")
    with r3:
        st.metric("Folds", str(len(unique_subjects)))
    with r4:
        st.metric("Best fold", f"S{unique_subjects[np.argmax(accs)]}", delta=f"{max(accs):.1f}%")

    # Per-fold table
    st.markdown(spacer(), unsafe_allow_html=True)
    st.markdown("**Per-fold results:**")
    st.dataframe(pd.DataFrame(fold_results), use_container_width=True, hide_index=True)

    # Aggregated confusion matrix
    st.markdown(spacer(), unsafe_allow_html=True)
    st.markdown("**Aggregated confusion matrix (across all LOSO folds):**")
    from sklearn.metrics import confusion_matrix
    all_classes = sorted(np.unique(all_true + all_pred).tolist())
    cm = confusion_matrix(all_true, all_pred, labels=all_classes)
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in all_classes],
                         columns=[f"Pred_{c}" for c in all_classes])
    st.dataframe(cm_df, use_container_width=True)

    # Bar chart
    st.markdown(spacer(), unsafe_allow_html=True)
    st.markdown("**Per-fold accuracy:**")
    chart_df = pd.DataFrame({
        "Fold": [r["Fold"] for r in fold_results],
        "Accuracy": accs,
        "Macro-F1": f1s,
    }).set_index("Fold")
    st.bar_chart(chart_df, use_container_width=True, height=300)

# ---------------------------------------------------------------------------
# Section: CLI equivalent
# ---------------------------------------------------------------------------
st.markdown(spacer(), unsafe_allow_html=True)
st.markdown(section_header("04", "CLI equivalent"), unsafe_allow_html=True)
st.markdown("For full LOSO on real NinaPro data:")
st.code(
    f"""myoadapt eval \\
  --model-path ./models/{model_type}_db2.pkl \\
  --db DB2 \\
  --protocol loso \\
  --output ./results/loso_db2""",
    language="bash",
)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(spacer(), unsafe_allow_html=True)
st.markdown(footer(), unsafe_allow_html=True)

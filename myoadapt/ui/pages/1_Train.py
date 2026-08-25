"""Page 1 — Train a model on tiny synthetic EMG (fast, < 3 seconds)."""
from __future__ import annotations

import io
import pickle
import time

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
st.set_page_config(page_title="Train — MyoAdapt", layout="wide", page_icon="🚂")
apply_theme(st)
sidebar_brand(st)

st.markdown(hero(
    eyebrow="Workflow · 01",
    title="Train a model",
    subtitle=(
        "Configure a classifier and train on tiny synthetic EMG (40 windows, "
        "4 channels, 100 samples, 3 classes, 4 subjects). The full pipeline "
        "runs in under 3 seconds — the same code path works on real NinaPro "
        "data via the CLI."
    ),
), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached synthetic-EMG generator — TINY for speed.
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
    """Extract time-domain features (fast, ~0.04s)."""
    from myoadapt.features import extract_features
    feats, names = extract_features(
        windows, modules=["time_domain"], return_names=True,
    )
    return feats, names


# ---------------------------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Training configuration")

model_type = st.sidebar.selectbox(
    "Model",
    ["random_forest", "lda", "svm", "logistic"],
    help="Classical models only in the live demo. Use the CLI for deep models.",
)
n_estimators = st.sidebar.slider(
    "n_estimators (RF only)", 5, 50, 10,
    help="Number of trees for RandomForest. Ignored for LDA/SVM/Logistic.",
)
k_features = st.sidebar.slider(
    "k_features (SelectKBest)", 10, 80, 20,
    help="Top-k features kept after univariate F-test selection.",
)

# ---------------------------------------------------------------------------
# Section: configuration summary
# ---------------------------------------------------------------------------
st.markdown(section_header("01", "Configuration"), unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Model", model_type)
with c2:
    st.metric("n_estimators", n_estimators if model_type == "random_forest" else "n/a")
with c3:
    st.metric("k_features", k_features)
with c4:
    st.metric("Feature module", "time_domain")

st.markdown(
    "Pipeline: generate tiny synthetic EMG → extract time-domain features "
    "(22 per channel × 4 channels = 88) → SelectKBest → train → report "
    "in-sample accuracy and per-class breakdown."
)

# ---------------------------------------------------------------------------
# Section: run training
# ---------------------------------------------------------------------------
st.markdown(section_header("02", "Train model"), unsafe_allow_html=True)

train_clicked = st.button("🚂 Train model", type="primary", use_container_width=False)

if train_clicked:
    progress = st.progress(0, text="Starting...")
    status = st.empty()

    # Step 1: data
    status.info("Generating tiny synthetic EMG...")
    windows, y, groups = generate_tiny_emg()
    progress.progress(25, text=f"Data ready: {windows.shape[0]} windows × {windows.shape[1]} ch × {windows.shape[2]} samples")

    # Step 2: features
    status.info("Extracting time-domain features...")
    feats, feat_names = extract_tiny_features(windows)
    progress.progress(50, text=f"Features: {feats.shape[1]} dimensions")

    # Step 3: train
    status.info(f"Training {model_type}...")
    from myoadapt.models.classical import EMGClassifier
    clf = EMGClassifier(
        model_type=model_type,
        n_estimators=n_estimators,
        k_features=min(k_features, feats.shape[1]),
        random_state=42,
    )
    t0 = time.perf_counter()
    clf.fit(feats, y)
    train_seconds = time.perf_counter() - t0
    progress.progress(75, text=f"Trained in {train_seconds:.3f}s")

    # Step 4: metrics
    from myoadapt.evaluation.metrics import compute_metrics
    y_pred = clf.predict(feats)
    classes = [str(i) for i in range(3)]
    m = compute_metrics(y, y_pred, classes=classes, rest_class=None)

    # Model size
    buf = io.BytesIO()
    pickle.dump(clf, buf)
    model_size_bytes = buf.tell()

    progress.progress(100, text="Done ✅")
    status.success(f"Training complete — accuracy: {m['accuracy']*100:.1f}%")

    # Results KPI strip
    st.markdown(spacer(), unsafe_allow_html=True)
    st.markdown(section_header("03", "Training results"), unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("Train accuracy", f"{m['accuracy']*100:.1f}%", delta="in-sample")
    with r2:
        st.metric("Macro-F1", f"{m['macro_f1']*100:.1f}%", delta="in-sample")
    with r3:
        st.metric("Feature dim", f"{min(k_features, feats.shape[1])}", delta=f"of {feats.shape[1]}")
    with r4:
        size_str = f"{model_size_bytes/1024:.1f} KB" if model_size_bytes >= 1024 else f"{model_size_bytes} B"
        st.metric("Model size", size_str, delta=f"{train_seconds:.3f}s")

    # Per-class table
    st.markdown(spacer(), unsafe_allow_html=True)
    st.markdown("**Per-class metrics:**")
    per_class_rows = []
    for cls, m_cls in m["per_class"].items():
        per_class_rows.append({
            "Class": f"Class {cls}",
            "Precision": f"{m_cls['precision']:.3f}",
            "Recall": f"{m_cls['recall']:.3f}",
            "F1": f"{m_cls['f1']:.3f}",
            "Support": m_cls["support"],
        })
    pc_df = pd.DataFrame(per_class_rows)
    st.dataframe(pc_df, use_container_width=True, hide_index=True)

    # Bar chart
    st.markdown(spacer(), unsafe_allow_html=True)
    st.markdown("**Per-class F1 scores:**")
    chart_df = pd.DataFrame({
        "Class": [r["Class"] for r in per_class_rows],
        "Precision": [float(r["Precision"]) for r in per_class_rows],
        "Recall": [float(r["Recall"]) for r in per_class_rows],
        "F1": [float(r["F1"]) for r in per_class_rows],
    }).set_index("Class")
    st.bar_chart(chart_df, use_container_width=True, height=300)

# ---------------------------------------------------------------------------
# Section: CLI equivalent
# ---------------------------------------------------------------------------
st.markdown(spacer(), unsafe_allow_html=True)
st.markdown(section_header("04", "CLI equivalent"), unsafe_allow_html=True)
st.markdown(
    "For production runs on real NinaPro data, use the CLI:"
)
st.code(
    f"""myoadapt train \\
  --db DB2 \\
  --model {model_type} \\
  --data-root ./data \\
  --output ./models/{model_type}_db2.pkl \\
  --k-features {k_features}""",
    language="bash",
)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(spacer(), unsafe_allow_html=True)
st.markdown(footer(), unsafe_allow_html=True)

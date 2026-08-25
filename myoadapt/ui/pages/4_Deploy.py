"""Page 4 — Deploy: ONNX export + 5-CPU benchmark (fast)."""
from __future__ import annotations

import io
import pickle

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
st.set_page_config(page_title="Deploy — MyoAdapt", layout="wide", page_icon="📦")
apply_theme(st)
sidebar_brand(st)

st.markdown(hero(
    eyebrow="Workflow · 04",
    title="Deploy to edge",
    subtitle=(
        "Export a trained model to ONNX with SHA-256 tamper detection, "
        "then run the 5-CPU latency benchmark to project real-time feasibility "
        "across Intel Celeron G5900 → i7-10700 → AMD Ryzen 5 3600."
    ),
), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached model (same tiny data as Train page)
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
# Section: model summary
# ---------------------------------------------------------------------------
st.markdown(section_header("01", "Trained model"), unsafe_allow_html=True)

clf, X_sample = get_trained_model()

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Model type", clf.model_type)
with c2:
    st.metric("n_features", X_sample.shape[1])
with c3:
    buf = io.BytesIO()
    pickle.dump(clf, buf)
    size = buf.tell()
    st.metric("Pickle size", f"{size/1024:.1f} KB")

# ---------------------------------------------------------------------------
# Section: ONNX export
# ---------------------------------------------------------------------------
st.markdown(spacer(), unsafe_allow_html=True)
st.markdown(section_header("02", "ONNX export + SHA-256"), unsafe_allow_html=True)

export_clicked = st.button("📦 Export to ONNX", type="primary", use_container_width=False)

if export_clicked:
    progress = st.progress(0, text="Exporting...")
    try:
        import os
        import tempfile

        from myoadapt.deployment.onnx_export import OnnxExporter

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = os.path.join(tmp, "model.onnx")
            progress.progress(30, text="Converting to ONNX...")
            info = OnnxExporter.export(clf, onnx_path)

            progress.progress(60, text="Computing SHA-256...")
            sha = info.get("sha256", "N/A")

            progress.progress(80, text="Verifying...")
            verify = OnnxExporter.verify(onnx_path)

            progress.progress(100, text="Done ✅")
            st.success(f"ONNX model exported — {info['size_bytes']/1024:.1f} KB")

            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("ONNX size", f"{info['size_bytes']/1024:.1f} KB")
            with r2:
                st.metric("Opset", str(info.get("opset", 17)))
            with r3:
                st.metric("SHA-256", sha[:12] + "...")

            st.code(f"SHA-256: {sha}", language="text")
            st.info(f"Verification: {'passed ✅' if verify.get('test_passed') else 'failed ❌'}")

    except ImportError:
        st.error("ONNX packages not installed. Run: pip install onnx onnxruntime skl2onnx")
    except Exception as e:
        st.error(f"Export failed: {e}")

# ---------------------------------------------------------------------------
# Section: 5-CPU benchmark
# ---------------------------------------------------------------------------
st.markdown(spacer(), unsafe_allow_html=True)
st.markdown(section_header("03", "5-CPU hardware benchmark"), unsafe_allow_html=True)
st.markdown(
    "Projects inference latency to 5 representative CPUs using PassMark "
    "score ratios. The local CPU's PassMark is auto-detected or set to a "
    "default of 8000 (mainstream desktop)."
)

bench_clicked = st.button("🏃 Run benchmark (10 runs)", type="primary", use_container_width=False)

if bench_clicked:
    progress = st.progress(0, text="Running benchmark...")
    try:
        from myoadapt.deployment.hardware_bench import CPU_PASSMARK, HardwareBenchmark
        bench = HardwareBenchmark(clf, n_runs=10)
        results = bench.run()

        progress.progress(100, text="Done ✅")
        st.success("Benchmark complete")

        # Local CPU
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Local CPU", results.get("local_cpu", "Unknown"))
        with r2:
            st.metric("Local PassMark", str(results.get("local_passmark", 8000)))
        with r3:
            st.metric("Local latency", f"{results.get('local_mean_ms', 0):.2f} ms")

        # Per-CPU projection table
        st.markdown(spacer(), unsafe_allow_html=True)
        st.markdown("**5-CPU projection:**")
        proj_rows = []
        for cpu, proj in results.get("per_cpu_projected", {}).items():
            proj_rows.append({
                "CPU": cpu,
                "PassMark": CPU_PASSMARK.get(cpu, "N/A"),
                "Projected latency (ms)": f"{proj.get('projected_mean_ms', 0):.2f}",
                "Real-time (<50ms)": "✓" if proj.get("realtime_capable") else "✗",
            })
        st.dataframe(pd.DataFrame(proj_rows), use_container_width=True, hide_index=True)

        st.info(f"Verdict: {results.get('verdict', 'N/A')}")

    except Exception as e:
        st.error(f"Benchmark failed: {e}")

# ---------------------------------------------------------------------------
# Section: CLI equivalent
# ---------------------------------------------------------------------------
st.markdown(spacer(), unsafe_allow_html=True)
st.markdown(section_header("04", "CLI equivalent"), unsafe_allow_html=True)
st.code(
    """myoadapt export --model-path ./models/model.pkl --output ./models/model.onnx
myoadapt benchmark --model-path ./models/model.pkl --n-runs 100""",
    language="bash",
)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(spacer(), unsafe_allow_html=True)
st.markdown(footer(), unsafe_allow_html=True)

"""
app.py — MyoAdapt home page (premium dark-mode UI).
"""
from __future__ import annotations

import streamlit as st

import myoadapt
from myoadapt.ui.theme import (
    COLORS,
    apply_theme,
    footer,
    hero,
    info_card,
    section_header,
    sidebar_brand,
    spacer,
)

st.set_page_config(
    page_title="MyoAdapt — sEMG Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme(st)
sidebar_brand(st, version=myoadapt.__version__)

# --- HERO ---
st.markdown(hero(
    eyebrow="Open-source neuromuscular intelligence",
    title="Decode muscle intent. On any CPU.",
    subtitle=(
        "MyoAdapt is a research-grade sEMG platform that goes beyond "
        "classification — supporting pose regression, zero-shot cross-user "
        "evaluation, domain adaptation, semantic retrieval, and synthetic "
        "data augmentation. All on commodity CPUs."
    ),
), unsafe_allow_html=True)

# --- KPI STRIP ---
kpi_cards = """
<div class="ma-kpi-grid">
    <div class="ma-kpi-card">
        <div class="accent-bar" style="background: {primary}"></div>
        <div class="label">Modules</div>
        <div class="value">11</div>
        <div class="delta">data · features · models · tasks · adaptation · evaluation · deployment · tracking · api · ui · cli</div>
    </div>
    <div class="ma-kpi-card">
        <div class="accent-bar" style="background: {accent}"></div>
        <div class="label">Datasets</div>
        <div class="value">8</div>
        <div class="delta">NinaPro DB1/2/3/7 · CapgMyo · UCI · Meta emg2pose/emg2qwerty</div>
    </div>
    <div class="ma-kpi-card">
        <div class="accent-bar" style="background: {accent2}"></div>
        <div class="label">Models</div>
        <div class="value">11</div>
        <div class="delta">classical + CNN1D + Lite-DAN + WaveFormer + task heads</div>
    </div>
    <div class="ma-kpi-card">
        <div class="accent-bar" style="background: {violet}"></div>
        <div class="label">Tests</div>
        <div class="value">344</div>
        <div class="delta">passing, full .[all] install</div>
    </div>
    <div class="ma-kpi-card">
        <div class="accent-bar" style="background: {success}"></div>
        <div class="label">License</div>
        <div class="value">Apache 2.0</div>
        <div class="delta">permissive, commercial use OK</div>
    </div>
</div>
""".format(
    primary=COLORS["primary"], accent=COLORS["accent"],
    accent2=COLORS["accent2"], violet=COLORS["violet"],
    success=COLORS["success"],
)
st.markdown(kpi_cards, unsafe_allow_html=True)

# --- 4-PILLAR ARCHITECTURE ---
st.markdown(section_header("01", "Four-pillar architecture"), unsafe_allow_html=True)
st.markdown(
    "MyoAdapt is the first open-source sEMG platform to unify four "
    "complementary paradigms in a single, coherent codebase."
)

pillars = """
<div class="ma-arch">
    <div class="ma-pillar p1">
        <div class="num">01</div>
        <h3>Classification</h3>
        <p>Multi-class gesture recognition via XGBoost, LDA, SVM, CNN-1D, Lite-DAN, WaveFormer</p>
    </div>
    <div class="ma-pillar p2">
        <div class="num">02</div>
        <h3>Regression</h3>
        <p>Continuous hand-pose (21-DOF) and kinematics decoding via MLP + Transformer</p>
    </div>
    <div class="ma-pillar p3">
        <div class="num">03</div>
        <h3>Zero-Shot</h3>
        <p>Large-scale cross-user generalization without adaptation — 600+ subject benchmark</p>
    </div>
    <div class="ma-pillar p4">
        <div class="num">04</div>
        <h3>Domain Adaptation</h3>
        <p>CORAL, TCA, SA, EA, adversarial GRL + statistical rigor suite</p>
    </div>
</div>
"""
st.markdown(pillars, unsafe_allow_html=True)

# --- WHAT'S NEW ---
st.markdown(spacer(), unsafe_allow_html=True)
st.markdown(section_header("02", "What's new in v2.0"), unsafe_allow_html=True)

st.markdown(
    "MyoAdapt v2.0 adds **8 new modules** (4,600+ lines of code) that "
    "close the gap with state-of-the-art research and introduce signature "
    "features found in no other open-source sEMG platform."
)

new_features = """
<div class="ma-bento">
    <div class="ma-tile">
        <h3>📊 Meta Dataset Loaders</h3>
        <p>emg2pose + emg2qwerty loaders — train and compare on Meta Reality Labs data alongside NinaPro.</p>
        <span class="badge ma-badge-blue">myoadapt decode</span>
    </div>
    <div class="ma-tile">
        <h3>🖐️ Pose Regression</h3>
        <p>21-DOF hand-pose decoding via MLP + Transformer. Goes beyond gesture IDs to full kinematics.</p>
        <span class="badge ma-badge-blue">myoadapt decode</span>
    </div>
    <div class="ma-tile">
        <h3>😮‍💨 Fatigue Tracking</h3>
        <p>MNF/MDF slope regression → composite fatigue index, level classification, plain-language narrative.</p>
        <span class="badge ma-badge-blue">myoadapt fatigue</span>
    </div>
    <div class="ma-tile">
        <h3>✋ Intent Detection</h3>
        <p>Rest-vs-active gatekeeper — energy-threshold or learned classifier, independent of gesture identity.</p>
        <span class="badge ma-badge-blue">myoadapt intent</span>
    </div>
    <div class="ma-tile">
        <h3>📉 Drift Detection</h3>
        <p>Feature / prediction / performance drift with MMD, chi-square, and Page-Hinkley change-point detection.</p>
        <span class="badge ma-badge-blue">myoadapt drift</span>
    </div>
    <div class="ma-tile">
        <h3>🎛️ Hyperparameter Tuning</h3>
        <p>Optuna TPE search over window length, features, and model hyperparameters.</p>
        <span class="badge ma-badge-blue">myoadapt tune</span>
    </div>
    <div class="ma-tile">
        <h3>👥 Demographic Fairness</h3>
        <p>Audit accuracy across age, dominant hand, or any demographic column — alongside per-subject fairness.</p>
        <span class="badge ma-badge-blue">myoadapt audit --demographics-csv</span>
    </div>
    <div class="ma-tile">
        <h3>⚡ WaveFormer</h3>
        <p>Lightweight Transformer with INT8 quantization — &lt;7ms inference on CPU, ~100K parameters.</p>
        <span class="badge ma-badge-blue">myoadapt train --model waveformer</span>
    </div>
    <div class="ma-tile">
        <h3>🎯 Zero-Shot Evaluator</h3>
        <p>Train on one large pool, test zero-shot on disjoint unseen users — protocol compatible with 600+ subject databases like EMG-EPN612.</p>
        <span class="badge ma-badge-blue">myoadapt zeroshot</span>
    </div>
    <div class="ma-tile">
        <h3>🔄 Electrode-Shift Robustness</h3>
        <p>Simulate armband rotation, channel dropout, noise injection — measure real-world reliability.</p>
        <span class="badge ma-badge-blue">myoadapt robustness</span>
    </div>
    <div class="ma-tile">
        <h3>🔍 EMG-Semantic Retrieval</h3>
        <p>Cross-modal EMG↔text retrieval — describe a gesture in English, find matching EMG windows.</p>
        <span class="badge ma-badge-blue">myoadapt retrieve</span>
    </div>
    <div class="ma-tile">
        <h3>🎭 Synthetic Augmentation</h3>
        <p>GAN + Diffusion-based EMG generation — expand small datasets, balance underrepresented classes.</p>
        <span class="badge ma-badge-blue">myoadapt augment</span>
    </div>
</div>
"""
st.markdown(new_features, unsafe_allow_html=True)
st.caption("Every capability above is runnable from the command line — see `myoadapt --help`.")

# --- RESEARCH PAPER CHAIN ---
st.markdown(spacer(), unsafe_allow_html=True)
st.markdown(section_header("03", "Research paper chain"), unsafe_allow_html=True)
st.markdown(
    "MyoAdapt is the software artifact for a 5-paper research chain on "
    "cross-subject sEMG pattern recognition. Each paper contributes a module."
)

papers = [
    ("P1", "Rest-Class Metric Inflation in Zero-Calibration sEMG", "LOSO benchmark across DB2/DB3/DB7 (73 subjects)", "LOSO benchmark · Rest inflation metric · 38-44pp TD advantage", "evaluation.metrics, evaluation.loso", "done"),
    ("P2", "Why MiniROCKET PPV Features Resist Domain Adaptation", "Mechanistic study · CORAL worsens near-singular covariance", "Eigenvalue diagnostic · CORAL failure mode · MiniRocketVerifier", "features.minirocket, adaptation.coral", "done"),
    ("P3", "Subject-Invariant EMG via 52K-Parameter Lite Adversarial Network", "CPU-only zero-calibration myocontrol", "Lite-DAN · gradual λ · GRL ablation · 5-CPU benchmark", "models.lite_dan, deployment.hardware_bench", "progress"),
    ("P4", "When Does Domain Adaptation Help Cross-Subject sEMG?", "Meta-analytic framework across feature spaces & populations", "LODO meta-analysis · Friedman/Nemenyi · effect sizes", "evaluation.statistics, evaluation.diagrams", "progress"),
    ("P5", "EMG-FM: A Foundation Model for Surface Electromyography", "Self-supervised pretraining on NinaPro DB1-7", "Transformer encoder · masked-patch SSL · fine-tuning hooks", "models.emg_foundation", "planned"),
]
_STATUS_LABEL = {"done": "Completed", "progress": "In progress", "planned": "Planned"}

paper_html = '<div class="ma-paper-chain">'
for tag, title, venue, contrib, module, status in papers:
    card = (
        '<div class="ma-paper-card">'
        f'<div class="meta">{tag} · {venue}'
        f'<span class="status ma-status-{status}">{_STATUS_LABEL[status]}</span></div>'
        f'<h4>{title}</h4>'
        f'<div class="contrib">{contrib}</div>'
        f'<div class="module">{module}</div>'
        '</div>'
    )
    paper_html += card
paper_html += '</div>'
st.markdown(paper_html, unsafe_allow_html=True)
st.caption(
    "Software platform status tracks independently of paper status — the "
    "modules a paper *contributes* (right column) may already be built and "
    "tested even while that paper's own results are still in progress."
)

# --- REGULATORY POSTURE ---
st.markdown(spacer(), unsafe_allow_html=True)
st.markdown(section_header("04", "Regulatory & compliance posture"), unsafe_allow_html=True)

r1, r2, r3 = st.columns(3)
with r1:
    st.markdown(info_card(
        "EU AI Act (Article 13)",
        "Per-prediction SHAP reports · trust scores · audit logs.",
        "research-grade",
        "blue",
    ), unsafe_allow_html=True)
with r2:
    st.markdown(info_card(
        "FDA SaMD pathway",
        "ONNX + SHA-256 · frozen artefact · 5-CPU latency budget.",
        "not filed",
        "amber",
    ), unsafe_allow_html=True)
with r3:
    st.markdown(info_card(
        "GDPR data governance",
        "No raw EMG leaves training env · feature-vector inference.",
        "data minimization",
        "teal",
    ), unsafe_allow_html=True)

# --- GET STARTED ---
st.markdown(spacer(), unsafe_allow_html=True)
st.markdown(section_header("05", "Get started in 60 seconds"), unsafe_allow_html=True)

st.markdown("**1. Install:**")
st.code("pip install -e '.[all]'", language="bash")
st.markdown("**2. Verify:**")
st.code("myoadapt info", language="bash")
st.markdown("**3. Train:**")
st.code("myoadapt train --db DB2 --model xgboost --output ./models/model.pkl", language="bash")
st.markdown("**4. Evaluate + Export + Explain:**")
st.code("""myoadapt eval --model-path ./models/model.pkl --db DB2 --protocol loso
myoadapt export --model-path ./models/model.pkl --output ./models/model.onnx
myoadapt audit --model-path ./models/model.pkl --db DB2
myoadapt ui --port 8501""", language="bash")
st.markdown("**5. Signal-level tools** *(new this pass — fatigue, intent gating, drift monitoring, tuning):*")
st.code("""myoadapt fatigue --db DB2 --subject 0 --channel 0
myoadapt intent --db DB2 --method learned
myoadapt drift --db DB2
myoadapt tune --db DB2 --n-trials 20""", language="bash")

# --- FOOTER ---
st.markdown(spacer(), unsafe_allow_html=True)
st.markdown(footer(), unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Pick a page to run a live demo. Each page exercises the real "
    "MyoAdapt pipeline on tiny synthetic data — fast and reproducible."
)

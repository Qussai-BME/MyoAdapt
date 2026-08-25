# MyoAdapt v2.0 — SOTA Gap Analysis (2024–2026)

**Date:** 2026-08-16
**Scope:** Identify what MyoAdapt v2.0 is missing versus the best open-source
sEMG / biosignal / ML platforms as of 2024–2026, so we can close the gaps.
**Method:** Web research against (1) LibEMG v2.0.1, (2) EMGBench (NeurIPS 2024),
(3) Meta `emg2pose` / `emg2qwerty` (NeurIPS 2024), (4) mainstream MLOps stacks
(MLflow / W&B / Optuna / DVC / BentoML / evidently), (5) sEMG research frontier
(force, fatigue, intent, prosthesis sim, multimodal), (6) community / docs
norms (mkdocstrings, Jupyter, leaderboards, academic templates).

**Correction (see below):** this document claimed to be "verified against the
actual codebase" but was not — Gaps 6 and 8 below both describe modules that
already existed in full (`evaluation/hyperopt.py`, `evaluation/fatigue.py`),
just unwired from the CLI/UI and untested. Treat every other gap in this
document as **unverified** until it's individually re-checked against the
code; do not cite this document's "no false-positive gaps" claim anywhere,
including grant material.

Where MyoAdapt already has partial coverage, the gap is described as
the *delta* (e.g. "MNF/MDF features exist in `features/frequency_domain.py` but
no high-level fatigue-index module exists").

---

## Executive summary — TOP 15 gaps ranked by impact

| #  | Gap | Competitor | Effort | Priority | Who cares |
|----|-----|-----------|-------|----------|-----------|
| 1  | Live device streaming + hardware streamers (Myo / Delsys / OT Bioquadrant) | LibEMG v2.0.1 | L | must-have | clinicians, investors |
| 2  | EMGBench standardized 9-dataset OOD benchmark (inter-subject + TSTS adaptation splits) | EMGBench (NeurIPS 2024) | M | must-have | researchers |
| 3  | Online evaluation tools — Screen-Guided Training + Iso-Fitts' law test | LibEMG v2.0.1 | M | must-have | clinicians, researchers |
| 4  | Production drift detection (evidently / NannyML) | every MLOps stack | M | must-have | investors, clinicians |
| 5  | Intent detection / no-movement rejection (rest vs active) | ReactEMG 2025, MyoGestic (Sci. Adv. 2025) | M | must-have | clinicians |
| 6  | Hyperparameter optimization (Optuna / Ray Tune) | every ML platform | S-M | must-have | researchers |
| 7  | W&B-style interactive experiment dashboard | MLflow / W&B / Aim | M | must-have | researchers |
| 8  | Fatigue tracking module (MNF/MDF slope regression + Instantaneous MF Band) | Miaoulis 2025, Karthick 2016 | S-M | must-have | clinicians, sports scientists |
| 9  | ASR-style sequence modeling for `emg2qwerty` (CTC + beam search + LM fusion → <10% CER) | Meta emg2qwerty (NeurIPS 2024) | L | must-have | researchers |
| 10 | Model registry + serving (MLflow Model Registry + BentoML / Seldon) | MLflow / BentoML / Seldon | M | nice-to-have | investors, engineers |
| 11 | Prosthesis simulation environment (Unity + MuJoCo physics) | LibEMG v2.0.1 (Unity examples) | L | nice-to-have | clinicians, researchers |
| 12 | Multi-modal fusion (EMG + IMU + force) | Purohit 2026, Santos 2026 | M | nice-to-have | researchers |
| 13 | Auto-generated API docs (mkdocstrings + MkDocs Material) | modern scientific Python | S | must-have | community |
| 14 | Interactive tutorials with real data + public leaderboard | Kaggle / EMGBench | M | nice-to-have | community, researchers |
| 15 | CI/CD for models + academic templates (LaTeX paper / poster / grant text) | GitHub Actions + CML | M | nice-to-have | researchers, community |

**Honorable mentions** (real gaps, lower rank): active / online learning during
deployment, transfer-learning registry / model zoo (HuggingFace-style), A/B
testing framework, movement-onset detection, video walkthroughs, Discord /
community forum, Ref-EMGBench normalization methods, per-user adaptation with
~30 min of data, public NinaPro/CapgMyo/HySer/FlexWear-HD leaderboard.

---

## Detailed gap cards

### Gap 1 — Live device streaming + hardware streamers
**What:** Out-of-the-box adapters that stream raw sEMG from the Myo armband,
Delsys Trigno, and OT Bioquadrant over shared memory / LSL, with a unified
`OnlineDataHandler` middleware so any classifier can consume a live feed.
**Why it matters:** Without hardware integration the platform cannot be
deployed on a real wearable — the central value proposition of sEMG.
LibEMG ships this and is therefore the default choice for any lab building a
real-time controller. **Competitor:** LibEMG v2.0.1 (`emg_toolbox.OnlineDataHandler`
+ device streamers; "hardware-agnostic, multiple devices supported by
default"). **MyoAdapt delta:** `deployment/realtime.py` already implements a
`RealtimeInference` engine with circular buffer + callbacks — but it consumes
an in-memory `np.ndarray`, not a live device feed. No device adapters, no
shared-memory / LSL middleware. **Effort:** L (one adapter per device, ~2–4
days each, plus a shared streaming bus). **Priority:** must-have.

### Gap 2 — EMGBench standardized 9-dataset OOD benchmark
**What:** A single entry point that loads NinaPro DB1–DB7, CapgMyo, Myo, UCI,
HySer, FlexWear-HD, plus EMGBench's new HD-EMG wearable dataset, and runs two
standardized splits: (a) **inter-subject** generalization (leave-one-subject-
out), (b) **train-test splits for time-series (TSTS)** adaptation — including
adaptation by fine-tuning on an initial subset of the held-out subject's data.
**Why it matters:** EMGBench is the NeurIPS 2024 community standard; any paper
that doesn't report these splits is increasingly un-citable. MyoAdapt's
`evaluation/loso.py` + `evaluation/lodo.py` + `evaluation/electrode_shift.py`
cover related but non-standardized protocols. **Competitor:** EMGBench
(NeurIPS 2024, Yang et al., CMU; `github.com/jehanyang/emgbench`). **Effort:**
M (dataset loaders largely exist; need the standardized split definitions + a
benchmark runner). **Priority:** must-have.

### Gap 3 — Online evaluation tools (Screen-Guided Training + Iso-Fitts' law)
**What:** A user-in-the-loop training UI that prompts gestures on screen and a
Fitts' law throughput test (ISO 9241-9) that measures real-world usability —
bits/s, completion time, error rate, path efficiency — during live control.
**Why it matters:** Offline accuracy is a poor predictor of real-time
usability; LibEMG ships both as the *defining* feature that distinguishes it
from offline-only libraries. Without this, MyoAdapt cannot claim a deployment
story. **Competitor:** LibEMG v2.0.1 (`Screen Guided Training` +
`Online Evaluation (Iso Fitts)` modules + example apps Snake / Mouse /
Hololens 2 / Unity). **MyoAdapt delta:** `ui/` is a Streamlit model-management
app — Train / Evaluate / Compare / Deploy / Explain — but no live-gesture
prompter and no Fitts' law test. **Effort:** M. **Priority:** must-have.

### Gap 4 — Production drift detection
**What:** Automatic monitoring of input-feature distribution drift and
prediction-probability drift in a deployed model, with alerting when a
re-calibration threshold is crossed (KS / PSI / Page-Hinkley on features,
ECE / Brier degradation on predictions).
**Why it matters:** Concept drift from electrode shift, sweat, fatigue, and
muscle adaptation is the single biggest cause of EMG system failure in the
field — EMGBench explicitly motivates its TSTS task by "non-stationary signals
that historically have made generalization difficult." MyoAdapt already
computes ECE / Brier at evaluation time but does not run them continuously in
deployment. **Competitor:** evidently, NannyML, Alibi Detect (mainstream
MLOps). **MyoAdapt delta:** `docs/deployment.md` line 84 lists
"Monitoring: prediction log + drift detection" as an unchecked TODO. **Effort:**
M. **Priority:** must-have.

### Gap 5 — Intent detection / no-movement rejection (rest vs active)
**What:** A dedicated module that classifies each window as `rest` vs `active`
*before* gesture classification, suppressing spurious outputs during rest, and
detecting movement onset/offset. Should ship a trained rest-state model and a
configurable decision rule (energy threshold + ML classifier).
**Why it matters:** Without rest-state rejection, prostheses and HCI devices
fire gestures whenever the user's arm is relaxed — a top user-complaint in
real-world deployment. ReactEMG (2025) and MyoGestic (Science Advances, 2025)
establish this as a first-class output, not an afterthought. **Competitor:**
ReactEMG (`reactemg.github.io`), MyoGestic (`sciadv.ads9150`, Sîmpetru et al.
2025), Trigili et al. 2019. **MyoAdapt delta:** `RealtimeInference` has a
`confidence_threshold` heuristic but no dedicated rest-state detector and no
onset/offset segmentation. **Effort:** M. **Priority:** must-have.

### Gap 6 — Hyperparameter optimization (Optuna / Ray Tune) — ❌ FALSE, ALREADY EXISTS
**Correction:** `myoadapt.evaluation.hyperopt.HyperparameterOptimizer` is a
real, 516-line Optuna-based implementation (`_suggest_param`, TPE search).
The actual gap is narrower than originally claimed: it is exported from
`evaluation/__init__.py` but has **no CLI command** (`myoadapt tune` does not
exist), **no test coverage**, and no multi-objective / pruning support yet.
**Effort to close the real gap:** S (CLI wiring + tests: ~half a day;
multi-objective/pruning: ~2–3 days on top). **Priority:** must-have.

### Gap 7 — W&B-style interactive experiment dashboard
**What:** A polished comparison view with parallel-coordinate plots, scatter
matrix of metrics, run-diff, model-comparison tables, and one-click
reproduction of any past run. Should integrate with the existing
`MLflowTracker` adapter.
**Why it matters:** The `tracking` module's MLflow adapter is a thin wrapper
(`set_experiment`, `start_run`, `log_params`, `log_metrics`) — it lacks the
dashboard polish that researchers now expect from W&B / Aim / MLflow UI.
**Competitor:** Weights & Biases, Aim, MLflow UI. **Effort:** M (expose the
MLflow UI behind a `myoadapt ui` command + add comparison plots to the
Streamlit `3_Compare.py` page). **Priority:** must-have.

### Gap 8 — Fatigue tracking module — ❌ FALSE, ALREADY EXISTS
**Correction:** `myoadapt.evaluation.fatigue.FatigueTracker` already ships a
full MNF/MDF/RMS slope regression, composite fatigue index, level
classification, and narrative generator (fixed a real 1-D shape bug and
wired it into the CLI as `myoadapt fatigue` — see CHANGELOG [Unreleased]).
**Remaining real gap:** no per-channel timeline *plot* (only the numeric
series), no alarm thresholds, and no 2025 Instantaneous Medium Frequency
Band (IMFB) variant yet. **Competitor citations (still useful):** Miaoulis
et al. 2025 (IMFB, MDPI Electronics), Karthick & Ghosh 2016 (Biomed Signal
Process Ctrl), Nair 2024 (UCC thesis). **Effort for the real remainder:**
S (plot + thresholds: ~1 day; IMFB variant: ~2 days). **Priority:** nice-to-have
now that the core module works — the must-have part is done.

### Gap 9 — ASR-style sequence modeling for `emg2qwerty`
**What:** A connectionist temporal classification (CTC) baseline that takes
raw 32-channel wrist sEMG and emits a keystroke sequence, plus optional beam
search and language-model fusion (4-gram or neural LM) to push character error
rate below 10% — Meta's threshold for a usable text model. Should reproduce the
`emg2qwerty` paper baselines so MyoAdapt users can compete on the benchmark.
**Why it matters:** MyoAdapt already loads `emg2qwerty` via `meta_loaders.py`
but ships no sequence-to-sequence model — so the platform cannot participate
in the typing benchmark that Meta and the community now treat as the canonical
sEMG sequence task. **Competitor:** Meta emg2qwerty (NeurIPS 2024,
`github.com/facebookresearch/emg2qwerty`); CER <10% reported with LM fusion.
**Effort:** L (CTC training loop + beam-search decoder + LM integration;
~2–3 weeks). **Priority:** must-have.

### Gap 10 — Model registry + serving
**What:** A `myoadapt.registry` API on top of MLflow Model Registry
(stage transitions: staging → production → archived, with model-card metadata)
and a `myoadapt.serve` command that packages any trained model as a REST /
WebSocket endpoint via BentoML or Seldon Core, with health-check + latency
SLOs.
**Why it matters:** MyoAdapt already exports ONNX (`deployment/onnx_export.py`)
and benchmarks hardware (`deployment/hardware_bench.py`), but there is no
versioned registry or serving runtime — so "deploy to production" is a manual
step. Investors and clinical-engineering partners expect a serving story.
**Competitor:** MLflow Model Registry, BentoML, Seldon Core, TorchServe.
**Effort:** M. **Priority:** nice-to-have (but moves to must-have if pursuing
clinical-grade deployment).

### Gap 11 — Prosthesis simulation environment (Unity + MuJoCo)
**What:** A closed-loop physics simulation where a MyoAdapt-trained controller
drives a virtual prosthetic hand (MPL or similar) in Unity with MuJoCo physics,
enabling grasp / Fitts' / box-and-blocks tests without hardware.
**Why it matters:** Required for prosthesis-control research and for any
clinical-trial-as-a-simulation story. LibEMG ships four Unity/Mixed-Reality
examples out of the box. **Competitor:** LibEMG v2.0.1 (Snake, Mouse, Hololens 2,
Unity examples); the myoelectric-simulator literature (VR + MuJoCo + eye
tracking, ResearchGate 2018). **Effort:** L. **Priority:** nice-to-have.

### Gap 12 — Multi-modal fusion (EMG + IMU + force)
**What:** A `myoadapt.tasks.multimodal` module with fusion architectures
(early / late / cross-attention) that consume synchronized EMG + IMU + force
streams; should ship a loader for NinaPro DB2 (which contains all three).
**Why it matters:** 2025–2026 sEMG research is converging on multimodal fusion
— Purohit et al. 2026 (dual-output LSTM for fatigue during neonatal CPR),
Santos et al. 2026 (validated wearable EMG+IMU device). MyoAdapt is currently
EMG-only. **Competitor:** Purohit 2026, Santos 2026, WearableSystems.org
publications. **Effort:** M. **Priority:** nice-to-have.

### Gap 13 — Auto-generated API docs (mkdocstrings + MkDocs Material)
**What:** Replace the hand-written `docs/api_reference.md` with
`mkdocstrings`-generated reference pages built from the docstrings in
`myoadapt/`, served via the existing `mkdocs.yml` with the Material theme, with
cross-references, signature tables, and source links.
**Why it matters:** `docs/api_reference.md` is already hand-maintained and will
diverge from the codebase as v2.0 ships its 11 + 8 new modules. Every modern
scientific Python package (scikit-learn, polars, pydantic) auto-generates docs;
hand-written references are a maintenance liability and a community-adoption
friction. **Competitor:** mkdocstrings + MkDocs Material (de facto standard).
**MyoAdapt delta:** `mkdocs.yml` already exists; only the auto-generation
handler is missing. **Effort:** S (~1 day). **Priority:** must-have.

### Gap 14 — Interactive tutorials with real data + public leaderboard
**What:** (a) A set of executable Jupyter notebooks (`.ipynb`, not `.py`)
walking through NinaPro DB2 LOSO, emg2pose regression, and a deployment demo
with a downloadable real-data sample bundled in the package; (b) a public
leaderboard page (auto-updated by CI) reporting the EMGBench inter-subject and
TSTS scores of every shipped model.
**Why it matters:** Notebooks are the #1 adoption driver for scientific Python
packages; `notebooks/` currently ships `.py` scripts only. A public leaderboard
is what made EMGBench / Kaggle-style benchmarks influential — it creates
community pressure to improve. **Competitor:** EMGBench leaderboard, Kaggle
EMG gesture competitions, LibEMG example notebooks. **Effort:** M. **Priority:**
nice-to-have.

### Gap 15 — CI/CD for models + academic templates
**What:** (a) A GitHub Actions workflow using CML (Continuous Machine
Learning) that runs the model suite on every PR, posts a metrics diff comment,
and *blocks merges* when accuracy / fairness / latency regress beyond a
threshold (an "evaluation gate"); (b) an `academics/` folder with LaTeX
templates (NeurIPS / IEEE / Nature Methods paper skeleton, conference poster
beamer file, and a 1-page grant-application boilerplate referencing MyoAdapt).
**Why it matters:** Evaluation gates prevent silent model regressions and are
now expected of any production ML repo. Academic templates reduce adoption
friction for the principal buyer of an open-source research platform —
graduate students and PIs writing papers. **Competitor:** GitHub Actions +
CML for the gate; HuggingFace and AllenAI repos ship paper templates.
**Effort:** M. **Priority:** nice-to-have.

---

## Honorable mentions (lower-priority real gaps)

- **Active / online learning during deployment** — acquisition functions that
  flag low-confidence windows for human labeling and incrementally fine-tune
  the deployed model. Alibi Detect + modAL are the reference stacks.
- **Transfer-learning registry / model zoo** — a HuggingFace-style hub of
  pretrained EMG encoders (the planned EMG-FM Paper 5 weights, plus
  per-dataset fine-tuned heads). Currently the foundation model ships as code
  only, no pretrained weights.
- **A/B testing framework** — online comparison of two deployed controllers
  with statistical significance (sequential testing, always-valid p-values).
- **Movement-onset detection** — separate from rest-state rejection (Gap 5),
  this is the temporal segmentation of `rest → transition → steady → release`.
- **Video walkthroughs** — short screen-recorded demos of the Streamlit UI
  and the deployment flow.
- **Discord / community forum** — a synchronous community channel beyond
  GitHub Issues.
- **Ref-EMGBench normalization methods** — five statistical normalization
  techniques benchmarked by Jin et al. for mitigating inter-subject
  variability; MyoAdapt's `adaptation/ea.py` (Euclidean Alignment) covers
  one of them.
- **Per-user adaptation with ~30 min of data** — Meta reports a "jump" in
  emg2qwerty performance when personalizing with half an hour of typing;
  MyoAdapt's adaptation module targets the cross-subject setting, not this
  few-shot personalization regime.
- **Public NinaPro / CapgMyo / HySer / FlexWear-HD leaderboard** — broader
  than Gap 14; a community-maintained results table across all major sEMG
  datasets.

---

## Recommended next actions (ranked)

1. **Gap 13 first** (1 day, S effort) — auto-generated API docs is the
   cheapest win and unlocks community contribution. Wire `mkdocstrings` into
   the existing `mkdocs.yml`.
2. **Gap 6** (1–3 days, S-M) — Optuna integration into a new
   `myoadapt.tune` module; immediately useful to existing users.
3. **Gap 8** (3–5 days, S-M) — fatigue-index module building on the
   already-shipped MNF/MDF features; opens a new user segment (clinicians,
   sports scientists).
4. **Gap 2** (1–2 weeks, M) — EMGBench split definitions + benchmark runner;
   required for v2.1 to remain citable.
5. **Gap 4 + Gap 5** (1–2 weeks each, M) — drift detection and intent
   detection together make the deployment story credible.
6. **Gap 1 + Gap 3** (3–4 weeks, L) — device streamers and online Fitts'
   evaluation are the heaviest lift but the biggest strategic win; they are
   what distinguishes LibEMG and what investors will ask about first.
7. **Gap 7** (1 week, M) — dashboard polish on top of the existing MLflow
   adapter.
8. **Gap 9** (2–3 weeks, L) — ASR-style emg2qwerty baseline; needed only if
   MyoAdapt wants to own the typing benchmark, not if it stays a gesture /
   pose platform.

---

## Sources (verified 2026-08-16)

- LibEMG v2.0.1 documentation — `libemg.github.io/libemg` (introduction, data
  handler, screen-guided training, online examples Snake/Mouse/Hololens2/Unity).
- Delsys — "LibEMG: An open-source Python toolbox for myoelectric control"
  (Sep 2023) and Delsys API live-data-sharing docs (Oct 2023).
- EMGBench — Yang, Soh, Lieu, Weber, Erickson (NeurIPS 2024 D&B Track),
  `arxiv.org/abs/2410.23625`, `emgbench.github.io`, `github.com/jehanyang/emgbench`.
- emg2pose — Salter et al. (NeurIPS 2024), `arxiv.org/abs/2412.02725`,
  `github.com/facebookresearch/emg2pose`.
- emg2qwerty — `arxiv.org/abs/2410.20081`, `github.com/facebookresearch/emg2qwerty`,
  Meta AI blog (Dec 5 2024).
- ReactEMG — `reactemg.github.io`, "Stable, Low-Latency Intent Detection from sEMG".
- MyoGestic — Sîmpetru et al., Science Advances 2025 (`sciadv.ads9150`).
- Fatigue — Miaoulis et al. 2025 (MDPI Electronics 14:2097, IMFB),
  Karthick & Ghosh 2016 (Biomed Signal Process Ctrl).
- Multimodal — Purohit et al. 2026 (dual-output LSTM EMG-IMU),
  Santos et al. 2026 (validated wearable EMG+IMU device).
- MLOps — MLflow Model Registry docs, BentoML docs, Seldon Core docs,
  evidently AI docs, CML (Continuous Machine Learning) docs.
- mkdocstrings + MkDocs Material documentation.
- Verified against the actual MyoAdapt v2.0 codebase at
  `/home/z/my-project/workspace/myoadapt/` (179 tests; modules data, features,
  models, adaptation, evaluation, deployment, tracking, api, ui, cli,
  reproducibility + Phase A/B/C modules meta_loaders, pose_regression,
  waveformer, zero_shot, demographic_fairness, electrode_shift,
  semantic_retrieval, synthetic_augmentation).

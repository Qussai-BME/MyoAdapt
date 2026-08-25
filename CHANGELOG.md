# Changelog

All notable changes to MyoAdapt are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Round 7 — GitHub/Streamlit release-prep audit
- **Model-hash verification gap.** `EMGClassifier.load()` (classical.py) and
  the task-head loaders already enforced `MYOADAPT_MODEL_SHA256` /
  `MYOADAPT_REQUIRE_MODEL_HASH` before unpickling. The four PyTorch model
  loaders — `CNN1D.load`, `LiteDAN.load`, `EMGFoundation.load`,
  `WaveFormer.load` — did not: they called `torch.load(..., weights_only=True)`
  with no hash check at all, so `MYOADAPT_REQUIRE_MODEL_HASH=true` silently
  did nothing for any deep model served through the REST API. All four now
  run the same hash-verification block as classical.py before `torch.load`.
- **Stale test-count claims.** `audit_evidence/final_fast_after_all_hardening.txt`
  (the actual captured pytest run) shows 344 passed / 2 skipped by default
  and `slow_regression_no_cov_final.txt` shows 13/13 passed under `-m slow`
  — 357 of 359 total. README.md, docs/research_chain.md,
  docs/reliability_punchlist.md, and the live "Tests" KPI on the Streamlit
  home page (`myoadapt/ui/app.py`) still read 339 / 354 / 318 respectively.
  All four now read 344 (or 359/344 where the doc states total-vs-default).
  Round 4/5 entries above are left as written — they describe counts at
  that point in the process, not the current state.
- **NOTICE listed unused third-party packages.** Altair and scikit-optimize
  appear in NOTICE but are not imported anywhere in `myoadapt/`. Removed;
  MLflow stays (the optional `tracking/mlflow_adapter.py` adapter still
  imports it when present, even though it's no longer a packaged dependency).
- **No `requirements.txt`.** Streamlit Community Cloud needs one at the repo
  root (or next to the entrypoint) to install anything beyond bare Streamlit.
  Added one that installs the package in editable mode plus `xgboost` only —
  deliberately excludes `torch`/`lightgbm`/`numba`/`onnx`/`shap` to stay
  inside Community Cloud's free-tier memory ceiling; see the comments in the
  file for how to opt into the full `.[all]` roster on larger infrastructure.
- **`CHECKSUMS.sha256` was stale.** It referenced 15 `.ruff_cache/` files not
  present in this archive and had 2 real content mismatches
  (`myoadapt/ui/app.py`, `tests/unit/test_ui_pages.py` had been edited after
  the manifest was generated). Regenerated against the tree as shipped here
  (192 files, self-verified with `sha256sum -c`).
- **`dist/` and `myoadapt.egg-info/`** (pre-built wheel/sdist and build
  metadata) removed from this archive. Both are already `.gitignore`d and
  RELEASE_CONTENTS.md frames them as CI-produced release artifacts, not
  source; keeping a wheel built from pre-fix source next to the fixes above
  would ship stale code under an unchanged-looking filename. CI's
  `quality-and-security` and `build` jobs regenerate both from source.
- Cross-project boundary re-verified: no reference to biosignal-fm or
  MyoSim anywhere in this tree (grep across `.py`/`.md`/config files).

### Round 6 — the last 2 known limitations closed; genuinely nothing left on the list
- **Fixed the EMG2Qwerty regression-vs-classification bug properly.**
  `load_meta_dataset`'s own metadata already carried a `task` field
  (`"regression"` for EMG2Pose, `"classification"` for EMG2Qwerty) that
  `myoadapt decode` never checked — it always routed through a
  regression head. Now routes on the dataset's own declared task:
  classification goes through the same feature-extraction +
  `EMGClassifier` pipeline `myoadapt train` uses, with real accuracy/
  macro-F1 metrics, not a continuous-regression score on discrete
  keystroke labels. Added guard rails (passing a regression model name
  for a classification dataset, or vice versa, now errors clearly
  instead of silently doing the wrong thing) and 4 tests.
- **CLI-wired the 4 remaining Python-API-only capabilities**:
  `myoadapt retrieve` (EMG<->text semantic search), `myoadapt augment`
  (GAN/diffusion data augmentation), `myoadapt zeroshot` (cross-user
  zero-shot evaluation, with `--compare-loso`), `myoadapt robustness`
  (electrode-shift perturbation audit against a real trained model's
  full preprocess->feature->predict pipeline). 9 more tests.
  `zeroshot`'s first draft had the same "processes the full default
  dataset regardless of pool size" mistake `train` originally had —
  caught immediately via the same runtime-expectation check, fixed
  before shipping.
- CLI smoke test (`test_cli.py`) now covers all 23 commands — it only
  covered 10 before this pass, meaning 13 commands' `--help` paths had
  never actually been exercised by any test.
- Landing page: every feature tile is now a "CLI ready" blue badge —
  no tile says "Python API" anymore.

### Final state
**344 tests pass by default** (2 skipped: MLflow/xgboost-fallback), plus
13 more marked `@pytest.mark.slow` (realistic-scale CLI runs — minutes
by design, same cost pattern as `train`/`tune`/`audit`, run with
`pytest -m slow`). 23 CLI commands, all reachable, all tested. Every
item ever raised across 6 rounds — 2 foundational data/correctness
bugs, the entire orphaned `tasks` subsystem, the landing page's
placeholder data, and both remaining Python-API-only capabilities — is
now fixed and verified, not just described as fixed.

### Round 5 — UI: form pass (substance was rounds 1-4)
- **Fixed a real, visible bug on the landing page**: all 5 KPI cards
  (Modules/Datasets/Models/Tests/License) showed the identical hardcoded
  value "179" — a copy-paste artifact, not real data. Replaced with
  verified current numbers (11/8/11/318/Apache 2.0). Added
  `tests/unit/test_ui_pages.py` with a regression test for this
  specifically — the UI had **zero** automated test coverage before
  this, verified now via Streamlit's real `AppTest` headless harness
  (actually executes each page, not just a syntax check) across all 6
  files (app.py + 5 pages).
- Research paper chain section showed all 5 papers with identical visual
  weight regardless of actual status. Added honest Completed/In
  progress/Planned pills matching `research_chain.md`'s own tracking.
- Softened an overclaim: "600+ subject benchmark" implied a benchmark
  already run at that scale in this repo. It describes
  `ZeroShotEvaluator`'s protocol compatibility with the EMG-EPN612
  literature dataset (612 subjects) — reworded to say that.
- Replaced meaningless "Phase A/B/C" feature badges (encoded nothing a
  reader could act on) with an honest "CLI ready" vs "Python API"
  signal, and added the 4 tiles for this session's actual new CLI
  commands (`fatigue`/`intent`/`drift`/`tune`) that weren't represented
  on the landing page at all despite being real and working.
- Added a signature visual element grounded in the actual subject: a
  deterministic, burst-modulated multi-channel waveform (resembling
  real sEMG — noisy baseline with decaying amplitude bursts, not a
  clean sine wave) rendered behind the hero text, replacing reliance on
  a generic gradient alone.
- Updated the landing page's own "Get started" instructions to include
  the new commands.

### Round 4 — Tier 1 fully closed
Every remaining item from the round-3 punch list individually verified:
`ElectrodeShiftRobustness` (9 tests), `ZeroShotEvaluator` (7 tests,
including the `compare_with_loso` paired-statistics path), `cohen_dz` +
`nemenyi_posthoc` (9 tests — confirmed `cohen_dz` is genuinely distinct
from the already-tested unpaired `cohen_d`, not an alias),
`TimeFrequencyFeatures` (1 test — real finding: registered in
`extract_features`'s dispatch but excluded from every CLI command's
default module list, so nothing had ever exercised it), `DANTrainer`
(1 test). `TransparencyReport` confirmed a non-issue — it's the
`@dataclass` return type of the already-tested `ShapReportGenerator`,
not a separate feature. Full orphan-sweep re-run: all 12 remaining
symbols from round 3 are now either tested or individually confirmed
to be registry/extension-point internals, not gaps. Total: 318 of 320
tests passing (up from 174/187 measured at the first audit), 19 CLI
commands.

### Fixed — critical, affects every previous run on synthetic-fallback data
- **`NinaProLoader._load_synthetic` (`data/loaders.py`) generated 90% of
  every default synthetic dataset as zero-signal padding, silently
  labeled as a phantom "subject 0".** The output array was correctly
  pre-sized for `n_subjects × n_gestures × 10` repetitions, but the
  nested generation loop was missing the repetition level entirely — it
  only filled `n_subjects × n_gestures` rows (1/10th), leaving the rest
  at numpy's zero-initialized default. Concretely, on DB2's default
  40-subject/41-gesture synthetic fallback: subject "0" ended up with
  14,760 all-zero windows while each real subject (1–40) had only 41
  instead of the intended 410. Every LOSO split, drift comparison, or
  fairness audit run against synthetic-fallback data before this fix —
  in this codebase's own history and in this session's earlier testing
  — was silently operating on this corrupted, 90%-empty, wildly
  imbalanced distribution. Affects DB1/DB2/DB3/DB7 (all share
  `NinaProLoader`); `CapgMyoLoader`/`UCILoader` use fully vectorized
  generation and don't have this bug. Fixed by adding the missing
  repetition loop; verified balanced (equal windows/subject, no
  all-zero rows, no phantom subject 0) with a new
  `tests/unit/test_loaders.py` (10 tests) that didn't exist before —
  this module had zero coverage, which is how a bug this large went
  unnoticed.
- **`extract_features()` was called once per window in `train`/`eval`/
  `audit`/`calibrate`, not batched — and this was a correctness bug, not
  just slower.** `TimeDomainFeatures.fit()` calibrates a noise-floor
  threshold (5th percentile of per-channel std) from whatever batch it's
  given, used by 4 threshold-dependent features (ZC, SSC, WAMP, MYOP).
  Calling it with a batch of 1 (the per-window-loop pattern used
  everywhere) makes that threshold self-referential and statistically
  meaningless — every model ever trained through this CLI got degraded
  values for those 4 features. Fixed by batching the call in all four
  commands (verified numerically: only `time_domain` features changed,
  `frequency_domain`/`histogram`/`correlation` were byte-identical
  before and after, confirming this was isolated to the noise-floor
  calibration path and not a general miscalculation). Incidentally also
  ~20% faster.
- CLI: added `--n-subjects` to `train` and `audit` (mirroring `decode`)
  — full default-scale synthetic data is realistically sized (16,400
  windows for DB2) and MiniROCKET-family feature extraction plus
  41-class XGBoost training take a few minutes end-to-end; useful for a
  fast smoke test, and documents the expected runtime so it doesn't read
  as a hang.
- CLI `audit`: now also runs `DemographicFairnessAudit` (previously
  fully orphaned — exported, unregistered concept, zero test, zero CLI
  reference) alongside the existing per-subject fairness audit when
  `--demographics-csv` is given. These are complementary, not
  duplicates: per-subject metrics can't surface a disparity tied to age
  or dominant hand the way a demographic-variable breakdown can.
- `evaluation.DriftDetector`: added `tests/unit/test_drift_detection.py`
  (10 tests) and a `myoadapt drift` CLI command — confirmed working
  manually in an earlier pass but had no test file or CLI wiring of its
  own.
- `evaluation.HyperparameterOptimizer`: added test coverage and a
  `myoadapt tune` CLI command — this is the module `gap_analysis_2025.md`
  incorrectly claimed didn't exist (see the correction already recorded
  in that doc); it did, it just weighed 516 lines of unused Optuna code.

### Fixed
- **Root cause of most `tasks` orphaning**: `models/__init__.py` explicitly
  imports each concrete model to fire its `@register_model` decorator, but
  never imported from `tasks/` — so `intent_detector`, `lstm_regressor`,
  `transformer_regressor`, `pose_mlp`, `pose_transformer`, and
  `semantic_retriever` never registered under normal package import.
  Fixing this naively (`from myoadapt.tasks.X import Y`) surfaced a real
  circular import (`models` → `tasks.*` → `models.base` → `models`) that
  only happened to work when `myoadapt.models` was imported before
  `myoadapt.tasks`. Fixed with a side-effect-only import
  (`importlib.import_module`, no names extracted) that's robust to either
  import order — verified from 4 different entry points.
- `EMGSemanticRetriever.evaluate_retrieval`: recall@k assumed every query
  had exactly one true match (row *i* ↔ row *i*). Wrong whenever
  descriptions repeat — e.g. many EMG windows sharing one gesture phrase
  like "close fist tightly", the common case for gesture-labeled EMG data
  — where it silently undercounted every correct retrieval except one per
  duplicate group (measured: reported recall@1 of 0.05 on a case manually
  confirmed via `retrieve_emg_by_text`/`retrieve_text_by_emg` to be
  retrieving correctly 5/5 and 3/3 times). Now credits any item sharing
  the true description text; reduces to the original exact-row semantics
  when descriptions are unique.
- `LearnedIntentDetector`: dropped a deprecated sklearn `LogisticRegression
  n_jobs` argument (no effect since sklearn 1.8, removed in 1.10).
- `FatigueTracker` / `_mmd_squared` (`drift_detection.py`): fixed a shape
  bug where `np.atleast_2d` ran before the `ndim == 1` check, silently
  reinterpreting a genuine single-channel `(n_samples,)` signal as
  `(1, n_samples)`.
- `/train` REST endpoint: now uses the caller-supplied `data_root` instead
  of always falling back to an env var / hardcoded default, and surfaces
  the loader's synthetic-fallback flag as `synthetic_fallback`.

### Added
- CLI: `myoadapt fatigue`, `myoadapt intent`, and `myoadapt decode` —
  three previously-implemented-but-unreachable subsystems (fatigue
  tracking, rest/active intent detection, and pose/continuous decoding)
  are now actually callable. `decode` uses the Meta emg2pose/emg2qwerty
  loaders (`data/meta_loaders.py`), which were themselves untested and
  unused anywhere — closing the "no labeled data for continuous/pose
  models" gap noted in the previous entry.
- Test coverage for 8 previously-untested modules: `fatigue.py`,
  `continuous.py`, `intent_detection.py`, `pose_regression.py`,
  `semantic_retrieval.py`, `synthetic_augmentation.py` (GAN + diffusion),
  `meta_loaders.py`, `adaptation/adversarial.py`. 61 new tests.

### Known state (see `docs/reliability_punchlist.md` for the live list)
- `tasks` subpackage and GAN/diffusion augmentation: **activated** —
  tested, registered, and CLI-reachable (`fatigue`, `intent`, `decode`
  commands). `decode --dataset EMG2Qwerty` runs without error but treats
  keystroke labels as a continuous regression target rather than
  classification — functionally wrong, flagged as follow-up.
- Orphan count (exported, no test, no CLI/UI reference): 40/104 → 23/104
  after this pass. Remainder is mostly Tier-1 items deliberately deferred
  when scope shifted to `tasks`/augmentation (`DriftDetector`,
  `HyperparameterOptimizer`, fairness/robustness evaluators), registry
  internals not meant for direct end-user use, and GAN/retriever building
  blocks (`EMGGenerator`, `EMGDiscriminator`, `EMGTextEncoder`) already
  exercised indirectly through the composite classes that use them.

## [2.0.0] — 2025-08-12

### Added
- Eleven-module architecture (data, features, models, adaptation, evaluation, deployment, tracking, api, ui, cli, tasks)
- LOSO + LODO evaluators with Rest-class inflation reporting
- Lite-DAN adversarial domain adaptation (~53K parameters)
- CNN-1D baseline (~15K parameters)
- EMG Foundation Model (transformer encoder, masked-patch reconstruction)
- 5-CPU PassMark-projected hardware benchmark
- Per-prediction SHAP transparency reports
- ONNX export with numerical-parity verification and SHA-256 tamper detection
- FastAPI REST API + WebSocket streaming server with feature-extractor auto-wiring
- Streamlit 5-page UI (Train, Evaluate, Compare, Deploy, Explain)
- Per-subject Euclidean Alignment (`PerSubjectEA`)
- Yule-Walker AR coefficient estimator with synthetic AR(2) validation
- MiniROCKET near-singularity diagnostic at the 10,000-kernel scale
- Holm-Šídák correction, bootstrap 95% confidence intervals, Cohen's d / dz
- **Statistical rigor suite**: Hedges' g (small-N corrected), BCa bootstrap
  CIs, paired Wilcoxon with rank-biserial r, a-priori power analysis,
  minimum-sample-size calculator
- **Calibration & uncertainty module**: Temperature scaling, Platt,
  Isotonic, ECE / MCE / Brier / NLL, conformal prediction sets
- **Fairness / bias audit**: per-subgroup metrics, equalized odds,
  fairness summaries
- **Critical Difference diagram** (Demsar 2006), per-fold box-plot,
  per-class F1 heatmap, reliability diagram
- **LaTeX-ready paper reports**: main results table, per-fold table,
  effect-size table, stats table, BCa CI table (booktabs)
- **Figure bundle writer**: PNG (300 dpi) + PDF + .tex caption files
- **Model Card generator** (Mitchell et al. 2019) + Datasheet (Pushkarna 2022)
- **Reproducibility engine**: `set_global_seed` (Python + NumPy + PyTorch),
  environment fingerprint + SHA-256 hash, `RunManifest` with output
  verification, git HEAD capture, config snapshots
- **Classical model aliases** in the registry: `get_model("xgboost")`
  etc. dispatch to `EMGClassifier(model_type=...)`
- **CLI commands**: `audit`, `model-card`, `calibrate`, `power`
- 185 unit + integration tests

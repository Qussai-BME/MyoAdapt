# Reliability Punch List — final, 2026-08-19 (round 6)

**Status: nothing left. Every item raised across 6 rounds is fixed and verified.**

344 tests pass by default (2 skipped: MLflow needs a live server,
xgboost-fallback skips when xgboost is installed). 13 more marked
`@pytest.mark.slow` (realistic-scale CLI runs, minutes by design —
run with `pytest -m slow`). 23 CLI commands, all reachable, all tested.
68/68 modules import cleanly. Every UI page verified via Streamlit's
real AppTest harness. Zero known open items.

## This round: the last 2 limitations, closed
- [x] `myoadapt decode --dataset EMG2Qwerty` now correctly classifies
      (feature extraction + EMGClassifier, real accuracy/macro-F1)
      instead of forcing a regression head onto discrete keystroke
      labels. The loader's own `task` metadata field was always there;
      the CLI just never checked it. Guard rails added for
      model/dataset-task mismatches. 4 tests.
- [x] `myoadapt retrieve`, `myoadapt augment`, `myoadapt zeroshot`,
      `myoadapt robustness` — the 4 remaining Python-API-only
      capabilities are now CLI commands. 9 tests. `zeroshot` had the
      same "ignores the pool-size flags, processes the full dataset
      regardless" bug `train` originally had — caught and fixed before
      shipping this round, not left for a 7th round to find.
- [x] Landing page: every feature tile is CLI-ready (blue badge) — no
      "Python API only" tiles remain.
- [x] CLI smoke test now covers all 23 commands (was 10 — 13 commands'
      `--help` output had never been checked by any test).

## Full history (rounds 1-5) — see CHANGELOG for complete detail
Two foundational bugs: the synthetic-data generator silently zero-
padded 90% of every default dataset under a phantom "subject 0"
(affected every prior run on synthetic fallback data), and a noise-
floor calibration bug degraded 4 time-domain features in every model
ever trained through the CLI. The entire `tasks` subsystem (continuous
decoding, pose regression, intent detection, semantic retrieval) and
GAN/diffusion augmentation activated from a fully orphaned/dead state.
`ElectrodeShiftRobustness`, `ZeroShotEvaluator`, `DriftDetector`,
`HyperparameterOptimizer`, `DemographicFairnessAudit`, `cohen_dz`,
`nemenyi_posthoc` tested and wired in. Landing page's "179" KPI
placeholder bug (all 5 metrics showed the identical fake value) fixed;
paper-chain status made honest; a subject-grounded signature waveform
visual added; the whole UI given automated test coverage for the
first time.

## Registry/extension internals — not gaps, just not directly user-facing
`features.register_feature`/`list_features`, `models.BaseModel`/
`register_model`, `tracking.get_tracker`, `adaptation.ADAPTATION_REGISTRY`,
`data.DATASET_REGISTRY`, `data.EMGGenerator`/`EMGDiscriminator` (internal
to `EMGGANAugmenter`), `tasks.EMGTextEncoder` (internal to
`EMGSemanticRetriever`) — extension-point machinery and implementation
details, exercised indirectly by everything that calls into them.

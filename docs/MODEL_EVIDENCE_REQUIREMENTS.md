# Model Evidence Requirements

This document defines the minimum evidence needed before making a research claim about a MyoAdapt model. It does not define clinical acceptance criteria and must not be used to infer clinical safety or effectiveness.

## Core rule

A claim may not be broader than its evidence. State the exact model revision, artifact hash, data source and permission, participant/sample inclusion rules, sensors, channel configuration, electrode placement procedure, sampling rate, preprocessing, feature pipeline, labels, split protocol, seed, hardware, software environment, and metric definitions. A result on one corpus, task, session, or population does not establish the same result elsewhere.

## Required evidence bundle

| Item | Minimum content | Required for |
|---|---|---|
| Intended research use | Target task, user/population, context, operator, input/output, and explicit out-of-scope uses | Every model |
| Data record | Source/license/consent scope, collection protocol, participants/sessions, exclusions, labels, imbalance, missingness, sensor configuration | Every model |
| Version record | Source revision, dependency/environment fingerprint, config, seed, model artifact SHA-256, training timestamps | Every model |
| Split protocol | Predeclared train/validation/test or LOSO/LODO method; group/session/time leakage controls | Any performance claim |
| Metrics | Primary metric, per-class results, macro/weighted aggregation, uncertainty intervals, confusion matrix, failure cases | Any performance claim |
| Calibration | Independent calibration/evaluation partitions, ECE/Brier/NLL, prediction-set coverage where applicable | Any probability/confidence claim |
| Robustness | Predeclared perturbations: session, subject, electrode/channel, noise, amplitude, missing channel, sensor/device, time | Any robustness/zero-calibration claim |
| Fairness/subgroups | Scientifically and ethically justified subgroup definitions, sample counts, uncertainty, interpretation limits | Any subgroup/fairness claim |
| Operational evidence | Latency, memory, resource behavior, input rejection, failure response, deployment hardware/config | Any deployment claim |
| Model card | Evidence-linked, versioned card with intended use, limitations, data, metrics, ethical considerations, and contacts | Every released artifact |

## Evaluation protocol expectations

Use group-aware splits that prevent a participant, session, recording, or temporally adjacent samples from leaking across evaluation boundaries. Do not calibrate on a sample-order split and present the result as operational calibration. For cross-subject claims, use participant-held-out evaluation; for cross-session claims, hold sessions out; for transfer claims, test the stated target domain without leaking target labels.

Report not only mean accuracy but also class-wise outcomes, uncertainty, the worst relevant subgroup/fold, rejection/failure behavior, and known negative results. If synthetic data are used, label them prominently and do not present synthetic-only performance as evidence of real-world performance.

Recent sEMG research that claims calibration-free or transferable performance explicitly evaluates leave-one-subject-out and cross-dataset protocols and reports perturbation analysis.[1] Adopt comparable transparency about protocols rather than copying a headline metric.

## Claim language examples

| Evidence available | Acceptable wording | Unacceptable wording |
|---|---|---|
| Unit tests and local demo | “The code implements a LOSO evaluator.” | “The model generalizes across users.” |
| One documented dataset/protocol | “On the stated held-out protocol, version X achieved metric Y (interval Z).” | “MyoAdapt achieves Y% accuracy.” |
| Perturbation experiment | “Under the stated channel/noise perturbation, performance changed by X.” | “The system is robust to electrode shift.” |
| Confidence output | “The model returns a confidence-derived informational score.” | “The trust score makes decisions safe.” |
| Hash/manifest | “The artifact was checked against an independently supplied SHA-256.” | “The artifact is tamper-proof/signed.” |

## Reference

[1] [Yang et al., *Calibration-free sEMG intention recognition via self-supervised pretraining and adversarial domain alignment for upper-limb rehabilitation*, Scientific Reports (2025)](https://www.nature.com/articles/s41598-025-28302-0)

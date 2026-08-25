# MyoAdapt Pre-Release Audit Report

**Audit date:** 22 August 2026  
**Audited release:** MyoAdapt v2.0, hardened research distribution  
**Audit scope:** Source code, tests, package metadata, dependency declarations, REST and WebSocket interfaces, artifact loading, container configuration, public documentation, scientific-claim language, research-governance materials, and selected current external references.

> **Decision:** This distribution is suitable for a **controlled research-software release** once its GitHub Actions release gates pass for the exact tagged commit. It is **not suitable for representation as a clinical product, medical device, regulated AI system, privacy-compliant service, autonomous controller, or validated prosthetic-control system.**

The audit applied a claim-discipline principle: implementation of an evaluator, report generator, explanation method, checksum, or statistical procedure is not evidence that a particular model has the corresponding real-world property. Current sEMG work evaluates transfer through explicit held-out protocols and perturbation analysis rather than treating code availability as proof of generalization.[1] FDA Good Machine Learning Practice is likewise lifecycle-oriented, not a code-feature checklist.[2]

## 1. Verification summary

| Verification activity | Result | Evidence retained in release workspace |
|---|---:|---|
| Fast regression suite, excluding `slow` marker | **Passed** | `audit_evidence/final_fast_after_all_hardening.txt` |
| Fast-suite coverage | **66%** on Python 3.12.3 | `audit_evidence/final_fast_after_all_hardening.txt` |
| Slow regression suite, no coverage overhead | **Passed in 697 seconds** | `audit_evidence/slow_regression_no_cov_final.txt` |
| Targeted API, CLI, persistence, and safe-loading regressions | **Passed** | `audit_evidence/*regression*.txt` |
| Full import sweep | **68 imported; 0 failed** | `audit_evidence/final_import_sweep.txt` |
| Operational Ruff rules (`F821`, `F811`, `E9`) | **Passed** | `audit_evidence/ruff_operational_final.txt` |
| Bandit medium/high gate | **Passed; 0 medium, 0 high** | `audit_evidence/bandit_ci_gate_pass.txt` |
| Declared-dependency audit | **No known vulnerabilities found** | `audit_evidence/pip_audit_resolved.txt` |
| Strict documentation build | **Passed** | `audit_evidence/mkdocs_build_final.txt` |
| Wheel and source-distribution build | **Passed** | `audit_evidence/distribution_build_permission_retry.txt` |
| Twine package metadata validation | **Passed** | `audit_evidence/twine_check_permission_retry.txt` |
| Docker Compose syntax / image runtime | **Not locally verified** | Docker was unavailable in this audit environment; CI gate is included. |

The fast suite retained two intentional skips: the XGBoost fallback cannot be exercised while XGBoost is installed, and the MLflow adapter requires a running MLflow service. The tracking extra and Compose MLflow service were removed from this hardened package because the available dependency resolution conflicted with the required cryptography security floor and produced disclosed-vulnerability findings. Local JSON tracking remains available.

## 2. Remediation delivered

| Area | Finding | Remediation included in this release | Residual responsibility |
|---|---|---|---|
| REST API | Caller-provided API key was not consistently enforced; request validation could expose internal behavior around non-finite values. | Enforced configured API key, bounded shape/value validation, safe validation-error handling, constrained error responses, and explicit rejection of non-finite/ragged/oversized input. | Apply TLS, gateway body limits, rate limits, identity, logging policy, and monitoring externally. |
| WebSocket | Shared-resource and unbounded-message risk in streaming path. | Per-connection engine isolation, API-key support, message/sample/channel bounds, and constrained errors. | Enforce connection quotas and rate limits at the gateway. |
| Network defaults | CLI services bound to all interfaces by default. | REST, WebSocket, and UI CLI defaults now bind to `127.0.0.1`; public exposure requires an explicit host/gateway decision. | Operate behind authenticated TLS reverse proxy; do not publish raw application ports. |
| Artifact integrity | Classical and task models used unrestricted pickle; PyTorch loaders used `weights_only=False`. | Added optional expected SHA-256 and enforceable `MYOADAPT_REQUIRE_MODEL_HASH`; converted all discovered PyTorch loaders to `weights_only=True` and safe serializable state. | Treat pickle as trusted-only; obtain checksum through an independent protected release channel. |
| Dependency security | Vulnerability audit identified a cryptography floor and an incompatible vulnerable MLflow/pyarrow path. | Enforced `cryptography>=50.0.0`; removed incompatible MLflow dependency/service; post-change declared-dependency audit is clean. | Re-run dependency and container scans for every tag. |
| Container posture | Container and Compose examples were too permissive for a networked research service. | Non-root image, local port binding, read-only root filesystem, no-new-privileges, dropped capabilities, bounded `/tmp`, PID limits, required API-key environment, and read-only data/model mounts. | Build, scan, and run in CI; harden host and gateway. |
| Scientific claims | README and documents made stronger claims than delivered evidence established. | Added research-only boundary and evidence-oriented language; corrected integrity, reproducibility, calibration, and generalization statements. | Publish protocol-specific data, methods, results, uncertainty, and independent review before stronger claims. |
| Calibration | CLI used a first-60%/last-40% sample-order split without an explicit warning. | Default is now participant-held-out exploratory calibration; order split requires explicit acknowledgement and reports its leakage risk. | Validate calibration against a predeclared, independent target-context protocol. |
| Model cards | Default card implied an evaluation protocol without attaching results. | Default now produces a clearly incomplete research draft and demands a versioned held-out evaluation artifact. | Complete and review each released model card. |
| Test runtime | Slow tests were coupled to global coverage collection and the synthetic zero-shot smoke comparison exceeded practical aggregate time. | Coverage moved to fast CI only; slow CI excludes coverage. Added explicit, visibly labeled per-subject window cap for bounded smoke experiments only. | Run full protocol experiments separately; never use capped smoke outputs as evidence. |
| Governance | Documentation did not provide a release decision, privacy operations checklist, or residual-risk register. | Added release-readiness, risk-register, research-use, security-deployment, privacy, and model-evidence documents. | The deploying organization must implement and evidence the organizational controls. |

## 3. Scientific, safety, and governance assessment

MyoAdapt contains useful research utilities, including group-aware evaluators, robustness helpers, calibration metrics, statistical tests, model-card templates, explanation reports, and reproducibility records. Those features can support an evidence package, but they do not create one automatically. In particular, a calibration plot is not a validated safety control; a `trust_score` is not a clinical confidence measure; SHAP output is not a clinical explanation; and a local SHA-256 sidecar is not signed software provenance.

The current research literature illustrates why boundary discipline matters. A 2025 sEMG study that describes calibration-free transfer evaluates leave-one-subject-out and cross-dataset settings, along with robustness perturbations; its methods and evidence cannot be transferred automatically to a different code path, data collection, model, sensor, user group, or operating condition.[1] The same logic applies to all headline metrics in this repository. Only a versioned, target-context experimental evidence bundle can support a publication claim.

For any healthcare- or medical-purpose future use, the repository is materially incomplete. FDA GMLP frames AI/ML medical-device work as lifecycle principles requiring multidisciplinary expertise, good software engineering and security practices, representative data, independent testing, and monitoring.[2] European Commission healthcare-AI guidance highlights risk mitigation, high-quality data, clear information, and human oversight for relevant high-risk contexts.[3] The present distribution does not provide product-specific intended use, clinical validation, risk management, quality-system records, human-factors evidence, regulatory assessment, post-market controls, or organizational governance.

## 4. Privacy and data handling

sEMG recordings, subject codes, prediction histories, and associated metadata may be personal data when connected directly or indirectly to an individual. The applicable legal classification, lawful basis, safeguards, and data-subject obligations depend on the actual controller, purpose, linkage, and jurisdiction. The EDPB describes data-protection duties in terms of lawful, fair, transparent, purpose-limited, minimized, time-limited, and secure processing.[4]

This release therefore avoids a generic claim that all EMG is or is not a special category of data. Before collecting, hosting, or sharing real participant data, the operator must complete `PRIVACY_OPERATOR_CHECKLIST.md`, confirm data-license and consent scope, determine applicable privacy/ethics requirements with qualified personnel, and implement access, retention, deletion, incident, and transfer procedures.

## 5. Security posture and operational release conditions

The application-level controls are defense in depth, not a public perimeter. OWASP identifies broken authentication, unrestricted resource consumption, security misconfiguration, and unsafe API consumption among recurring API security risks.[5] Deploy the service behind a managed TLS-enabled gateway with identity-aware access where applicable, secret rotation, body/message limits, per-client rate limits, connection limits, protected logs, alerts, backups, and incident response.

Do not load an artifact from a client-provided path or an untrusted upload. The classical/scikit-learn persistence format uses pickle, which is intrinsically unsafe for untrusted files. This release can compare a model to `MYOADAPT_MODEL_SHA256`, and production configuration can require that value. That check is meaningful only if the expected hash arrives through a protected independent channel. It is not a signature, complete provenance scheme, authorization control, or malware scanner.

## 6. Conditions before publication

| Required action | Why it is required |
|---|---|
| Tag the exact audited revision and require the new CI workflow to pass. | Local verification cannot replace a protected remote CI record for the released commit. |
| Publish wheel, source archive, checksums, generated SBOM, and release notes from CI. | Supports artifact identification and supply-chain review; SLSA describes provenance controls as progressive practice.[6] |
| Run Docker build, Compose validation, non-root inspection, and image scanning in CI. | Docker is unavailable in the local audit environment. |
| Preserve research-only notices in README, API UI, hosted UI, marketing, and model cards. | Prevents unsafe reliance and unsupported product claims. |
| Use a TLS gateway and set `MYOADAPT_REQUIRE_API_KEY=true`, `MYOADAPT_API_KEY`, `MYOADAPT_REQUIRE_MODEL_HASH=true`, `MYOADAPT_MODEL_SHA256`, and explicit CORS origins for any network service. | The package has bounded application controls but does not replace network identity and abuse controls. |
| Attach a model-evidence bundle before publishing accuracy, robustness, fairness, calibration, or generalization claims. | Prevents a code capability being mistaken for a demonstrated result. |
| Obtain qualified privacy, ethics, clinical, legal, and regulatory review before any use with real participants, health data, safety-critical integration, or medical purpose. | These organizational and product-specific obligations lie outside the source repository. |

## 7. Known limitations retained deliberately

The repository retains broad optional capability and therefore retains technical debt. The audited operational Ruff rules pass, while broader style debt remains and should be reduced incrementally under ordinary code review. The fast-suite coverage is 66%; important lower-covered areas include the WebSocket server, API model-loading paths, UI, ONNX export, several CLI commands, and optional external integrations. The CI pipeline now makes the meaningful gates executable, but actual GitHub execution, container inspection, SBOM publication, code-signing/provenance, external penetration testing, and cross-version Python matrix results remain publisher actions.

The slow test suite passes without coverage in approximately 11.6 minutes on this audit environment. That duration is acceptable for a separate bounded CI job but is intentionally not imposed on every local rapid-feedback loop. The zero-shot smoke cap exists solely to make an explicit demo/test workload bounded; it changes the protocol and must not be used to report a complete scientific result.

## 8. Release artifact contents

The final archive includes source, tests, project metadata, hardened Docker/Compose files, `.env.example`, CI workflow, built wheel and sdist, the full English documentation set, this audit report, and selected verification logs. It deliberately excludes raw datasets, participant data, secrets, generated model artifacts, external credentials, Python caches, local coverage databases, and the prior audit workspace.

## References

[1] [Yang et al., “Calibration-free sEMG intention recognition via self-supervised pretraining and adversarial domain alignment for upper-limb rehabilitation,” *Scientific Reports* (2025)](https://www.nature.com/articles/s41598-025-28302-0)

[2] [U.S. Food and Drug Administration, “Good Machine Learning Practice for Medical Device Development: Guiding Principles”](https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles)

[3] [European Commission, “Artificial intelligence in healthcare”](https://health.ec.europa.eu/ehealth-digital-health-and-care/artificial-intelligence-healthcare_en)

[4] [European Data Protection Board, “Data protection basics”](https://www.edpb.europa.eu/sme/learn-the-basics/data-protection-basics_en)

[5] [OWASP, “API Security Top 10 – 2023”](https://owasp.org/API-Security/editions/2023/en/0x00-header/)

[6] [SLSA, “Supply-chain Levels for Software Artifacts”](https://slsa.dev/)

# MyoAdapt Release Readiness

**Release scope:** MyoAdapt v2.0 is distributed as **research-only software** for sEMG signal-processing, model-development, evaluation, and controlled deployment experiments. It is **not** validated for clinical diagnosis, treatment, patient management, or autonomous prosthetic control. This document is an engineering release assessment, not medical, legal, regulatory, or security certification.

> **Release decision.** The source package is suitable for a controlled research release after the mandatory technical gates below pass in the publisher's own clean CI environment. It is **not ready to be represented as a clinical product, a safety-critical controller, a regulated medical device, or a GDPR/FDA/EU-AI-Act-compliant system.**

## Release gates

| Gate | Required evidence | Status in this package | Publisher action |
|---|---|---:|---|
| Research-use boundary | Prominent limitation notice in public documentation and deployed API/UI | Implemented | Preserve unchanged in public distribution and any hosted interface. |
| Source integrity | Immutable Git tag, reviewed changes, release archive checksum, and independent artifact record | Template only | Create a signed Git tag and GitHub Release; publish checksums from CI. |
| Automated verification | Clean install, unit/integration tests, packaging, import sweep, lint/type baseline, and security audit in CI | Workflow expanded in this release; must run remotely | Require green CI for the exact tagged commit. |
| Dependency security | Vulnerability scan with no accepted unfixed production findings | Security floor added; scan must run in CI | Re-run `pip-audit` at tag time and record any accepted risk. |
| Container hardening | Non-root image, explicit API key, local port binding, read-only filesystem, limited capabilities, TLS gateway | Hardened examples included | Validate image and Compose in a container-capable CI environment before deployment. |
| Model-artifact trust | Model origin, intended use, SHA-256 from an independent channel, and evaluation manifest | Code supports optional enforcement | Set `MYOADAPT_REQUIRE_MODEL_HASH=true` and `MYOADAPT_MODEL_SHA256` for any network deployment. |
| Scientific claims | Versioned protocol and results on representative data/population with held-out evaluation | Not shipped | Do not claim accuracy, robustness, zero-calibration performance, patient benefit, or clinical readiness. |
| Privacy governance | Controller-specific lawful basis, notice, retention, access control, processing record, and security review | Not shipped | Complete before collecting or hosting personal data. |
| Clinical/regulatory use | Intended-use definition, risk management, clinical evidence, quality system, usability, cybersecurity, and regulatory assessment | Not shipped | Obtain qualified clinical/regulatory review before any such use. |

## Deployment boundary

The built-in API is deliberately bounded but is not a perimeter security service. A public deployment must terminate TLS at a managed gateway and apply request-size limits, network rate limits, monitoring, log redaction, backup/incident procedures, and access management. The Compose example binds services to loopback by default; do not expose the underlying ports directly to the Internet.

The API key in this package is a basic shared-secret control, not an identity, authorization, audit, rotation, or rate-limiting system. Use an identity-aware gateway where individual users, patient data, or multi-tenant access are in scope.

## Scientific and safety boundary

Recent sEMG research evaluates cross-subject and cross-dataset transfer under explicit protocols, including perturbations such as noise and channel changes. Implementing an evaluator does not reproduce those results or establish the same property for a different model, population, sensor, task, or deployment context.[1] Confidence, calibration, SHAP values, and the exposed `trust_score` are informational outputs; they are not validated safety controls.

FDA guidance describes Good Machine Learning Practice as lifecycle-oriented principles for safe, effective, high-quality AI/ML medical devices rather than a checklist satisfied by individual code features.[2] The European Commission similarly notes that medical-purpose AI may be high risk and requires risk mitigation, data quality, clear user information, and human oversight.[3] Accordingly, this repository must not be used to claim regulatory readiness or clinical validation.

## Minimum publisher checklist

Before publishing a release, the publisher should complete the following actions in the release repository and retain the resulting evidence. The evidence is intentionally operational: a statement that a feature exists is not a substitute for an executed, versioned result.

1. Run the documented release command set in a clean environment and attach its logs.
2. Create a semantic version tag and a GitHub Release containing the source archive, wheel, SHA-256 checksums, SBOM, and release notes.
3. Enable branch protection, code review, security advisories, dependency updates, and CI required checks.
4. Run all applicable slow tests on a declared CPU/GPU budget and publish any exclusions or failures.
5. Build and scan the container image in a container-capable CI job; verify the non-root runtime and Compose policy.
6. Publish a model-evidence bundle before making any performance, robustness, or deployment claim.
7. Review every reference to healthcare, prosthetics, EU AI Act, FDA, GMLP, GDPR, safety, accuracy, robustness, or compliance against the delivered evidence.

## References

[1] [Yang et al., *Calibration-free sEMG intention recognition via self-supervised pretraining and adversarial domain alignment for upper-limb rehabilitation*, Scientific Reports (2025)](https://www.nature.com/articles/s41598-025-28302-0)

[2] [U.S. Food and Drug Administration, *Good Machine Learning Practice for Medical Device Development: Guiding Principles*](https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles)

[3] [European Commission, *Artificial intelligence in healthcare*](https://health.ec.europa.eu/ehealth-digital-health-and-care/artificial-intelligence-healthcare_en)

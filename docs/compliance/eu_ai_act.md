# EU AI Act Context for MyoAdapt

## Status statement

MyoAdapt is **research-only software**. This repository does not determine whether a particular deployment is an AI system under Regulation (EU) 2024/1689, whether it is high risk, or whether any obligation applies. It does not provide conformity assessment, CE marking, registration, a quality-management system, a post-market monitoring system, or legal advice.

The European Commission notes that AI-based software intended for medical purposes can be high risk and highlights risk-mitigation systems, high-quality data, clear user information, and human oversight as relevant safeguards.[1] Whether those statements apply to a specific implementation depends on the intended purpose, deployment, integration, jurisdiction, and applicable sectoral law. Obtain qualified regulatory and legal assessment before any medical-purpose, clinical, or safety-critical use.

## What the repository provides

The codebase contains research utilities that may help create engineering evidence when correctly run and independently reviewed:

| Utility | What it can support | What it does **not** establish |
|---|---|---|
| LOSO/LODO and subgroup evaluation | Documented experimental results for a stated protocol | Generalization, accuracy, fairness, or robustness in a target deployment |
| SHAP reports and feature/channel summaries | Investigation of a particular model output | Human-understandable, clinically valid, or legally sufficient explanation |
| Confidence/trust-score display | An informational model-output signal | A safety mechanism, human oversight implementation, or acceptable risk |
| Local tracker/run manifest | Engineering traceability records | Complete logging, auditability, provenance, quality-system records, or compliance |
| ONNX sidecar hash | An integrity comparison when expected hash is independently trusted | Signing, tamper prevention, secure supply-chain provenance, or deployment authorization |

## Gaps that remain outside the repository

A medical-purpose or high-risk deployment may require, among other things, intended-use definition, risk management, data governance, technical documentation, record-keeping, transparency, human oversight, performance/cybersecurity evidence, quality management, change control, monitoring, incident handling, and sector-specific conformity activities. This source package does not provide those organizational and lifecycle controls.

Do not state that MyoAdapt “complies with,” “supports compliance with,” or “is ready for” the EU AI Act based solely on the tools above. A more accurate statement is that the project contains research utilities that may contribute evidence to a separate, deployment-specific assessment.

## Operator actions before any deployment involving personal or health-related data

1. Define intended purpose, users, outputs, foreseeable misuse, and out-of-scope uses.
2. Obtain a context-specific classification and applicability assessment from qualified counsel/regulatory specialists.
3. Create a risk register, data-governance plan, evaluation protocol, human-oversight procedure, cybersecurity plan, and controlled change process.
4. Produce evidence on representative data and operating conditions; do not generalize from implementation tests or synthetic data.
5. Keep the research-use notice visible in the UI, API, documentation, and release material.

## Reference

[1] [European Commission, *Artificial intelligence in healthcare*](https://health.ec.europa.eu/ehealth-digital-health-and-care/artificial-intelligence-healthcare_en)

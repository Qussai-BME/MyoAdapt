# GDPR Context and Data-Protection Boundary

## Status statement

MyoAdapt is research software and does not provide a GDPR compliance program. The organization that determines the purpose and means of processing must assess its own role, lawful basis, notices, processor relationships, security measures, retention, transfer rules, participant rights, and documentation. This document is informational only and is not legal advice.

EMG recordings, linked subject identifiers, prediction histories, and related metadata may be personal data when they relate to an identified or identifiable individual. Whether a data set also concerns health, biometric, or other special-category data depends on the purpose and technical processing. Do not use a universal repository statement to decide that question for a specific study or service.

The European Data Protection Board explains that personal data may identify a person directly or indirectly and that controllers must process personal data lawfully, fairly, transparently, for specified purposes, with data minimization, retention limits, and appropriate security.[1] If special-category data is involved, additional conditions can apply.[1]

## What the repository does and does not do

| Area | Repository support | Not delivered |
|---|---|---|
| Local data processing | Core library does not require a remote service for basic use | Lawful basis, transparency notice, consent management, or access control |
| Pseudonymized IDs | Utilities can operate on non-identifying labels | A re-identification policy, key separation, or proof that data are anonymous |
| Files and models | Outputs can be written locally | Encryption at rest, retention enforcement, backup policy, deletion workflow, or access audit trail |
| API/UI | Bounded requests and basic shared-secret option | Identity management, role-based authorization, consent, data-subject portal, or multi-tenant segregation |
| Evaluation artifacts | Technical run records may assist research review | Record of processing activities, DPIA, data-processing agreement, or transfer assessment |

## Operator checklist

Before processing personal or participant data, complete the [Privacy and Data-Governance Operator Checklist](../PRIVACY_OPERATOR_CHECKLIST.md). At a minimum, define the purpose, data flow, controller/processor roles, lawful basis, participant information, retention schedule, access controls, security safeguards, data sharing restrictions, and incident procedure.

Avoid raw EMG payloads, direct identifiers, API keys, or sensitive context in logs, issue reports, notebooks, and public model artifacts. Confirm the license, consent scope, and redistribution terms for every input corpus before storing it in a repository, container image, release archive, or hosted service.

## Research context

Scientific research may have context-specific provisions, but it does not remove the need to assess legal basis, safeguards, transparency, security, retention, and participant rights. Obtain institutional and privacy review appropriate to the location and study before collecting new data or providing services to individuals.

## Reference

[1] [European Data Protection Board, *Data protection basics*](https://www.edpb.europa.eu/sme/learn-the-basics/data-protection-basics_en)

# Privacy and Data-Governance Operator Checklist

This checklist is for the organization that determines why and how MyoAdapt processes data. It is not legal advice and does not make a deployment GDPR-compliant. EMG recordings, subject codes, prediction logs, and linked metadata may be personal data; the legal classification and required safeguards depend on the processing purpose, context, linkage, jurisdiction, and controller/processor roles.

The European Data Protection Board states that personal data includes information relating to an identified or identifiable person and that controllers must be able to demonstrate lawful, fair, transparent, purpose-limited, minimal, time-limited, and secure processing.[1] Review this checklist with qualified privacy, ethics, and information-security personnel before handling participant or patient data.

## Before collection or ingestion

| Check | Evidence to retain | Owner |
|---|---|---|
| Define the research purpose and fields needed | Protocol/data dictionary | Research lead |
| Identify controller, processors, hosting region, and data flows | Data-flow diagram and agreements | Data owner |
| Determine lawful basis and any special-category/health/biometric implications | Written privacy assessment | Privacy lead |
| Confirm participant notice, consent where used, withdrawal handling, and contact route | Approved notice/consent materials | Ethics/privacy lead |
| Confirm dataset license, consent scope, redistribution rights, and attribution | License/data-use record | Data steward |
| Decide whether a DPIA or local equivalent is required | Documented screening/assessment | Privacy lead |
| Prohibit direct identifiers in training folders unless strictly necessary | Ingestion policy | Data engineer |

## During processing

| Check | Minimum control |
|---|---|
| Data minimization | Load and retain only fields/channels needed for the stated protocol. |
| Pseudonymization | Use randomized study IDs; maintain the re-identification key separately with restricted access. |
| Access control | Least privilege, unique identities, MFA at the hosting layer, periodic access review. |
| Encryption | Use transport encryption and encryption at rest appropriate to the deployment; protect keys separately. |
| Logging | Avoid EMG payloads, direct identifiers, API keys, and raw exception dumps in operational logs. |
| Retention | Set written retention periods for raw signals, features, models, logs, backups, and reports. |
| Sharing | Review each export, notebook, artifact, and release bundle for personal data and data-license restrictions. |
| Processors/transfers | Document subprocessors and cross-border transfers; apply required safeguards. |

## Deletion, rights, and incidents

Define a tested method to locate data by study ID, correct or restrict data where applicable, export records if required, and delete active copies and backups on the documented schedule. Keep a decision log where a request cannot be fulfilled due to a lawful research exception or conflicting obligation.

Prepare an incident procedure that covers containment, preservation of evidence, internal escalation, legal/privacy review, notification assessment, remediation, and lessons learned. Do not rely on the open-source repository to deliver any of these organizational controls.

## Special note on research data

Research context does not remove the need to assess lawful basis, transparency, safeguards, retention, security, and participant rights. Do not write a generic statement that EMG is or is not special-category data for every project. Instead, document the actual purpose and technical processing, then obtain a context-specific determination from the responsible organization.

## Reference

[1] [European Data Protection Board, *Data protection basics*](https://www.edpb.europa.eu/sme/learn-the-basics/data-protection-basics_en)

# Security Policy

## Reporting a Vulnerability

We take security vulnerabilities in MyoAdapt seriously. If you believe you
have found a security issue, **please do not open a public GitHub issue**.

Instead, report it privately:

- **Email**: [adlbiqussai@gmail.com](mailto:adlbiqussai@gmail.com)
- **Subject line**: `[SECURITY] <short description>`

If possible, encrypt your report using our PGP public key (see below). Please
include:

1. A description of the issue and its potential impact.
2. Affected version(s) (`myoadapt info` output).
3. Steps to reproduce, including a minimal proof-of-concept.
4. Suggested mitigation or fix, if you have one.

## PGP

If you prefer end-to-end encrypted communication, fetch our public key from
the project maintainers and encrypt your report. (A PGP key fingerprint will
be published on the project website at https://github.com/Qussai-BME/MyoAdapt/.pgp when
available; until then, plain email is acceptable but please redact any
sensitive tokens from the report.)

## Response Timeline

| Step                                  | Target SLA         |
| ------------------------------------- | ------------------ |
| Acknowledge receipt of report         | 2 business days    |
| Initial assessment & severity rating  | 5 business days    |
| Fix or mitigation released            | 90 days from report |
| Public disclosure (coordinated)       | After fix release  |

We follow a **90-day coordinated disclosure window**: we will work with you
to ship a fix within 90 days of the first acknowledgment, then publish a
security advisory. If a fix is not feasible within 90 days, we will
coordinate an extension with you before disclosing.

## Supported Versions

MyoAdapt is currently in beta. Only the latest minor release receives
security fixes.

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| < 2.0   | :x:                |

## Scope

In scope:

- The `myoadapt` Python package and its CLI.
- The FastAPI REST API shipped at `myoadapt.api.rest`.
- The Streamlit UI shipped at `myoadapt.ui`.
- The Docker images built from this repository's `Dockerfile`.

Out of scope:

- Vulnerabilities in third-party dependencies — report these upstream.
- Theoretical attacks without a working proof-of-concept against MyoAdapt.
- Issues that require already-compromised access to the host (e.g. arbitrary
  code execution after an attacker has root).

## Acknowledgements

We credit security reporters in the release notes for the fix unless they
prefer to remain anonymous. Thank you for helping keep MyoAdapt safe.

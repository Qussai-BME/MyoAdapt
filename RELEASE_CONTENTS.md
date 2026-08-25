# Release Package Contents

This archive is a hardened **research-only** MyoAdapt source distribution. Read `docs/RELEASE_AUDIT_2026-08-22.md` first. The archive is not a clinical-validation package, a regulatory submission, or a substitute for deployment-specific privacy/security controls.

| Location | Contents | Review purpose |
|---|---|---|
| `README.md` | Project introduction and prominent research-use boundary | Understand what the software does and does not claim. |
| `docs/RELEASE_AUDIT_2026-08-22.md` | Audited scope, results, remediation, constraints, and release conditions | Primary pre-release decision record. |
| `docs/RELEASE_READINESS.md` | Release-gate matrix and publisher checklist | Determine remaining operational work before publication. |
| `docs/RISK_REGISTER.md` | Identified risks, controls, residual risk, and ownership | Plan review and operational mitigation. |
| `docs/SECURITY_DEPLOYMENT.md` | Deployment hardening and gateway guidance | Configure a controlled network deployment. |
| `docs/PRIVACY_OPERATOR_CHECKLIST.md` | Data-governance checklist | Complete before handling real participant data. |
| `docs/MODEL_EVIDENCE_REQUIREMENTS.md` | Evidence and claim requirements | Use before publishing metrics or model claims. |
| `docs/compliance/` | Contextual FDA, EU AI Act, and GDPR boundary documents | Do not mistake contextual guidance for compliance certification. |
| `.github/workflows/ci.yml` | Required release quality gates | Enable on the release repository and require green checks. |
| `Dockerfile`, `docker-compose.yml`, `.env.example` | Hardened local research-deployment example | Review and configure behind a TLS gateway. |
| `dist/` | Built wheel and source distribution | Optional package publication artifacts; validate checksums before use. |
| `audit_evidence/` | Selected local verification logs | Inspect the exact audit evidence summarized in the report. |
| `CHECKSUMS.sha256` | SHA-256 checksums generated for this archive's content set | Verify copied artifacts after download or transfer. |

## Local verification

Create a clean environment, install the package, and run the explicit checks below. These commands should be re-run by the publisher on the exact tagged revision and in remote CI.

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
pytest tests -m "not slow" --cov=myoadapt --cov-report=term-missing
pytest tests -m slow --no-cov
ruff check myoadapt --select F821,F811,E9
pip-audit
bandit -r myoadapt -ll -ii
python -m build
twine check dist/*
mkdocs build --strict
```

The container gate cannot be substituted by a local textual review. Build the image and validate Compose in CI, then inspect that the final image runs as the non-root `myoadapt` user and that external exposure occurs only through a separately hardened TLS-enabled gateway.

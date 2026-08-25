# Security Deployment Guide

This guide describes a minimum baseline for a **controlled research deployment**. It does not make a public deployment secure by itself and does not provide a regulated-device cybersecurity file.

## 1. Release the artifact safely

Build from a reviewed, immutable source revision. Create a semantic Git tag, build the source archive and wheel in CI, record SHA-256 checksums, generate an SBOM, and publish the artifacts through the project release channel. Use a dependency audit for every release. SLSA describes provenance and integrity controls as progressive supply-chain practices; a locally generated hash sidecar is not substitute provenance.[1]

Do not publish model artifacts until their training data permission, model card, evaluation evidence, source revision, and expected SHA-256 are recorded. Never accept model file paths or artifact uploads from API clients.

## 2. Configure mandatory secrets

Copy `.env.example` to `.env`, create a long random API secret, and keep the `.env` file outside source control. For any networked deployment, use all of the following settings:

```dotenv
MYOADAPT_REQUIRE_API_KEY=true
MYOADAPT_API_KEY=<long-random-secret>
MYOADAPT_REQUIRE_MODEL_HASH=true
MYOADAPT_MODEL_SHA256=<sha256-from-independent-release-record>
MYOADAPT_CORS_ORIGINS=https://research.example.org
```

An API key is a basic shared secret. It is not user authentication, authorization, audit logging, multi-tenancy, rate limiting, or secret rotation. Place the service behind an identity-aware gateway when more than one operator or any personal data is involved.

## 3. Use a gateway; do not expose application ports directly

Bind Compose ports to `127.0.0.1` as supplied and place a managed reverse proxy or API gateway in front of the service. The gateway should terminate TLS, redirect HTTP to HTTPS, enforce a request-body limit below the application limit, set connection and request timeouts, apply per-client rate limits, restrict trusted origins, and log security events without recording sensitive EMG payloads or API secrets.

OWASP's API Security Top 10 includes broken authentication, unrestricted resource consumption, security misconfiguration, and unsafe API consumption.[2] Application-level feature/batch/message limits are defense in depth; gateway controls are still required to address network-layer abuse.

## 4. Run the hardened Compose profile

```bash
cp .env.example .env
# Edit .env securely, then start only the REST API.
docker compose up -d api
# Optional local UI or tracking services are intentionally separate profiles.
docker compose --profile ui up -d
```

The supplied Compose configuration runs the main image as a non-root user, makes the root filesystem read-only, drops Linux capabilities, enables `no-new-privileges`, limits PIDs, uses a bounded writable `/tmp`, mounts raw data/models read-only, and binds published ports to loopback. Confirm that the host mounts and named volumes use appropriate ownership and backups.

## 5. Operational controls

| Control | Minimum research deployment action |
|---|---|
| Access | Restrict network paths; protect the gateway; rotate API secrets; revoke access promptly. |
| TLS | Use managed certificates; disable plaintext external access; review modern TLS policy. |
| Logging | Send operational logs to protected storage; redact EMG payloads, IDs, tokens, and model paths. |
| Monitoring | Alert on failed authentication, request-limit rejection, restart loops, latency increases, resource exhaustion, and checksum mismatch. |
| Patching | Rebuild on base-image and dependency advisories; re-run tests, `pip-audit`, and image scan. |
| Backups | Encrypt backup stores where required; test restoration; define retention and deletion procedures. |
| Incident response | Document owner, contact path, containment, notification, evidence preservation, and recovery steps. |
| Change control | Version source, config, model, data schema, dependencies, and expected artifact hash together. |

## 6. WebSocket protocol notes

The streaming endpoint requires a trusted local model path and creates an inference buffer per connection. It enforces message, sample-row, and channel limits and rejects non-finite/ragged payloads. Prefer the `X-API-Key` header instead of a query parameter because URLs are often logged. The endpoint remains unsuitable for unbounded public traffic without gateway-level connection quotas and rate limits.

## References

[1] [SLSA, *Supply-chain Levels for Software Artifacts*](https://slsa.dev/)

[2] [OWASP, *API Security Top 10 – 2023*](https://owasp.org/API-Security/editions/2023/en/0x00-header/)

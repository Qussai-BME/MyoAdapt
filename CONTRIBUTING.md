# Contributing to MyoAdapt

Thanks for your interest in improving MyoAdapt! This document describes how
to set up a development environment and the workflow for landing changes.

## Code of Conduct

Participation in this project is governed by the
[Contributor Covenant 2.1](CODE_OF_CONDUCT.md). Please be excellent to each
other.

## Development setup

```bash
git clone https://github.com/Qussai-BME/MyoAdapt.git
cd myoadapt
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[all]"
pre-commit install
```

The `.[all]` extra pulls in the ML, ONNX, explainability, tracking, and dev
dependencies so the full test suite can run on your machine.

## Workflow

1. **Fork** the repository and create a feature branch off `main`:
   ```bash
   git checkout -b feat/short-description
   ```
2. **Write code.** Keep commits focused and write clear messages
   (Conventional Commits style is appreciated but not required).
3. **Add tests.** New behaviour needs new tests under `tests/`. Bug fixes
   should include a regression test that fails before the fix and passes
   after.
4. **Run the test suite locally:**
   ```bash
   pytest tests/ --no-cov -q
   ```
   All tests must pass. CI runs the same suite on Python 3.10, 3.11, and
   3.12.
5. **Lint your changes:**
   ```bash
   ruff check myoadapt tests
   ```
6. **Open a Pull Request** against `main`. Fill in the PR template and link
   any related issues.

## Pull request checklist

- [ ] Tests pass locally (`pytest tests/ --no-cov -q`)
- [ ] `ruff check` is clean
- [ ] New code has tests
- [ ] Public API changes are documented in `docs/`
- [ ] `CHANGELOG.md` updated (under the `[Unreleased]` section)

## Branching & release model

- `main` — always shippable; protected, requires green CI + 1 review.
- `feat/*`, `fix/*`, `docs/*` — short-lived feature branches.
- Releases are tagged `vMAJOR.MINOR.PATCH` and documented in `CHANGELOG.md`.

## Reporting bugs

Open a [GitHub Issue](https://github.com/Qussai-BME/MyoAdapt/issues) with:

- MyoAdapt version (`myoadapt info`)
- Python version and OS
- Minimal reproduction snippet
- Expected vs. actual behaviour

Security vulnerabilities should **not** be reported via public issues — see
[SECURITY.md](SECURITY.md) for the disclosure process.

## Licensing

By contributing you agree that your contributions are licensed under the
Apache 2.0 license, as described in the project [LICENSE](LICENSE).

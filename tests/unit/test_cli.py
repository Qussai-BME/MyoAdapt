"""Smoke tests for the Typer CLI.

Verifies that every documented subcommand parses, prints its --help text,
and exits 0. We do NOT actually train/serve/stream — we only exercise
Typer's argument-parsing layer so that the test is fast and side-effect
free. If a subcommand's signature changes such that `--help` breaks, this
test will catch it.
"""
from __future__ import annotations

import pytest
from typer.testing import CliRunner

from myoadapt.cli.main import app


runner = CliRunner()

# Commands documented in `myoadapt.cli.main` that we want to smoke-test.
# Each entry is the subcommand name as the user would type it.
SUBCOMMANDS = [
    "train",
    "eval",
    "export",
    "serve",
    "stream",
    "benchmark",
    "explain",
    "ui",
    "info",
    "verify",
    "audit",
    "model-card",
    "calibrate",
    "power",
    "fatigue",
    "intent",
    "decode",
    "drift",
    "tune",
    "retrieve",
    "augment",
    "zeroshot",
    "robustness",
]


@pytest.mark.parametrize("cmd", SUBCOMMANDS)
def test_subcommand_help_exits_zero(cmd: str) -> None:
    """Every subcommand must accept --help and exit 0."""
    result = runner.invoke(app, [cmd, "--help"])
    assert result.exit_code == 0, (
        f"`myoadapt {cmd} --help` failed with exit code {result.exit_code}.\n"
        f"stdout:\n{result.stdout}\n"
        f"exception:\n{result.exception}"
    )
    # Help text should at least mention the command name or "Usage".
    assert ("Usage" in result.stdout) or (cmd in result.stdout), (
        f"`myoadapt {cmd} --help` did not print a recognizable help block.\n"
        f"stdout:\n{result.stdout}"
    )


def test_top_level_help() -> None:
    """The top-level CLI must print help listing the subcommands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    # All subcommands should be discoverable from the top-level help.
    for cmd in SUBCOMMANDS:
        assert cmd in result.stdout, (
            f"subcommand '{cmd}' not listed in top-level help.\n"
            f"stdout:\n{result.stdout}"
        )


def test_no_args_prints_help() -> None:
    """With no args at all, the CLI should print help (no_args_is_help=True).

    Typer's ``no_args_is_help`` behaviour is to print the help text and exit
    with code 2 (the standard "missing command" convention). Both the help
    text and the non-zero status are the documented contract.
    """
    result = runner.invoke(app, [])
    assert result.exit_code == 2
    assert "Usage" in result.stdout or "Commands" in result.stdout


def test_info_command_runs() -> None:
    """`myoadapt info` is safe to run end-to-end (no side effects) — verify
    it actually executes and prints the version banner."""
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0, (
        f"`myoadapt info` failed: {result.exception}"
    )
    assert "MyoAdapt" in result.stdout

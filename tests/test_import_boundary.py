"""The label-leakage boundary, checked rather than trusted.

A signal that has seen ground truth still returns a number, and that number still
correlates with ECE -- more strongly, in fact, which is the direction that makes the
paper wrong rather than merely broken. Nothing else in this repo fails when it happens.

This runs `import-linter` against the contracts in `pyproject.toml`. There is no CI
(`ROADMAP.md` 3), so the test suite is the only thing that sees every change.

The contract cannot see a label passed as an *argument*, since that involves no import.
That half is closed by `signals.base.Evidence`, which has no parameter for one.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_signals_cannot_reach_a_label():
    if shutil.which("lint-imports") is None:
        pytest.fail("import-linter is not installed; run `uv sync --extra dev`")

    result = subprocess.run(
        ["lint-imports"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

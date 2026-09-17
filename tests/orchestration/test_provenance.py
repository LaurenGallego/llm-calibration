import re
import subprocess
from datetime import UTC, datetime

import pytest

from madcal.orchestration import code_sha, new_run_id, provenance


def test_run_id_encodes_the_utc_start_time():
    run_id = new_run_id(datetime(2026, 8, 24, 14, 25, 30, tzinfo=UTC))
    assert re.fullmatch(r"20260824T142530Z-[0-9a-f]{6}", run_id)


def test_run_ids_started_in_the_same_second_differ():
    moment = datetime(2026, 8, 24, 14, 25, 30, tzinfo=UTC)
    assert len({new_run_id(moment) for _ in range(100)}) == 100


def test_code_sha_is_the_checked_out_commit():
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert code_sha() == expected


def test_code_sha_raises_outside_a_git_checkout(monkeypatch, tmp_path):
    monkeypatch.setattr(provenance.madcal, "__file__", str(tmp_path / "madcal" / "__init__.py"))
    with pytest.raises(RuntimeError, match="not inside a git checkout"):
        code_sha()

"""Run identifiers and code provenance."""

import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import madcal


def new_run_id(now: datetime | None = None) -> str:
    """Return a time-sortable, collision-safe run id such as `20260824T142530Z-3f9a1c`."""
    moment = now or datetime.now(UTC)
    return f"{moment.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:6]}"


def code_sha() -> str:
    """Return the git commit of the running code, raising outside a git checkout."""
    package_root = Path(madcal.__file__).resolve().parent
    try:
        result = subprocess.run(
            ["git", "-C", str(package_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"cannot resolve a commit SHA for the code at {package_root}: not inside a git "
            f"checkout; install madcal from a clone rather than a copied tree"
        ) from error
    return result.stdout.strip()

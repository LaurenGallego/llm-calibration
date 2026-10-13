"""Command line: fetch what a config needs, or run it on an idle GPU."""

import argparse
import fcntl
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from madcal.analysis import ANALYSIS_STATUS, build_report, to_arrow
from madcal.config import RunConfig, TransformersModelConfig
from madcal.orchestration import build_benchmark, run
from madcal.storage import ResultStore

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE = REPO_ROOT / ".cache"
GPU_LOCKS = CACHE / "gpu-locks"
BUSY_MEMORY_MIB = 1024
OFFLINE = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")


def idle_gpus(gpus: str, apps: str, busy_mib: int = BUSY_MEMORY_MIB) -> list[int]:
    """Return the indices of GPUs with no compute process and under `busy_mib` of memory in use."""
    busy = {line.strip() for line in apps.splitlines() if line.strip()}
    idle = []
    for line in gpus.splitlines():
        if not line.strip():
            continue
        index, uuid, used = (field.strip() for field in line.split(","))
        if uuid not in busy and int(used) < busy_mib:
            idle.append(int(index))
    return idle


def claim_gpu(locks: Path = GPU_LOCKS) -> str | None:
    """Pin this process to an idle GPU no other madcal process holds, unless one is already set."""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return os.environ["CUDA_VISIBLE_DEVICES"]
    if shutil.which("nvidia-smi") is None:
        return None
    gpus = _smi("--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits")
    apps = _smi("--query-compute-apps=gpu_uuid", "--format=csv,noheader")
    locks.mkdir(parents=True, exist_ok=True)
    for index in idle_gpus(gpus, apps):
        handle = os.open(locks / f"{index}.lock", os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(handle)
            continue
        os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
        return str(index)
    raise SystemExit(f"no idle GPU:\n{gpus}")


def _smi(*query: str) -> str:
    return subprocess.run(["nvidia-smi", *query], check=True, capture_output=True, text=True).stdout


def main(argv: Sequence[str] | None = None) -> None:
    """Fetch or run every config named on the command line, or report on a results root."""
    parser = argparse.ArgumentParser(prog="madcal")
    parser.add_argument("command", choices=("fetch", "run", "report"))
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)

    if args.command == "report":
        for root in args.paths:
            _report(root)
        return

    configs = [RunConfig.from_yaml(path) for path in args.paths]
    os.chdir(REPO_ROOT)
    os.environ["HF_HOME"] = str(CACHE / "huggingface")
    if args.command == "run":
        os.environ.update(dict.fromkeys(OFFLINE, "1"))
        print(f"gpu {claim_gpu()}", flush=True)

    for path, config in zip(args.paths, configs, strict=True):
        print(f"{args.command} {path}", flush=True)
        if args.command == "fetch":
            _fetch(config)
        else:
            _run(config)


def _fetch(config: RunConfig) -> None:
    from huggingface_hub import snapshot_download

    if isinstance(config.model, TransformersModelConfig):
        snapshot_download(
            config.model.model_id,
            revision=config.model.revision,
            ignore_patterns=["original/*", "*.pth"],
        )
    benchmark = build_benchmark(config.benchmark)
    questions = benchmark.load()
    if config.prompt.n_shots:
        benchmark.exemplars(questions[0], config.prompt.n_shots)
    print(f"  {len(questions)} questions cached", flush=True)


def _run(config: RunConfig) -> None:
    outcome = run(config)
    print(
        f"  {outcome.status} run {outcome.run_id} config {outcome.config_hash}: "
        f"{outcome.n_written} written, {outcome.n_skipped} already stored",
        flush=True,
    )


def _report(root: Path) -> None:
    report = to_arrow(build_report(ResultStore(root)))
    pq.write_table(report, root / "report.parquet")
    duckdb.sql(
        "SELECT model_slug, variant, protocol, level, agent_id, round, metric, "
        "round(estimate, 4) AS estimate, round(ci_low, 4) AS ci_low, "
        "round(ci_high, 4) AS ci_high, n_analysable FROM report ORDER BY ALL"
    ).show(max_rows=10_000, max_width=250)
    print(f"all estimates are {ANALYSIS_STATUS}", flush=True)
    print(f"written {root / 'report.parquet'}", flush=True)


if __name__ == "__main__":
    main()

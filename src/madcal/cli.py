"""Command line: fetch what a config needs, run it on an idle GPU, or analyse its results."""

import argparse
import fcntl
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import duckdb
import pyarrow.compute as pc
import pyarrow.parquet as pq

from madcal.analysis import (
    ANALYSIS_STATUS,
    AnalysisFrame,
    build_report,
    load_arms,
    load_frame,
    propagation_rows,
    propagation_table,
    references,
    semantic_rows,
    semantic_table,
    to_arrow,
)
from madcal.config import (
    CalibratedTransformersModelConfig,
    FitConfig,
    HarvestConfig,
    OnPolicyConfig,
    RunConfig,
    TransformersModelConfig,
)
from madcal.orchestration import build_benchmark, run
from madcal.storage import ResultStore, model_slug

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
    """Fetch, run or trace propagation for every config named, or report on a results root."""
    parser = argparse.ArgumentParser(prog="madcal")
    parser.add_argument(
        "command",
        choices=(
            "fetch",
            "run",
            "report",
            "propagation",
            "semantic",
            "harvest",
            "fit",
            "compare",
            "on-policy",
        ),
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--reference", action="append", default=[])
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--seed", type=int, action="append", default=[])
    args = parser.parse_args(argv)

    if args.command == "compare":
        _compare(args.paths[0], args.paths[1:])
        return

    if args.command == "fit":
        for path in args.paths:
            config = FitConfig.from_yaml(path)
            if args.cache is not None:
                config = config.model_copy(update={"cache": args.cache})
            for seed in args.seed or [config.seed]:
                _fit(config.model_copy(update={"seed": seed}))
        return

    if args.command == "on-policy":
        trainings = [OnPolicyConfig.from_yaml(path) for path in args.paths]
        _offline_cache(offline=True)
        for path, training in zip(args.paths, trainings, strict=True):
            for seed in args.seed or [training.seed]:
                print(f"on-policy {path} seed {seed}", flush=True)
                _on_policy(training.model_copy(update={"seed": seed}))
        return

    if args.command == "harvest" or (args.command == "fetch" and _is_harvest(args.paths)):
        harvests = [HarvestConfig.from_yaml(path) for path in args.paths]
        _offline_cache(offline=args.command == "harvest")
        for path, harvest_config in zip(args.paths, harvests, strict=True):
            print(f"{args.command} {path}", flush=True)
            if args.command == "harvest":
                _harvest(harvest_config)
            else:
                _fetch_harvest(harvest_config)
        return

    if args.command == "semantic":
        for root in args.paths:
            _semantic(root, args.reference)
        return

    if args.command == "report":
        for root in args.paths:
            _report(root)
        return

    configs = [RunConfig.from_yaml(path) for path in args.paths]
    os.chdir(REPO_ROOT)
    os.environ["HF_HOME"] = str(CACHE / "huggingface")
    if args.command in ("run", "propagation"):
        os.environ.update(dict.fromkeys(OFFLINE, "1"))
    if args.command == "run":
        print(f"gpu {claim_gpu()}", flush=True)

    for path, config in zip(args.paths, configs, strict=True):
        print(f"{args.command} {path}", flush=True)
        if args.command == "fetch":
            _fetch(config)
        elif args.command == "propagation":
            _propagation(path, config)
        else:
            _run(config)


def _is_harvest(paths: Sequence[Path]) -> bool:
    import yaml

    kinds = {"train" in yaml.safe_load(path.read_text()) for path in paths}
    if len(kinds) != 1:
        raise SystemExit("fetch takes either run configs or harvest configs, not both")
    return kinds.pop()


def _offline_cache(offline: bool) -> None:
    os.chdir(REPO_ROOT)
    os.environ["HF_HOME"] = str(CACHE / "huggingface")
    if offline:
        os.environ.update(dict.fromkeys(OFFLINE, "1"))
        print(f"gpu {claim_gpu()}", flush=True)


def _fetch_harvest(config: HarvestConfig) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        config.model.model_id,
        revision=config.model.revision,
        ignore_patterns=["original/*", "*.pth"],
    )
    for benchmark_config in (config.train, config.evaluation):
        questions = build_benchmark(benchmark_config).load()
        print(f"  {benchmark_config.split}: {len(questions)} questions cached", flush=True)


def _harvest(config: HarvestConfig) -> None:
    from madcal.orchestration.harvest import harvest

    outcome = harvest(config)
    print(
        f"  cache {outcome.root}: {outcome.n_written} written, {outcome.n_skipped} already stored",
        flush=True,
    )


def _fit(config: FitConfig) -> None:
    from madcal.orchestration.fit import fit

    outcome = fit(config)
    print(
        f"  {config.arm.name} seed {config.seed} -> {outcome.root} "
        f"(best step {outcome.best_step}, selection Brier {outcome.selection_brier:.4f})",
        flush=True,
    )


def _on_policy(config: OnPolicyConfig) -> None:
    from madcal.orchestration.on_policy import on_policy

    outcome = on_policy(config)
    print(
        f"  {config.arm.name} seed {config.seed} -> {outcome.root} "
        f"(best step {outcome.best_step}, selection Brier {outcome.selection_brier:.4f})",
        flush=True,
    )


def _compare(cache_dir: Path, fit_dirs: Sequence[Path]) -> None:
    from madcal.analysis.heads import head_rows, heads_table, score_arms
    from madcal.orchestration.fit import resolve_device
    from madcal.storage.states import StateStore

    cache = StateStore(cache_dir).load()
    ids, correct, scored = score_arms(cache, fit_dirs, resolve_device("auto"))
    compare = heads_table(head_rows(cache.manifest["benchmark"], ids, correct, scored))
    destination = cache_dir.parent / f"compare-{cache.manifest['config_hash']}.parquet"
    pq.write_table(compare, destination)
    duckdb.sql(
        "SELECT arm, comparison, metric, round(estimate, 4) AS estimate, "
        "round(ci_low, 4) AS ci_low, round(ci_high, 4) AS ci_high, n_questions FROM compare "
        "WHERE metric IN ('accuracy', 'mean_confidence', 'ece_equal_mass', 'smooth_ece', "
        "'brier', 'auroc', 'answer_changed') ORDER BY comparison <> 'arm', arm, comparison, metric"
    ).show(max_rows=10_000, max_width=250)
    print(f"all estimates are {ANALYSIS_STATUS}", flush=True)
    print(f"written {destination}", flush=True)


def _fetch(config: RunConfig) -> None:
    from huggingface_hub import snapshot_download

    if isinstance(config.model, TransformersModelConfig | CalibratedTransformersModelConfig):
        snapshot_download(
            config.model.model_id,
            revision=config.model.revision,
            ignore_patterns=["original/*", "*.pth"],
        )
    if config.semantic is not None and config.semantic.nli_model is not None:
        snapshot_download(config.semantic.nli_model, revision=config.semantic.nli_revision)
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


def _propagation(path: Path, config: RunConfig) -> None:
    if config.debate is None:
        raise ValueError(f"{path} is not a debate config; propagation reads debates only")
    frame = load_frame(ResultStore(config.output_root))
    condition = (
        (pc.field("model_slug") == model_slug(config.model.model_id))
        & (pc.field("variant") == str(config.model.variant))
        & (pc.field("protocol") == config.debate.protocol)
        & (pc.field("benchmark") == config.benchmark.name)
    )
    selected = AnalysisFrame(frame.rows.filter(condition), frame.excluded_runs)
    labels = {
        question.id: (str(question.answer), len(question.choices or ()))
        for question in build_benchmark(config.benchmark).load()
    }
    propagation = propagation_table(propagation_rows(selected, labels))
    destination = Path(config.output_root) / f"propagation-{path.stem}.parquet"
    pq.write_table(propagation, destination)
    duckdb.sql(
        "SELECT comparison, metric, round(estimate, 4) AS estimate, round(ci_low, 4) AS ci_low, "
        "round(ci_high, 4) AS ci_high, n_questions, n_analysable FROM propagation ORDER BY ALL"
    ).show(max_rows=10_000, max_width=250)
    print(f"all estimates are {ANALYSIS_STATUS}", flush=True)
    print(f"written {destination}", flush=True)


def _semantic(root: Path, chosen: Sequence[str]) -> None:
    arms = load_arms(ResultStore(root))
    if not arms:
        raise SystemExit(f"no complete semantic runs under {root}")
    reference_arms = references(arms, chosen)
    for (model, benchmark), arm in reference_arms.items():
        named = "none: no single base arm, pass --reference" if arm is None else arm.config_hash
        print(f"reference for {model} {benchmark}: {named}", flush=True)
    semantic = semantic_table(semantic_rows(arms, reference_arms))
    destination = root / "semantic.parquet"
    pq.write_table(semantic, destination)
    duckdb.sql(
        "SELECT benchmark, arm, variant, comparison, measure, metric, "
        "round(estimate, 4) AS estimate, round(ci_low, 4) AS ci_low, "
        "round(ci_high, 4) AS ci_high, n_analysable FROM semantic "
        "WHERE metric IN ('accuracy', 'mean_confidence', 'ece', 'smooth_ece', 'brier', 'auroc') "
        "ORDER BY benchmark, comparison <> 'arm', arm, measure, metric"
    ).show(max_rows=10_000, max_width=250)
    print(f"all estimates are {ANALYSIS_STATUS}", flush=True)
    print(f"written {destination}", flush=True)


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

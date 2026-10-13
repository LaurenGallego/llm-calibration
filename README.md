# llm-calibration

Calibration propagation through multi-agent debate (`madcal`).

## Running

Everything a run needs stays inside the clone: models and datasets in `.cache/huggingface`,
results under the config's `output_root` (relative paths resolve against the repo root),
logs in `logs/`.

```bash
uv sync                                         # once per machine; Python 3.12
uv run madcal fetch configs/<config>.yaml       # on a node with internet (cluster: login node)
uv run madcal run configs/<config>.yaml         # offline; pins itself to an idle GPU
uv run madcal report results                    # metrics per cell -> results/report.parquet
```

`madcal run` keeps a GPU the scheduler already assigned (`CUDA_VISIBLE_DEVICES`). Otherwise it
takes the first GPU that `nvidia-smi` shows with no compute process and under 1 GiB in use, and
holds a lock in `.cache/gpu-locks/` for the lifetime of the process so two runs started from this
clone never share a GPU. Without `nvidia-smi` it runs on whatever torch finds.

A rerun of the same config resumes: questions already stored under its config hash are skipped.

Machine-specific settings (SLURM account and partition, `HF_TOKEN`) live in a gitignored `.env`:
`cp .env.example .env` and fill it in.

**SLURM cluster.** From the repo root on the login node, after `fetch`:

```bash
scripts/submit_slurm.sh configs/<config>.yaml
```

Any `SBATCH_*` variable in `.env` (e.g. `SBATCH_TIMELIMIT`) overrides the defaults in
`scripts/slurm.sbatch`.

**Multi-GPU machine without a scheduler.** Each config becomes its own background process on its
own GPU:

```bash
scripts/launch_local.sh configs/a.yaml configs/b.yaml
```

Gated models (Llama) need `HF_TOKEN` in the environment for `fetch`:
`set -a; source .env; set +a` first.

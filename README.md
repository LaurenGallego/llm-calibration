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
scripts/submit_slurm.sh [--sbatch-option ...] configs/<config>.yaml
```

Any `SBATCH_*` variable in `.env` (e.g. `SBATCH_TIMELIMIT`) overrides the defaults in
`scripts/slurm.sbatch`.

**Multi-GPU machine without a scheduler.** Each config becomes its own background process on its
own GPU:

```bash
scripts/launch_local.sh configs/a.yaml configs/b.yaml
```

**Answer-level heads.** One harvest per benchmark caches the model's greedy answers, their grades and
the output head's inputs at the answer positions; each arm is then fitted on the cache and compared
on the held-out evaluation questions:

```bash
uv run madcal fetch configs/harvest_qwen7b_trivia.yaml              # login node
uv run madcal harvest configs/harvest_qwen7b_trivia.yaml            # GPU -> results/harvest/<hash>
uv run madcal fit configs/fit_*.yaml --cache results/harvest/<hash> --seed 0 --seed 1 --seed 2
uv run madcal compare results/harvest/<hash> results/fits/*         # -> results/harvest/compare-<hash>.parquet
```

On SLURM, set the command with `MADCAL_COMMAND`, e.g.
`MADCAL_COMMAND=harvest scripts/submit_slurm.sh configs/harvest_qwen7b_trivia.yaml`. A fitted VAE's
`checkpoint.pt` loads into the `calibrated_transformers` model (`vae_checkpoint` + `vae_sha256` from its
`fit.json`) for semantic and debate runs.

Gated models (Llama) need `HF_TOKEN` in the environment for `fetch`:
`set -a; source .env; set +a` first.

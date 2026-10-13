# llm-calibration

Calibration propagation through multi-agent debate (`madcal`).

## Running

Everything a run needs stays inside the clone: models and datasets in `.cache/huggingface`,
results under the config's `output_root` (relative paths resolve against the repo root),
logs in `logs/`.

```bash
uv sync                                         # once per machine; Python 3.12
uv run madcal fetch configs/<config>.yaml       # on a node with internet (C3: login node)
uv run madcal run configs/<config>.yaml         # offline; pins itself to an idle GPU
uv run madcal report results                    # metrics per cell -> results/report.parquet
```

`madcal run` keeps a GPU the scheduler already assigned (`CUDA_VISIBLE_DEVICES`). Otherwise it
takes the first GPU that `nvidia-smi` shows with no compute process and under 1 GiB in use, and
holds a lock in `.cache/gpu-locks/` for the lifetime of the process so two runs started from this
clone never share a GPU. Without `nvidia-smi` it runs on whatever torch finds.

A rerun of the same config resumes: questions already stored under its config hash are skipped.

**C3 (SLURM).** From the repo root on the login node, after `fetch`:

```bash
sbatch scripts/c3.sbatch configs/<config>.yaml
```

**skoll (4 GPUs, no scheduler).** Each config becomes its own background process on its own GPU:

```bash
scripts/skoll.sh configs/a.yaml configs/b.yaml
```

Gated models (Llama) need `HF_TOKEN` exported before `fetch`.

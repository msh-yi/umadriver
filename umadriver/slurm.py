"""Write a SLURM batch script that reruns this command on a compute node.

`--submit` takes the command you just typed, strips its own flags out of it, and
writes the rest into a batch script with the environment UMA needs. It does not
call sbatch: the script is left for you to read and submit, since a job that is
about to occupy several GPUs for hours is worth looking at first.

The preamble is the one that has been working in practice — caches on netscratch
rather than home (the model is several GB and home is slow and quota'd), CUDA's
JIT cache likewise, `module load cuda/12.9 Mambaforge`, and the batch-safe conda
activation. `conda activate` from a non-interactive shell needs `conda.sh`
sourced first, and that has to happen with `set -u` off because the script sets
unbound variables.
"""

from __future__ import annotations

import os
import re
import shlex
import sys
from dataclasses import dataclass
from typing import List, Optional

from .constants import VAST_BASE

# gpu_test on FASRC Cannon: 12 h, 8 A100 MIG 3g.20GB slices per job, and per the
# docs "less than 8 CPUs/MIG GPU and 64GB/MIG GPU". Exceeding these does not fail
# at submission — it fails later, or the job never schedules.
GPU_TEST_MAX_HOURS = 12
GPU_TEST_MAX_GPUS = 8
GPU_TEST_CPUS_PER_GPU = 8
GPU_TEST_GB_PER_GPU = 64


@dataclass
class SlurmOpts:
    """Defaults reproduce the submit script this package has been run with."""

    partition: str = "gpu_test"
    gpus: int = 6
    cpus: int = 16
    mem: str = "160g"
    time: str = "12:00:00"
    job_name: Optional[str] = None


# Every one of these takes exactly one value; --submit takes none.
_VALUE_FLAGS = (
    "--slurm-partition",
    "--slurm-gpus",
    "--slurm-cpus",
    "--slurm-mem",
    "--slurm-time",
    "--slurm-job-name",
)


def strip_slurm_flags(argv: List[str]) -> List[str]:
    """The command minus the flags that asked for it to be written out."""
    out: List[str] = []
    skip = False
    for token in argv:
        if skip:
            skip = False
            continue
        if token == "--submit":
            continue
        if token in _VALUE_FLAGS:
            skip = True
            continue
        if any(token.startswith(f + "=") for f in _VALUE_FLAGS):
            continue
        out.append(token)
    return out


def _hours(t: str) -> float:
    """SLURM walltime as hours. Accepts D-HH:MM:SS, HH:MM:SS, MM:SS and MM."""
    days = 0
    if "-" in t:
        d, _, t = t.partition("-")
        days = int(d)
    parts = [float(p) for p in t.split(":")] if t else [0]
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    else:
        h, m, s = 0, parts[0], 0
    return days * 24 + h + m / 60 + s / 3600


def _mem_gb(mem: str) -> float:
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kmgt]?b?)\s*", mem, re.I)
    if not m:
        raise ValueError(f"--slurm-mem: cannot read {mem!r}; try something like 160g")
    scale = {"": 1e-3, "k": 1e-6, "m": 1e-3, "g": 1.0, "t": 1024.0}
    return float(m.group(1)) * scale[m.group(2)[:1].lower()]


def check_limits(opts: SlurmOpts) -> None:
    """Refuse a request gpu_test will not honour. Other partitions are the
    user's business — this only knows the one it defaults to."""
    if opts.partition != "gpu_test":
        return

    problems = []
    if _hours(opts.time) > GPU_TEST_MAX_HOURS:
        problems.append(f"time {opts.time} exceeds the {GPU_TEST_MAX_HOURS} h limit")
    if opts.gpus > GPU_TEST_MAX_GPUS:
        problems.append(f"{opts.gpus} GPUs exceeds the limit of {GPU_TEST_MAX_GPUS}")
    if opts.cpus >= GPU_TEST_CPUS_PER_GPU * opts.gpus:
        problems.append(
            f"{opts.cpus} CPUs for {opts.gpus} GPUs — gpu_test wants fewer than "
            f"{GPU_TEST_CPUS_PER_GPU} per MIG slice ({GPU_TEST_CPUS_PER_GPU * opts.gpus})"
        )
    if _mem_gb(opts.mem) >= GPU_TEST_GB_PER_GPU * opts.gpus:
        problems.append(
            f"{opts.mem} for {opts.gpus} GPUs — gpu_test wants under "
            f"{GPU_TEST_GB_PER_GPU} GB per MIG slice ({GPU_TEST_GB_PER_GPU * opts.gpus} GB)"
        )

    if problems:
        raise ValueError(
            "gpu_test limits: " + "; ".join(problems) + ".\n"
            "  Use --slurm-partition gpu for a longer or larger job (3 days, "
            "whole A100s), or lower the request."
        )


def conda_env_name() -> str:
    """The environment this interpreter is in, so the job runs what you tested.

    Prefers the running prefix over CONDA_DEFAULT_ENV, which still says "base"
    when a package is invoked by its full path out of an env's bin/.
    """
    prefix = sys.prefix
    if os.path.basename(os.path.dirname(prefix)) == "envs":
        return os.path.basename(prefix)
    return os.environ.get("CONDA_DEFAULT_ENV") or "base"


def render_script(
    command: List[str],
    opts: SlurmOpts,
    *,
    cache_base: Optional[str] = None,
    conda_env: Optional[str] = None,
) -> str:
    base = cache_base or VAST_BASE
    env = conda_env or conda_env_name()
    name = opts.job_name or "umadriver"
    # Threads are per worker; umadriver divides this budget across its workers.
    threads = max(1, opts.cpus // max(1, opts.gpus))

    return f"""#!/bin/bash
#SBATCH -p {opts.partition}
#SBATCH --gres=gpu:{opts.gpus}
#SBATCH -c {opts.cpus}
#SBATCH --mem={opts.mem}
#SBATCH -t {opts.time}
#SBATCH -J {name}
#SBATCH --mail-type=END,FAIL

# Written by `umadriver --submit`. Edit freely, then: sbatch this file.
set -Eeuo pipefail

# Node-local scratch for temporary files.
export TMPDIR="/scratch/$USER/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"

# Model and HF caches live on netscratch: the checkpoint is several GB, and home
# is both slow and quota'd. UMA_CACHE_BASE is what umadriver itself reads.
export UMA_CACHE_BASE="{base}"
export OMOL_CACHE_BASE="$UMA_CACHE_BASE"
export FAIRCHEM_CACHE="$UMA_CACHE_BASE/fairchem_cache"
export HUGGINGFACE_HUB_CACHE="$UMA_CACHE_BASE/hf_cache"
export TRANSFORMERS_CACHE="$HUGGINGFACE_HUB_CACHE"
export HF_HOME="$UMA_CACHE_BASE/.hf"
mkdir -p "$FAIRCHEM_CACHE" "$HUGGINGFACE_HUB_CACHE" "$HF_HOME"

export CUDA_MODULE_LOADING=LAZY
export CUDA_CACHE_PATH="$UMA_CACHE_BASE/.nv/ComputeCache"
export CUDA_CACHE_MAXSIZE=2147483648
mkdir -p "$CUDA_CACHE_PATH"

export OMP_NUM_THREADS={threads}
export MKL_NUM_THREADS={threads}
export OPENBLAS_NUM_THREADS={threads}
export NUMEXPR_NUM_THREADS={threads}

if [ -f /etc/profile.d/lmod.sh ]; then
  . /etc/profile.d/lmod.sh
elif [ -f /etc/profile.d/modules.sh ]; then
  . /etc/profile.d/modules.sh
fi

module purge
module load cuda/12.9
module load Mambaforge

hostname
date
nvidia-smi || true

# conda activate needs conda.sh sourced in a non-interactive shell, and both
# touch unbound variables, so -u comes off for the duration.
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate {env}
set -u

cd "${{SLURM_SUBMIT_DIR:-$PWD}}"

{shlex.join(command)}
"""


def write_script(
    command: List[str],
    opts: SlurmOpts,
    *,
    directory: Optional[str] = None,
    cache_base: Optional[str] = None,
) -> str:
    """Write the script beside where the command was typed; return its path."""
    check_limits(opts)
    name = opts.job_name or "umadriver"
    path = os.path.join(directory or os.getcwd(), f"{name}.sbatch")
    with open(path, "w") as f:
        f.write(render_script(command, opts, cache_base=cache_base))
    os.chmod(path, 0o755)
    return path


def job_name_from(inputs: List[str], manifest: Optional[str]) -> str:
    """`uma_<stem>` of the manifest, or of the first input."""
    source = manifest or (inputs[0] if inputs else None)
    if not source:
        return "umadriver"
    stem = os.path.splitext(os.path.basename(source))[0]
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in stem)[:48]
    return f"uma_{safe}" if safe else "umadriver"

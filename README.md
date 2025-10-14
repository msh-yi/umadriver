# umadriver

Wrapper around **Meta’s Universal Model for Atoms (UMA)** that drives an **ensemble** workflow per input XYZ file: geometry optimizations (incl. TS), single points, vibrational frequencies, quasi-RRHO thermochemistry, optional **Gaussian input** emission, and optional **IRC** tracing.

`umadriver` is exposed as a command-line tool and as a small Python API.

---

## Features

- **Ensemble-first pipeline**: every run goes through the conformer ensemble workflow and writes a results CSV.
- **Geometry**: standard optimization or **TS** (first-order saddle) via **Sella**.
- **Frequencies & Thermochemistry**: finite-difference frequencies, qRRHO by default (RRHO available), temperature/pressure controls.
- **Gaussian inputs**: emit M052X/6-31G* (optionally SMD solvent) inputs for downstream QM.
- **IRC**: forward/backward intrinsic reaction coordinate after a TS (or directly).
- **Batch mode**: run many jobs via a manifest file or globbing multiple XYZs.
- **HPC-friendly**: thread controls, scratch directories, safe defaults to avoid JAX/CPU thread contention.

---

## IMPORTANT: get access to the UMA model
(adapted from fairchem repo)

Create a free Hugging Face account, request access to the **[UMA model repository](https://huggingface.co/facebook/UMA)**, and have your personal access token ready.

```bash
pip install -U "huggingface_hub[cli]"
hf --help           # should print CLI usage (the 'hf' CLI is the modern tool)
# (Alternative older name: huggingface-cli --help)

# Interactive (recommended):
hf auth login

# Or non-interactive if you already exported HF_TOKEN:
# export HF_TOKEN=hf_xxx...    # put this in your shell profile to persist
hf auth login --token "$HF_TOKEN" --add-to-git-credential

# Verify:
hf auth whoami
```

Issues:

* **401 / permission denied**: You haven’t been granted access to `facebook/UMA` or you aren’t logged in. Run `hf auth whoami` and re-apply for repository access if needed.

## Installation

> Requires **Python ≥ 3.9**.


### From source
```bash
conda create -n umadriver -c conda-forge python=3.10 -y
conda activate umadriver
git clone https://github.com/msh-yi/umadriver.git
cd umadriver
pip install .
umadriver -h
```

### Runtime dependencies

Declared in `pyproject.toml`:

- `numpy>=1.23`
- `ase>=3.22.1`
- `fairchem-core>=2.4.0`

> Notes:
> - GPU use is optional. Set `--device cuda` to use a CUDA GPU if your `fairchem-core`/PyTorch install supports it.
> - The driver touches **JAX** only to set safe defaults so that JAX does not hog GPU memory from UMA/torch. No JAX code is required from your side.
> - If you have more than one GPU they will be parallelized at the ensemble level (i.e. if you have four .xyz files and three GPUs, each will handle all the structures in one .xyz file)

---

## Quickstart (CLI)

```bash
# Basic geometry optimization of all conformers in molecule.xyz
umadriver --xyz molecule.xyz

# Tight optimization + frequencies + qRRHO at 343.15 K
umadriver --xyz molecule.xyz --opt-mode Tight --freq --temp 343.15

# Single-point energies (no optimization)
umadriver --xyz molecule.xyz --sp

# Transition-state optimization, then freq and IRC
umadriver --xyz ts_guess.xyz --optts --freq --irc

# Emit Gaussian inputs (gas phase)
umadriver --xyz molecule.xyz --solv none

# Emit Gaussian inputs with SMD(acetonitrile), custom resources
umadriver --xyz molecule.xyz --solv acetonitrile --gauss-mem 80GB --gauss-nproc 8

# Explicit charge / multiplicity
umadriver --xyz anion.xyz --charge -1 --mult 1
```

**Outputs**  
By default, results go to `<basename>.ensemble/` (override with `--outdir`). The driver prints the path to a summary CSV at the end:

```
Ensemble complete. Results CSV: <outdir>/energies.csv
```

If `--freq` is requested you’ll also get per-structure, ORCA-style vibration outputs and a thermochemistry summary (qRRHO on by default).

---

## Common flags (selected)

- `--xyz PATH` (required unless using `batch`): input XYZ with one or more conformers/frames.
- **Geometry**
  - `--opt` *(default: Sella)*: optimizer (use `--sp` for single-point only).
  - `--optts`: TS optimization (1st-order saddle; uses Sella).
  - `--opt-mode {Loose,Normal,Tight,VeryTight}` *(default: Normal)*.
  - `--maxcycles INT` *(default: 300)*.
- **Frequencies / Thermo**
  - `--freq`: run vibrational analysis.
  - `--freq-delta FLOAT` *(Å, default 0.01)*, `--freq-nfree 2`, `--freq-scale FLOAT`.
  - `--temp K` *(default 298.15)*, `--pressure-atm` *(default 1)*.
  - `--qrrho/--no-qrrho` *(default qRRHO on)*, `--cutoff-cm1`, `--qrrho-ref-cm1` *(default 100)*, `--qrrho-alpha` *(default 4.0)*.
  - `--symmetry-number`, `--point-group` (optional thermochemistry inputs).
- **Gaussian inputs**
  - `--solv NAME` → writes M052X/6-31G* inputs; if `NAME` given, uses `scrf(SMD,solvent=<NAME>)`.
  - `--gauss-mem`, `--gauss-nproc`.
- **IRC**
  - `--irc`, `--irc-dx FLOAT` *(default 0.1)*.
- **Model / device / cache**
  - `--model` *(default: uma-m-1p1)*.
  - `--device {cuda,cpu,auto}` *(default: cuda; **batch default also cuda**)*.
  - `--cache-dir` *(default: driver’s `DEFAULT_FAIRCHEM_CACHE`)*.
- **Scratch & logging**
  - `--scratch-root PATH` *(default: `$UMA_SCRATCH_ROOT` or driver default)*.
  - `--use-local-scratch`.
  - `--verbose`, `--debug`.

Run `umadriver -h` to see the full help.

---

## Batch workflows

Use the **`batch`** subcommand to process multiple ensembles in one process (one GPU), either from a **manifest** or with **globs**.

### With globs
```bash
umadriver batch \
  --xyz-glob "inputs/*.xyz" "more/*.xyz" \
  --out-root runs \
  --resume \
  --model uma-m-1p1 --device cuda \
  --optimizer Sella --optts \
  --do-freq \
  --irc --irc-dx 0.1
```

- `--out-root` becomes the parent folder for each job’s `out_dir`.
- `--resume` (default **on**) skips jobs that already have an `energies.csv`. Use `--no-resume` to force reruns.
- Flags like `--optimizer`, `--optts`, `--do-freq`, `--solv`, `--irc`, `--irc-dx` act as **broadcast overrides** applied to all jobs in this batch.

### With a manifest file

Two styles are supported per job entry:

1) **Per-job `overrides:` map** (recommended for clarity)  
2) **Flattened keys** directly under a job (handy for SP/solv/freq shortcuts)

#### Minimal schema

```yaml
# manifest.yaml
jobs:
  - xyz: path/to/molecule_A.xyz
    out_dir: runs/molecule_A
    overrides:
      charge: 0
      optimizer: Sella
      opt_mode: Tight
      do_freq: true
      temp: 298.15

  - xyz: path/to/ts_guess.xyz
    out_dir: runs/ts_pipeline
    overrides:
      charge: -1
      optimizer: Sella
      optts: true      # TS optimization
      do_freq: true
      irc: true
      irc_dx: 0.1

  # Single-point + Gaussian input emission (no optimization, no freq)
  - xyz: path/to/solv_sp.xyz
    out_dir: runs/solv_sp
    optimizer: null    # same effect as CLI --sp
    optts: false
    do_freq: false
    solv: acetonitrile
    gauss_mem: 80GB
    gauss_nproc: "8"

  # Pure frequency/thermo at elevated T (no optimization)
  - xyz: path/to/freq_only.xyz
    out_dir: runs/freq_only
    overrides:
      optimizer: null  # no optimization
      do_freq: true
      temp: 343.15
      pressure_atm: 1.0
```

Run it:

```bash
umadriver batch --manifest manifest.yaml --out-root runs --device cuda
```

The CLI will print a one-line summary per job:
```
[OK] inputs/molecule_A.xyz -> runs/molecule_A
```

---

## Threading, JAX, and HPC notes

`umadriver` does a small **early parse** of two special flags before importing heavy deps:

- `--sella-threads INT`  
  Sets the Sella/BLAS thread pool size. If omitted, it falls back to:
  1) `$SLURM_CPUS_PER_TASK` (if present), else  
  2) `os.cpu_count()`.

  It also relaxes caps on `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` accordingly.

- `--jax-platform {cpu,cuda}` *(default: cpu)*  
  Pins `JAX_PLATFORMS`/`JAX_PLATFORM_NAME`. **Unless on multi-GPU**, CPU is safer so JAX doesn’t reserve GPU memory that UMA/torch needs.

**Examples**
```bash
# On a SLURM node with 8 CPUs:
srun -c 8 umadriver --xyz mol.xyz --sella-threads 8

# Force JAX to CPU; use CUDA for UMA
umadriver --xyz mol.xyz --jax-platform cpu --device cuda
```

---

## Output structure (typical)

```
<outdir>/
  energies.csv                 # ensemble summary (paths, energies, statuses, etc.)
  conformer_0001/              # per-conformer working folders
    opt.log
    freq/                      # if --freq
      vibrations.dat           # ORCA-style prints
      thermo_summary.json      # qRRHO/RRHO summary, temperature, etc.
  gaussian_inputs/             # if --solv was set
    conf0001.gjf
    ...
```

*(Exact layout may evolve; rely on the printed CSV path for aggregation.)*

---

## Python API (advanced)

If you prefer calling from Python:

```python
from umadriver.ensemble import run_conformer_workflow

csv_path = run_conformer_workflow(
    "molecule.xyz",
    out_dir="molecule.ensemble",
    charge=0, mult=1,
    model="uma-m-1p1", device="cuda",
    cache_dir=None, use_local_scratch=False,
    optimizer="Sella", opt_mode="Normal", optts=False,
    maxcycles=300,
    do_freq=False,
    freq_delta=0.01, freq_nfree=2, freq_scale=1.0,
    temp=298.15, pressure_atm=1.0,
    symmetry_number=1, point_group=None,
    qrrho=True, cutoff_cm1=None,
    qrrho_ref_cm1=100.0, qrrho_alpha=4.0,
    solv=None, gauss_mem="160GB", gauss_nproc="16",
    sella_internal=True, sella_eta=2e-2, sella_gamma=1e-4, sella_delta0=0.02,
    irc=False, irc_dx=0.1,
)
print("CSV:", csv_path)
```

> The arguments mirror the CLI options. See the source for the exact signature and defaults.

---

## Troubleshooting

- **CUDA OOM or contention**  
  - Try `--jax-platform cpu` (default) to keep JAX off the GPU.  
  - Reduce `--gauss-nproc` / threads; set `--sella-threads` explicitly.
- **Runs are being skipped in batch**  
  - `--resume` is **on** by default. Use `--no-resume` to force re-runs.
- **Slow frequencies**  
  - Increase `--freq-delta` slightly (with care), or run fewer conformers at once.
- **Thermochemistry doesn’t match expectations**  
  - Remember qRRHO is enabled by default (`--no-qrrho` to switch to RRHO).  
  - Provide `--symmetry-number` / `--point-group` when known.

---

## Citation & Acknowledgements

- **UMA (Universal Model for Atoms)** by Meta AI.  
- Uses **ASE** for atoms & vibrations, **Sella** for robust geometry/TS, and **fairchem-core** for the underlying ML potential.

---

## License

See `LICENSE` in this repository.

---

## Changelog

- **0.1.0** — initial public release.

---

### Appendix: Another manifest example (mirrors common patterns)

```yaml
jobs:
  # Gas-phase optimization with frequencies at 298 K
  - xyz: inputs/complex.xyz
    out_dir: runs/complex_opt
    overrides:
      charge: -1
      optimizer: Sella
      opt_mode: VeryTight
      do_freq: true
      temp: 298.15

  # TS → freq → IRC (anionic)
  - xyz: inputs/ts_guess.xyz
    out_dir: runs/ts_pipeline
    overrides:
      charge: -1
      optimizer: Sella
      optts: true
      do_freq: true
      irc: true
      irc_dx: 0.1

  # Single-point + Gaussian input emission in acetonitrile (no freq)
  - xyz: inputs/solv_only.xyz
    out_dir: runs/solv_only
    optimizer: null
    optts: false
    do_freq: false
    solv: acetonitrile
    gauss_mem: 80GB
    gauss_nproc: "8"
```

# umadriver

Wrapper around **Meta’s Universal Model for Atoms (UMA)** that drives an **ensemble** workflow per input XYZ file: geometry optimizations (incl. TS), single points, vibrational frequencies, quasi-RRHO thermochemistry, optional **implicit solvation**, and optional **IRC** tracing.

`umadriver` is exposed as a command-line tool and as a small Python API.

---

## Features

- **Ensemble-first pipeline**: every run goes through the conformer ensemble workflow and writes a results CSV.
- **Geometry**: standard optimization or **TS** (first-order saddle) via **Sella**.
- **Frequencies & Thermochemistry**: finite-difference frequencies, qRRHO by default (RRHO available), temperature/pressure controls.
- **Solvation**: ALPB implicit solvent from xtb, applied to every energy *and* force, so
  geometries optimize in solution.
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
- `sella` (default optimizer + all TS/IRC work)
- `pyyaml>=5.1` (YAML manifests)

> Notes:
> - GPU use is optional. Set `--device cuda` to use a CUDA GPU if your `fairchem-core`/PyTorch install supports it.
> - The driver touches **JAX** only to set safe defaults so that JAX does not hog GPU memory from UMA/torch. No JAX code is required from your side.
> - If you have more than one GPU they will be parallelized at the ensemble level (i.e. if you have four .xyz files and three GPUs, each will handle all the structures in one .xyz file)

---

## Quickstart (CLI)

> New here? [`examples/`](examples/) has five annotated manifests that run as-is
> against the structures in `tests/data/` — optimization, TS + IRC, implicit
> solvent, and the optimize-then-frequencies pipeline. Start with
> [`examples/01_optimize_conformers.yaml`](examples/01_optimize_conformers.yaml)
> to confirm your install works before pointing anything at a real system.

XYZ inputs are **positional** (one or more paths and/or glob patterns). There is no
`--xyz` flag and no `batch` subcommand — the ensemble/batch workflow is the only mode.

```bash
# Single-point energies (no --opt means no optimization)
umadriver molecule.xyz

# Geometry optimization of all conformers in molecule.xyz
umadriver molecule.xyz --opt Sella

# Tight optimization + frequencies + qRRHO at 343.15 K
umadriver molecule.xyz --opt Sella --opt-mode Tight --freq --temp 343.15

# Single-point energies (no optimization)
umadriver molecule.xyz --sp

# Transition-state optimization, then freq and IRC
umadriver ts_guess.xyz --optts --freq --irc

# Relaxed scan of the atom 1 - atom 2 distance, 1.4 -> 2.6 A in 13 points
# (atoms numbered from 1). Feed scan/*_scan_max.xyz to --optts for a TS guess.
umadriver mol.xyz --scan 1 2 1.4 2.6 13

# Optimize in implicit water (ALPB correction on every force and energy)
umadriver molecule.xyz --opt Sella --alpb water

# Explicit charge / multiplicity
umadriver anion.xyz --charge -1 --mult 1

# Multiple inputs / globs at once (quote globs so umadriver expands them)
umadriver "inputs/*.xyz" more/*.xyz --optts --freq --irc

# Manifest of jobs. An optional leading `batch` verb is accepted.
umadriver batch --manifest jobs.yaml
```

> **`--opt` is required to optimize.** With no `--opt`, `optimizer` is `None` and the
> route is a **single point**, not an optimization. `--optts` is the exception: it
> selects the TS route on its own and implies Sella (`order=1`).
>
> **`--optts` is a saddle *search*, not a label — the geometry moves.** For
> frequencies on a TS you do not want touched, use `--sp --freq --freq-ts`.
> `--sp --optts` and `--optts --opt LBFGS` are rejected rather than silently
> resolved, and a manifest pairing `optimizer: null` with `optts: true` is an
> error at load time.

**Outputs**  
By default, each input's results go to `<out-root>/<basename>.ensemble/` (`--out-root`
defaults to `runs`). The driver prints a one-line status per job at the end:

```
[ok] inputs/molecule.xyz -> runs/molecule.ensemble
```

The ranked summary CSV for each ensemble is `<out-root>/<basename>.ensemble/energies.csv`.
If `--freq` is requested you’ll also get per-structure, ORCA-style vibration outputs and a
thermochemistry summary (qRRHO on by default).

---

## Common flags (selected)

- `inputs` (positional, required unless `--manifest`): one or more XYZ paths and/or glob patterns.
- `--manifest PATH`: YAML/JSON manifest of jobs (if set, positional inputs are ignored).
- **Batch / outputs**
  - `--out-root PATH` *(default: `runs`)*: parent folder for each job's output dir.
  - `--resume` / `--no-resume` *(default: resume on)*: skip jobs that already have an `energies.csv`.
  - `--split-multi-structure` / `--no-split-multi-structure` *(default: split on)*: split a
    multi-structure XYZ into one job per structure for GPU load-balancing. Applies to
    **both** manifest jobs and positional/glob inputs. Results are re-compiled and ranked
    into a single `energies.csv` afterward.
  - `--workers-per-gpu INT` *(default: 1)*: worker processes per GPU. A single small
    structure does not saturate a large card, so 2–3 can be close to a linear win.
    Costs VRAM linearly, and multiplies with `--freq-batch-size`. The CPU thread budget
    is divided across workers automatically.
- **Geometry**
  - `--opt OPT`: optimizer (`Sella`, `LBFGS`, `BFGS`, `BFGSLineSearch`, `FIRE`, `QuasiNewton`).
    If omitted, defaults to optimization unless `--sp` is given.
  - `--sp`: single-point only (no optimization).
  - `--optts`: optimize to a 1st-order saddle. Implies Sella; conflicts with `--sp`
    and with any other `--opt`.
  - `--scan I J FROM TO STEPS`: relaxed scan of the distance between atoms `I` and
    `J` from `FROM` to `TO` Å in `STEPS` points. **Atoms are numbered from 1.** At
    each point the bond is held fixed and everything else is minimized, starting
    from the previous relaxed geometry. Writes `scan/<tag>_scan.csv` (the profile),
    `scan/<tag>_scan.xyz` (the path), and `scan/<tag>_scan_max.xyz` — the highest
    point, which is the structure to hand to `--optts`. Conflicts with `--optts`
    and `--sp`. In a manifest: `scan: {i: 1, j: 2, from: 1.4, to: 2.6, steps: 13}`
    or `scan: [1, 2, 1.4, 2.6, 13]`.
  - `--opt-mode {Loose,Normal,Tight,VeryTight}` *(default: Normal)*.
  - `--maxcycles INT` *(default: 300)*.
- **Frequencies / Thermo**
  - `--freq`: run vibrational analysis.
  - `--freq-ts`: score the result as a transition state (one imaginary mode expected)
    without optimizing. Use with `--sp` for frequencies on a saddle you want left
    exactly as given. By default the route decides.
  - `--freq-delta FLOAT` *(Å, default 0.01)*, `--freq-nfree 2`, `--freq-scale FLOAT`.
  - `--freq-batch-size INT` *(default 1)*: evaluate the `2×3N` finite-difference
    displacements in batches instead of one forward pass at a time. A 170-atom molecule
    needs 1020 gradient calls, so this is the largest single speedup available. Default
    1 keeps ASE's `Vibrations` path; the batched Hessian is verified to reproduce it.
    Works under `--alpb` too — see `--freq-xtb-workers`.
  - `--temp K` *(default 298.15)*, `--pressure-atm` *(default 1)*.
  - `--qrrho/--no-qrrho` *(default qRRHO on)*, `--cutoff-cm1`, `--qrrho-ref-cm1` *(default 100)*, `--qrrho-alpha` *(default 4.0)*.
  - `--symmetry-number`, `--point-group` (optional thermochemistry inputs).
- **Solvation**
  - `--alpb SOLVENT`: add `E_xtb,alpb - E_xtb,vacuum` from GFN2-xTB to every energy and
    force, so the geometry optimizes in solvent and frequencies include the solvation
    Hessian. Needs `tblite` (`pip install tblite`). The per-conformer correction is
    reported as `solv_corr_kcal`.
  - `--alpb-method {GFN2-xTB,GFN1-xTB,GFN-FF}` *(default GFN2-xTB)*. ALPB has no GFN0
    parameters.
  - `--no-alpb-concurrent`: evaluate UMA and the two xtb calls sequentially. Concurrent
    is the default and measured 2.5x faster on a 170-atom system, since UMA is on the GPU
    and xtb on the CPU.
  - `--freq-xtb-workers INT` *(default: cores // OMP_NUM_THREADS)*: concurrent xtb calls
    during a solvated frequency run. xtb parallelizes poorly *within* a call (24 threads
    buys ~1.4x on 170 atoms) but the displacements are independent, so throughput comes
    from many single-threaded calls at once. `--freq-batch-size` works under `--alpb`:
    UMA batches on the GPU while xtb runs on CPU threads, measured **3.2x** end to end.
  - `--free-volume-solvent {none,H2O,toluene,DMF,AcOH,chloroform}`: solvent for the
    free-volume translational-entropy correction. Unrelated to `--alpb`, and only has an
    effect together with `--conc-mol-l`.
- **IRC**
  - `--irc`, `--irc-dx FLOAT` *(default 0.1)*.
  - `--irc-steps` *(default 200)*.
  - `--irc-eta`, `--irc-gamma`: Sella IRC hyperparameters, defaulting to `--sella-eta` /
    `--sella-gamma`. **Do not loosen `--irc-gamma`.** Sella's own IRC default is `0.1`,
    which is too loose to resolve the negative mode: the initial diagonalization reports
    a positive lowest eigenvalue, Sella's convergence test passes at the saddle itself
    (the force term is trivially satisfied there), and the IRC "succeeds" without taking
    a single step — one frame per leg, both on the TS. A one-frame leg is logged as a
    warning.
  - An IRC is skipped when frequencies were computed and `n_imag != 1` — it only means
    something from a first-order saddle.
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

`umadriver` always processes 1+ ensembles in one invocation. Give it multiple XYZ
paths/globs positionally, or point it at a **manifest** with `--manifest`. A leading
`batch` verb is accepted and ignored, so `umadriver batch --manifest jobs.yaml` and
`umadriver --manifest jobs.yaml` are the same command.

**How work is distributed.** Every structure from every input is expanded into its own
job and placed on a single shared queue; `--workers-per-gpu` workers per GPU pull from
it. Because it is one queue rather than a static assignment, uneven job costs
self-balance — a worker that finishes early takes the next structure. This applies to
manifest jobs and glob inputs alike, so a manifest of multi-conformer files fans out
across all available GPUs. With no GPUs the scheduler falls back to a serial loop.

### With globs

```bash
umadriver "inputs/*.xyz" "more/*.xyz" \
  --out-root runs \
  --resume \
  --model uma-m-1p1 --device cuda \
  --opt Sella --optts \
  --freq \
  --irc --irc-dx 0.1
```

- `--out-root` becomes the parent folder for each job’s output dir.
- `--resume` (default **on**) skips jobs that already have an `energies.csv`. Use `--no-resume` to force reruns.
- Flags like `--opt`, `--optts`, `--freq`, `--alpb`, `--irc`, `--irc-dx` act as **broadcast overrides** applied to all jobs in this run.

### With a manifest file

Two styles are supported per job entry:

1) **Per-job `overrides:` map** (recommended for clarity)  
2) **Flattened keys** directly under a job (handy for SP/freq shortcuts)

Both are honored. Effective per-job settings are merged with precedence
**CLI flags < manifest `common:` < per-job flattened keys < per-job `overrides:`**, so a
CLI `--freq` applies everywhere unless a job overrides it. Keys map to
`run_conformer_workflow` kwargs (`optimizer`, `optts`, `do_freq`, `irc`, `irc_dx`, …).

#### Minimal schema

```yaml
# manifest.yaml
common:            # optional; batch-level defaults for all jobs
  model: uma-m-1p1
  device: cuda
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

  # Single-point in implicit water (flattened keys, no optimization, no freq)
  - xyz: path/to/solv_sp.xyz
    out_dir: runs/solv_sp
    optimizer: null    # same effect as CLI --sp
    optts: false
    do_freq: false
    alpb: water

  # Pure frequency/thermo at elevated T (no optimization)
  - xyz: path/to/freq_only.xyz
    out_dir: runs/freq_only
    overrides:
      optimizer: null  # no optimization
      do_freq: true
      temp: 343.15
      pressure_atm: 1.0

  # The same, on a transition state. Note `optts: false` — writing `optts: true`
  # here would re-run the saddle search and replace the geometry; `freq_ts` is how
  # you say "expect one imaginary mode" without optimizing. (`optimizer: null`
  # alongside `optts: true` is rejected at load time for exactly this reason.)
  - xyz: path/to/ts_freq_only.xyz
    out_dir: runs/ts_freq_only
    overrides:
      optimizer: null
      optts: false
      freq_ts: true
      do_freq: true
      temp: 343.15
```

Run it:

```bash
umadriver --manifest manifest.yaml --out-root runs --device cuda
```

The CLI will print a one-line summary per job:

```
[ok] inputs/molecule_A.xyz -> runs/molecule_A
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
srun -c 8 umadriver mol.xyz --sella-threads 8

# Force JAX to CPU; use CUDA for UMA
umadriver mol.xyz --jax-platform cpu --device cuda
```

---

## Output structure (typical)

```
<out-root>/<basename>.ensemble/
  energies.csv                        # ranked ensemble summary (rank, energies, gibbs, n_imag, …)
  energies_per_conformer_<tag>.csv    # per-conformer checkpoint (append+flush, used for resume)
  optimized_ranked.xyz                # optimized geometries in ranked order
  per_struct_<tag>/                   # per-conformer optimized XYZs
    <tag>_conf_0000.xyz
  freq_out/                           # if --freq
    conf_0000.out                     # ORCA-style vibrations + thermochemistry
  irc/                                # if --irc
    conf_0000_irc_traj_fwd.traj       # ASE trajectory, forward leg
    conf_0000_irc_traj_rev.traj       # ASE trajectory, reverse leg
    conf_0000_irc_path.xyz            # stitched: reverse reversed -> TS -> forward
    conf_0000_irc.csv                 # direction,step,arc_length_A,energy_Eh,rel_kcal
```

With `--split-multi-structure` (default), a multi-conformer input is run as one job per
structure under `<out_dir>/<label>/`, then re-compiled into the top-level
`<out_dir>/energies.csv` and `optimized_ranked.xyz` ranked across all structures.
For manifest jobs `<out_dir>` is the job's own `out_dir:`; for positional/glob inputs it
is `<out-root>/<basename>.ensemble/`. Temp single-structure XYZs live in
`<out_dir>/.split/` and are removed as each member finishes.

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
    alpb=None, alpb_method="GFN2-xTB", alpb_concurrent=True,
    free_volume_solvent=None,
    sella_internal=True, sella_eta=2e-2, sella_gamma=1e-4, sella_delta0=0.02,
    freq_batch_size=1,
    irc=False, irc_dx=0.1,
    irc_eta=None, irc_gamma=None, irc_steps=200,
)
print("CSV:", csv_path)
```

> The arguments mirror the CLI options. See the source for the exact signature and defaults.

---

## Tests

```bash
pip install -e ".[test]"

pytest tests/test_units.py tests/test_batching.py -v   # no GPU, no model; seconds
pytest tests/ -v                                       # adds real-UMA runs
pytest tests/ -v --runbig                              # adds the 170-atom catalyst TS
```

Tests run against the **real** UMA model — there are no mock calculators standing in for
it. Anything needing the model skips (rather than fails) when there is no CUDA device or
no usable checkpoint in the cache, and says which. The two exceptions are pure-function
suites: `test_units.py` (ranking, XYZ splitting, manifest merging, aggregation) and
`test_batching.py` (job expansion, worker slots, thread budget), plus the
finite-difference Hessian check in `test_frequencies.py`, which is validated against
`ase.vibrations.Vibrations` as the reference implementation.

---

## Troubleshooting

- **`401` / `GatedRepoError` at model load**  
  - Your HF token has expired: `hf auth whoami` to confirm, then `hf auth login`.  
  - Note the token is written under `$HF_HOME` if that is set, **not** `~/.cache/huggingface`.
- **Cache looks populated but the model still downloads / fails**  
  - Scratch purges delete the HF `blobs/` while leaving the `snapshots/` symlink tree
    behind, so the cache directory is non-empty but every checkpoint is a dangling link.
    Check with `find <cache> -name '*.pt' -size +100M` — if that prints nothing, the
    cache is empty regardless of how it looks.
- **CUDA OOM or contention**  
  - Try `--jax-platform cpu` (default) to keep JAX off the GPU.  
  - Set `--sella-threads` explicitly.  
  - Lower `--workers-per-gpu` and/or `--freq-batch-size`; their VRAM costs multiply.
- **Runs are being skipped in batch**  
  - `--resume` is **on** by default. Use `--no-resume` to force re-runs.
- **Slow frequencies**  
  - Set `--freq-batch-size 16` (or higher) — the `2×3N` displacements are independent
    and go through the model together instead of one at a time.  
  - Increase `--freq-delta` slightly (with care), or run fewer conformers at once.
- **Only one GPU is busy**  
  - Confirm the log shows `Splitting <file> into N structures`. If a job is a single
    structure there is nothing to fan out; raise `--workers-per-gpu` instead.
  - Check what `Detected GPUs:` actually lists. On a **MIG-partitioned** card those must
    be `MIG-<uuid>` tokens, not `[0, 1, 2, 3]`. Integer indices do not address MIG slices —
    every index resolves to the *first* slice — so a run that reports four GPUs would
    quietly put all four workers on one slice. `nvidia-smi -L` shows the real slices.
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

- **Unreleased**
  - **Relaxed bond scans** — `--scan I J FROM TO STEPS`. Walks the distance between
    two atoms, holding it fixed and minimizing everything else at each point, each
    point starting from the previous relaxed geometry. **Atoms are numbered from 1.**
    Emits the profile as CSV, the path as an XYZ trajectory, and the highest point
    on its own as a TS guess for `--optts`. Every point records the distance it
    actually reached, not the one requested, and warns when the two differ.
    Note for anyone using ASE constraints elsewhere with this model: UMA returns
    **float32** forces, and ASE's `FixBondLengths` enforces itself with a RATTLE
    iteration to a hard-coded 1e-13 tolerance that float32 cannot reach — it
    exhausts `maxiter` and raises on every force evaluation. `umadriver.scan`
    ships a float64 subclass that fixes it without loosening any tolerance.
  - **`optts` and `optimizer` can no longer disagree.** `optts` selects a saddle
    search and always used Sella, so any other `optimizer` was silently ignored and
    `optimizer: null` was ignored too — meaning a job written to say "frequencies on
    this TS, don't move it" re-ran the optimization and replaced the geometry, with
    nothing in the output to show it. `optts` now implies Sella; `--sp --optts`,
    `--optts --opt LBFGS`, and a manifest pairing `optimizer: null` with
    `optts: true` are all errors, the last one at load time before any GPU work.
    Frequencies on an unmoved saddle are `--sp --freq --freq-ts` (`optts: false` +
    `freq_ts: true`), and `--freq-ts` is newly exposed on the CLI.
  - **`--opt-mode Tight` was unreachable for some molecules.** Convergence was judged
    on raw Cartesian forces, which carry translation and rotation components UMA does
    not cancel exactly. Those change nothing about a structure, and Sella's internal
    coordinates cannot remove them at all — so a run could sit at a converged geometry
    forever, burning every cycle and reporting `converged=False`. On water the leftover
    torque is 6.0e-5 Eh/Bohr against Tight's 1.5e-5 cutoff, while the part of the
    gradient that actually deforms the molecule was already at 1.1e-8. Forces are now
    projected onto the rigid-body-free subspace before being scored. A run that does
    exhaust `--maxcycles` now logs which criteria it missed and by how much, instead of
    only `conv=False`.
  - **The lowest real vibration was being deleted from every minimum.** Rigid-body
    modes were identified as the 6 smallest `|f|` *excluding* the most negative one,
    which was reserved as "the imaginary mode". Finite-difference rotations routinely
    come out at a few negative cm⁻¹, so at a minimum that exclusion kept a rotation and
    zeroed a genuine vibration in its place — on water it deleted the 1623 cm⁻¹ bend
    (2.3 kcal/mol of ZPE, straight into *G*) and printed the leftover rotation as the
    lowest mode. A rotation noisier than −5 cm⁻¹ would additionally have been reported
    as `n_imag=1`, marking a minimum as a saddle. Rigid-body modes are now selected by
    their overlap with the translation/rotation subspace. **Thermochemistry from earlier
    runs is affected and should be recomputed**; the output format is unchanged.
  - **IRC now works.** It was written against a Sella API that does not exist (`IRC`
    takes `Atoms`, not a `Sella` object, and `run()` returns a bool, not coordinates),
    so `--irc` only ever produced a logged traceback. Both legs now run off one `IRC`
    instance so Sella restores the TS between directions, and the run emits a stitched
    `_irc_path.xyz` and an `_irc.csv` energy profile alongside the trajectories.
  - **Manifest jobs are split across GPUs.** Previously only positional/glob inputs were
    split, so a manifest of multi-conformer files ran one whole file per GPU with the
    rest idle.
  - **Split ensembles are aggregated.** `batch.py` used `csv` without importing it, and
    the failure was swallowed per member — so the top-level ranked `energies.csv` was
    silently never written, on any path.
  - **MIG slices are addressed correctly.** `CUDA_VISIBLE_DEVICES` tokens were coerced to
    `int`, which throws on `MIG-<uuid>` entries; the scheduler fell back to
    `torch.cuda.device_count()` and handed each worker an integer index. On a MIG card
    every integer index resolves to the *first* slice, so N workers silently shared one
    slice while the log reported N GPUs. Tokens are now preserved verbatim.
  - Added `--workers-per-gpu` (with the CPU thread budget divided across workers) and
    `--freq-batch-size` (batched finite-difference gradients).
  - `initialize_env()` is now actually called, so a populated cache means `HF_HUB_OFFLINE`
    and no Hub round-trip at model load; and the cache check no longer counts a purged
    symlink tree as populated.
  - Added a test suite (`pytest tests/`).
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

  # Single-point in implicit acetonitrile (no freq)
  - xyz: inputs/solv_only.xyz
    out_dir: runs/solv_only
    optimizer: null
    optts: false
    do_freq: false
    alpb: acetonitrile
```

# Example manifests

Five worked manifests, in the order worth reading them. Every one runs as-is
against the structures shipped in `tests/data/`, so you can confirm your install
works before pointing anything at a real system. Run them from the repository
root.

| File | What it teaches |
|---|---|
| [`01_optimize_conformers.yaml`](01_optimize_conformers.yaml) | Optimize an ensemble and rank it. Start here. |
| [`02_transition_state.yaml`](02_transition_state.yaml) | TS search, the frequency check that validates it, and the IRC. |
| [`03_implicit_solvent.yaml`](03_implicit_solvent.yaml) | ALPB solvation with `--alpb`, and how to read `solv_corr_kcal`. |
| [`04_phase1_optimize.yaml`](04_phase1_optimize.yaml) | Optimize a large batch across GPUs. |
| [`05_phase2_frequencies.yaml`](05_phase2_frequencies.yaml) | Frequencies and solution-phase thermochemistry on the survivors. |

```bash
umadriver --manifest examples/01_optimize_conformers.yaml --out-root runs
```

`umadriver batch --manifest ...` works too — the `batch` verb is accepted and
ignored.

## The schema in one minute

A manifest is a `jobs:` list plus an optional `common:` block:

```yaml
common:                    # defaults for every job
  model: uma-m-1p1
  device: cuda

jobs:
  - xyz: path/to/input.xyz
    out_dir: runs/whatever # optional; defaults to <out-root>/<stem>.ensemble
    overrides:             # settings for this job
      optimizer: Sella
      do_freq: true
```

Keys under `overrides:` are `run_conformer_workflow` keyword arguments — the same
things the CLI flags set, so `--opt-mode Tight` is `opt_mode: Tight` here. You can
also write them flat, directly under the job, which reads better for short
single-point entries; both styles work and can be mixed across jobs.

Precedence, lowest to highest:

```
CLI flags  <  manifest common:  <  per-job flat keys  <  per-job overrides:
```

So `umadriver --manifest jobs.yaml --freq` turns frequencies on everywhere except
in jobs that say otherwise.

## Two things that surprise people

**`optts: true` optimizes, even with `optimizer: null`.** It selects a saddle
search, not a labelling. For frequencies on a TS geometry you do not want moved,
use `optts: false` with `freq_ts: true` (CLI: `--sp --freq --freq-ts`). Example 05
spells this out. Writing `optimizer: null` next to `optts: true` used to run and
quietly replace your geometry; it is now rejected at manifest load, before any GPU
time is spent. `--sp --optts` and `--optts --opt LBFGS` are likewise refused.

**A TS is not a TS until the frequencies say so.** A saddle search converges to
plenty of things that are not first-order saddles. Check `n_imag == 1` in
`energies.csv` before using the number. The IRC is gated on this automatically and
will skip rather than follow a mode that is not there.

## The two-phase pattern

Frequencies cost 6N gradient calls per structure, so running them on a whole
ensemble is mostly wasted — you discard most conformers on energy anyway. Optimize
everything first (04), read `energies.csv`, then run frequencies on what survives
(05). Both phases resume: re-running skips jobs that already have an
`energies.csv`, so an interrupted batch picks up where it stopped.

## What lands in an output directory

```
runs/my_job/
  energies.csv                     ranked: energy_Eh, rel_kcal, gibbs_Eh, n_imag, solv_corr_kcal
  optimized_ranked.xyz             final geometries, ranked
  per_struct_<stem>/               one XYZ per conformer
  energies_per_conformer_*.csv     the resume ledger — delete to force a rerun
  freq_out/conf_*.out              ORCA-format frequencies + thermochemistry (with do_freq)
  irc/conf_*_irc_path.xyz          IRC path, both legs stitched (with irc)
  irc/conf_*_irc.csv               energy vs. arc length along the path
```

For the full flag list see the [main README](../README.md).

"""Multi-structure ensembles, end to end.

Job-list construction is covered in test_batching.py; this exercises the runs that
consume those job lists, including the split-ensemble aggregation that produces the
top-level ranked CSV.
"""

from __future__ import annotations

import os

import pytest
from ase.io import read as ase_read

from umadriver.batch import BatchCommon, run_batch_from_glob, run_batch_from_manifest
from umadriver.ensemble import run_conformer_workflow


def test_ensemble_ranks_all_three_conformers(
    tmp_path, h2o_ens_xyz, uma_calc, energies
):
    out = str(tmp_path / "ens")
    csv_path = run_conformer_workflow(
        h2o_ens_xyz,
        out_dir=out,
        optimizer="Sella",
        opt_mode="Loose",
        maxcycles=100,
        do_freq=False,
        calc=uma_calc,
    )

    rows = energies(csv_path)
    assert len(rows) == 3
    assert {r["tag"] for r in rows} == {"conf_0000", "conf_0001", "conf_0002"}

    e = [float(r["energy_Eh"]) for r in rows]
    assert e == sorted(e), "rows are not ranked by ascending energy"
    assert [int(r["rank"]) for r in rows] == [1, 2, 3]
    assert float(rows[0]["rel_kcal"]) == pytest.approx(0.0)

    per_struct = os.path.join(out, "per_struct_h2o_ens")
    assert len(os.listdir(per_struct)) == 3
    assert len(ase_read(os.path.join(out, "optimized_ranked.xyz"), index=":")) == 3


def test_resume_skips_completed_conformers(tmp_path, h2o_ens_xyz, uma_calc, energies):
    out = str(tmp_path / "ens")
    kwargs = dict(
        out_dir=out,
        optimizer="Sella",
        opt_mode="Loose",
        maxcycles=100,
        do_freq=False,
        calc=uma_calc,
        resume_from_per_conformer_csv=True,
    )

    first = energies(run_conformer_workflow(h2o_ens_xyz, **kwargs))

    per_conf = os.path.join(out, "energies_per_conformer_h2o_ens.csv")
    n_rows_before = len(open(per_conf).read().strip().splitlines())

    second = energies(run_conformer_workflow(h2o_ens_xyz, **kwargs))
    n_rows_after = len(open(per_conf).read().strip().splitlines())

    assert n_rows_after == n_rows_before, "resume re-ran and appended new rows"
    assert [r["energy_Eh"] for r in second] == [r["energy_Eh"] for r in first]


def test_glob_split_produces_members_and_aggregate(
    tmp_path, h2o_ens_xyz, energies, requires_model
):
    """Members under <stem>.ensemble/<label>/, ranked aggregate on top.

    Runs on CPU so the scheduler takes the serial loop rather than spawning
    worker processes, which is fragile under pytest. Three waters is seconds.
    """
    out_root = str(tmp_path / "runs")
    run_batch_from_glob(
        [h2o_ens_xyz],
        BatchCommon(out_root=out_root, device="cpu", resume=False),
        optimizer="Sella",
        opt_mode="Loose",
        maxcycles=30,
        do_freq=False,
    )

    ens = os.path.join(out_root, "h2o_ens.ensemble")
    for label in ("conf0000", "conf0001", "conf0002"):
        assert os.path.isfile(os.path.join(ens, label, "energies.csv"))

    agg = os.path.join(ens, "energies.csv")
    assert os.path.isfile(agg), "split ensemble was never aggregated"

    rows = energies(agg)
    assert len(rows) == 3
    e = [float(r["energy_Eh"]) for r in rows]
    assert e == sorted(e)
    assert [int(r["rank"]) for r in rows] == [1, 2, 3]

    assert len(ase_read(os.path.join(ens, "optimized_ranked.xyz"), index=":")) == 3


def test_manifest_split_fans_out_and_aggregates(
    tmp_path, h2o_ens_xyz, energies, requires_model
):
    """The regression that motivated all of this: manifest jobs used to run as one
    whole-file job on a single GPU, with no aggregate CSV."""
    out_dir = str(tmp_path / "uma_opt")
    manifest = tmp_path / "jobs.yaml"
    manifest.write_text(
        "jobs:\n"
        f"- xyz: {h2o_ens_xyz}\n"
        f"  out_dir: {out_dir}\n"
        "  overrides:\n"
        "    charge: 0\n"
        "    optimizer: Sella\n"
        "    opt_mode: Loose\n"
        "    maxcycles: 30\n"
    )

    run_batch_from_manifest(
        str(manifest),
        BatchCommon(out_root=str(tmp_path / "runs"), device="cpu", resume=False),
    )

    for label in ("conf0000", "conf0001", "conf0002"):
        assert os.path.isfile(os.path.join(out_dir, label, "energies.csv")), (
            f"{label} did not run as its own job"
        )

    rows = energies(os.path.join(out_dir, "energies.csv"))
    assert len(rows) == 3
    assert [int(r["rank"]) for r in rows] == [1, 2, 3]


def test_split_temp_files_are_cleaned_up(tmp_path, h2o_ens_xyz, requires_model):
    out_dir = str(tmp_path / "uma_opt")
    manifest = tmp_path / "jobs.yaml"
    manifest.write_text(
        f"jobs:\n- xyz: {h2o_ens_xyz}\n  out_dir: {out_dir}\n"
        "  overrides:\n    optimizer: null\n    do_freq: false\n"
    )

    run_batch_from_manifest(
        str(manifest),
        BatchCommon(out_root=str(tmp_path / "runs"), device="cpu", resume=False),
    )

    split_dir = os.path.join(out_dir, ".split")
    leftover = os.listdir(split_dir) if os.path.isdir(split_dir) else []
    assert leftover == [], f"temp split XYZs were not removed: {leftover}"

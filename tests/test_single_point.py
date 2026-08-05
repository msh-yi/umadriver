"""Single-point energies."""

from __future__ import annotations

import os

import numpy as np
import pytest
from ase.io import read as ase_read

from umadriver.ensemble import run_conformer_workflow


def test_single_point_writes_one_row(tmp_path, h2o_xyz, uma_calc, energies):
    out = str(tmp_path / "sp")
    csv_path = run_conformer_workflow(
        h2o_xyz,
        out_dir=out,
        charge=0,
        mult=1,
        optimizer=None,
        optts=False,
        do_freq=False,
        calc=uma_calc,
    )

    rows = energies(csv_path)
    assert len(rows) == 1

    r = rows[0]
    assert r["route"] == "SP"
    assert int(r["steps"]) == 0
    assert r["converged"] == "True"
    assert float(r["rel_kcal"]) == pytest.approx(0.0)

    E = float(r["energy_Eh"])
    assert np.isfinite(E) and E < 0.0


def test_single_point_does_not_move_the_geometry(tmp_path, h2o_xyz, uma_calc):
    """An SP must be a pure evaluation — no optimizer touches the coordinates."""
    out = str(tmp_path / "sp")
    before = ase_read(h2o_xyz).get_positions()

    run_conformer_workflow(
        h2o_xyz,
        out_dir=out,
        optimizer=None,
        optts=False,
        do_freq=False,
        calc=uma_calc,
    )

    written = os.path.join(out, "per_struct_h2o", "h2o_conf_0000.xyz")
    assert os.path.isfile(written)
    after = ase_read(written).get_positions()

    np.testing.assert_allclose(after, before, atol=1e-8)


def test_single_point_energy_is_reproducible(tmp_path, h2o_xyz, uma_calc, energies):
    """Same geometry, same model -> same energy. Guards against state leaking
    between runs through the cached predictor."""
    e = []
    for i in range(2):
        csv_path = run_conformer_workflow(
            h2o_xyz,
            out_dir=str(tmp_path / f"sp{i}"),
            optimizer=None,
            optts=False,
            do_freq=False,
            calc=uma_calc,
        )
        e.append(float(energies(csv_path)[0]["energy_Eh"]))

    assert e[0] == pytest.approx(e[1], abs=1e-9)


def test_charge_changes_the_energy(tmp_path, h2o_xyz, uma_calc, energies):
    """charge/mult reach the model via atoms.info; if they were dropped the
    anion and the neutral would come back identical."""
    out = {}
    for charge, mult in ((0, 1), (-1, 2)):
        csv_path = run_conformer_workflow(
            h2o_xyz,
            out_dir=str(tmp_path / f"q{charge}"),
            charge=charge,
            mult=mult,
            optimizer=None,
            optts=False,
            do_freq=False,
            calc=uma_calc,
        )
        out[charge] = float(energies(csv_path)[0]["energy_Eh"])

    assert out[0] != pytest.approx(out[-1], abs=1e-6)

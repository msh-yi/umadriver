"""IRC tracing from a transition state.

HCN <-> HNC is the fixture because its two endpoints are chemically distinct
isomers, which makes the "did the reverse leg actually restart from the TS"
question answerable rather than a tolerance argument.
"""

from __future__ import annotations

import csv
import os

import numpy as np
import pytest
from ase.io import read as ase_read
from ase.io.trajectory import Trajectory

from umadriver.ensemble import run_conformer_workflow


@pytest.fixture
def irc_run(tmp_path, hcn_ts_xyz, uma_calc):
    out = str(tmp_path / "irc")
    csv_path = run_conformer_workflow(
        hcn_ts_xyz,
        out_dir=out,
        charge=0,
        mult=1,
        optts=True,
        maxcycles=300,
        do_freq=True,
        irc=True,
        irc_dx=0.1,
        irc_steps=60,
        calc=uma_calc,
    )
    return out, csv_path


def _irc_dir(out):
    return os.path.join(out, "irc")


def test_irc_writes_both_trajectories(irc_run):
    out, _ = irc_run
    d = _irc_dir(out)

    fwd = os.path.join(d, "conf_0000_irc_traj_fwd.traj")
    rev = os.path.join(d, "conf_0000_irc_traj_rev.traj")
    assert os.path.isfile(fwd)
    assert os.path.isfile(rev)

    assert len(Trajectory(fwd)) >= 3
    assert len(Trajectory(rev)) >= 3


def test_irc_path_xyz_stitches_both_legs(irc_run):
    out, _ = irc_run
    d = _irc_dir(out)

    n_fwd = len(Trajectory(os.path.join(d, "conf_0000_irc_traj_fwd.traj")))
    n_rev = len(Trajectory(os.path.join(d, "conf_0000_irc_traj_rev.traj")))

    path = ase_read(os.path.join(d, "conf_0000_irc_path.xyz"), index=":")
    assert len(path) == n_rev + 1 + n_fwd


def test_irc_energy_falls_away_from_the_saddle(irc_run):
    """Monotonic downhill in both directions — that is what an IRC is."""
    out, _ = irc_run
    rows = list(csv.DictReader(open(os.path.join(_irc_dir(out), "conf_0000_irc.csv"))))

    ts = [r for r in rows if r["direction"] == "ts"]
    assert len(ts) == 1
    e_ts = float(ts[0]["energy_Eh"])

    for direction in ("forward", "reverse"):
        leg = [r for r in rows if r["direction"] == direction]
        assert leg, f"no {direction} frames"
        e = [float(r["energy_Eh"]) for r in leg]

        assert e[-1] < e_ts, f"{direction} endpoint is not below the TS"
        # allow a little slack for the first step off the saddle
        assert all(
            b <= a + 1e-6 for a, b in zip(e, e[1:])
        ), f"{direction} energy is not monotonically decreasing"


def test_irc_arc_length_is_monotonic(irc_run):
    out, _ = irc_run
    rows = list(csv.DictReader(open(os.path.join(_irc_dir(out), "conf_0000_irc.csv"))))

    for direction in ("forward", "reverse"):
        arcs = [float(r["arc_length_A"]) for r in rows if r["direction"] == direction]
        assert arcs == sorted(arcs)


def test_irc_endpoints_are_distinct_structures(irc_run):
    """The bug this replaces: both legs started from the same displaced geometry,
    so forward and reverse converged to the same place. For HCN<->HNC the two
    endpoints are different isomers."""
    out, _ = irc_run
    d = _irc_dir(out)

    fwd_end = Trajectory(os.path.join(d, "conf_0000_irc_traj_fwd.traj"))[-1]
    rev_end = Trajectory(os.path.join(d, "conf_0000_irc_traj_rev.traj"))[-1]

    rmsd = float(
        np.sqrt(
            ((fwd_end.get_positions() - rev_end.get_positions()) ** 2).sum(axis=1).mean()
        )
    )
    assert rmsd > 0.1, f"forward and reverse endpoints coincide (RMSD {rmsd:.4f} A)"


def test_irc_endpoints_are_hcn_and_hnc(irc_run):
    """Chemistry check: one endpoint should have H bound to C, the other to N."""
    out, _ = irc_run
    d = _irc_dir(out)

    def _h_partner(atoms):
        sym = atoms.get_chemical_symbols()
        h = sym.index("H")
        dists = {
            s: atoms.get_distance(h, i) for i, s in enumerate(sym) if s in ("C", "N")
        }
        return min(dists, key=dists.get)

    fwd_end = Trajectory(os.path.join(d, "conf_0000_irc_traj_fwd.traj"))[-1]
    rev_end = Trajectory(os.path.join(d, "conf_0000_irc_traj_rev.traj"))[-1]

    assert _h_partner(fwd_end) != _h_partner(rev_end), (
        "both IRC endpoints put H on the same heavy atom — the legs did not "
        "diverge into the two isomers"
    )


def test_ts_geometry_survives_the_irc(irc_run):
    """The legs walk `atoms` away from the saddle; the optimized TS geometry that
    was written out must still be the saddle, not an IRC endpoint."""
    out, _ = irc_run

    ts_xyz = os.path.join(out, "per_struct_hcn_ts", "hcn_ts_conf_0000.xyz")
    ts = ase_read(ts_xyz)

    path = ase_read(os.path.join(_irc_dir(out), "conf_0000_irc_path.xyz"), index=":")
    n_rev = len(Trajectory(os.path.join(_irc_dir(out), "conf_0000_irc_traj_rev.traj")))

    np.testing.assert_allclose(
        ts.get_positions(), path[n_rev].get_positions(), atol=1e-6
    )


def test_irc_skipped_when_not_a_saddle(tmp_path, h2o_xyz, uma_calc, energies):
    """An IRC from a minimum is meaningless. Previously this ran anyway whenever
    --optts was not set.

    Tight/300 is required to actually reach a minimum here: a looser optimization
    leaves water with a small imaginary mode, in which case n_imag == 1 and the
    gate is right to let the IRC through.
    """
    out = str(tmp_path / "min_irc")
    csv_path = run_conformer_workflow(
        h2o_xyz,
        out_dir=out,
        optimizer="Sella",
        opt_mode="Tight",
        maxcycles=300,
        do_freq=True,
        irc=True,
        calc=uma_calc,
    )

    assert int(energies(csv_path)[0]["n_imag"]) == 0, "fixture is not at a minimum"

    d = _irc_dir(out)
    produced = os.listdir(d) if os.path.isdir(d) else []
    assert produced == [], f"IRC ran on a minimum and wrote {produced}"

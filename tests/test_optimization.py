"""Geometry optimization.

Parametrized across optimizers to cover both branches of
``_minimize_atoms_inplace``: Sella is driven by a manual ``opt.step()`` loop, the
ASE optimizers by ``dyn.irun(fmax=1e-12, steps=maxcycles)`` with convergence judged
entirely by this package's Gaussian-style criteria.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from ase.io import read as ase_read

from umadriver.ensemble import run_conformer_workflow
from umadriver.utils import (
    force_metrics_HB,
    gaussian_cutoffs,
    internal_force_metrics_HB,
    project_out_rigid_body,
)

OPTIMIZERS = ["Sella", "LBFGS", "FIRE"]


# ------------------------------------------------- the convergence measure
def _water():
    from ase.build import molecule

    return molecule("H2O")


def test_a_pure_translation_projects_to_nothing():
    atoms = _water()
    forces = np.tile([0.3, -0.2, 0.7], (len(atoms), 1))

    left = project_out_rigid_body(atoms.get_positions(), forces)

    np.testing.assert_allclose(left, 0.0, atol=1e-12)


def test_a_pure_torque_projects_to_nothing():
    """The case that actually bit: UMA leaves a torque behind, and Sella's
    internal coordinates cannot remove it, so the run never converges."""
    atoms = _water()
    d = atoms.get_positions() - atoms.get_positions().mean(axis=0)
    forces = np.cross(np.array([0.0, 0.0, 0.4]), d)

    assert np.abs(forces).max() > 1e-3, "test would be vacuous with no torque"
    np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-12)  # no net force

    left = project_out_rigid_body(atoms.get_positions(), forces)

    np.testing.assert_allclose(left, 0.0, atol=1e-12)


def test_a_deforming_force_survives_projection():
    """The projection must not quietly shrink real forces — otherwise it would
    manufacture convergence instead of measuring it."""
    atoms = _water()
    forces = np.zeros((len(atoms), 3))
    forces[1] = [0.0, 0.5, 0.0]  # stretch one O-H
    forces[2] = [0.0, -0.5, 0.0]  # and compress the other: no net force, no torque

    left = project_out_rigid_body(atoms.get_positions(), forces)

    assert np.abs(left).max() > 0.4 * np.abs(forces).max()


def test_rigid_contamination_does_not_inflate_the_metric():
    """Adding a rigid-body force to a converged gradient must not change the
    number convergence is judged on."""
    atoms = _water()
    real = np.zeros((len(atoms), 3))
    real[1] = [0.0, 1e-6, 0.0]
    real[2] = [0.0, -1e-6, 0.0]

    d = atoms.get_positions() - atoms.get_positions().mean(axis=0)
    contaminated = real + np.cross(np.array([0.0, 0.0, 0.4]), d)

    clean = internal_force_metrics_HB(atoms.get_positions(), real)
    dirty = internal_force_metrics_HB(atoms.get_positions(), contaminated)

    np.testing.assert_allclose(dirty, clean, atol=1e-14)
    # and the raw metric is the one that would have been fooled
    assert force_metrics_HB(contaminated)[1] > 100 * force_metrics_HB(real)[1]


def test_projection_is_idempotent():
    atoms = _water()
    rng = np.random.default_rng(0)
    forces = rng.normal(size=(len(atoms), 3))

    once = project_out_rigid_body(atoms.get_positions(), forces)
    twice = project_out_rigid_body(atoms.get_positions(), once)

    np.testing.assert_allclose(twice, once, atol=1e-12)


def _sp_energy(tmp_path, xyz, calc, energies):
    csv_path = run_conformer_workflow(
        xyz,
        out_dir=str(tmp_path / "sp_ref"),
        optimizer=None,
        optts=False,
        do_freq=False,
        calc=calc,
    )
    return float(energies(csv_path)[0]["energy_Eh"])


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_optimization_converges_and_lowers_energy(
    tmp_path, h2o_xyz, uma_calc, energies, optimizer
):
    e_sp = _sp_energy(tmp_path, h2o_xyz, uma_calc, energies)

    csv_path = run_conformer_workflow(
        h2o_xyz,
        out_dir=str(tmp_path / f"opt_{optimizer}"),
        optimizer=optimizer,
        opt_mode="Loose",
        maxcycles=100,
        optts=False,
        do_freq=False,
        calc=uma_calc,
    )

    r = energies(csv_path)[0]
    assert r["route"] == "OPT"
    assert r["converged"] == "True", f"{optimizer} did not converge"
    assert int(r["steps"]) > 0

    e_opt = float(r["energy_Eh"])
    assert e_opt <= e_sp + 1e-9, "optimization raised the energy"


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_reported_convergence_is_real(
    tmp_path, h2o_xyz, uma_calc, energies, optimizer
):
    """Re-derive the convergence test from the final geometry rather than
    trusting the CSV flag."""
    out = str(tmp_path / f"opt_{optimizer}")
    run_conformer_workflow(
        h2o_xyz,
        out_dir=out,
        optimizer=optimizer,
        opt_mode="Loose",
        maxcycles=100,
        optts=False,
        do_freq=False,
        calc=uma_calc,
    )

    final = ase_read(os.path.join(out, "per_struct_h2o", "h2o_conf_0000.xyz"))
    final.calc = uma_calc
    grms, gmax = force_metrics_HB(final.get_forces())
    cuts = gaussian_cutoffs("Loose")

    assert grms < cuts.grms, f"RMS force {grms:.2e} exceeds {cuts.grms:.2e}"
    assert gmax < cuts.gmax, f"MAX force {gmax:.2e} exceeds {cuts.gmax:.2e}"


def test_optimization_writes_ranked_xyz(tmp_path, h2o_xyz, uma_calc):
    out = str(tmp_path / "opt")
    run_conformer_workflow(
        h2o_xyz,
        out_dir=out,
        optimizer="Sella",
        opt_mode="Loose",
        maxcycles=100,
        do_freq=False,
        calc=uma_calc,
    )

    ranked = os.path.join(out, "optimized_ranked.xyz")
    assert os.path.isfile(ranked)
    assert len(ase_read(ranked, index=":")) == 1


def test_tighter_mode_gives_smaller_residual_forces(
    tmp_path, h2o_xyz, uma_calc
):
    """--opt-mode has to actually change the convergence threshold."""
    residuals = {}
    for mode in ("Loose", "Tight"):
        out = str(tmp_path / f"opt_{mode}")
        run_conformer_workflow(
            h2o_xyz,
            out_dir=out,
            optimizer="Sella",
            opt_mode=mode,
            maxcycles=300,
            do_freq=False,
            calc=uma_calc,
        )
        final = ase_read(os.path.join(out, "per_struct_h2o", "h2o_conf_0000.xyz"))
        final.calc = uma_calc
        residuals[mode] = internal_force_metrics_HB(
            final.get_positions(), final.get_forces()
        )[0]

    assert residuals["Tight"] <= residuals["Loose"]


def test_tight_actually_converges(tmp_path, h2o_xyz, uma_calc, energies):
    """--opt-mode Tight has to be reachable, not just tighter.

    Water used to burn all 300 cycles and report converged=False: its residual
    Cartesian gradient is almost entirely a torque UMA does not quite cancel
    (6.0e-5 Eh/Bohr against a 1.5e-5 cutoff), while the part that deforms the
    molecule was already at 1.1e-8. Scoring rigid-body-projected forces makes the
    criterion measure the thing the optimizer can actually change.
    """
    csv_path = run_conformer_workflow(
        h2o_xyz,
        out_dir=str(tmp_path / "tight"),
        optimizer="Sella",
        opt_mode="Tight",
        maxcycles=300,
        do_freq=False,
        calc=uma_calc,
    )

    row = energies(csv_path)[0]
    assert row["converged"] == "True", "Tight is unreachable again"
    assert int(row["steps"]) < 300, "converged only by exhausting maxcycles"


def test_frequencies_on_a_minimum(tmp_path, h2o_xyz, uma_calc, energies):
    out = str(tmp_path / "optfreq")
    csv_path = run_conformer_workflow(
        h2o_xyz,
        out_dir=out,
        optimizer="Sella",
        opt_mode="Tight",
        maxcycles=300,
        do_freq=True,
        calc=uma_calc,
    )

    r = energies(csv_path)[0]
    assert int(r["n_imag"]) == 0, "a minimum must have no imaginary modes"
    assert r["imag_ok"] == "True"
    assert np.isfinite(float(r["gibbs_Eh"]))
    assert os.path.isfile(os.path.join(out, "freq_out", "conf_0000.out"))

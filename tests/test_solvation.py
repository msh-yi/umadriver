"""ALPB solvation correction.

The algebra and the guards need no GPU — EMT stands in for UMA purely to exercise
the mixing weights. The real-model cases live at the bottom and skip without a
checkpoint.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from umadriver.solvation import (
    ALPB_METHODS,
    SolvationUnavailable,
    SolvatedCalculator,
    base_calculator,
    make_solvated_calculator,
    solvation_correction_eV,
)

tblite = pytest.importorskip("tblite.ase", reason="tblite not installed")

from ase.build import molecule  # noqa: E402
from ase.calculators.emt import EMT  # noqa: E402
from tblite.ase import TBLite  # noqa: E402

EV_TO_KCAL = 23.060548867


def _solo(calc, name="H2O"):
    a = molecule(name)
    a.calc = calc
    return a.get_potential_energy(), a.get_forces()


# ---------------------------------------------------------------- algebra
def test_correction_is_base_plus_alpb_minus_vacuum():
    """E_tot = E_base + (E_alpb - E_vac), exactly."""
    atoms = molecule("H2O")
    atoms.calc = make_solvated_calculator(EMT(), "water", charge=0, mult=1)
    E = atoms.get_potential_energy()
    F = atoms.get_forces()

    e_b, f_b = _solo(EMT())
    e_a, f_a = _solo(TBLite(method="GFN2-xTB", solvation=("alpb", "water"), verbosity=0))
    e_v, f_v = _solo(TBLite(method="GFN2-xTB", verbosity=0))

    assert E == pytest.approx(e_b + e_a - e_v, abs=1e-10)
    np.testing.assert_allclose(F, f_b + f_a - f_v, atol=1e-10)


def test_concurrent_and_sequential_agree():
    """Concurrency is an execution detail; it must not change the numbers."""
    out = {}
    for concurrent in (False, True):
        a = molecule("H2O")
        a.calc = make_solvated_calculator(EMT(), "water", concurrent=concurrent)
        out[concurrent] = (a.get_potential_energy(), a.get_forces())

    assert out[True][0] == pytest.approx(out[False][0], abs=1e-12)
    np.testing.assert_allclose(out[True][1], out[False][1], atol=1e-12)


def test_solvation_correction_is_negative_for_a_neutral():
    """Sign check. E_base + E_alpb - E_vac and its inverse are equally
    self-consistent, so only the sign of the correction catches a swap."""
    atoms = molecule("H2O")
    atoms.calc = make_solvated_calculator(EMT(), "water")
    atoms.get_potential_energy()

    d = solvation_correction_eV(atoms.calc)
    assert d is not None
    kcal = d * EV_TO_KCAL
    assert kcal < 0.0, f"solvating a neutral should be favourable, got {kcal:+.2f}"
    assert -40.0 < kcal < -1.0, f"implausible magnitude: {kcal:+.2f} kcal/mol"


def test_correction_reported_without_extra_scf():
    """The term comes from Mixer's cached per-calculator contributions."""
    atoms = molecule("H2O")
    atoms.calc = make_solvated_calculator(EMT(), "water")
    atoms.get_potential_energy()

    contribs = atoms.calc.results["energy_contributions"]
    assert len(contribs) == 3
    assert solvation_correction_eV(atoms.calc) == pytest.approx(
        contribs[1] - contribs[2], abs=1e-12
    )


def test_different_solvents_give_different_corrections():
    vals = {}
    for solvent in ("water", "toluene"):
        a = molecule("H2O")
        a.calc = make_solvated_calculator(EMT(), solvent)
        a.get_potential_energy()
        vals[solvent] = solvation_correction_eV(a.calc)

    assert vals["water"] != pytest.approx(vals["toluene"], abs=1e-6)


# ---------------------------------------------------------------- guards
def test_wrapper_does_not_leak_fairchem_internals():
    """The batched-Hessian path talks to FAIRChem internals directly. If the
    wrapper forwarded them, the solvation term would be silently dropped from the
    Hessian and the result would look completely normal."""
    calc = make_solvated_calculator(EMT(), "water")
    assert not hasattr(calc, "a2g")
    assert not hasattr(calc, "predictor")


def test_batched_path_rejects_a_wrapped_calculator():
    from umadriver.vib_thermo import _is_fairchem_calculator

    assert not _is_fairchem_calculator(make_solvated_calculator(EMT(), "water"))
    assert not _is_fairchem_calculator(EMT())


def test_base_calculator_unwraps():
    inner = EMT()
    assert base_calculator(make_solvated_calculator(inner, "water")) is inner
    assert base_calculator(inner) is inner


def test_free_energy_survives_the_mix():
    """TBLite has no free_energy and Mixer intersects property sets, so the
    combination would drop it — breaking any force-consistent energy request."""
    atoms = molecule("H2O")
    atoms.calc = make_solvated_calculator(EMT(), "water")

    assert "free_energy" in atoms.calc.implemented_properties
    assert atoms.get_potential_energy(force_consistent=True) == pytest.approx(
        atoms.get_potential_energy()
    )


def test_rejects_method_without_alpb_parameters():
    with pytest.raises(ValueError, match="not parameterized"):
        make_solvated_calculator(EMT(), "water", method="GFN0-xTB")
    assert "GFN0-xTB" not in ALPB_METHODS


def test_missing_tblite_gives_an_actionable_error(monkeypatch):
    import umadriver.solvation as sol

    def _boom():
        raise SolvationUnavailable(sol._TBLITE_HINT)

    monkeypatch.setattr(sol, "_import_tblite", _boom)
    with pytest.raises(SolvationUnavailable, match="pip install tblite"):
        sol.make_solvated_calculator(EMT(), "water")


def test_charge_reaches_xtb():
    """xtb takes charge as a constructor argument, not from atoms.info — if it
    were dropped, the anion and the neutral would agree."""
    vals = {}
    for charge, mult in ((0, 1), (-1, 2)):
        a = molecule("H2O")
        a.calc = make_solvated_calculator(EMT(), "water", charge=charge, mult=mult)
        a.get_potential_energy()
        vals[charge] = solvation_correction_eV(a.calc)

    assert vals[0] != pytest.approx(vals[-1], abs=1e-6)
    # a charged solute is far more strongly solvated than a neutral
    assert vals[-1] < vals[0]


# ---------------------------------------------------------------- workflow
def test_workflow_energy_differs_from_gas_phase(tmp_path, h2o_xyz, uma_calc, energies):
    from umadriver.ensemble import run_conformer_workflow

    out = {}
    for label, alpb in (("gas", None), ("solv", "water")):
        csv_path = run_conformer_workflow(
            h2o_xyz,
            out_dir=str(tmp_path / label),
            optimizer="Sella",
            opt_mode="Loose",
            maxcycles=100,
            do_freq=False,
            alpb=alpb,
            calc=uma_calc,
        )
        out[label] = energies(csv_path)[0]

    assert float(out["gas"]["energy_Eh"]) != pytest.approx(
        float(out["solv"]["energy_Eh"]), abs=1e-8
    )
    assert out["gas"]["converged"] == "True"
    assert out["solv"]["converged"] == "True"

    corr = float(out["solv"]["solv_corr_kcal"])
    assert corr < 0.0
    assert out["gas"]["solv_corr_kcal"] in ("", None)


def test_resume_refuses_rows_from_a_different_solvation(
    tmp_path, h2o_xyz, uma_calc, energies
):
    """A stored gas-phase energy must not be reused for a solvated run."""
    from umadriver.ensemble import run_conformer_workflow

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

    gas = energies(run_conformer_workflow(h2o_xyz, alpb=None, **kwargs))[0]
    solv = energies(run_conformer_workflow(h2o_xyz, alpb="water", **kwargs))[0]

    assert float(solv["energy_Eh"]) != pytest.approx(
        float(gas["energy_Eh"]), abs=1e-8
    ), "solvated run reused the gas-phase energy"


def test_solvated_ts_keeps_one_imaginary_mode(tmp_path, hcn_ts_xyz, uma_calc, energies):
    from umadriver.ensemble import run_conformer_workflow

    csv_path = run_conformer_workflow(
        hcn_ts_xyz,
        out_dir=str(tmp_path / "ts"),
        optts=True,
        maxcycles=300,
        do_freq=True,
        alpb="water",
        calc=uma_calc,
    )
    r = energies(csv_path)[0]
    assert r["route"] == "TS"
    assert int(r["n_imag"]) == 1, "solvated TS is no longer a clean saddle"


def test_mixed_batched_forces_match_the_mixer(monkeypatch):
    """The batched Hessian re-implements the weighted sum outside the mixer.

    If those two ever disagree, the Hessian silently stops matching the forces
    that produced the geometry. This pins them together without needing a GPU:
    the base is faked as batchable, and the result is compared against the
    calculator's own get_forces().
    """
    import fairchem.core.datasets as fcd
    import umadriver.vib_thermo as vt
    from ase.build import molecule

    atoms = molecule("H2O")
    mixed = make_solvated_calculator(EMT(), "water", concurrent=False)
    atoms.calc = mixed

    positions = [atoms.get_positions() + 0.003 * i for i in range(5)]

    # reference: straight through the mixer, one geometry at a time
    reference = []
    for pos in positions:
        probe = molecule("H2O")
        probe.set_positions(pos)
        probe.calc = make_solvated_calculator(EMT(), "water", concurrent=False)
        reference.append(probe.get_forces())
    reference = np.array(reference)

    # make the EMT base look batchable, and give the batch path a trivial collater
    monkeypatch.setattr(fcd, "data_list_collater", lambda objs, otf_graph=True: objs)
    monkeypatch.setattr(
        vt, "_is_fairchem_calculator", lambda c: isinstance(c, EMT)
    )

    class _Array:
        """Stands in for the torch tensor `predict` returns."""

        def __init__(self, arr):
            self._arr = arr

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

    class _Pred:
        def predict(self, batch):
            forces = []
            for img in batch:
                img.calc = EMT()
                forces.append(img.get_forces())
            return {"forces": _Array(np.concatenate(forces, axis=0))}

    base = mixed.mixer.calcs[0]
    monkeypatch.setattr(base, "a2g", lambda img: img, raising=False)
    monkeypatch.setattr(base, "predictor", _Pred(), raising=False)

    got = vt._forces_for_geometries(atoms, positions, batch_size=3, xtb_workers=2)

    assert got.shape == reference.shape
    # Exact, because the batched path cold-starts every xtb SCF and the reference
    # uses a fresh calculator per geometry. If it ever warm-started instead, this
    # would drift by ~1e-6 relative and the Hessian would depend on the order the
    # displacements happened to be scheduled in.
    np.testing.assert_allclose(got, reference, atol=1e-10)


def test_mixer_parts_refuses_a_non_batchable_base():
    """Only a mix whose first component is a bare FAIRChemCalculator qualifies."""
    from umadriver.vib_thermo import _mixer_parts

    assert _mixer_parts(EMT()) is None
    assert _mixer_parts(make_solvated_calculator(EMT(), "water")) is None


def test_batching_under_alpb_matches_unbatched(tmp_path, hcn_ts_xyz, uma_calc):
    """Catches the silent raw-UMA fast path: if the batched route ignored the
    wrapper, these frequencies would differ."""
    from umadriver.ensemble import run_conformer_workflow
    from tests.test_ts_sella import parse_frequencies

    freqs = {}
    for bs in (1, 16):
        out = str(tmp_path / f"bs{bs}")
        run_conformer_workflow(
            hcn_ts_xyz,
            out_dir=out,
            optts=True,
            maxcycles=300,
            do_freq=True,
            freq_batch_size=bs,
            alpb="water",
            calc=uma_calc,
        )
        freqs[bs] = parse_frequencies(os.path.join(out, "freq_out", "conf_0000.out"))

    np.testing.assert_allclose(freqs[16], freqs[1], atol=0.1)

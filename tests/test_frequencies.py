"""Frequency computation.

Two layers:

1. The finite-difference math is checked against ase.vibrations.Vibrations, the
   reference implementation it replaces. This needs no model — a plain ASE
   calculator supplies forces, and with a non-fairchem calculator the batched path
   falls back to a serial force loop, so what is under test is exactly the
   displacement schedule, stencil, symmetrization and frequency conversion.

2. The real gate: with UMA, batched and unbatched must produce the same
   frequencies and the same ORCA-format output. --freq-batch-size must not default
   to >1 until this passes, because a downstream viewer parses those files.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from umadriver.ensemble import run_conformer_workflow
from umadriver.vib_thermo import (
    compute_hessian_fd,
    compute_mass_weighted_modes,
    frequencies_and_modes_from_hessian,
    order_modes,
    rigid_body_subspace,
)

TOL_CM1 = 0.1

ORCA_SECTIONS = [
    "VIBRATIONAL FREQUENCIES",
    "NORMAL MODES",
    "IR SPECTRUM",
    "THERMOCHEMISTRY AT",
    "****ORCA TERMINATED NORMALLY****",
]


def _ase_reference(atoms, delta, nfree):
    from ase.vibrations import Vibrations

    scratch = tempfile.mkdtemp()
    vib = Vibrations(atoms, name=os.path.join(scratch, "vib"), delta=delta, nfree=nfree)
    vib.run()
    raw = np.asarray(vib.get_frequencies())
    freqs = np.array(
        [-abs(f.imag) if abs(f.imag) > 1e-8 else f.real for f in raw], dtype=float
    )
    hess = vib.get_vibrations().get_hessian().reshape(3 * len(atoms), 3 * len(atoms))
    vib.clean()
    return freqs, hess


# ---------------------------------------------------------------- math layer
@pytest.mark.parametrize("name,nfree", [("H2O", 2), ("H2O", 4), ("CH4", 2)])
def test_fd_hessian_matches_ase(name, nfree):
    """Frequencies are the product here — this arithmetic must not drift."""
    from ase.build import molecule
    from ase.calculators.emt import EMT

    atoms = molecule(name)
    atoms.calc = EMT()
    delta = 0.01

    ref_freqs, ref_hess = _ase_reference(atoms, delta, nfree)

    H = compute_hessian_fd(atoms, delta=delta, nfree=nfree, batch_size=8)
    got_freqs, _modes, _w2 = frequencies_and_modes_from_hessian(H, atoms)

    np.testing.assert_allclose(H, ref_hess, atol=1e-10)
    np.testing.assert_allclose(
        np.sort(got_freqs), np.sort(ref_freqs), atol=TOL_CM1
    )


def test_fd_hessian_leaves_geometry_untouched():
    from ase.build import molecule
    from ase.calculators.emt import EMT

    atoms = molecule("H2O")
    atoms.calc = EMT()
    before = atoms.get_positions().copy()

    compute_hessian_fd(atoms, delta=0.01, nfree=2, batch_size=4)

    np.testing.assert_allclose(atoms.get_positions(), before, atol=1e-12)


def test_fd_hessian_is_symmetric():
    from ase.build import molecule
    from ase.calculators.emt import EMT

    atoms = molecule("H2O")
    atoms.calc = EMT()
    H = compute_hessian_fd(atoms, delta=0.01, nfree=2, batch_size=4)

    np.testing.assert_allclose(H, H.T, atol=1e-10)


def test_fd_hessian_rejects_unsupported_nfree():
    from ase.build import molecule
    from ase.calculators.emt import EMT

    atoms = molecule("H2O")
    atoms.calc = EMT()
    with pytest.raises(ValueError, match="nfree"):
        compute_hessian_fd(atoms, delta=0.01, nfree=3, batch_size=1)


def test_batch_size_does_not_change_the_result():
    """Chunking is an implementation detail; results must be identical."""
    from ase.build import molecule
    from ase.calculators.emt import EMT

    atoms = molecule("CH4")
    atoms.calc = EMT()

    hessians = [
        compute_hessian_fd(atoms, delta=0.01, nfree=2, batch_size=bs)
        for bs in (1, 4, 7, 1000)
    ]
    for H in hessians[1:]:
        np.testing.assert_allclose(H, hessians[0], atol=1e-12)


# ------------------------------------------------- which modes are rigid-body
def _orthonormal_complement(P: np.ndarray) -> np.ndarray:
    """Columns spanning everything P does not."""
    M = np.eye(P.shape[0]) - P @ P.T
    U, s, _ = np.linalg.svd(M)
    return U[:, s > 0.5]


def _modes_and_freqs(atoms, rigid_freqs, vib_freqs):
    """A synthetic mode set: exact rigid-body vectors at `rigid_freqs`, and
    genuine vibrations at `vib_freqs`. Nothing to diagonalize — the pairing of
    frequency to mode character is the whole point, so it is stated directly."""
    P = rigid_body_subspace(atoms)
    assert P.shape[1] == len(rigid_freqs)
    modes = np.column_stack([P, _orthonormal_complement(P)])
    return modes, np.array(list(rigid_freqs) + list(vib_freqs), dtype=float)


def test_rigid_modes_are_chosen_by_character_not_by_smallest_frequency():
    """Regression: the water bend used to be deleted from every H2O frequency job.

    These are the nine frequencies UMA actually produced for the optimized test
    water. Six are rigid-body (rotations pick up real curvature at a geometry
    that is not perfectly stationary — one of them lands at 260 cm^-1), and three
    are vibrations. Selecting the rigid block by smallest |f| while reserving the
    most negative mode as "the imaginary" zeroed the 1623 cm^-1 bend instead of
    the -3.63 cm^-1 rotation, costing 2.3 kcal/mol of ZPE and reporting the
    leftover rotation as the molecule's lowest vibration.
    """
    from ase.build import molecule

    atoms = molecule("H2O")
    rigid = [-3.63, -1.01, -0.18, 27.84, 29.32, 260.46]
    vib = [1622.98, 3822.10, 3911.82]
    modes, freqs = _modes_and_freqs(atoms, rigid, vib)

    perm, zero_first = order_modes(freqs, modes, atoms, ts=False)

    assert zero_first == 6
    assert sorted(freqs[perm][:zero_first]) == sorted(rigid), "wrong modes zeroed"
    np.testing.assert_allclose(freqs[perm][zero_first:], vib)


def test_a_negative_rigid_mode_is_not_reported_as_imaginary():
    """The same bug's other face: a rotation at -3.63 cm^-1 was promoted out of
    the rigid block, and had it been noisier than -5 cm^-1 it would have been
    counted as an imaginary frequency — turning a minimum into a fake saddle."""
    from ase.build import molecule

    atoms = molecule("H2O")
    modes, freqs = _modes_and_freqs(
        atoms, [-12.0, -4.0, -0.2, 3.0, 8.0, 40.0], [1600.0, 3800.0, 3900.0]
    )

    perm, zero_first = order_modes(freqs, modes, atoms, ts=False)
    printed = freqs[perm]
    printed[:zero_first] = 0.0

    assert sum(1 for f in printed if f < 0.0) == 0


def test_ts_imaginary_mode_still_leads_the_vibrations():
    from ase.build import molecule

    atoms = molecule("H2O")
    modes, freqs = _modes_and_freqs(
        atoms, [-2.0, -0.5, 0.1, 15.0, 18.0, 40.0], [-1200.0, 900.0, 1500.0]
    )

    perm, zero_first = order_modes(freqs, modes, atoms, ts=True)

    assert zero_first == 6
    assert freqs[perm][zero_first] == -1200.0


def test_linear_molecules_get_five_rigid_modes():
    from ase.build import molecule

    atoms = molecule("CO2")
    assert rigid_body_subspace(atoms).shape[1] == 5

    modes, freqs = _modes_and_freqs(
        atoms, [-1.0, -0.3, 0.2, 6.0, 9.0], [667.0, 668.0, 1333.0, 2349.0]
    )
    perm, zero_first = order_modes(freqs, modes, atoms, ts=False)

    assert zero_first == 5
    np.testing.assert_allclose(freqs[perm][zero_first:], [667.0, 668.0, 1333.0, 2349.0])


# ---------------------------------------------------------------- UMA gate
def test_batched_frequencies_match_serial_with_uma(tmp_path, h2o_xyz, uma_calc):
    """THE GATE. Do not default --freq-batch-size above 1 until this passes.

    Both batch sizes must see *the same geometry*, so the optimization happens
    once, up front. Running it inside each branch does not guarantee that — the
    two are separate optimizations, and any difference between where they stop
    gets attributed to batching. That is exactly what happened here: neither run
    converged (the convergence measure was scoring rigid-body forces the
    optimizer could not remove), so each stopped wherever cycle 300 left it,
    0.074 kcal/mol and a visibly different structure apart.
    """
    from tests.test_ts_sella import parse_frequencies

    opt = str(tmp_path / "opt")
    run_conformer_workflow(
        h2o_xyz,
        out_dir=opt,
        optimizer="Sella",
        opt_mode="Normal",
        maxcycles=300,
        do_freq=False,
        calc=uma_calc,
    )
    geometry = os.path.join(opt, "optimized_ranked.xyz")

    freqs = {}
    for bs in (1, 16):
        out = str(tmp_path / f"freq_bs{bs}")
        run_conformer_workflow(
            geometry,
            out_dir=out,
            optimizer=None,
            do_freq=True,
            freq_batch_size=bs,
            calc=uma_calc,
        )
        freqs[bs] = parse_frequencies(os.path.join(out, "freq_out", "conf_0000.out"))

    assert len(freqs[1]) == len(freqs[16])
    np.testing.assert_allclose(freqs[16], freqs[1], atol=TOL_CM1)


def test_batched_ts_frequencies_match_serial_with_uma(tmp_path, hcn_ts_xyz, uma_calc):
    """Same gate on a saddle — the imaginary mode is the fragile case."""
    from tests.test_ts_sella import parse_frequencies

    freqs = {}
    for bs in (1, 16):
        out = str(tmp_path / f"ts_bs{bs}")
        run_conformer_workflow(
            hcn_ts_xyz,
            out_dir=out,
            optts=True,
            maxcycles=300,
            do_freq=True,
            freq_batch_size=bs,
            calc=uma_calc,
        )
        freqs[bs] = parse_frequencies(os.path.join(out, "freq_out", "conf_0000.out"))

    np.testing.assert_allclose(freqs[16], freqs[1], atol=TOL_CM1)
    assert sum(1 for f in freqs[16] if f < 0) == 1


class _Array:
    """Minimal stand-in for the torch tensor `predict` returns."""

    def __init__(self, arr):
        self._arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


def test_batch_oom_backs_off_instead_of_failing(monkeypatch):
    """An oversized batch must degrade, not kill the frequency calculation.

    The workable batch size depends on system size and card — 16 is fine for a
    small molecule and OOMs at 170 atoms on a 20 GB MIG slice — so it cannot be
    picked correctly up front.
    """
    import fairchem.core.datasets as fcd
    import umadriver.vib_thermo as vt
    from ase.build import molecule
    from ase.calculators.emt import EMT
    from umadriver.vib_thermo import _forces_for_geometries

    monkeypatch.setattr(fcd, "data_list_collater", lambda objs, otf_graph=True: objs)
    # The batch path is deliberately gated on the calculator being a genuine
    # FAIRChemCalculator (so a wrapper can never silently bypass its own
    # contribution). Open that gate here rather than weakening it in production.
    monkeypatch.setattr(vt, "_is_fairchem_calculator", lambda calc: True)

    attempted = []

    class RefusesBatchesOverFour:
        """Looks like a FAIRChemCalculator to _forces_for_geometries.

        Batched images are produced by atoms.copy(), which does not carry a
        calculator, so this attaches one before evaluating.
        """

        def __init__(self, real):
            self._real = real

        a2g = staticmethod(lambda img: img)

        @property
        def predictor(self):
            return self

        def predict(self, batch):
            attempted.append(len(batch))
            if len(batch) > 4:
                raise RuntimeError("CUDA out of memory. Tried to allocate 1024.00 MiB")
            forces = []
            for img in batch:
                img.calc = self._real
                forces.append(img.get_forces())
            return {"forces": _Array(np.concatenate(forces, axis=0))}

    atoms = molecule("H2O")
    atoms.calc = EMT()
    positions = [atoms.get_positions() + 0.01 * i for i in range(9)]
    reference = np.array([_forces_at(atoms, p) for p in positions])

    atoms.calc = RefusesBatchesOverFour(EMT())
    out = _forces_for_geometries(atoms, positions, batch_size=16)

    assert max(attempted) > 4, "should have tried the requested size first"
    assert attempted[-1] <= 4, "should have backed off to a size that fits"
    assert out.shape == (9, 3, 3)
    np.testing.assert_allclose(out, reference, atol=1e-10)


def _forces_at(template, positions):
    probe = template.copy()
    probe.calc = template.calc
    probe.set_positions(positions)
    return probe.get_forces()


@pytest.mark.big
def test_batched_frequencies_match_serial_on_catalyst(
    tmp_path, catalyst_ts_xyz, uma_calc
):
    """The gate at a size where batching actually matters: 510 modes, 1020
    displacements.

    The small gates cannot exercise chunking at all — 6N is 18 for a 3-atom
    molecule, so everything fits in a single batch and the loop that splits work
    across batches never runs.

    The fixture is already a converged TS, so this skips the optimization
    (optimizer=None) and uses freq_ts=True to tell the frequency/thermo path to
    expect one imaginary mode — the same freq-only pattern as
    sample_jobs_freq_phase3.yaml.
    """
    from tests.test_ts_sella import parse_frequencies

    freqs = {}
    for bs in (1, 16):
        out = str(tmp_path / f"cat_bs{bs}")
        run_conformer_workflow(
            catalyst_ts_xyz,
            out_dir=out,
            charge=0,
            mult=1,
            optimizer=None,
            optts=False,
            freq_ts=True,
            do_freq=True,
            freq_batch_size=bs,
            calc=uma_calc,
        )
        freqs[bs] = parse_frequencies(os.path.join(out, "freq_out", "conf_0000.out"))

    assert len(freqs[1]) == 3 * 170
    np.testing.assert_allclose(freqs[16], freqs[1], atol=TOL_CM1)

    # and the physics survives batching
    assert sum(1 for f in freqs[16] if f < 0) == 1


def test_water_has_three_real_vibrations(tmp_path, h2o_xyz, uma_calc):
    """End-to-end physics check on the smallest possible case.

    Every batching gate compares one code path against another, so both can be
    wrong together — and were: water was reported with a bend of 3.55 cm^-1
    instead of ~1600, in both. Assert the answer, not just its reproducibility.
    """
    from tests.test_ts_sella import parse_frequencies

    out = str(tmp_path / "freq")
    run_conformer_workflow(
        h2o_xyz,
        out_dir=out,
        optimizer="Sella",
        opt_mode="Normal",
        maxcycles=300,
        do_freq=True,
        calc=uma_calc,
    )
    freqs = parse_frequencies(os.path.join(out, "freq_out", "conf_0000.out"))

    assert freqs[:6] == [0.0] * 6, "expected 6 rigid-body zeros"
    bend, sym, asym = freqs[6:]
    assert 1400 < bend < 1800, f"H-O-H bend is missing or wrong: {bend} cm-1"
    assert 3400 < sym < 4100 and 3400 < asym < 4100, f"O-H stretches: {sym}, {asym}"


def test_orca_output_sections_are_present(tmp_path, h2o_xyz, uma_calc):
    """Format regression guard for the downstream viewer."""
    out = str(tmp_path / "freq")
    run_conformer_workflow(
        h2o_xyz,
        out_dir=out,
        optimizer="Sella",
        opt_mode="Tight",
        maxcycles=300,
        do_freq=True,
        calc=uma_calc,
    )

    txt = open(os.path.join(out, "freq_out", "conf_0000.out")).read()
    for section in ORCA_SECTIONS:
        assert section in txt, f"missing ORCA section: {section}"

    from tests.test_ts_sella import parse_frequencies

    freqs = parse_frequencies(os.path.join(out, "freq_out", "conf_0000.out"))
    assert len(freqs) == 9  # 3N for water

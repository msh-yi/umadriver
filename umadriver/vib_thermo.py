from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import logging
import math, numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
from ase import Atoms
from ase.vibrations import Vibrations
import os
import glob
from typing import Optional

from .constants import (
    HARTREE_PER_EV,
    EV_PER_HARTREE,
    KCAL_PER_MOL_PER_EV,
    kB_J_K,
    h_J_s,
    c_cm_s,
    amu_kg,
    angstrom_m,
    kB_eV_K,
    eV_per_cm1,
    theta_per_cm1_K,
)
from .writer import ORCAWriter

LOG = logging.getLogger("uma.vib")

# Fallback-safe Avogadro
try:
    from . import constants as _const_mod

    AVOGADRO = getattr(_const_mod, "AVOGADRO", 6.02214076e23)
except Exception:
    AVOGADRO = 6.02214076e23


# The only solvents the free-volume correction knows about. Exported so the CLI can
# restrict --free-volume-solvent to them: an unrecognised name here silently means
# "the whole litre is free", i.e. no correction at all, which is easy to miss.
FREE_VOLUME_SOLVENTS = ["none", "H2O", "toluene", "DMF", "AcOH", "chloroform"]


def _free_space_mL_per_L(solv: Optional[str]) -> float:
    """
    Return accessible free space (mL per L) for a solute in bulk solvent.
    Based on Shakhnovich & Whitesides (J. Org. Chem. 1998, 63, 3821) and
    the GoodVibes implementation.
    """
    if not solv:
        return 1000.0

    solvent_list = FREE_VOLUME_SOLVENTS
    molarity = [1.0, 55.6, 9.4, 12.9, 17.4, 12.5]  # mol/L
    molecular_vol = [1.0, 27.944, 149.070, 77.442, 86.10, 97.0]  # Å^3

    try:
        i = solvent_list.index(solv)
    except ValueError:
        # Unknown solvent -> assume full liter is “free”
        LOG.warning(
            "Free-volume solvent %r is not in %s; treating the whole litre as free "
            "(no entropy correction).",
            solv,
            solvent_list,
        )
        return 1000.0

    if i == 0:  # 'none'
        return 1000.0

    solv_m = molarity[i]
    solv_volA3 = molecular_vol[i]
    # v_free (Å^3 per molecule) for accessible volume
    v_free = (
        8.0
        * ((1e27 / (solv_m * AVOGADRO)) ** (1.0 / 3.0) - solv_volA3 ** (1.0 / 3.0)) ** 3
    )
    # Convert to mL free space per liter of bulk solvent
    freespace_mL_per_L = v_free * solv_m * AVOGADRO * 1e-24
    return float(freespace_mL_per_L)


# ---------- Geometry classification / rotation ----------
def principal_moments_amuA2(atoms: Atoms) -> np.ndarray:
    return atoms.get_moments_of_inertia()


def rotational_constants_cm1_from_I(I_amuA2: np.ndarray) -> Tuple[float, float, float]:
    I_SI = I_amuA2 * amu_kg * (angstrom_m**2)
    B_cm = []
    for I in I_SI:
        B_cm.append(0.0 if I <= 0 else h_J_s / (8.0 * math.pi**2 * c_cm_s * I))
    return (B_cm[0], B_cm[1], B_cm[2])


def classify_geometry(I_amuA2: np.ndarray) -> str:
    """
    Return one of: 'atom', 'linear', 'nonlinear'
    - 'atom' if all principal moments are (near) zero
    - 'linear' if one moment is ~0 but not all (diatomics, etc.)
    - 'nonlinear' otherwise
    """
    I = np.array(I_amuA2, dtype=float)
    imax = float(np.max(np.abs(I)))
    # Treat as atom if all moments are essentially zero
    if imax < 1e-12:
        return "atom"
    if (float(np.min(np.abs(I))) / imax) < 1e-6:
        return "linear"
    return "nonlinear"


# ---------- Mass-weighted normal modes ----------
def compute_mass_weighted_modes(vib, atoms: Atoms) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (U, w2) where:
      U: (3N, 3N) mass-weighted eigenvectors (columns are modes)
      w2: (3N,) eigenvalues of the dynamical matrix (ascending)
    Uses the same Hessian as ASE Vibrations, so ordering aligns with ASE freqs.
    """
    vd = vib.get_vibrations()  # VibrationsData
    K4 = vd.get_hessian()  # (N,3,N,3) in eV/Å^2
    N = len(atoms)
    K = K4.reshape(3 * N, 3 * N)

    masses = np.repeat(atoms.get_masses(), 3)  # amu
    inv_sqrt_m = 1.0 / np.sqrt(masses)
    # dynamical matrix D = M^{-1/2} K M^{-1/2}
    D = (inv_sqrt_m[:, None]) * K * (inv_sqrt_m[None, :])
    D = 0.5 * (D + D.T)

    w2, U = np.linalg.eigh(D)  # ascending by eigenvalue
    return U, w2


def rigid_body_subspace(atoms: Atoms) -> np.ndarray:
    """Orthonormal basis (3N x k) of the translation/rotation subspace.

    In mass-weighted coordinates q_i = sqrt(m_i) r_i, a rigid displacement
    dr_i = t + w x (r_i - r_cm) becomes sqrt(m_i) * dr_i. k is 6 for a general
    molecule, 5 for a linear one and 3 for a single atom — the SVD drops the
    columns that are numerically null rather than relying on a geometry test.
    """
    m = np.asarray(atoms.get_masses(), dtype=float)
    sm = np.sqrt(m)
    pos = atoms.get_positions()
    com = (m[:, None] * pos).sum(axis=0) / m.sum()
    d = pos - com

    n = len(atoms)
    B = np.zeros((3 * n, 6), dtype=float)
    for k in range(3):
        v = np.zeros((n, 3))
        v[:, k] = 1.0
        B[:, k] = (sm[:, None] * v).ravel()
    for k in range(3):
        e = np.zeros(3)
        e[k] = 1.0
        B[:, 3 + k] = (sm[:, None] * np.cross(e, d)).ravel()

    U, s, _ = np.linalg.svd(B, full_matrices=False)
    if s[0] <= 0.0:
        return U[:, :0]
    return U[:, s > 1e-8 * s[0]]


def rigid_mode_indices(modes_mw: np.ndarray, atoms: Atoms, count: int) -> List[int]:
    """Indices of the ``count`` modes that are most nearly rigid-body motions.

    Selection is by overlap with the translation/rotation subspace — the actual
    definition of a rigid-body mode — rather than by which frequencies happen to
    be smallest. At a geometry that is not perfectly converged, rotations pick up
    real curvature (on the test water they land at 260 cm^-1), so ordering by
    |f| can rank a genuine low-frequency vibration below them and delete it.
    """
    if count <= 0:
        return []
    P = rigid_body_subspace(atoms)
    if P.shape[1] == 0:
        return list(range(min(count, modes_mw.shape[1])))
    overlap = np.linalg.norm(P.T @ modes_mw, axis=0) ** 2
    return [int(i) for i in np.argsort(-overlap)[:count]]


def order_modes(
    freqs_cm: np.ndarray, modes_mw: np.ndarray, atoms: Atoms, *, ts: bool
) -> Tuple[List[int], int]:
    """(permutation, number of leading rigid-body modes) for printing.

    The permutation is: rigid-body modes first, then — if this is a TS — the
    imaginary mode, then everything else by increasing |f|.

    The previous rule took the ``zero_first`` smallest |f| while *unconditionally*
    excluding the most negative mode, so that it stayed available as "the
    imaginary". Rigid-body modes routinely come out at a few negative cm^-1 from
    finite-difference noise, so at a minimum that exclusion reserved a rotation
    and pushed a genuine vibration into the zeroed block. On water it deleted the
    1623 cm^-1 bend — 2.3 kcal/mol of ZPE — and reported the leftover rotation as
    the lowest mode; a rotation noisier than -5 cm^-1 would also have been counted
    as an imaginary frequency. Identifying the rigid block by overlap removes the
    need for that exclusion: at a TS the imaginary mode has essentially no
    rigid-body character, so it is never selected as rigid in the first place.
    """
    geom = classify_geometry(principal_moments_amuA2(atoms))
    zero_first = {"atom": 3, "linear": 5}.get(geom, 6)

    nmode = len(freqs_cm)
    rigid = rigid_mode_indices(modes_mw, atoms, zero_first)
    zero_first = len(rigid)

    is_rigid = set(rigid)
    rest = [i for i in range(nmode) if i not in is_rigid]
    negative = [i for i in rest if freqs_cm[i] < 0.0]
    idx_imag = min(negative, key=lambda i: freqs_cm[i]) if negative else None

    perm = list(rigid)
    if ts and idx_imag is not None:
        perm.append(idx_imag)
        rest = [i for i in rest if i != idx_imag]
    perm.extend(sorted(rest, key=lambda i: abs(freqs_cm[i])))
    return perm, zero_first


def _is_fairchem_calculator(calc) -> bool:
    """True only for a bare FAIRChemCalculator, never a wrapper around one."""
    try:
        from fairchem.core import FAIRChemCalculator
    except Exception:
        return False
    return type(calc) is FAIRChemCalculator


def _is_oom(exc: BaseException) -> bool:
    """CUDA out-of-memory, however torch chooses to spell it this version."""
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    return "out of memory" in str(exc).lower()


def _free_cuda():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _forces_for_geometries(
    atoms: Atoms,
    positions_list: List[np.ndarray],
    batch_size: int,
    xtb_workers: Optional[int] = None,
) -> np.ndarray:
    """Forces (eV/Å) for a list of displaced geometries — shape (M, N, 3).

    Finite-difference displacements are independent, so they can go through the
    model together. ASE's Vibrations evaluates them one at a time, which for a
    170-atom molecule is 1020 sequential forward passes.

    Batching goes through the same path FAIRChemCalculator.calculate uses
    internally (a2g -> data_list_collater -> predictor.predict), just with more
    than one structure in the batch. Any other calculator falls back to the serial
    loop, which keeps this usable with plain ASE calculators.
    """
    calc = atoms.calc
    if calc is None:
        raise ValueError("atoms has no calculator")

    # A mixed (solvated) calculator: batch its base on the GPU and evaluate the
    # remaining contributions on CPU threads, overlapping the two.
    mixed = _mixer_parts(calc)
    if batch_size > 1 and mixed is not None:
        return _mixed_forces(atoms, mixed, positions_list, batch_size, xtb_workers)

    # Deliberately an identity check, not duck-typing on `a2g`/`predictor`.
    #
    # The batch path below bypasses the ASE Calculator interface and talks to
    # FAIRChem internals directly. Any wrapper that merely *forwards* those two
    # attributes would therefore have its own contribution silently skipped,
    # yielding a Hessian built from bare UMA forces that looks entirely normal.
    can_batch = batch_size > 1 and _is_fairchem_calculator(calc)

    if batch_size > 1 and not can_batch:
        LOG.warning(
            "  [FRQ] batch_size=%d requested but the calculator is %s, which cannot "
            "use the batched path. Falling back to one displacement at a time — "
            "correct, but slower.",
            batch_size,
            type(calc).__name__,
        )

    if not can_batch:
        return _serial_forces(atoms, calc, positions_list)

    return _batched_fairchem_forces(atoms, calc, positions_list, batch_size)


def _serial_forces(atoms: Atoms, calc, positions_list) -> np.ndarray:
    """One ASE force call per geometry. Always correct, never fast."""
    out = np.empty((len(positions_list), len(atoms), 3), dtype=float)
    probe = atoms.copy()
    probe.calc = calc
    for i, pos in enumerate(positions_list):
        probe.set_positions(pos)
        out[i] = probe.get_forces()
    return out


def _batched_fairchem_forces(
    atoms: Atoms, calc, positions_list, batch_size: int
) -> np.ndarray:
    """Many displaced geometries through one FAIRChem forward pass at a time."""
    from fairchem.core.datasets import data_list_collater

    out = np.empty((len(positions_list), len(atoms), 3), dtype=float)
    n_at = len(atoms)
    bs = batch_size
    start = 0

    while start < len(positions_list):
        chunk = positions_list[start : start + bs]
        images = []
        for pos in chunk:
            img = atoms.copy()
            img.set_positions(pos)
            img.info.update(atoms.info)  # charge/spin must ride along
            images.append(img)

        try:
            batch = data_list_collater(
                [calc.a2g(img) for img in images], otf_graph=True
            )
            pred = calc.predictor.predict(batch)
            forces = pred["forces"].detach().cpu().numpy()
        except Exception as e:
            # A batch that does not fit is a tuning problem, not a failure: back off
            # and keep going. Without this the whole frequency calculation dies, and
            # the workable batch size depends on system size and card — 16 is fine
            # for a small molecule and OOMs at 170 atoms on a 20 GB MIG slice.
            if bs > 1 and _is_oom(e):
                bs = max(1, bs // 2)
                LOG.warning(
                    "  [FRQ] batch did not fit in GPU memory; halving batch_size to %d",
                    bs,
                )
                _free_cuda()
                continue
            raise

        # forces come back concatenated over the batch
        for j in range(len(chunk)):
            out[start + j] = forces[j * n_at : (j + 1) * n_at]
        start += len(chunk)

    return out


def _mixer_parts(calc):
    """(base, base_weight, extra_calcs, extra_weights) for a batchable mixed calc.

    Returns None unless the calculator is a weighted mix whose *first* component is
    a bare FAIRChemCalculator — the only component the batch path can accelerate.

    The weights are read off the mixer rather than assumed, so the batched Hessian
    and the mixer's own `get_forces()` share one definition of the combination. Only
    the loop is duplicated, not the arithmetic.
    """
    mixer = getattr(calc, "mixer", None)
    if mixer is None:
        return None
    calcs = list(getattr(mixer, "calcs", []) or [])
    weights = list(getattr(mixer, "weights", []) or [])
    if len(calcs) < 2 or len(calcs) != len(weights):
        return None
    if not _is_fairchem_calculator(calcs[0]):
        return None
    factory = getattr(calc, "new_extra_calculators", None)
    return calcs[0], weights[0], calcs[1:], weights[1:], factory


def _extra_forces_threaded(
    atoms: Atoms, calcs, weights, positions_list, workers: int, factory=None
) -> np.ndarray:
    """Weighted sum of the non-batchable contributions, spread over threads.

    xtb parallelizes poorly *within* a call (24 threads buys ~1.4x on 170 atoms) but
    the displacements are completely independent, so throughput comes from running
    many single-threaded calls at once instead.

    Each worker gets its *own* calculator instances. Sharing one across threads
    segfaults — ASE calculators hold mutable per-call state and tblite caches an
    API object — whereas separate instances give bit-identical results.
    """
    out = np.zeros((len(positions_list), len(atoms), 3), dtype=float)

    if factory is None:
        # No way to make per-thread copies, so threading would be unsafe.
        workers = 1

    local = threading.local()

    def _worker_calcs():
        if factory is None:
            return calcs
        got = getattr(local, "calcs", None)
        if got is None:
            # deterministic=True: a Hessian differences forces at +delta and
            # -delta, so an SCF warm-started from whichever geometry this worker
            # happened to see last would bias the result in a way that does not
            # cancel. Cold-starting each geometry costs ~1.5x here and is hidden
            # behind the GPU batches anyway.
            got = factory(True)
            local.calcs = got
        return got

    def _one(i_pos):
        i, pos = i_pos
        _limit_openmp_threads_here()
        mine = _worker_calcs()
        acc = np.zeros((len(atoms), 3), dtype=float)
        for c, w in zip(mine, weights):
            probe = atoms.copy()
            probe.set_positions(pos)
            probe.info.update(atoms.info)
            probe.calc = c
            reset = getattr(c, "reset", None)
            if reset is not None:
                reset()  # drop the previous wavefunction; force a cold SCF
            acc += w * probe.get_forces()
        return i, acc

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, acc in pool.map(_one, list(enumerate(positions_list))):
            out[i] = acc
    return out


def _mixed_forces(
    atoms: Atoms, parts, positions_list, batch_size: int, xtb_workers: Optional[int]
) -> np.ndarray:
    """Batch the base on the GPU while the rest runs on CPU threads.

    The two halves use different hardware, so the wall time is roughly the slower of
    them rather than their sum.
    """
    base, w_base, extras, w_extras, factory = parts
    workers = xtb_workers or _default_extra_workers()

    LOG.info(
        "  [FRQ] mixed batched Hessian: %d displacements, GPU batch_size=%d, "
        "%d CPU worker(s) for %d extra contribution(s)",
        len(positions_list),
        batch_size,
        workers,
        len(extras),
    )

    # One worker: the GPU stream. The CPU side runs on this thread meanwhile.
    with ThreadPoolExecutor(max_workers=1) as gpu:
        fut = gpu.submit(
            _batched_fairchem_forces, atoms, base, positions_list, batch_size
        )
        extra = _extra_forces_threaded(
            atoms, extras, w_extras, positions_list, workers, factory
        )
        f_base = fut.result()

    return w_base * f_base + extra


def _default_extra_workers() -> int:
    """As many concurrent single-threaded calls as we have cores for."""
    cores = int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or (os.cpu_count() or 1)
    per_call = max(1, int(os.environ.get("OMP_NUM_THREADS", "1") or 1))
    return max(1, cores // per_call)


def _limit_openmp_threads_here() -> None:
    """Ask OpenMP for one thread on *this* thread only.

    nthreads-var is a per-thread internal control variable, so this bounds each
    worker's own parallel regions without touching the rest of the process. Without
    it, N workers each spawning OMP_NUM_THREADS threads oversubscribes the node.
    """
    global _OMP_SET_NUM_THREADS
    if _OMP_SET_NUM_THREADS is None:
        try:
            import ctypes

            _OMP_SET_NUM_THREADS = ctypes.CDLL("libgomp.so.1").omp_set_num_threads
        except Exception:
            _OMP_SET_NUM_THREADS = False
    if _OMP_SET_NUM_THREADS:
        try:
            _OMP_SET_NUM_THREADS(1)
        except Exception:
            pass


_OMP_SET_NUM_THREADS = None


def compute_hessian_fd(
    atoms: Atoms,
    *,
    delta: float,
    nfree: int,
    batch_size: int,
    xtb_workers: Optional[int] = None,
) -> np.ndarray:
    """Cartesian Hessian (3N x 3N, eV/Å²) by finite differences of the forces.

    Deliberately reproduces ase.vibrations.Vibrations' arithmetic term for term,
    including its convention of accumulating the half-Hessian and then adding the
    transpose (rather than averaging) — the frequencies this feeds are parsed
    downstream, so they must not shift.
    """
    if nfree not in (2, 4):
        raise ValueError(f"nfree must be 2 or 4, got {nfree}")

    n = len(atoms)
    x0 = atoms.get_positions().copy()

    # Displacement schedule: for each Cartesian coordinate, the stencil points.
    offsets = (-1, 1) if nfree == 2 else (-2, -1, 1, 2)
    geometries: List[np.ndarray] = []
    for a in range(n):
        for i in range(3):
            for k in offsets:
                pos = x0.copy()
                pos[a, i] += k * delta
                geometries.append(pos)

    forces = _forces_for_geometries(atoms, geometries, batch_size, xtb_workers)
    atoms.set_positions(x0)  # leave the caller's geometry untouched

    H = np.empty((3 * n, 3 * n), dtype=float)
    per_coord = len(offsets)
    for r in range(3 * n):
        block = forces[r * per_coord : (r + 1) * per_coord]
        if nfree == 2:
            fminus, fplus = block
            row = 0.5 * (fminus - fplus).ravel()
        else:
            fmm, fminus, fplus, fpp = block
            row = (-fmm + 8 * fminus - 8 * fplus + fpp).ravel() / 12.0
        H[r] = row / (2 * delta)

    H += H.copy().T
    return H


def frequencies_and_modes_from_hessian(
    H: np.ndarray, atoms: Atoms
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(freqs_cm1, mass-weighted modes, eigenvalues) from a Cartesian Hessian.

    Imaginary frequencies come back as negative reals, matching what the rest of
    this module expects from ``Vibrations.get_frequencies()``.
    """
    from ase import units

    im = np.repeat(atoms.get_masses() ** -0.5, 3)
    D = im[:, None] * H * im[None, :]
    D = 0.5 * (D + D.T)

    w2, U = np.linalg.eigh(D)

    conv = units._hbar * units.m / np.sqrt(units._e * units._amu)
    energies = conv * w2.astype(complex) ** 0.5

    freqs = np.array(
        [
            -float(abs(e.imag)) if abs(e.imag) > 1e-8 else float(e.real)
            for e in energies
        ]
    ) / units.invcm

    return freqs, U, w2


def run_frequencies_and_write(
    writer: ORCAWriter,
    atoms: Atoms,
    *,
    delta: float,
    nfree: int,
    scale: float,
    scratch_dir: str | None = None,
    ts: bool = False,
    batch_size: int = 1,
    xtb_workers: Optional[int] = None,
) -> List[float]:
    """
    Prints:
      - 3 (atom), 5 (linear) or 6 (nonlinear) rigid-body zeros at indices 0..zero_first-1,
      - if ts=True, the single imaginary at index zero_first,
      - remaining modes in increasing |frequency| order,
    and clamps tiny post-facto negatives to +abs for readability.
    """
    import os, glob, shutil
    from ase.vibrations import Vibrations

    # Below this, a leftover negative is finite-difference noise rather than a real
    # imaginary mode, and is printed as +|f|.
    eps_small_cm1 = 5.0

    writer.write_energy_grad_banner()
    writer._coords_block6(atoms)

    # vib scratch
    base_tag = os.path.splitext(os.path.basename(writer.path))[0]
    if scratch_dir is None:
        scratch_dir = os.getcwd()
    os.makedirs(scratch_dir, exist_ok=True)
    vib_prefix = os.path.join(scratch_dir, f"vib_{base_tag}")
    for p in glob.glob(vib_prefix + "*"):
        try:
            os.remove(p)
        except IsADirectoryError:
            shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass

    vib = None
    if batch_size > 1:
        # Batched finite differences: same arithmetic as ASE, fewer forward passes.
        LOG.info(
            "  [FRQ] batched Hessian: %d displacements, batch_size=%d",
            2 * (nfree // 2) * 3 * len(atoms),
            batch_size,
        )
        H = compute_hessian_fd(
            atoms, delta=delta, nfree=nfree, batch_size=batch_size,
            xtb_workers=xtb_workers,
        )
        freqs_cm, modes_mw, w2 = frequencies_and_modes_from_hessian(H, atoms)
    else:
        vib = Vibrations(atoms, name=vib_prefix, delta=delta, nfree=nfree)
        vib.run()

        # Frequencies (cm^-1)
        freqs_raw = vib.get_frequencies()
        freqs_cm = []
        for f in np.asarray(freqs_raw):
            if np.iscomplexobj(f) and abs(f.imag) > 1e-8:
                freqs_cm.append(-float(abs(f.imag)))  # imaginary -> negative
            else:
                freqs_cm.append(float(np.real(f)))
        freqs_cm = np.array(freqs_cm, dtype=float)

        # Mass-weighted eigenvectors (same underlying Hessian)
        modes_mw, w2 = compute_mass_weighted_modes(vib, atoms)  # (3N,3N), (3N,)

    # Rigid-body modes first, then (if TS) the imaginary, then the rest by |f|.
    perm, zero_first = order_modes(freqs_cm, modes_mw, atoms, ts=ts)
    nmode = len(freqs_cm)

    # Apply permutation to freqs and modes
    freqs_cm = freqs_cm[perm]
    modes_mw = modes_mw[:, perm]

    # Post-facto cleanup for printing:
    #   - force the first zero_first to exactly 0.0
    #   - clamp tiny |f| < eps_small_cm1 to +abs(f) for the rest
    freqs_cm[:zero_first] = 0.0
    for i in range(zero_first, nmode):
        if abs(freqs_cm[i]) < eps_small_cm1:
            freqs_cm[i] = abs(freqs_cm[i])

    # Scale for printing
    freqs_print = [f * scale for f in freqs_cm]

    # Emit sections
    writer.write_vibrational_frequencies(freqs_print, scale=scale)
    writer.write_normal_modes_preamble()
    writer.write_normal_modes_matrix(modes_mw, zero_first=zero_first)

    # IR spectrum: positive modes only, index = column number
    vib_modes = [
        (i, freqs_print[i]) for i in range(zero_first, nmode) if freqs_print[i] > 0.0
    ]
    writer.write_ir_spectrum(vib_modes)

    if vib is not None:
        vib.clean()  # the batched path never writes displacement caches

    # Keep the scratch dir for debugging; if completely empty, remove it.
    try:
        leftover = glob.glob(os.path.join(scratch_dir, "*"))
        if not leftover:
            shutil.rmtree(scratch_dir, ignore_errors=True)
    except Exception:
        pass

    # Optional CUDA memory hygiene: drop the Vibrations object and free the GPU
    # allocator cache between conformers to avoid OOM on long ensembles. We skip a
    # forced gc.collect() here — it ran on every conformer and dominated allocator
    # churn while refcounting already releases `vib` immediately.
    try:
        import torch

        del vib
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return freqs_print


# ---------- Thermochemistry (RRHO / Quasi-RRHO) ----------
def rrho_thermo(
    atoms: Atoms,
    freqs_cm1: List[float],
    T: float,
    P_atm: float,
    sigma: int,
    E_el_Eh: float,
    *,
    qrrho: bool = True,
    cutoff_cm1: Optional[float] = None,
    qrrho_ref_cm1: float = 100.0,
    qrrho_alpha: float = 4.0,
    # NEW:
    conc_mol_L: Optional[float] = None,
    solv: Optional[str] = None,
    multiplicity: int = 1,
) -> Dict[str, float]:
    # Remove 3/5/6 zero modes
    I = principal_moments_amuA2(atoms)
    geom = classify_geometry(I)  # 'atom', 'linear', 'nonlinear'
    if geom == "atom":
        start = 3
    elif geom == "linear":
        start = 5
    else:
        start = 6
    vib_cm_all = [f for i, f in enumerate(freqs_cm1) if i >= start and f > 0.0]

    if cutoff_cm1 is None:
        cutoff_cm1 = 1.0 if qrrho else 35.0
    vib_cm = [f for f in vib_cm_all if f > cutoff_cm1]

    mass_amu = float(np.sum(atoms.get_masses()))
    mass_kg = mass_amu * amu_kg

    B_A, B_B, B_C = rotational_constants_cm1_from_I(I)
    theta = [theta_per_cm1_K * b for b in (abs(B_A), abs(B_B), abs(B_C))]
    theta = [t if t > 0 else 1e-30 for t in theta]

    vib_e = [f * eV_per_cm1 for f in vib_cm]
    ZPE_eV = 0.5 * sum(vib_e)

    # Rot/trans energies (unchanged)
    Erot_eV = (
        0.0 if geom == "atom" else ((1.0 if geom == "linear" else 1.5) * kB_eV_K * T)
    )
    Etrans_eV = 1.5 * kB_eV_K * T

    # ---------- Translational entropy: pressure OR concentration ----------
    # S/k = ln( (2π m kT)^{3/2} / (h^3 n) ) + 5/2  where n is number density [1/m^3].
    # - Gas phase:    n = P / (kT)
    # - Solution:     n = (conc [mol/L]) * 1000 [L/m^3] * Na / (free_space_fraction),
    #                 with free-space from Shakhnovich–Whitesides.
    lambda_factor = ((2.0 * math.pi * mass_kg * kB_J_K * T) ** 1.5) / (h_J_s**3)

    if conc_mol_L is not None:
        free_mL_per_L = _free_space_mL_per_L(solv)
        # free-space fraction in a liter:
        free_frac = max(free_mL_per_L / 1000.0, 1e-9)  # avoid zero
        number_density = conc_mol_L * 1000.0 * AVOGADRO / free_frac  # 1/m^3
    else:
        P_Pa = P_atm * 101325.0
        number_density = P_Pa / (kB_J_K * T)

    S_trans_over_k = math.log(lambda_factor / number_density) + 2.5
    TS_trans_eV = kB_eV_K * T * S_trans_over_k

    # ---------- Rotational entropy ----------
    if geom == "atom":
        S_rot_over_k = 0.0
    elif geom == "linear":
        theta_rot = theta[1]
        S_rot_over_k = math.log(T / (sigma * theta_rot)) + 1.0
    else:
        prod_theta = theta[0] * theta[1] * theta[2]
        S_rot_over_k = (
            math.log(math.sqrt(math.pi) * (T**1.5) / (sigma * math.sqrt(prod_theta)))
            + 1.5
        )
    TS_rot_eV = kB_eV_K * T * S_rot_over_k

    # ---------- Vibrational entropy + thermal vibrational energy ----------
    I_SI = I * amu_kg * (angstrom_m**2)
    I_av = float(np.mean(I_SI)) if np.any(I_SI > 0) else 1e-46

    S_vib_over_k = 0.0
    Evib_corr_eV = 0.0

    for f_cm in vib_cm:
        e_eV = f_cm * eV_per_cm1
        x = e_eV / (kB_eV_K * T) if T > 0 else float("inf")

        if T > 0:
            S_HO_over_k = (x / (math.expm1(x))) - math.log1p(-math.exp(-x))
            E_th_HO_eV = e_eV / (math.expm1(x))
        else:
            S_HO_over_k = 0.0
            E_th_HO_eV = 0.0

        if not qrrho:
            S_vib_over_k += S_HO_over_k
            Evib_corr_eV += E_th_HO_eV
            continue

        w = 1.0 / (1.0 + (qrrho_ref_cm1 / f_cm) ** qrrho_alpha)

        nu_Hz = c_cm_s * f_cm
        muK = h_J_s / (8.0 * math.pi**2 * nu_Hz)  # kg·m^2
        muEff = (muK * I_av) / (muK + I_av)

        S_FR_over_k = 0.5 + 0.5 * math.log(
            (8.0 * math.pi**2 * muEff * kB_J_K * T) / (h_J_s**2)
        )

        S_vib_over_k += w * S_HO_over_k + (1.0 - w) * S_FR_over_k
        Evib_corr_eV += w * E_th_HO_eV + (1.0 - w) * (0.5 * kB_eV_K * T)

    # ---------- Electronic entropy (optional; = 0 if multiplicity==1) ----------
    TS_el_eV = (
        kB_eV_K
        * T
        * (0.0 if multiplicity <= 1 else math.log(max(float(multiplicity), 1.0)))
    )

    E_el_eV = E_el_Eh * EV_PER_HARTREE

    U_total_eV = E_el_eV + ZPE_eV + Evib_corr_eV + Erot_eV + Etrans_eV
    Hcorr_eV = kB_eV_K * T
    H_total_eV = U_total_eV + Hcorr_eV
    TS_vib_eV = kB_eV_K * T * S_vib_over_k
    TS_tot_eV = TS_el_eV + TS_vib_eV + TS_rot_eV + TS_trans_eV
    G_total_eV = H_total_eV - TS_tot_eV
    G_minus_Eel_eV = G_total_eV - E_el_eV

    # ---- conversions (unchanged) ----
    ZPE_Eh = ZPE_eV * HARTREE_PER_EV
    Evib_corr_Eh = Evib_corr_eV * HARTREE_PER_EV
    Erot_Eh = Erot_eV * HARTREE_PER_EV
    Etrans_Eh = Etrans_eV * HARTREE_PER_EV
    U_total_Eh = U_total_eV * HARTREE_PER_EV
    H_total_Eh = H_total_eV * HARTREE_PER_EV
    Hcorr_Eh = Hcorr_eV * HARTREE_PER_EV
    TS_el_Eh = TS_el_eV * HARTREE_PER_EV
    TS_vib_Eh = TS_vib_eV * HARTREE_PER_EV
    TS_rot_Eh = TS_rot_eV * HARTREE_PER_EV
    TS_trans_Eh = TS_trans_eV * HARTREE_PER_EV
    G_total_Eh = G_total_eV * HARTREE_PER_EV
    G_minus_Eel_Eh = G_minus_Eel_eV * HARTREE_PER_EV

    # symmetry sweep (unchanged)
    rot_table = []
    for sn in range(1, 13):
        if geom == "atom":
            TS_rot_sn_Eh = 0.0
        elif geom == "linear":
            S_rot_sn = math.log(T / (sn * (theta[1]))) + 1.0
            TS_rot_sn_Eh = (kB_eV_K * T * S_rot_sn) * HARTREE_PER_EV
        else:
            prod_theta = theta[0] * theta[1] * theta[2]
            S_rot_sn = (
                math.log(math.sqrt(math.pi) * (T**1.5) / (sn * math.sqrt(prod_theta)))
                + 1.5
            )
            TS_rot_sn_Eh = (kB_eV_K * T * S_rot_sn) * HARTREE_PER_EV
        rot_table.append((sn, TS_rot_sn_Eh))

    return dict(
        mass_amu=float(np.sum(atoms.get_masses())),
        rotconsts_cm1=(B_A, B_B, B_C),
        ZPE_Eh=ZPE_Eh,
        Evib_corr_Eh=Evib_corr_Eh,
        Erot_Eh=Erot_Eh,
        Etrans_Eh=Etrans_Eh,
        U_total_Eh=U_total_Eh,
        H_total_Eh=H_total_Eh,
        Hcorr_Eh=Hcorr_Eh,
        TS_el_Eh=TS_el_Eh,
        TS_vib_Eh=TS_vib_Eh,
        TS_rot_Eh=TS_rot_Eh,
        TS_trans_Eh=TS_trans_Eh,
        G_total_Eh=G_total_Eh,
        G_minus_Eel_Eh=G_minus_Eel_Eh,
        rot_table_Eh=rot_table,
        # Echo config for writer:
        qrrho=qrrho,
        cutoff_cm1=float(cutoff_cm1),
        qrrho_ref_cm1=float(qrrho_ref_cm1) if qrrho else None,
        qrrho_alpha=float(qrrho_alpha) if qrrho else None,
        # NEW echoes:
        conc_mol_L=conc_mol_L,
        solv=(solv or "none"),
        multiplicity=int(multiplicity),
    )

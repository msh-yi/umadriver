# umadriver/solvation.py
"""ALPB solvation as an additive correction on top of the base potential.

    E_tot = E_base + (E_xtb,alpb - E_xtb,vacuum)

The correction is a *difference* of two xtb calculations at the same geometry with
the same Hamiltonian, so most of xtb's intrinsic error cancels and what survives is
the solvation response — which is what ALPB is parameterized for.

Because this is a real ASE calculator, the correction applies to forces as well as
energies: geometries optimize in solvent, and frequencies include the solvation
Hessian.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence

from ase.calculators.mixing import LinearCombinationCalculator, Mixer

LOG = logging.getLogger("uma.solvation")

# ALPB is parameterized for GFN1-xTB, GFN2-xTB and GFN-FF — but not GFN0.
ALPB_METHODS = ("GFN2-xTB", "GFN1-xTB", "GFN-FF")

_TBLITE_HINT = (
    "ALPB solvation needs the `tblite` package.\n"
    "    pip install tblite\n"
    "  (or: conda install -c conda-forge tblite-python)"
)


class SolvationUnavailable(RuntimeError):
    """tblite is not importable — raised with an actionable message."""


def _import_tblite():
    try:
        from tblite.ase import TBLite
    except ImportError as e:  # pragma: no cover - exercised via monkeypatch
        raise SolvationUnavailable(_TBLITE_HINT) from e
    return TBLite


class _ConcurrentMixer(Mixer):
    """Mixer that evaluates its calculators concurrently.

    ASE's Mixer walks the calculators in a list comprehension, so a UMA call on the
    GPU and two xtb SCFs on the CPU run one after another even though they occupy
    different hardware. Measured on a 170-atom TS: UMA 1113 ms, alpb 938 ms,
    vacuum 938 ms — nearly 3 s sequential against ~1.1 s if overlapped.

    Only the *warm-up* is concurrent. Each calculator caches its results, so the
    weighted sum afterwards is pure arithmetic on cached values and is left to the
    base class.
    """

    def __init__(self, calcs, weights, concurrent: bool = True):
        super().__init__(calcs, weights)
        self.concurrent = concurrent

    def get_properties(self, properties, atoms):
        if self.concurrent and len(self.calcs) > 1:
            wanted = [p for p in properties if p in self.implemented_properties]
            if wanted:
                def _warm(calc):
                    for prop in wanted:
                        calc.get_property(prop, atoms)

                # Each calculator owns its own results dict and only reads `atoms`,
                # so this is safe; the compiled cores (torch, tblite) release the GIL.
                with ThreadPoolExecutor(max_workers=len(self.calcs)) as pool:
                    list(pool.map(_warm, self.calcs))

        return super().get_properties(properties, atoms)


class SolvatedCalculator(LinearCombinationCalculator):
    """base + alpb - vacuum, with the solvation term recoverable for reporting."""

    def __init__(self, calcs, weights, concurrent: bool = True, extra_factory=None):
        super().__init__(calcs, weights)
        self.mixer = _ConcurrentMixer(calcs, weights, concurrent=concurrent)
        self._extra_factory = extra_factory

        # TBLite does not implement `free_energy`, and Mixer intersects the property
        # sets, so the combination would drop it even though the base calculator has
        # it. Anything asking for a force-consistent energy would then raise. UMA
        # already defines free_energy as a copy of energy, so mirror that.
        props = list(self.mixer.implemented_properties)
        if "energy" in props and "free_energy" not in props:
            props.append("free_energy")
        self.implemented_properties = props

    def new_extra_calculators(self, deterministic: bool = False) -> Optional[List]:
        """Fresh, independent copies of the non-base contributions, or None.

        ASE calculators carry mutable per-call state (`self.atoms`, `self.results`)
        and tblite keeps its last wavefunction, so evaluating one instance from
        several threads at once segfaults. Separate instances are safe — verified
        to give bit-identical forces under a thread pool — so concurrent code takes
        one set per worker from here rather than sharing.

        ``deterministic=True`` additionally makes each instance resettable, so a
        caller can force a cold SCF per geometry and get order-independent forces.
        Measured cost on a 170-atom system with 12 workers: 609 vs 398 ms per
        displacement, which is hidden entirely behind the GPU in the batched path.
        """
        if self._extra_factory is None:
            return None
        return self._extra_factory(deterministic)

    def calculate(self, atoms, properties, system_changes):
        wanted = [p for p in properties if p != "free_energy"]
        if "free_energy" in properties and "energy" not in wanted:
            wanted.append("energy")
        super().calculate(atoms, wanted, system_changes)
        if "energy" in self.results:
            self.results.setdefault("free_energy", self.results["energy"])


def make_solvated_calculator(
    base_calc,
    solvent: str,
    *,
    method: str = "GFN2-xTB",
    charge: int = 0,
    mult: int = 1,
    concurrent: bool = True,
    accuracy: float = 1.0,
) -> SolvatedCalculator:
    """Wrap ``base_calc`` so every energy and force carries the ALPB correction.

    ``charge``/``mult`` must match what the base calculator is using — xtb takes
    them as constructor arguments rather than reading ``atoms.info``.
    """
    if method not in ALPB_METHODS:
        raise ValueError(
            f"ALPB is not parameterized for {method!r}; choose one of {ALPB_METHODS}"
        )

    TBLite = _import_tblite()

    def _xtb(solvation, deterministic: bool = False):
        # tblite feeds the previous result into the next SCF
        # (`self._res = self._xtb.singlepoint(self._res)`), so a reused instance
        # warm-starts from whatever geometry it saw last and its forces depend on
        # evaluation order at ~2e-3 eV/A. That is harmless while iterating to a
        # stationary point, but a finite-difference Hessian subtracts forces at
        # +delta and -delta, where an order-dependent bias does not cancel.
        #
        # cache_api=False is what lets reset() clear that state; callers wanting
        # reproducible forces pair this with a reset() before each geometry.
        return TBLite(
            method=method,
            charge=charge,
            multiplicity=mult,
            accuracy=accuracy,
            solvation=solvation,
            verbosity=0,
            cache_api=not deterministic,
        )

    def _fresh_extras(deterministic: bool = False):
        return [
            _xtb(("alpb", solvent), deterministic),
            _xtb(None, deterministic),
        ]

    alpb, vac = _fresh_extras()

    LOG.info(
        "Solvation: %s/ALPB(%s) correction on top of the base potential "
        "(charge=%d mult=%d concurrent=%s)",
        method,
        solvent,
        charge,
        mult,
        concurrent,
    )
    return SolvatedCalculator(
        [base_calc, alpb, vac],
        [1.0, 1.0, -1.0],
        concurrent=concurrent,
        extra_factory=_fresh_extras,
    )


def solvation_correction_eV(calc) -> Optional[float]:
    """The (E_alpb - E_vac) term from the last evaluation, or None.

    ASE's Mixer stores the unweighted per-calculator contributions, so this costs
    nothing extra — no additional SCF is run.
    """
    contribs = getattr(calc, "results", {}).get("energy_contributions")
    if not contribs or len(contribs) != 3:
        return None
    return float(contribs[1]) - float(contribs[2])


def base_calculator(calc):
    """The unsolvated calculator underneath, or ``calc`` if it is not wrapped.

    The batched-Hessian fast path talks to FAIRChem internals directly, so it needs
    the raw calculator — and must never be handed a wrapper by accident.
    """
    if isinstance(calc, SolvatedCalculator):
        return calc.mixer.calcs[0]
    return calc


def xtb_calculators(calc) -> List:
    """The [alpb, vacuum] pair of a solvated calculator, else []."""
    if isinstance(calc, SolvatedCalculator):
        return list(calc.mixer.calcs[1:])
    return []

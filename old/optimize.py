from __future__ import annotations
from typing import Optional, Dict, Literal
import time
import numpy as np
from ase import Atoms
from ase.optimize import LBFGS, BFGS, BFGSLineSearch, FIRE, QuasiNewton
import logging

from .utils import (
    load_xyz,
    set_charge_mult,
    resolve_device,
    build_calculator,
    setup_logging,
    gaussian_cutoffs,
    gaussian_converged,
    force_metrics_HB,
    disp_metrics_bohr,
    GaussCutoffs,
)
from .writer import ORCAWriter
from .constants import HARTREE_PER_EV
from .vib_thermo import run_frequencies_and_write, rrho_thermo

LOG = logging.getLogger("omol_driver")

OptimName = Literal["LBFGS", "BFGS", "BFGSLineSearch", "FIRE", "QuasiNewton"]
OPTIMIZERS = {
    "LBFGS": LBFGS,
    "BFGS": BFGS,
    "BFGSLineSearch": BFGSLineSearch,
    "FIRE": FIRE,
    "QuasiNewton": QuasiNewton,
}


def optimize_xyz(
    xyz_path: str,
    *,
    out_path: str,
    charge: int = 0,
    multiplicity: int = 1,
    model: str = "uma-m-1p1",
    device: str = "cuda",
    opt_mode: Literal["Loose", "Normal", "Tight", "VeryTight"] = "Normal",
    maxcycles: int = 300,
    optimizer: OptimName | Literal["Sella"] = "LBFGS",
    maxstep: Optional[float] = None,
    damp: Optional[float] = None,
    cache_dir: Optional[str] = None,
    use_local_scratch: bool = False,
    # freq options
    do_freq: bool = False,
    freq_only: bool = False,
    freq_delta: float = 0.01,
    freq_nfree: int = 2,
    freq_scale: float = 1.0,
    # thermo options
    temp: float = 298.15,
    pressure_atm: float = 1.0,
    symmetry_number: int = 1,
    point_group: Optional[str] = None,
    thermo_scale: Optional[float] = None,
    # qRRHO
    qrrho: bool = True,
    cutoff_cm1: Optional[float] = None,
    qrrho_ref_cm1: float = 100.0,
    qrrho_alpha: float = 4.0,
    # input settings passthrough
    input_settings: Optional[dict] = None,
    # NEW — Sella controls
    optts: bool = False,  # transition-state optimization
    sella_internal: bool = True,
    sella_order: Optional[
        int
    ] = None,  # defaulted below: 1 if optts else 0 (when using Sella)
    sella_eta: float = 2e-2,
    sella_gamma: float = 1e-4,
    sella_delta0: float = 0.02,
    scratch_dir: str | None = None,  # <— NEW
) -> Dict:

    atoms = load_xyz(xyz_path)
    set_charge_mult(atoms, charge, multiplicity)

    dev = resolve_device(device)
    atoms.calc = build_calculator(
        model=model,
        device=dev,
        cache_dir=cache_dir,
        use_local_scratch=use_local_scratch,
    )

    writer = ORCAWriter(out_path, xyz_path, model, dev, opt_banner=not freq_only)
    t_start = time.time()

    # Print the run settings once near the top
    if input_settings:
        writer.write_input_parameters(input_settings)

    # Frequencies-only path
    if freq_only:
        LOG.info("Calculating frequencies")
        freqs = run_frequencies_and_write(
            writer,
            atoms,
            delta=freq_delta,
            nfree=freq_nfree,
            scale=freq_scale,
            scratch_dir=scratch_dir,
        )
        used_scale = thermo_scale if (thermo_scale is not None) else freq_scale
        freqs_th = [f * used_scale for f in freqs]
        E_h = atoms.get_potential_energy() * HARTREE_PER_EV
        th = rrho_thermo(
            atoms,
            freqs_th,
            temp,
            pressure_atm,
            symmetry_number,
            E_h,
            qrrho=qrrho,
            cutoff_cm1=cutoff_cm1,
            qrrho_ref_cm1=qrrho_ref_cm1,
            qrrho_alpha=qrrho_alpha,
        )
        writer.write_thermochemistry(
            T=temp,
            P_atm=pressure_atm,
            mass_amu=th["mass_amu"],
            point_group=point_group or "C1",
            sigma=symmetry_number,
            rotconsts_cm1=th["rotconsts_cm1"],
            use_qrrho=th["qrrho"],
            cutoff_cm1=th["cutoff_cm1"],
            qrrho_ref_cm1=th["qrrho_ref_cm1"],
            qrrho_alpha=th["qrrho_alpha"],
            E_el_Eh=E_h,
            ZPE_Eh=th["ZPE_Eh"],
            Evib_corr_Eh=th["Evib_corr_Eh"],
            Erot_Eh=th["Erot_Eh"],
            Etrans_Eh=th["Etrans_Eh"],
            Hcorr_Eh=th["Hcorr_Eh"],
            TS_el_Eh=0.0,
            TS_vib_Eh=th["TS_vib_Eh"],
            TS_rot_Eh=th["TS_rot_Eh"],
            TS_trans_Eh=th["TS_trans_Eh"],
            G_total_Eh=th["G_total_Eh"],
            H_total_Eh=th["H_total_Eh"],
            U_total_Eh=th["U_total_Eh"],
            G_minus_Eel_Eh=th["G_minus_Eel_Eh"],
            rot_entropy_table_Eh=th["rot_table_Eh"],
        )
        wall = time.time() - t_start
        writer.write_termination(wall)
        writer.close()
        return dict(
            converged=None,
            steps=0,
            energy_H=None,
            walltime_s=wall,
            model=model,
            device=dev,
            out=out_path,
            freq=True,
            freq_only=True,
        )

    # --- Optimization path ---
    use_sella = (optimizer == "Sella") or optts
    if use_sella:
        try:
            from sella import Sella as SellaOpt
        except Exception as e:
            raise SystemExit(
                f"ERROR: Sella requested but not available: {e}\n"
                "Install with: pip install sella"
            )
        # Default Sella order if not set explicitly
        effective_order = 1 if optts else 0
        if sella_order is not None:
            effective_order = int(sella_order)

        # Sella ignores maxstep/damp; it has its own knobs
        dyn = SellaOpt(
            atoms,
            trajectory=None,
            logfile=None,
            internal=sella_internal,
            order=effective_order,
            eta=sella_eta,
            gamma=sella_gamma,
            delta0=sella_delta0,
        )

    else:
        Opt = OPTIMIZERS[optimizer]
        opt_kwargs: Dict = {}
        if maxstep is not None:
            opt_kwargs["maxstep"] = maxstep
        if optimizer == "FIRE" and damp is not None:
            opt_kwargs["damp"] = damp

        # NOTE: if you prefer "no hard cap" when maxcycles <= 0, set steps=None here
        dyn = Opt(atoms, trajectory=None, logfile=None, **opt_kwargs)

    ase_fmax = 1e-12  # ridiculously small so ase never stops, we manage our own convergence ourselves
    cuts = gaussian_cutoffs(mode=opt_mode)
    prev_pos = None
    converged = False
    steps = 0
    E_h = atoms.get_potential_energy() * HARTREE_PER_EV  # initialize for scope

    for cycle, _ in enumerate(dyn.irun(fmax=ase_fmax, steps=maxcycles), start=1):
        steps = cycle
        forces = atoms.get_forces()  # eV/Å
        grms, gmax = force_metrics_HB(forces)  # Hartree/Bohr
        curr_pos = atoms.get_positions()  # Å
        drms, dmax = disp_metrics_bohr(prev_pos, curr_pos)  # Bohr
        E_h = atoms.get_potential_energy() * HARTREE_PER_EV  # Hartree

        writer.write_cycle(cycle, atoms, E_h, grms, gmax, drms, dmax, cuts)

        if gaussian_converged(grms, gmax, drms, dmax, cuts):
            converged = True
            break

        prev_pos = curr_pos.copy()

    wall = time.time() - t_start

    # Detect hard stop due to maxcycles (only meaningful if maxcycles > 0)
    hit_maxcycles = (
        (not converged)
        and (maxcycles is not None)
        and (maxcycles > 0)
        and (steps >= maxcycles)
    )

    if hit_maxcycles:
        # Loud banner & skip any post-opt work
        writer.write_maxcycles_abort(steps, maxcycles)

    # Always print final geometry + energy (converged banner only if converged)
    if converged:
        writer.write_final_geom_and_energy(atoms, E_h, converged)

    # Frequencies & Thermochemistry ONLY if converged
    if do_freq and converged:
        LOG.info("Optimization complete, calculating frequencies")
        freqs = run_frequencies_and_write(
            writer,
            atoms,
            delta=freq_delta,
            nfree=freq_nfree,
            scale=freq_scale,
            scratch_dir=scratch_dir,
            ts=optts
        )
        used_scale = thermo_scale if (thermo_scale is not None) else freq_scale
        freqs_th = [f * used_scale for f in freqs]
        th = rrho_thermo(
            atoms,
            freqs_th,
            temp,
            pressure_atm,
            symmetry_number,
            E_h,
            qrrho=qrrho,
            cutoff_cm1=cutoff_cm1,
            qrrho_ref_cm1=qrrho_ref_cm1,
            qrrho_alpha=qrrho_alpha,
        )
        writer.write_thermochemistry(
            T=temp,
            P_atm=pressure_atm,
            mass_amu=th["mass_amu"],
            point_group=point_group or "C1",
            sigma=symmetry_number,
            rotconsts_cm1=th["rotconsts_cm1"],
            use_qrrho=th["qrrho"],
            cutoff_cm1=th["cutoff_cm1"],
            qrrho_ref_cm1=th["qrrho_ref_cm1"],
            qrrho_alpha=th["qrrho_alpha"],
            E_el_Eh=E_h,
            ZPE_Eh=th["ZPE_Eh"],
            Evib_corr_Eh=th["Evib_corr_Eh"],
            Erot_Eh=th["Erot_Eh"],
            Etrans_Eh=th["Etrans_Eh"],
            Hcorr_Eh=th["Hcorr_Eh"],
            TS_el_Eh=0.0,
            TS_vib_Eh=th["TS_vib_Eh"],
            TS_rot_Eh=th["TS_rot_Eh"],
            TS_trans_Eh=th["TS_trans_Eh"],
            G_total_Eh=th["G_total_Eh"],
            H_total_Eh=th["H_total_Eh"],
            U_total_Eh=th["U_total_Eh"],
            G_minus_Eel_Eh=th["G_minus_Eel_Eh"],
            rot_entropy_table_Eh=th["rot_table_Eh"],
        )
    elif do_freq and not converged:
        # Optional: make it explicit in the output file why we skipped freq
        LOG.info(
            "Frequencies were requested but optimization did not converge; skipping frequency analysis."
        )
        writer._w(
            "Frequencies were requested but optimization did not converge; skipping frequency analysis."
        )
        writer._w("")

    # Termination & time
    writer.write_termination(wall)
    writer.close()

    res = dict(
        converged=converged,
        steps=steps,
        energy_H=float(E_h),
        walltime_s=wall,
        cutoffs=cuts.__dict__,
        opt_mode=opt_mode,
        model=model,
        device=dev,
        out=out_path,
        freq=do_freq,
        freq_only=False,
        T=temp,
        P_atm=pressure_atm,
        sigma=symmetry_number,
        stopped="maxcycles" if hit_maxcycles else None,
        optimizer="Sella" if use_sella else optimizer,
        optts=optts,
        sella_internal=sella_internal if use_sella else None,
        sella_order=effective_order if use_sella else None,
        sella_eta=sella_eta if use_sella else None,
        sella_gamma=sella_gamma if use_sella else None,
        sella_delta0=sella_delta0 if use_sella else None,
    )
    LOG.info(
        "Result: converged=%s steps=%d wall=%.1fs E(H)=%.6f stopped=%s",
        converged,
        steps,
        res["walltime_s"],
        res["energy_H"],
        res["stopped"],
    )
    return res

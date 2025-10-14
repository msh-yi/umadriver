from __future__ import annotations
from datetime import datetime
from typing import List, Tuple, Optional
from ase import Atoms

from .constants import EV_PER_HARTREE, KCAL_PER_MOL_PER_EV


class ORCAWriter:
    def __init__(
        self,
        path: str,
        xyz_path: str,
        model: str,
        device: str,
        *,
        opt_banner: bool = True,
    ):
        self.path = path
        self.f = open(path, "w", buffering=1)
        self._w("                                 *****************")
        self._w("                                 * O   R   C   A *")
        self._w("                                 *****************")
        self._w("")
        self._w(f"OMol/ASE; model={model}; device={device}")
        self._w(datetime.now().strftime("Start  : %a %b %d %H:%M:%S  %Y"))
        self._w(f"Input  : {xyz_path}")
        self._w("")
        if opt_banner:
            self._w("                       *****************************")
            self._w("                       * Geometry Optimization Run *")
            self._w("                       *****************************")
            self._w("")

    def _w(self, s=""):
        self.f.write(s + ("\n" if not s.endswith("\n") else ""))

    def _cycle_banner(self, n: int):
        left = " " * 9
        stars = "*" * 61
        self._w(f"{left}{stars}")
        title = f"GEOMETRY OPTIMIZATION CYCLE   {n}"
        interior = 61 - 2
        prefix_spaces = 16
        pad = max(0, interior - prefix_spaces - len(title))
        self._w(f"{left}*{' ' * prefix_spaces}{title}{' ' * pad}*")
        self._w(f"{left}{stars}")

    def _coords_block10(self, atoms: Atoms):
        self._w("---------------------------------")
        self._w("CARTESIAN COORDINATES (ANGSTROEM)")
        self._w("---------------------------------")
        syms = atoms.get_chemical_symbols()
        pos = atoms.get_positions()
        for s, (x, y, z) in zip(syms, pos):
            self._w(f"  {s:<3s}{x:16.10f} {y:16.10f} {z:16.10f}")
        self._w("")

    def _coords_block6(self, atoms: Atoms):
        self._w("---------------------------------")
        self._w("CARTESIAN COORDINATES (ANGSTROEM)")
        self._w("---------------------------------")
        syms = atoms.get_chemical_symbols()
        pos = atoms.get_positions()
        for s, (x, y, z) in zip(syms, pos):
            self._w(f"  {s:<3s}{x:12.6f} {y:11.6f} {z:12.6f}")
        self._w("")

    def _energy_box(self, energy_h: float):
        self._w("-------------------------   --------------------")
        self._w(f"FINAL SINGLE POINT ENERGY     {energy_h:.15f}")
        self._w("-------------------------   --------------------")
        self._w("")

    def _geom_conv_box(self, grms, gmax, drms, dmax, cuts):
        self._w("                                .--------------------.")
        self._w(
            "          ----------------------|Geometry convergence|-------------------------"
        )
        self._w(
            "          Item                value                   Tolerance       Converged"
        )
        self._w(
            "          ---------------------------------------------------------------------"
        )
        self._w(
            f"          RMS gradient        {grms:12.10f}            {cuts.grms:12.10f}      {'YES' if grms < cuts.grms else 'NO'}"
        )
        self._w(
            f"          MAX gradient        {gmax:12.10f}            {cuts.gmax:12.10f}      {'YES' if gmax < cuts.gmax else 'NO'}"
        )
        self._w(
            f"          RMS step            {drms:12.10f}            {cuts.drms:12.10f}      {'YES' if drms < cuts.drms else 'NO'}"
        )
        self._w(
            f"          MAX step            {dmax:12.10f}            {cuts.dmax:12.10f}      {'YES' if dmax < cuts.dmax else 'NO'}"
        )
        self._w("")
        self._w("          ........................................................")
        self._w("          Max(Bonds)      0.0000      Max(Angles)    0.00")
        self._w("          Max(Dihed)        0.00      Max(Improp)    0.00")
        self._w(
            "          ---------------------------------------------------------------------"
        )
        self._w("")

    def write_input_parameters(self, params: dict):
        """
        Print a stable, human-readable summary of all run inputs.
        Keys can be raw CLI arg names; we'll pretty-print labels.
        """
        label_map = {
            "action": "Action",
            "xyz": "Input XYZ",
            "charge": "Charge",
            "multiplicity": "Multiplicity",
            "model": "Model",
            "device": "Device",
            "cache_dir": "Cache dir",
            "use_local_scratch": "Use local scratch",
            "opt_mode": "Opt convergence mode",
            "optimizer": "Optimizer",
            "maxcycles": "Max cycles",
            "maxstep": "Max step",
            "damp": "FIRE damping",
            "freq": "Do frequencies",
            "freq_delta": "FD step (Å)",
            "freq_nfree": "nfree",
            "freq_scale": "Print scale",
            "temp": "Thermo T (K)",
            "pressure_atm": "Thermo P (atm)",
            "symmetry_number": "Symmetry number σ",
            "point_group": "Point group",
            "thermo_scale": "Thermo scale",
            "qrrho": "Quasi-RRHO",
            "cutoff_cm1": "CutOffFreq (cm^-1)",
            "qrrho_ref_cm1": "QRRHORefFreq (cm^-1)",
            "qrrho_alpha": "QRRHO α",
            "optts": "Optimize TS",
            "sella_internal": "Sella internal coords",
            "sella_order": "Sella order",
            "sella_eta": "Sella eta",
            "sella_gamma": "Sella gamma",
            "sella_delta0": "Sella delta0",
        }

        order = [
            # General
            "action",
            "xyz",
            "charge",
            "multiplicity",
            "model",
            "device",
            "cache_dir",
            "use_local_scratch",
            # Optimization
            "opt_mode",
            "optimizer",
            "maxcycles",
            "maxstep",
            "damp",
            # Frequencies
            "freq",
            "freq_delta",
            "freq_nfree",
            "freq_scale",
            # Thermochemistry
            "temp",
            "pressure_atm",
            "symmetry_number",
            "point_group",
            "thermo_scale",
            # qRRHO
            "qrrho",
            "cutoff_cm1",
            "qrrho_ref_cm1",
            "qrrho_alpha",
            "optts",
            "sella_internal",
            "sella_order",
            "sella_eta",
            "sella_gamma",
            "sella_delta0",
        ]

        self._w("---------------------")
        self._w("INPUT PARAMETERS")
        self._w("---------------------")
        self._w("")

        def fmt(v):
            if isinstance(v, float):
                # short but precise default
                return f"{v:.6g}"
            return str(v)

        # align like other boxes: "<label>  ...  value"
        for k in order:
            if k in params:
                lab = label_map.get(k, k)
                val = params[k]
                if val is None:
                    continue  # omit unset
                self._w(f"{lab:<28} ... {fmt(val)}")
        self._w("")

    def write_maxcycles_abort(self, steps: int, maxcycles: int):
        # ORCA-ish loud banner + a blank line
        self._w("                    *********************** STOP ********************")
        self._w(
            f"                    ***  MAX CYCLES REACHED ({steps}/{maxcycles})  ***"
        )
        self._w(
            "                    ***      OPTIMIZATION ABORTED (NOT CONVERGED)  ***"
        )
        self._w("                    *************************************************")
        self._w("")

    # --- Frequency sections (match your exact formats) ---
    def write_energy_grad_banner(self):
        self._w("                     *******************************")
        self._w("                     * Energy+Gradient Calculation *")
        self._w("                     *******************************")
        self._w("")

    def write_vibrational_frequencies(self, freqs_cm1: List[float], scale: float):
        self._w("-----------------------")
        self._w("VIBRATIONAL FREQUENCIES")
        self._w("-----------------------")
        self._w("")
        self._w(f"Scaling factor for frequencies =  {scale:.9f} (already applied!)")
        self._w("")
        for i, f in enumerate(freqs_cm1):
            self._w(f"{i:4d}:{f:13.2f} cm**-1")
        self._w("")

    def write_normal_modes_preamble(self):
        self._w("------------")
        self._w("NORMAL MODES")
        self._w("------------")
        self._w("")
        self._w(
            "These modes are the cartesian displacements weighted by the diagonal matrix"
        )
        self._w("M(i,i)=1/sqrt(m[i]) where m[i] is the mass of the displaced atom")
        self._w("Thus, these vectors are normalized but *not* orthogonal")
        self._w("")

    def write_normal_modes_matrix(self, modes_mw, zero_first: int = 0):
        """
        modes_mw: (3N, 3N) mass-weighted eigenvectors (columns are modes)
        zero_first: number of initial columns to zero (5 for linear, 6 for nonlinear)
        Uses EXACT format you provided.
        """
        import numpy as np

        V = np.array(modes_mw, copy=True)
        if zero_first > 0:
            V[:, :zero_first] = 0.0

        nrows, ncols = V.shape
        block = 6

        for c0 in range(0, ncols, block):
            c1 = min(c0 + block, ncols)

            indent = " " * 18
            pieces = [f"{j:<11d}" for j in range(c0, c1)]
            header = indent + "".join(pieces).rstrip() + "    "
            self._w(header)

            for r in range(nrows):
                line = (
                    f"{r:7d}"
                    + "    "
                    + "".join(f"{V[r, j]:11.6f}" for j in range(c0, c1))
                )
                self._w(line)

    def write_ir_spectrum(self, vib_modes: list[tuple[int, float]]):
        self._w("-----------")
        self._w("IR SPECTRUM")
        self._w("-----------")
        self._w("")
        self._w(
            " Mode   freq       eps      Int      T**2         TX        TY        TZ"
        )
        self._w("       cm**-1   L/(mol*cm) km/mol    a.u.")
        self._w(
            "----------------------------------------------------------------------------"
        )
        for idx, f in vib_modes:
            self._w(
                f"{idx:4d}: {f:8.2f}   {0.000000:0.6f}   {0.00:6.2f}  {0.000000:0.6f}  ({-0.000000: .6f} {-0.000000: .6f} {0.000000: .6f})"
            )
        self._w("")

    # High-level writers used by optimize()
    def write_cycle(
        self,
        cycle: int,
        atoms: Atoms,
        energy_h: float,
        grms: float,
        gmax: float,
        drms: float,
        dmax: float,
        cuts,
    ):
        self._cycle_banner(cycle)
        self._coords_block10(atoms)
        self._energy_box(energy_h)
        self._geom_conv_box(grms, gmax, drms, dmax, cuts)

    def write_final_geom_and_energy(
        self, atoms: Atoms, energy_h: float, converged: bool
    ):
        if converged:
            self._w(
                "                    ***********************HURRAY********************"
            )
            self._w(
                "                    ***        THE OPTIMIZATION HAS CONVERGED     ***"
            )
            self._w(
                "                    *************************************************"
            )
            self._w("")
        self._coords_block10(atoms)
        self._energy_box(energy_h)

    def write_termination(self, wall_s: float):
        self._w("                             ****ORCA TERMINATED NORMALLY****")
        days = int(wall_s // 86400)
        rem = wall_s - days * 86400
        hours = int(rem // 3600)
        rem -= hours * 3600
        mins = int(rem // 60)
        rem -= mins * 60
        secs = int(rem)
        msec = int((rem - secs) * 1000)
        self._w(
            f"TOTAL RUN TIME: {days} days {hours} hours {mins} minutes {secs} seconds {msec} msec"
        )
        self.f.flush()

    # --- Thermochemistry block (advertises RRHO vs qRRHO + knobs) ---
    def write_thermochemistry(
        self,
        T: float,
        P_atm: float,
        mass_amu: float,
        point_group: str,
        sigma: int,
        rotconsts_cm1: Tuple[float, float, float],
        use_qrrho: bool,
        cutoff_cm1: float,
        qrrho_ref_cm1: Optional[float],
        qrrho_alpha: Optional[float],
        # contributions in Eh and kcal/mol
        E_el_Eh: float,
        ZPE_Eh: float,
        Evib_corr_Eh: float,
        Erot_Eh: float,
        Etrans_Eh: float,
        Hcorr_Eh: float,
        TS_el_Eh: float,
        TS_vib_Eh: float,
        TS_rot_Eh: float,
        TS_trans_Eh: float,
        G_total_Eh: float,
        H_total_Eh: float,
        U_total_Eh: float,
        G_minus_Eel_Eh: float,
        rot_entropy_table_Eh: list[tuple[int, float]],
    ):
        self._w("--------------------------")
        self._w(f"THERMOCHEMISTRY AT {T:.2f}K")
        self._w("--------------------------")
        self._w("")
        self._w(f"Temperature         ... {T:.2f} K")
        self._w(f"Pressure            ... {P_atm:.2f} atm")
        self._w(f"Total Mass          ... {mass_amu:.2f} AMU")
        self._w("")
        self._w("Throughout the following assumptions are being made:")
        self._w("  (1) The electronic state is orbitally nondegenerate")
        self._w("  (2) There are no thermally accessible electronically excited states")
        self._w("  (3) Hindered rotations indicated by low frequency modes are not")
        self._w("      treated as such but are treated as vibrations and this may")
        self._w("      cause some error")
        self._w("  (4) All equations used are the standard statistical mechanics")
        self._w("      equations for an ideal gas")
        self._w("  (5) All vibrations are strictly harmonic")
        self._w("")
        self._w("")
        self._w("------------")
        self._w("INNER ENERGY")
        self._w("------------")
        self._w("")
        self._w("The inner energy is: U= E(el) + E(ZPE) + E(vib) + E(rot) + E(trans)")
        self._w(
            "    E(el)   - is the total energy from the electronic structure calculation"
        )
        self._w("              = E(kin-el) + E(nuc-el) + E(el-el) + E(nuc-nuc)")
        self._w(
            "    E(ZPE)  - the the zero temperature vibrational energy from the frequency calculation"
        )
        self._w(
            "    E(vib)  - the the finite temperature correction to E(ZPE) due to population"
        )
        self._w("              of excited vibrational states")
        self._w("    E(rot)  - is the rotational thermal energy")
        self._w("    E(trans)- is the translational thermal energy")
        self._w("")

        def line(lbl, Eh, kcal):
            self._w(f"{lbl:<32} ... {Eh:12.8f} Eh  {kcal:9.2f} kcal/mol")

        line("Summary of contributions to the inner energy U:", 0.0, 0.0)
        line(
            "Electronic energy", E_el_Eh, E_el_Eh * EV_PER_HARTREE * KCAL_PER_MOL_PER_EV
        )
        line("Zero point energy", ZPE_Eh, ZPE_Eh * EV_PER_HARTREE * KCAL_PER_MOL_PER_EV)
        line(
            "Thermal vibrational correction",
            Evib_corr_Eh,
            Evib_corr_Eh * EV_PER_HARTREE * KCAL_PER_MOL_PER_EV,
        )
        line(
            "Thermal rotational correction",
            Erot_Eh,
            Erot_Eh * EV_PER_HARTREE * KCAL_PER_MOL_PER_EV,
        )
        line(
            "Thermal translational correction",
            Etrans_Eh,
            Etrans_Eh * EV_PER_HARTREE * KCAL_PER_MOL_PER_EV,
        )
        self._w(
            "-----------------------------------------------------------------------"
        )
        self._w(f"Total thermal energy                    {U_total_Eh:12.8f} Eh")
        self._w("")
        self._w("")
        self._w("Summary of corrections to the electronic energy:")
        self._w("(perhaps to be used in another calculation)")
        self._w(
            f"Total thermal correction                  {Evib_corr_Eh+Erot_Eh+Etrans_Eh:12.8f} Eh"
        )
        self._w(f"Non-thermal (ZPE) correction              {ZPE_Eh:12.8f} Eh")
        self._w(
            "-----------------------------------------------------------------------"
        )
        self._w(
            f"Total correction                          {ZPE_Eh+Evib_corr_Eh+Etrans_Eh:12.8f} Eh"
        )
        self._w("")
        self._w("")
        self._w("--------")
        self._w("ENTHALPY")
        self._w("--------")
        self._w("")
        self._w("The enthalpy is H = U + kB*T")
        self._w("                kB is Boltzmann's constant")
        self._w(f"Total free energy                 ...    {U_total_Eh:12.8f} Eh ")
        self._w(
            f"Thermal Enthalpy correction       ...      {Hcorr_Eh:10.8f} Eh      {Hcorr_Eh*EV_PER_HARTREE*KCAL_PER_MOL_PER_EV:6.2f} kcal/mol"
        )
        self._w(
            "-----------------------------------------------------------------------"
        )
        self._w(f"Total Enthalpy                    ...    {H_total_Eh:12.8f} Eh")
        self._w("")
        if use_qrrho:
            self._w("Vibrational entropy computed via Quasi-RRHO (Grimme).")
            self._w("Reference: Chem. Eur. J. 2012, 18, 9955.")
            self._w(f"QRRHORefFreq  ... {qrrho_ref_cm1:.1f} cm-1")
            self._w(f"Mix exponent α ... {qrrho_alpha:.1f}")
            self._w(
                f"CutOffFreq    ... {cutoff_cm1:.1f} cm-1 (modes below excluded from thermo)"
            )
        else:
            self._w("Vibrational entropy computed via RRHO (harmonic oscillator).")
            self._w(
                f"CutOffFreq    ... {cutoff_cm1:.1f} cm-1 (modes below excluded from thermo)"
            )
        self._w("")
        self._w("Note: Rotational entropy computed according to Herzberg ")
        self._w("Infrared and Raman Spectra, Chapter V,1, Van Nostrand Reinhold, 1945 ")
        pg = point_group or "C1"
        self._w(f"Point Group:  {pg}, Symmetry Number:  {sigma:3d}  ")
        self._w(
            f"Rotational constants in cm-1:   {rotconsts_cm1[0]:10.6f}   {rotconsts_cm1[1]:10.6f}   {rotconsts_cm1[2]:10.6f} "
        )
        self._w("")
        self._w("-------")
        self._w("ENTROPY")
        self._w("-------")
        self._w("")
        self._w("The entropy contributions are T*S = T*(S(el)+S(vib)+S(rot)+S(trans))")
        self._w("     S(el)   - electronic entropy")
        self._w("     S(vib)  - vibrational entropy")
        self._w("     S(rot)  - rotational entropy")
        self._w("     S(trans)- translational entropy")
        self._w("The entropies will be listed as multiplied by the temperature to get")
        self._w("units of energy")
        self._w("")

        def lineTS(lbl, TS_Eh):
            self._w(
                f"{lbl:<32} ...      {TS_Eh:10.8f} Eh  {TS_Eh*EV_PER_HARTREE*KCAL_PER_MOL_PER_EV:6.2f} kcal/mol"
            )

        lineTS("Electronic entropy", TS_el_Eh)
        lineTS("Vibrational entropy", TS_vib_Eh)
        lineTS("Rotational entropy", TS_rot_Eh)
        lineTS("Translational entropy", TS_trans_Eh)
        self._w(
            "-----------------------------------------------------------------------"
        )
        TS_tot = TS_el_Eh + TS_vib_Eh + TS_rot_Eh + TS_trans_Eh
        self._w(
            f"Final entropy term                ...      {TS_tot:10.8f} Eh  {TS_tot*EV_PER_HARTREE*KCAL_PER_MOL_PER_EV:6.2f} kcal/mol"
        )
        self._w("")
        self._w(
            "In case the symmetry of your molecule has not been determined correctly"
        )
        self._w(
            "or in case you have a reason to use a different symmetry number we print "
        )
        self._w("out the resulting rotational entropy values for sn=1,12 :")
        self._w(" --------------------------------------------------------")
        for sn, TSrot in rot_entropy_table_Eh:
            self._w(
                f"|  sn={sn:2d} | S(rot)=       {TSrot:0.8f} Eh  {TSrot*EV_PER_HARTREE*KCAL_PER_MOL_PER_EV:6.2f} kcal/mol|"
            )
        self._w(" --------------------------------------------------------")
        self._w("")
        self._w("")
        self._w("-------------------")
        self._w("GIBBS FREE ENERGY")
        self._w("-------------------")
        self._w("")
        self._w("The Gibbs free energy is G = H - T*S")
        self._w("")
        self._w(f"Total enthalpy                    ...    {H_total_Eh:12.8f} Eh ")
        self._w(
            f"Total entropy correction          ...     {-TS_tot:11.8f} Eh     {-TS_tot*EV_PER_HARTREE*KCAL_PER_MOL_PER_EV:6.2f} kcal/mol"
        )
        self._w(
            "-----------------------------------------------------------------------"
        )
        self._w(f"Final Gibbs free energy         ...    {G_total_Eh:12.8f} Eh")
        self._w("")
        self._w("For completeness - the Gibbs free energy minus the electronic energy")
        self._w(
            f"G-E(el)                           ...      {G_minus_Eel_Eh:10.8f} Eh      {G_minus_Eel_Eh*EV_PER_HARTREE*KCAL_PER_MOL_PER_EV:6.2f} kcal/mol"
        )
        self._w("")

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

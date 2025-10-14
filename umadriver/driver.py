# uma/uma_driver.py

# TODO: time per step

import argparse
import logging
import os, sys


## have to be before sella import!
def _early_parse_threads(argv):
    # lightweight pre-parse just for --sella-threads and --jax-platform
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--sella-threads", type=int, default=0)
    # unless we are on a multi-GPU system, always use CPU because otherwise JAX hogs all the memory leaving none for UMA/torch
    p.add_argument("--jax-platform", choices=("cpu", "cuda"), default="cpu")

    ns, _ = p.parse_known_args(argv)
    # Platform pinning
    os.environ.setdefault("JAX_PLATFORMS", ns.jax_platform)
    os.environ.setdefault("JAX_PLATFORM_NAME", ns.jax_platform)  # legacy alias
    # Thread pool
    n = (
        ns.sella_threads
        or int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
        or (os.cpu_count() or 1)
    )
    flags = os.environ.get("XLA_FLAGS", "")
    add = f"--xla_cpu_multi_thread_eigen=true"
    if ns.jax_platform == "cpu" and add not in flags:
        os.environ["XLA_FLAGS"] = (flags + " " + add).strip()
    # also relax BLAS caps for Sella path
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, str(n))


_early_parse_threads(sys.argv[1:])

from .utils import (
    setup_logging,
    mode_type,
    optimizer_type,
)
from .constants import DEFAULT_FAIRCHEM_CACHE, VAST_BASE
from .ensemble import run_conformer_workflow

LOG = logging.getLogger("omol_driver")


def main():
    p = argparse.ArgumentParser(
        description=(
            "UMA driver (always through ensemble pipeline): OPT/SP/TS per conformer, "
            "optional frequencies + qRRHO thermochemistry, optional Gaussian inputs and IRC"
        )
    )

    # Subparsers MUST be created before parse_args
    sub = p.add_subparsers(dest="cmd", required=False)

    # --- batch subcommand (unchanged) ---
    pb = sub.add_parser("batch", help="Run multiple ensemble XYZ files in one process (one GPU).")
    pb.add_argument("--xyz-glob", nargs="*", default=None,
                    help="One or more globs or paths, e.g. 'inputs/*.xyz'.")
    pb.add_argument("--manifest", type=str, default=None,
                    help="YAML or JSON manifest specifying jobs.")
    pb.add_argument("--out-root", type=str, default="runs",
                    help="Root directory for job outputs (default: runs/).")
    pb.add_argument("--resume", action="store_true", default=True,
                    help="Skip jobs that already have energies.csv (default on).")
    pb.add_argument("--no-resume", dest="resume", action="store_false")

    # Common model/device knobs (apply to all jobs)
    pb.add_argument("--model", default="uma-m-1p1")
    pb.add_argument("--device", default="cuda")
    pb.add_argument("--cache-dir", default=None)
    pb.add_argument("--use-local-scratch", action="store_true")

    # Broadcast overrides (optional)
    pb.add_argument("--optimizer", default="Sella")
    pb.add_argument("--optts", action="store_true")
    pb.add_argument("--do-freq", action="store_true")
    pb.add_argument("--solv", default=None)
    pb.add_argument("--irc", action="store_true")
    pb.add_argument("--irc-dx", type=float, default=0.1)

    # --- root/common (single) ---
    p.add_argument("--xyz", help="Input XYZ file")

    p.add_argument(
        "--scratch-root",
        default=os.environ.get("UMA_SCRATCH_ROOT", VAST_BASE),
        help=(
            "Root dir for per-job scratch (ASE vibrations cache). "
            "Default: $UMA_SCRATCH_ROOT or netscratch base."
        ),
    )
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--mult", type=int, default=1, help="Spin multiplicity (2S+1)")

    # Optimization and actions
    p.add_argument(
        "--opt-mode",
        type=mode_type,
        default=mode_type("Normal"),
        help="Optimization convergence mode: Loose, Normal, Tight, VeryTight",
    )
    p.add_argument(
        "--freq",
        action="store_true",
        help="Run frequency analysis per conformer.",
    )
    p.add_argument(
        "--opt",
        type=optimizer_type,
        default="Sella",
        help=(
            "Optimizer to use (default: Sella). Use --sp for single-point (no optimization)."
        ),
    )
    p.add_argument(
        "--sp",
        action="store_true",
        help="Force single-point per conformer (no geometry optimization).",
    )
    p.add_argument(
        "--optts",
        action="store_true",
        help="Optimize a transition state (first-order saddle) per conformer. Uses Sella.",
    )

    # Sella controls (no explicit order flag; set by --optts)
    p.add_argument("--sella-internal", dest="sella_internal", action="store_true",
                   help="Use internal coordinates with Sella (default).")
    p.add_argument("--no-sella-internal", dest="sella_internal", action="store_false")
    p.set_defaults(sella_internal=True)
    p.add_argument("--sella-eta", type=float, default=2e-2)
    p.add_argument("--sella-gamma", type=float, default=1e-4)
    p.add_argument("--sella-delta0", type=float, default=0.02)

    p.add_argument("--maxcycles", type=int, default=300)

    # Model/device/cache
    p.add_argument("--model", default="uma-m-1p1")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    p.add_argument("--cache-dir", default=DEFAULT_FAIRCHEM_CACHE)
    p.add_argument("--use-local-scratch", action="store_true")

    # vibrations / printing
    p.add_argument("--freq-delta", type=float, default=0.01, help="Finite-difference step in Å")
    p.add_argument("--freq-nfree", type=int, default=2, help="nfree for Vibrations (2=central)")
    p.add_argument("--freq-scale", type=float, default=1.0,
                   help="Scaling factor printed/applied to freqs")

    # thermochemistry
    p.add_argument("--temp", type=float, default=298.15)
    p.add_argument("--pressure-atm", type=float, default=1.00)
    p.add_argument("--symmetry-number", type=int, default=1)
    p.add_argument("--point-group", type=str, default=None)

    # qRRHO
    p.add_argument("--qrrho", dest="qrrho", action="store_true", help="Use Quasi-RRHO (default)")
    p.add_argument("--no-qrrho", dest="qrrho", action="store_false")
    p.set_defaults(qrrho=True)
    p.add_argument("--cutoff-cm1", type=float, default=None,
                   help="Thermo cutoff (cm^-1); default 1 (qRRHO) or 35 (RRHO)")
    p.add_argument("--qrrho-ref-cm1", type=float, default=100.0,
                   help="QRRHO reference frequency ω0 (cm^-1)")
    p.add_argument("--qrrho-alpha", type=float, default=4.0,
                   help="QRRHO damping exponent α")

    # logging
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--debug", action="store_true")

    # always-ensemble outputs
    p.add_argument("--outdir", default=None,
                   help="Output directory for ensemble workflow (default: <basename>.ensemble)")
    p.add_argument("--solv", default=None,
                   help="Emit Gaussian inputs for M052X/6-31G* and M052X/6-31G* scrf(SMD,solvent=<solv>)")
    p.add_argument("--gauss-mem", default="160GB")
    p.add_argument("--gauss-nproc", default="16")

    # IRC
    p.add_argument("--irc", action="store_true",
                   help="Run intrinsic reaction coordinate (IRC) after TS/freq or directly.")
    p.add_argument("--irc-dx", type=float, default=0.1, help="IRC step size dx (default 0.1).")

    # ---- parse now (after all subparsers are defined) ----
    args = p.parse_args()

    # configure logging BEFORE any LOG.*
    setup_logging(verbose=args.verbose, debug=args.debug)
    LOG.info("CLI args: %s", vars(args))

    # ---- batch mode ----
    if args.cmd == "batch":
        from .batch import BatchCommon, run_batch_from_manifest, run_batch_from_glob
        common = BatchCommon(
            model=args.model,
            device=args.device,
            cache_dir=args.cache_dir,
            use_local_scratch=args.use_local_scratch,
            out_root=args.out_root,
            resume=args.resume,
        )
        overrides = {}
        if args.optimizer is not None: overrides["optimizer"] = args.optimizer
        if args.optts: overrides["optts"] = True
        if args.do_freq: overrides["do_freq"] = True
        if args.solv is not None: overrides["solv"] = args.solv
        if args.irc: overrides["irc"] = True
        overrides["irc_dx"] = args.irc_dx

        if args.manifest:
            summary = run_batch_from_manifest(args.manifest, common, **overrides)
        elif args.xyz_glob:
            summary = run_batch_from_glob(args.xyz_glob, common, **overrides)
        else:
            p.error("batch requires either --manifest or --xyz-glob")

        for item in summary:
            print(f"[{item['status']}] {item['xyz']} -> {item['out_dir']}")
        return

    #-------------------------------------------------------------------
    #              NON-BATCH — always go through ENSEMBLE workflow
    #-------------------------------------------------------------------
    if not args.xyz:
        p.error("--xyz is required unless using the 'batch' subcommand")

    # Respect explicit scratch-root if provided (ensemble reads UMA_SCRATCH_ROOT)
    os.environ.setdefault("UMA_SCRATCH_ROOT", args.scratch_root)

    # Default outdir if not provided
    if not args.outdir:
        args.outdir = os.path.splitext(os.path.basename(args.xyz))[0] + ".ensemble"
    os.makedirs(args.outdir, exist_ok=True)

    LOG.info("Route: ENSEMBLE (always) | frames from %s | outdir=%s", args.xyz, args.outdir)
    if args.solv:
        LOG.info("Gaussian inputs: solvent=%s | mem=%s | nproc=%s",
                 args.solv, args.gauss_mem, args.gauss_nproc)
    if args.freq:
        LOG.info("Frequencies per conformer will be computed (ORCA-style outputs per structure).")

    csv_path = run_conformer_workflow(
        args.xyz, out_dir=args.outdir,
        charge=args.charge, mult=args.mult,
        model=args.model, device=args.device,
        cache_dir=args.cache_dir, use_local_scratch=args.use_local_scratch,
        optimizer=(None if args.sp else args.opt), opt_mode=args.opt_mode, optts=args.optts,
        maxcycles=args.maxcycles,
        do_freq=args.freq,
        freq_delta=args.freq_delta, freq_nfree=args.freq_nfree, freq_scale=args.freq_scale,
        temp=args.temp,
        pressure_atm=args.pressure_atm,
        symmetry_number=args.symmetry_number, point_group=args.point_group,
        qrrho=args.qrrho, cutoff_cm1=args.cutoff_cm1,
        qrrho_ref_cm1=args.qrrho_ref_cm1, qrrho_alpha=args.qrrho_alpha,
        solv=args.solv, gauss_mem=args.gauss_mem, gauss_nproc=args.gauss_nproc,
        sella_internal=args.sella_internal, sella_eta=args.sella_eta,
        sella_gamma=args.sella_gamma, sella_delta0=args.sella_delta0,
        irc=args.irc, irc_dx=args.irc_dx,
    )
    print(f"\nEnsemble complete. Results CSV: {csv_path}")
    return


if __name__ == "__main__":
    main()

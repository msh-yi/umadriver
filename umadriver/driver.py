import argparse
import logging
import os, sys
from pathlib import Path

# TODO: time per step


## have to be before sella import!
def _early_parse_threads(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--sella-threads", type=int, default=0)
    p.add_argument("--jax-platform", choices=("cpu", "cuda"), default="cpu")
    ns, _ = p.parse_known_args(argv)

    os.environ.setdefault("JAX_PLATFORMS", ns.jax_platform)
    os.environ.setdefault("JAX_PLATFORM_NAME", ns.jax_platform)

    n = (
        ns.sella_threads
        or int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
        or (os.cpu_count() or 1)
    )

    flags = os.environ.get("XLA_FLAGS", "")
    add = "--xla_cpu_multi_thread_eigen=true"
    if ns.jax_platform == "cpu" and add not in flags:
        os.environ["XLA_FLAGS"] = (flags + " " + add).strip()

    # Assign, do not setdefault. SLURM (and many site profiles) export
    # OMP_NUM_THREADS=1 into the job environment, and setdefault is a no-op when
    # the variable already exists — so this block silently did nothing and
    # --sella-threads had no effect at all. Sella's BLAS work, and anything else
    # OpenMP-parallel, ran single-threaded regardless of what was requested.
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = str(n)


def _strip_batch_token(argv):
    """Accept an optional leading ``batch`` verb: `umadriver batch --manifest x.yaml`.

    Historically this "worked" only by accident — there are no subcommands, so
    `batch` was swallowed as a positional input and then ignored because
    --manifest short-circuits the positional list. That silently broke
    `umadriver batch mol.xyz`, which tried to open a file named "batch". Strip it
    explicitly instead; the bare `umadriver mol.xyz ...` form is unaffected.
    """
    if argv and argv[0] == "batch":
        return argv[1:]
    return argv


_early_parse_threads(_strip_batch_token(sys.argv[1:]))

from .utils import setup_logging, mode_type, optimizer_type, initialize_env
from .constants import DEFAULT_FAIRCHEM_CACHE, VAST_BASE
from .batch import BatchCommon, run_batch_from_manifest, run_batch_from_glob
from .vib_thermo import FREE_VOLUME_SOLVENTS
from .scan import parse_scan_spec
from .solvation import ALPB_METHODS

LOG = logging.getLogger("umadriver")


def main():
    p = argparse.ArgumentParser(
        description="UMA driver: always runs the ensemble workflow over 1+ XYZ inputs (batch is the only mode)."
    )

    # Register these so argparse won't complain if user passes them (already used in early parse)
    p.add_argument("--sella-threads", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument(
        "--jax-platform", choices=("cpu", "cuda"), default="cpu", help=argparse.SUPPRESS
    )

    # Inputs
    p.add_argument(
        "inputs",
        nargs="*",
        help="XYZ paths and/or glob patterns (e.g. inputs/*.xyz).",
    )
    p.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="YAML/JSON manifest (if set, positional inputs ignored).",
    )

    # Batch/common
    p.add_argument(
        "--out-root", type=str, default="runs", help="Root directory for job outputs."
    )
    p.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip jobs with existing energies.csv (default on).",
    )
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--model", default="uma-m-1p1")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    p.add_argument("--cache-dir", default=DEFAULT_FAIRCHEM_CACHE)
    p.add_argument("--use-local-scratch", action="store_true")

    p.add_argument(
        "--split-multi-structure",
        dest="split_multi_structure",
        action="store_true",
        default=True,
        help="Automatically split multi-structure XYZ files into separate jobs for parallel execution (default on).",
    )
    p.add_argument(
        "--no-split-multi-structure", dest="split_multi_structure", action="store_false"
    )
    p.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="Worker processes per GPU (default 1). >1 hides single-structure "
        "inference latency on large cards; costs VRAM linearly.",
    )

    # Per-job parameters (applied uniformly to every input unless you use manifest overrides)
    p.add_argument(
        "--scratch-root", default=os.environ.get("UMA_SCRATCH_ROOT", VAST_BASE)
    )
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--mult", type=int, default=1)

    p.add_argument("--opt-mode", type=mode_type, default=mode_type("Normal"))
    p.add_argument("--freq", action="store_true")
    p.add_argument(
        "--opt",
        type=optimizer_type,
        default=None,
        help="Optimizer to use. If omitted: default is OPT unless --freq is passed alone (then SP).",
    )
    p.add_argument(
        "--sp",
        action="store_true",
        help="Single point: no optimization. Pair with --freq-ts for frequencies "
        "on a transition state you do not want moved.",
    )
    p.add_argument(
        "--optts",
        action="store_true",
        help="Optimize to a first-order saddle. Implies Sella (order=1) — it is a "
        "saddle search, not a label, so the geometry WILL move.",
    )
    p.add_argument(
        "--freq-ts",
        dest="freq_ts",
        action="store_true",
        default=None,
        help="Score the frequency job as a transition state (expect one imaginary "
        "mode) without optimizing. Use with --sp; the route decides by default.",
    )
    p.add_argument("--maxcycles", type=int, default=300)
    p.add_argument(
        "--scan",
        nargs=5,
        metavar=("I", "J", "FROM", "TO", "STEPS"),
        default=None,
        help="Relaxed scan of the distance between atoms I and J, from FROM to TO "
        "angstroms in STEPS points. Atoms are numbered from 1 (the first atom in "
        "the file is 1). At each point the bond is held fixed and everything else "
        "is minimized. Writes scan/<tag>_scan.csv, the path as an XYZ trajectory, "
        "and the highest point on its own as a TS guess for --optts.",
    )

    p.add_argument("--sella-internal", dest="sella_internal", action="store_true")
    p.add_argument("--no-sella-internal", dest="sella_internal", action="store_false")
    p.set_defaults(sella_internal=True)
    p.add_argument("--sella-eta", type=float, default=2e-2)
    p.add_argument("--sella-gamma", type=float, default=1e-4)
    p.add_argument("--sella-delta0", type=float, default=0.02)

    p.add_argument("--freq-delta", type=float, default=0.01)
    p.add_argument("--freq-nfree", type=int, default=2)
    p.add_argument("--freq-scale", type=float, default=1.0)
    p.add_argument(
        "--freq-batch-size",
        type=int,
        default=1,
        help="Evaluate finite-difference displacements in batches of this size "
        "(default 1 = ASE's one-at-a-time Vibrations path). Large speedup for "
        "frequencies; costs VRAM, and multiplies with --workers-per-gpu.",
    )

    p.add_argument(
        "--freq-xtb-workers",
        type=int,
        default=None,
        help="Concurrent xtb calls during a solvated frequency run (default: "
        "cores // OMP_NUM_THREADS). xtb parallelizes poorly within a call, so "
        "throughput comes from running many single-threaded calls at once.",
    )

    p.add_argument("--temp", type=float, default=298.15)
    p.add_argument("--pressure-atm", type=float, default=1.00)
    p.add_argument("--symmetry-number", type=int, default=1)
    p.add_argument("--point-group", type=str, default=None)

    p.add_argument("--qrrho", dest="qrrho", action="store_true")
    p.add_argument("--no-qrrho", dest="qrrho", action="store_false")
    p.set_defaults(qrrho=True)
    p.add_argument("--cutoff-cm1", type=float, default=None)
    p.add_argument("--qrrho-ref-cm1", type=float, default=100.0)
    p.add_argument("--qrrho-alpha", type=float, default=4.0)

    p.add_argument("--conc-mol-l", type=float, default=None)

    p.add_argument(
        "--alpb",
        default=None,
        metavar="SOLVENT",
        help="Add an ALPB solvation correction from xtb: "
        "E = E_UMA + (E_xtb,alpb - E_xtb,vacuum). Applies to every energy AND "
        "force, so geometries optimize in solvent. Requires `tblite`.",
    )
    p.add_argument(
        "--alpb-method",
        default="GFN2-xTB",
        choices=list(ALPB_METHODS),
        help="xTB Hamiltonian used for the solvation correction (default GFN2-xTB). "
        "ALPB is not parameterized for GFN0.",
    )
    p.add_argument(
        "--no-alpb-concurrent",
        dest="alpb_concurrent",
        action="store_false",
        help="Evaluate UMA and the two xtb calls one after another instead of "
        "concurrently. Concurrency measured 2.5x faster on a 170-atom system.",
    )
    p.set_defaults(alpb_concurrent=True)

    p.add_argument(
        "--free-volume-solvent",
        default=None,
        choices=FREE_VOLUME_SOLVENTS,
        help="Solvent used for the free-volume translational-entropy correction. "
        "Only has an effect together with --conc-mol-l, and is unrelated to --alpb.",
    )

    p.add_argument("--irc", action="store_true")
    p.add_argument("--irc-dx", type=float, default=0.1)
    p.add_argument(
        "--irc-eta",
        type=float,
        default=None,
        help="Sella IRC eta (default: --sella-eta).",
    )
    p.add_argument(
        "--irc-gamma",
        type=float,
        default=None,
        help="Sella IRC gamma (default: --sella-gamma). Do not loosen this: Sella's own IRC default of 0.1 fails to resolve the negative mode and the IRC exits without stepping.",
    )
    p.add_argument("--irc-steps", type=int, default=200)

    p.add_argument("--verbose", action="store_true")
    p.add_argument("--debug", action="store_true")

    args = p.parse_args(_strip_batch_token(sys.argv[1:]))

    # --sp says "do not move the geometry" and --optts says "search for a saddle".
    # They cannot both be honored, and silently letting one win is how a job ends
    # up optimizing a structure the user meant to leave alone.
    if args.sp and args.optts:
        p.error(
            "--sp and --optts conflict: --optts optimizes to a saddle. For "
            "frequencies on a TS geometry you do not want moved, use --sp "
            "--freq --freq-ts."
        )
    if args.optts and args.opt not in (None, "Sella"):
        p.error(
            f"--optts is a saddle search and is Sella-only; got --opt {args.opt}. "
            "Drop --opt, or drop --optts to minimize instead."
        )
    if args.scan is not None:
        if args.optts:
            p.error(
                "--scan and --optts are different routes: a scan walks a "
                "coordinate with the bond constrained, --optts searches for a "
                "saddle with everything free. Scan first, then run --optts on the "
                "scan/*_scan_max.xyz it writes."
            )
        if args.sp:
            p.error(
                "--scan and --sp conflict: every scan point is a constrained "
                "minimization, so a scan moves atoms by definition."
            )
        # Fail here, on one structure's worth of argv, rather than inside a worker
        # once the batch is already running.
        try:
            parse_scan_spec(args.scan)
        except ValueError as e:
            p.error(str(e))

    setup_logging(verbose=args.verbose, debug=args.debug)
    # Sets HF_HUB_OFFLINE=1 when the model cache is already populated, so a healthy
    # run doesn't depend on the Hub being reachable (or the token being current).
    initialize_env()
    LOG.info("CLI args: %s", vars(args))

    # ensure scratch root visible to workers
    os.environ.setdefault("UMA_SCRATCH_ROOT", args.scratch_root)

    common = BatchCommon(
        model=args.model,
        device=args.device,
        cache_dir=args.cache_dir,
        use_local_scratch=args.use_local_scratch,
        out_root=args.out_root,
        resume=args.resume,
        split_multi_structure=args.split_multi_structure,
        workers_per_gpu=args.workers_per_gpu,
    )

    # Overrides are parameters of run_conformer_workflow()
    overrides = dict(
        charge=args.charge,
        mult=args.mult,
        # --optts implies Sella; say so here rather than passing None and relying
        # on the workflow to infer it.
        optimizer=("Sella" if args.optts else (None if args.sp else args.opt)),
        opt_mode=args.opt_mode,
        optts=args.optts,
        maxcycles=args.maxcycles,
        scan=args.scan,
        do_freq=args.freq,
        freq_ts=args.freq_ts,
        freq_delta=args.freq_delta,
        freq_nfree=args.freq_nfree,
        freq_scale=args.freq_scale,
        freq_batch_size=args.freq_batch_size,
        freq_xtb_workers=args.freq_xtb_workers,
        temp=args.temp,
        pressure_atm=args.pressure_atm,
        symmetry_number=args.symmetry_number,
        point_group=args.point_group,
        qrrho=args.qrrho,
        cutoff_cm1=args.cutoff_cm1,
        qrrho_ref_cm1=args.qrrho_ref_cm1,
        qrrho_alpha=args.qrrho_alpha,
        alpb=args.alpb,
        alpb_method=args.alpb_method,
        alpb_concurrent=args.alpb_concurrent,
        free_volume_solvent=args.free_volume_solvent,
        sella_internal=args.sella_internal,
        sella_eta=args.sella_eta,
        sella_gamma=args.sella_gamma,
        sella_delta0=args.sella_delta0,
        irc=args.irc,
        irc_dx=args.irc_dx,
        irc_eta=args.irc_eta,
        irc_gamma=args.irc_gamma,
        irc_steps=args.irc_steps,
        conc_mol_L=args.conc_mol_l,
        resume_from_per_conformer_csv=True,
    )

    if args.manifest:
        summary = run_batch_from_manifest(args.manifest, common, **overrides)
    else:
        if not args.inputs:
            p.error("Provide XYZ inputs (paths or globs), or use --manifest.")
        summary = run_batch_from_glob(args.inputs, common, **overrides)

    for item in summary:
        print(f"[{item['status']}] {item['xyz']} -> {item['out_dir']}")


if __name__ == "__main__":
    main()

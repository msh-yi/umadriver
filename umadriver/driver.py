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

    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, str(n))


_early_parse_threads(sys.argv[1:])

from .utils import setup_logging, mode_type, optimizer_type
from .constants import DEFAULT_FAIRCHEM_CACHE, VAST_BASE
from .batch import BatchCommon, run_batch_from_manifest, run_batch_from_glob

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
    p.add_argument("--sp", action="store_true")
    p.add_argument("--optts", action="store_true")
    p.add_argument("--maxcycles", type=int, default=300)

    p.add_argument("--sella-internal", dest="sella_internal", action="store_true")
    p.add_argument("--no-sella-internal", dest="sella_internal", action="store_false")
    p.set_defaults(sella_internal=True)
    p.add_argument("--sella-eta", type=float, default=2e-2)
    p.add_argument("--sella-gamma", type=float, default=1e-4)
    p.add_argument("--sella-delta0", type=float, default=0.02)

    p.add_argument("--freq-delta", type=float, default=0.01)
    p.add_argument("--freq-nfree", type=int, default=2)
    p.add_argument("--freq-scale", type=float, default=1.0)

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

    p.add_argument("--solv", default=None)
    p.add_argument("--gauss-mem", default="160GB")
    p.add_argument("--gauss-nproc", default="16")

    p.add_argument("--irc", action="store_true")
    p.add_argument("--irc-dx", type=float, default=0.1)

    p.add_argument("--verbose", action="store_true")
    p.add_argument("--debug", action="store_true")

    args = p.parse_args()

    setup_logging(verbose=args.verbose, debug=args.debug)
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
    )

    # Overrides are parameters of run_conformer_workflow()
    overrides = dict(
        charge=args.charge,
        mult=args.mult,
        optimizer=(None if args.sp else args.opt),
        opt_mode=args.opt_mode,
        optts=args.optts,
        maxcycles=args.maxcycles,
        do_freq=args.freq,
        freq_delta=args.freq_delta,
        freq_nfree=args.freq_nfree,
        freq_scale=args.freq_scale,
        temp=args.temp,
        pressure_atm=args.pressure_atm,
        symmetry_number=args.symmetry_number,
        point_group=args.point_group,
        qrrho=args.qrrho,
        cutoff_cm1=args.cutoff_cm1,
        qrrho_ref_cm1=args.qrrho_ref_cm1,
        qrrho_alpha=args.qrrho_alpha,
        solv=args.solv,
        gauss_mem=args.gauss_mem,
        gauss_nproc=args.gauss_nproc,
        sella_internal=args.sella_internal,
        sella_eta=args.sella_eta,
        sella_gamma=args.sella_gamma,
        sella_delta0=args.sella_delta0,
        irc=args.irc,
        irc_dx=args.irc_dx,
        conc_mol_L=args.conc_mol_l,
        resume_from_per_conformer_csv=True,
        split_multi_structure=args.split_multi_structure,
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

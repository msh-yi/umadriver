"""umadriver — UMA-driven geometry/frequency/IRC workflows.

Importing this package preloads the system OpenMP runtime when tblite is present.
That has to happen before torch is imported: torch ships its own bundled libgomp
which lacks the ``GOMP_5.0`` symbols tblite's compiled core needs, and whichever
copy loads first wins for the rest of the process. Without this, importing torch
before tblite leaves solvation dead with

    tblite C extension unimportable, cannot use C-API

The preload is skipped entirely when tblite is not installed, so this is a no-op
for gas-phase users.
"""

from __future__ import annotations


def _preload_system_openmp() -> None:
    import importlib.util

    if importlib.util.find_spec("tblite") is None:
        return  # no solvation support installed; leave the process alone

    import ctypes

    for soname in ("libgomp.so.1", "libgomp.so"):
        try:
            ctypes.CDLL(soname, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue
    # Nothing preloadable: tblite may still work if imported before torch.


_preload_system_openmp()
del _preload_system_openmp

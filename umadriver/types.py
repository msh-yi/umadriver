# umadriver/types.py
from dataclasses import dataclass
from typing import Optional, Union

LogLike = Optional[Union[str, int]]  # ASE accepts filename, "-", int fd, or None

@dataclass
class SellaOpts:
    internal: bool = True
    order: int = 0        # 0=min, 1=TS
    eta: float = 2e-2
    gamma: float = 1e-4
    delta0: float = 0.02
    logfile: LogLike = None  # NEW: pass through to Sella/ASE Optimizer

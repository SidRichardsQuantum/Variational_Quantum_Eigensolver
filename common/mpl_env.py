from __future__ import annotations

import os
from pathlib import Path


def ensure_mpl_config_dir() -> str:
    current = os.environ.get("MPLCONFIGDIR", "").strip()
    if current:
        target = Path(current)
    else:
        target = Path("/tmp/mplconfig")
        os.environ["MPLCONFIGDIR"] = str(target)

    target.mkdir(parents=True, exist_ok=True)
    return str(target)


ensure_mpl_config_dir()

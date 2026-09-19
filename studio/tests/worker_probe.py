"""Subprocess integration fixture: real solver with a filesystem-controlled pause."""

import os
import time
from pathlib import Path

from studio import adapter, worker

real_execute = adapter.execute


def execute(config, *, progress_callback):
    control = Path(os.environ["STUDIO_TEST_CONTROL"])
    if (control / "fail").exists():
        raise ValueError("actual solver failure")

    def report(event):
        progress_callback(event)
        if event.get("iteration") == 1 and not (control / "release").exists():
            (control / "observed").write_text(str(os.getpid()))
            deadline = time.monotonic() + 60
            while not (control / "release").exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("Test did not release worker")
                time.sleep(0.02)

    return real_execute(config, progress_callback=report)


if __name__ == "__main__":
    adapter.execute = execute
    worker.main()

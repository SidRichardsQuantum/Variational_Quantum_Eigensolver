"""Process ownership of a Studio data directory, released on process exit."""

import errno
import os


class DirectoryLock:
    def __init__(self, directory):
        # Keep the file in place: unlinking it could let two processes lock
        # different inodes. The OS lock, not the file's existence, is ownership.
        self.file = (directory / ".owner.lock").open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise RuntimeError(
                    f"Another Studio instance owns {directory}. "
                    "Stop it or choose a different VQE_PENNYLANE_DATA_DIR."
                ) from exc
            raise

    def close(self):
        # Closing the handle also releases the lock on both supported platforms.
        self.file.close()

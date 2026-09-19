"""Private subprocess protocol; scientific output stays in a staging directory."""

import contextlib
import json
import os
import sys
import threading


def main():
    config = json.loads(sys.stdin.readline())

    def watch_parent():
        # The server owns the only writer. An abrupt server exit closes it too.
        while os.read(sys.stdin.fileno(), 1):
            pass
        os._exit(1)

    threading.Thread(target=watch_parent, daemon=True).start()
    protocol = sys.stdout

    def send(kind, value):
        protocol.write(json.dumps({"kind": kind, "value": value}) + "\n")
        protocol.flush()

    with contextlib.redirect_stdout(sys.stderr):
        from studio.adapter import execute, json_bytes
        from studio.history import artifacts, find_artifact

        try:
            result = execute(
                config,
                progress_callback=lambda event: send(
                    "progress", json.loads(json_bytes(event))
                ),
            )
            json_bytes(result)
            name = find_artifact(config, result)
            send(
                "result",
                {
                    "artifact_name": name,
                    "artifact_digest": artifacts()[name]["artifact_digest"],
                    "invocation": {
                        k: result[k]
                        for k in ("runtime_s", "compute_runtime_s", "cache_hit")
                        if k in result
                    },
                },
            )
        except Exception as exc:
            send("error", f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()

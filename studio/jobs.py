"""Single-worker lifecycle. Manifests contain references, never scientific results."""

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common.paths import data_root
from common.persist import atomic_write_json, read_json, stable_hash_dict

from .adapter import json_bytes, normalize
from .history import artifact_digest, artifacts, timestamp
from .ownership import DirectoryLock

WORKER_COMMAND = [sys.executable, "-m", "studio.worker"]


class Jobs:
    def __init__(self):
        self.directory = data_root() / "results" / "studio"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.owner = DirectoryLock(self.directory)
        self.lock = threading.RLock()
        self.rows = {}
        self.progress = {}
        self.processes = {}
        self.closed = False
        try:
            self.recover()
            self.pool = ThreadPoolExecutor(max_workers=1)
        except BaseException:
            self.owner.close()
            raise

    def recover(self):
        """Recover interrupted work only while holding directory ownership."""
        for stale in self.directory.glob("worker-*"):
            if stale.is_dir() and not stale.is_symlink():
                shutil.rmtree(stale)
        for path in self.directory.glob("*.json"):
            try:
                row = read_json(path)
                if row["status"] in {"submitted", "running"}:
                    row.update(
                        status="failed",
                        finished_at=timestamp(),
                        error="Studio stopped before execution was recorded as complete.",
                    )
                    atomic_write_json(path, row)
                self.rows[row["id"]] = row
            except (OSError, ValueError, KeyError, TypeError):
                continue

    def save(self, row):
        atomic_write_json(self.directory / f"{row['id']}.json", row)

    def submit(self, raw):
        config = normalize(raw)
        with self.lock:
            if self.closed:
                raise ValueError("Studio is shutting down")
            if (
                sum(r["status"] in {"submitted", "running"} for r in self.rows.values())
                >= 8
            ):
                raise ValueError(
                    "Execution queue is full (8 runs); wait for a run to finish"
                )
            row = {
                "id": uuid.uuid4().hex,
                "status": "submitted",
                "config": config,
                "config_id": stable_hash_dict(config),
                "timestamp": timestamp(),
                "timestamp_source": "submission",
                "method": config["method"],
            }
            self.save(row)
            self.rows[row["id"]] = row
            response = copy.deepcopy(row)
            self.pool.submit(self.work, row["id"])
            return response

    def cancel(self, run_id):
        """First terminal transition wins; repeated cancellation is idempotent."""
        with self.lock:
            if run_id not in self.rows:
                raise KeyError(run_id)
            row = self.rows[run_id]
            if row["status"] in {"submitted", "running"}:
                row.update(
                    status="cancelled",
                    cancelled_at=timestamp(),
                    finished_at=timestamp(),
                )
                self.save(row)
                self.progress.pop(run_id, None)
                process = self.processes.get(run_id)
                if process is not None and process.poll() is None:
                    process.kill()
            return copy.deepcopy(row)

    def work(self, run_id):
        with self.lock:
            row = self.rows[run_id]
            if row["status"] != "submitted":
                return
            row.update(status="running", started_at=timestamp())
            self.save(row)
        process = None
        try:
            with tempfile.TemporaryDirectory(
                prefix="worker-", dir=self.directory
            ) as staging:
                root = Path(staging)
                cache = root / "results" / "vqe"
                cache.mkdir(parents=True)
                # Copy, never hard-link: a worker cannot overwrite the shared cache.
                for source in (self.directory.parent / "vqe").glob("*.json"):
                    if source.is_file() and not source.is_symlink():
                        shutil.copy2(source, cache / source.name)
                env = os.environ.copy()
                env["VQE_PENNYLANE_DATA_DIR"] = staging
                env["PYTHONPATH"] = (
                    str(Path(__file__).resolve().parent.parent)
                    + os.pathsep
                    + env.get("PYTHONPATH", "")
                )
                with (root / "worker.log").open("w+") as log:
                    with self.lock:
                        if row["status"] == "cancelled":
                            return
                        process = subprocess.Popen(
                            WORKER_COMMAND,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=log,
                            text=True,
                            cwd=staging,
                            env=env,
                        )
                        self.processes[run_id] = process
                    try:
                        process.stdin.write(json_bytes(row["config"]).decode() + "\n")
                        process.stdin.flush()
                        result = None
                        error = None
                        for line in process.stdout:
                            message = json.loads(line)
                            if message["kind"] == "progress":
                                self.report(run_id, message["value"])
                            elif message["kind"] == "result":
                                result = message["value"]
                            elif message["kind"] == "error":
                                error = message["value"]
                        code = process.wait()
                        if error:
                            raise RuntimeError(error)
                        if code or result is None:
                            raise RuntimeError(
                                f"Scientific worker exited without a result (exit {code})"
                            )
                        name = result["artifact_name"]
                        if Path(name).name != name:
                            raise ValueError("Invalid worker artifact name")
                        record = read_json(cache / name)
                        if artifact_digest(record) != result["artifact_digest"]:
                            raise ValueError("Worker artifact digest mismatch")
                        with self.lock:
                            if row["status"] != "running":
                                return
                            destination = self.directory.parent / "vqe" / name
                            # Publishing and completing share the cancellation lock.
                            if (
                                not destination.exists()
                                or artifact_digest(read_json(destination))
                                != result["artifact_digest"]
                            ):
                                atomic_write_json(destination, record)
                            row.update(
                                **result, status="completed", finished_at=timestamp()
                            )
                            self.progress.pop(run_id, None)
                            self.save(row)
                    finally:
                        if process.poll() is None:
                            process.kill()
                        process.wait()
                        process.stdin.close()
                        process.stdout.close()
        except Exception as exc:
            with self.lock:
                if row["status"] == "running":
                    row.update(status="failed", error=str(exc), finished_at=timestamp())
                    self.save(row)
        finally:
            with self.lock:
                self.processes.pop(run_id, None)
                self.progress.pop(run_id, None)

    def report(self, run_id, event):
        """Keep actual intermediate observations in memory, separate from results."""
        event = json.loads(json_bytes(event))
        with self.lock:
            if self.rows[run_id]["status"] != "running":
                return
            progress = self.progress.setdefault(
                run_id, {"energies": [], "outer_energies": []}
            )
            previous_outer = progress.get("outer_iteration")
            if (
                event["phase"] == "inner_optimization"
                and previous_outer != event["outer_iteration"]
            ):
                progress["energies"] = []
            if event["phase"] in {"optimization", "inner_optimization"}:
                progress["energies"].append(event["energy"])
            if event["phase"] == "outer_completed":
                progress["outer_energies"].append(event["energy"])
            progress.update(event, updated_at=timestamp())

    def history(self):
        with self.lock:
            rows = copy.deepcopy(list(self.rows.values()))
            for row in rows:
                if row["id"] in self.progress:
                    row["progress"] = copy.deepcopy(self.progress[row["id"]])
        records = artifacts()
        referenced = set()
        for row in rows:
            name = row.get("artifact_name")
            if name:
                referenced.add(name)
                artifact = records.get(name)
                if artifact and artifact["artifact_digest"] == row.get(
                    "artifact_digest"
                ):
                    row.update(
                        {
                            k: artifact[k]
                            for k in (
                                "artifact",
                                "signature",
                                "resolved_config",
                                "result",
                            )
                        }
                    )
                else:
                    row.update(
                        status="failed",
                        error="Referenced scientific artifact is missing, changed, invalid, or from an older cache schema.",
                    )
        rows.extend(r for name, r in records.items() if name not in referenced)
        return sorted(rows, key=lambda row: row["timestamp"], reverse=True)

    def close(self, *, cancel=True):
        """Stop accepting work; cancel by default, or explicitly drain for batch use."""
        with self.lock:
            self.closed = True
            if cancel:
                for run_id in self.rows:
                    self.cancel(run_id)
        self.pool.shutdown(wait=True)
        with self.lock:
            self.owner.close()

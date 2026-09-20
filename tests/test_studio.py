"""Studio boundary tests, including real H2 cache integration."""

import inspect
import json
import threading
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from studio import adapter
from studio.history import artifacts, find_artifact
from studio.jobs import Jobs
from studio.server import make_server
from vqe.core import run_vqe


def test_h2_maps_to_public_api(monkeypatch):
    calls = []
    monkeypatch.setattr(
        adapter, "run_vqe", lambda **kwargs: calls.append(kwargs) or {"energy": -1.0}
    )
    raw = {
        "method": "vqe",
        "problem": {"molecule": "H2", "mapping": "parity"},
        "settings": {
            "ansatz": "UCCSD",
            "optimizer": "Adam",
            "steps": 3,
            "stepsize": None,
            "seed": 42,
        },
    }
    assert adapter.execute(raw) == {"energy": -1.0}
    assert calls == [
        dict(
            molecule="H2",
            mapping="parity",
            basis="sto-3g",
            ansatz_name="UCCSD",
            optimizer_name="Adam",
            steps=3,
            stepsize=None,
            seed=42,
            plot=False,
        )
    ]


@pytest.mark.parametrize(
    "raw",
    [
        None,
        [],
        {"method": "qpe"},
        {"force": True},
        {"problem": {"molecule": "invented"}},
        {"problem": {"basis": "6-31g"}},
        {"problem": {"mapping": "bad"}},
        {"settings": {"steps": True}},
        {"settings": {"steps": 1.5}},
        {"settings": {"steps": -1}},
        {"settings": {"steps": 10**400}},
        {"settings": {"stepsize": 0}},
        {"settings": {"stepsize": float("nan")}},
        {"settings": {"seed": 2**32}},
        {"settings": {"seed": "0"}},
        {"settings": {"optimizer": "fake"}},
        {"settings": {"ansatz": "fake"}},
        {"settings": {"extra": 1}},
        {"settings": None},
    ],
)
def test_invalid_configuration(raw):
    with pytest.raises(ValueError):
        adapter.normalize(raw)


def test_defaults_match_python_and_registry():
    from common.molecules import MOLECULES

    kwargs = adapter.to_kwargs({})
    for name, value in kwargs.items():
        if name != "plot":
            assert value == inspect.signature(run_vqe).parameters[name].default
    cat = adapter.catalogue()
    molecule = next(f for f in cat["methods"][0]["fields"] if f["name"] == "molecule")
    assert molecule["values"] == list(MOLECULES)
    assert (
        adapter.normalize({"problem": {"molecule": "LiH"}})["problem"]["basis"]
        == MOLECULES["LiH"]["basis"].lower()
    )


def test_serialization_deterministic_json_safe():
    a = {"energy": np.float64(-1.23), "values": np.array([1, 2]), "seed": np.int64(0)}
    b = {"seed": 0, "values": [1, 2], "energy": -1.23}
    assert adapter.json_bytes(a) == adapter.json_bytes(b)
    assert json.loads(adapter.json_bytes(a)) == b
    with pytest.raises(ValueError):
        adapter.json_bytes({"energy": float("nan")})


def test_real_h2_defaults_cache_and_artifact():
    config = adapter.normalize({"settings": {"steps": 1}})
    fresh = adapter.execute(config)
    cached = adapter.execute(config)
    direct = run_vqe(steps=1, plot=False)
    assert fresh["cache_hit"] is False
    assert cached["cache_hit"] is direct["cache_hit"] is True
    assert fresh["energies"] == cached["energies"] == direct["energies"]
    assert cached["compute_runtime_s"] == fresh["compute_runtime_s"]
    name = find_artifact(config, cached)
    rows = artifacts()
    assert len(rows) == 1
    assert rows[name]["result"]["cache_hit"] is False
    assert adapter.json_bytes(cached)


def wait_for(predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for worker")


def use_probe(monkeypatch, tmp_path):
    control = tmp_path / "control"
    control.mkdir()
    monkeypatch.setenv("STUDIO_TEST_CONTROL", str(control))
    monkeypatch.setattr(
        "studio.jobs.WORKER_COMMAND",
        [sys.executable, "-m", "studio.tests.worker_probe"],
    )
    return control


def test_jobs_lifecycle_and_restart():
    jobs = Jobs()
    submitted = jobs.submit({"settings": {"steps": 0}})
    assert submitted["status"] == "submitted"
    jobs.close(cancel=False)
    row = jobs.history()[0]
    assert row["status"] == "completed", row.get("error")
    assert row["invocation"]["cache_hit"] is False
    manifest = json.loads((jobs.directory / f"{row['id']}.json").read_text())
    assert "result" not in manifest
    restarted = Jobs()
    try:
        assert restarted.history()[0]["signature"] == row["signature"]
        second = restarted.submit(row["config"])
    finally:
        restarted.close(cancel=False)
    second_row = next(r for r in restarted.history() if r["id"] == second["id"])
    assert second_row["invocation"]["cache_hit"] is True
    assert second_row["config_id"] == row["config_id"]
    assert restarted.cancel(second["id"])["status"] == "completed"


def test_second_instance_preserves_active_jobs(monkeypatch, tmp_path):
    import subprocess

    control = use_probe(monkeypatch, tmp_path)
    jobs = Jobs()
    try:
        running = jobs.submit({"settings": {"steps": 2}})
        wait_for((control / "observed").exists)
        queued = jobs.submit({"settings": {"steps": 3}})
        manifests = {p: p.read_bytes() for p in jobs.directory.glob("*.json")}
        staging = list(jobs.directory.glob("worker-*"))
        assert staging
        with pytest.raises(RuntimeError, match="Another Studio instance"):
            Jobs()
        check = subprocess.run(
            [sys.executable, "-c", "from studio.jobs import Jobs; Jobs()"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert check.returncode != 0
        assert "Another Studio instance" in check.stderr
        assert {p: p.read_bytes() for p in manifests} == manifests
        assert all(p.is_dir() for p in staging)
        assert jobs.rows[running["id"]]["status"] == "running"
        assert jobs.rows[queued["id"]]["status"] == "submitted"
    finally:
        (control / "release").touch()
        jobs.close(cancel=False)
    assert all(row["status"] == "completed" for row in jobs.history())


def test_startup_failure_releases_directory_ownership(monkeypatch):
    def fail(*args):
        raise OSError("startup failed")

    with monkeypatch.context() as patch:
        patch.setattr(Jobs, "recover", fail)
        with pytest.raises(OSError, match="startup failed"):
            Jobs()
    with monkeypatch.context() as patch:
        patch.setattr("studio.server.ThreadingHTTPServer", fail)
        with pytest.raises(OSError, match="startup failed"):
            make_server()
    jobs = Jobs()
    jobs.close()
    jobs.close()


def test_failed_and_interrupted_jobs(monkeypatch, tmp_path):
    control = use_probe(monkeypatch, tmp_path)
    (control / "fail").touch()
    jobs = Jobs()
    row = jobs.submit({})
    jobs.close(cancel=False)
    assert jobs.history()[0]["error"] == "ValueError: actual solver failure"
    row["status"] = "running"
    jobs.save(row)
    restarted = Jobs()
    try:
        assert restarted.history()[0]["status"] == "failed"
        assert "stopped" in restarted.history()[0]["error"]
        assert restarted.history()[0]["finished_at"]
    finally:
        restarted.close()


def test_missing_and_corrupt_artifacts(tmp_path):
    jobs = Jobs()
    row = jobs.submit({"settings": {"steps": 0}})
    jobs.close(cancel=False)
    artifact = tmp_path / jobs.history()[0]["artifact"]
    artifact.write_text("invalid JSON")
    assert artifacts() == {}
    assert jobs.history()[0]["status"] == "failed"
    assert jobs.history()[0]["id"] == row["id"]


def test_http_catalogue_validation_and_static_files():
    server = make_server(port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def get(path):
        with urlopen(base + path, timeout=10) as response:
            return response.read()

    try:
        catalogue = json.loads(get("/api/catalogue"))
        assert catalogue["methods"][0]["id"] == "vqe"
        assert b"Experiment composer" in get("/")
        assert b"renderComposer" in get("/app.js")
        assert json.loads(get("/api/runs")) == []
        for path in ["/../../pyproject.toml", "/api/runs/unknown"]:
            with pytest.raises(HTTPError) as error:
                get(path)
            assert error.value.code == 404
        request = Request(
            base + "/api/runs", data=b"{}", headers={"Content-Type": "application/json"}
        )
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 403
        request.add_header("X-Studio-Token", catalogue["submission_token"])
        request.data = b'{"settings":{"steps":-1}}'
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 400
        request.data = b'{"settings":{"steps":0}}'
        with urlopen(request) as response:
            assert response.status == 202
            submitted = json.load(response)
        assert submitted["status"] == "submitted"
        cancel_url = base + f"/api/runs/{submitted['id']}/cancel"
        with pytest.raises(HTTPError) as error:
            urlopen(Request(cancel_url, method="POST"))
        assert error.value.code == 403
        headers = {"X-Studio-Token": catalogue["submission_token"]}
        with urlopen(Request(cancel_url, method="POST", headers=headers)) as response:
            cancelled = json.load(response)
        assert cancelled["status"] in {"cancelled", "completed"}
        with urlopen(Request(cancel_url, method="POST", headers=headers)) as response:
            assert json.load(response) == cancelled
        with pytest.raises(HTTPError) as error:
            urlopen(
                Request(
                    base + "/api/runs/unknown/cancel", method="POST", headers=headers
                )
            )
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        server.jobs.close(cancel=False)
        thread.join()


def test_replaced_artifact_is_not_presented_as_original_result(tmp_path):
    jobs = Jobs()
    jobs.submit({"settings": {"steps": 0}})
    jobs.close(cancel=False)
    path = tmp_path / jobs.history()[0]["artifact"]
    record = json.loads(path.read_text())
    record["result"]["energy"] = -99.0
    path.write_text(json.dumps(record))
    row = jobs.history()[0]
    assert row["status"] == "failed"
    assert "changed" in row["error"]
    assert "result" not in row


def test_adapt_catalogue_and_adapter_defaults(monkeypatch):
    from vqe.adapt import run_adapt_vqe

    kwargs = adapter.to_kwargs({"method": "adapt_vqe"})
    assert "basis" not in kwargs
    assert "ansatz_name" not in kwargs
    for name, value in kwargs.items():
        if name != "plot":
            assert value == inspect.signature(run_adapt_vqe).parameters[name].default
    calls = []

    def callback(event):
        pass

    monkeypatch.setattr(adapter, "run_adapt_vqe", lambda **kw: calls.append(kw) or {})
    adapter.execute({"method": "adapt_vqe"}, progress_callback=callback)
    assert calls == [{**kwargs, "progress_callback": callback}]
    for settings in (
        {"steps": 2},
        {"ansatz": "UCCSD"},
        {"pool": "invalid"},
        {"inner_stepsize": None},
        {"max_ops": -1},
        {"grad_tol": -1},
    ):
        with pytest.raises(ValueError):
            adapter.normalize({"method": "adapt_vqe", "settings": settings})


def test_adapt_jobs_artifacts_and_rerun_cache():
    jobs = Jobs()
    first = jobs.submit(
        {"method": "adapt_vqe", "settings": {"max_ops": 1, "inner_steps": 2}}
    )
    jobs.close(cancel=False)
    row = jobs.history()[0]
    assert row["id"] == first["id"]
    assert row["status"] == "completed", row.get("error")
    assert row["method"] == "adapt_vqe"
    assert row["resolved_config"]["adapt_pool"] == "uccsd"
    assert row["result"]["selected_operators"]
    assert row["invocation"]["cache_hit"] is False
    restarted = Jobs()
    restarted.submit(row["config"])
    restarted.close(cancel=False)
    assert restarted.history()[0]["invocation"]["cache_hit"] is True
    assert all(r["method"] == "adapt_vqe" for r in artifacts().values())


@pytest.mark.parametrize(
    "method, settings",
    [("vqe", {"steps": 2}), ("adapt_vqe", {"max_ops": 1, "inner_steps": 2})],
)
def test_live_observations_are_visible_and_not_persisted(
    monkeypatch, tmp_path, method, settings
):
    control = use_probe(monkeypatch, tmp_path)
    jobs = Jobs()
    try:
        submitted = jobs.submit({"method": method, "settings": settings})
        wait_for((control / "observed").exists)
        row = wait_for(
            lambda: next(
                (
                    r
                    for r in jobs.history()
                    if r.get("progress", {}).get("iteration") == 1
                ),
                None,
            )
        )
        assert row["status"] == "running"
        assert "result" not in row
        assert len(row["progress"]["energies"]) == 2
        assert "progress" not in json.loads(
            (jobs.directory / f"{submitted['id']}.json").read_text()
        )
    finally:
        (control / "release").touch()
        jobs.close(cancel=False)
    row = jobs.history()[0]
    assert row["status"] == "completed", row.get("error")
    assert "progress" not in row


def test_cancel_queue_running_and_restart(monkeypatch, tmp_path):
    control = use_probe(monkeypatch, tmp_path)
    # Existing scientific artifacts must survive cancellation byte-for-byte.
    adapter.execute({"settings": {"steps": 0}})
    existing = {p: p.read_bytes() for p in (tmp_path / "results/vqe").glob("*.json")}
    jobs = Jobs()
    try:
        running = jobs.submit({"settings": {"steps": 2}})
        wait_for((control / "observed").exists)
        process = jobs.processes[running["id"]]
        queued = jobs.submit({"settings": {"steps": 3}})
        next_run = jobs.submit({"settings": {"steps": 0}})
        cancelled = jobs.cancel(queued["id"])
        assert cancelled["status"] == "cancelled"
        assert "started_at" not in cancelled
        assert jobs.cancel(queued["id"]) == cancelled
        assert jobs.cancel(running["id"])["status"] == "cancelled"
        wait_for(lambda: process.poll() is not None)
        jobs.close(cancel=False)
        rows = {r["id"]: r for r in jobs.history()}
        assert rows[next_run["id"]]["status"] == "completed"
        for run in (running, queued):
            row = rows[run["id"]]
            assert row["status"] == "cancelled"
            assert row["cancelled_at"] and row["finished_at"]
            assert "result" not in row and "progress" not in row
        assert {p: p.read_bytes() for p in existing} == existing
        assert len(artifacts()) == 1
        assert not list(jobs.directory.glob("worker-*"))
    finally:
        jobs.close()
    restarted = Jobs()
    try:
        assert sum(r["status"] == "cancelled" for r in restarted.history()) == 2
        with pytest.raises(KeyError):
            restarted.cancel("unknown")
    finally:
        restarted.close()


def test_shutdown_cancels_and_rejects_new_work(monkeypatch, tmp_path):
    control = use_probe(monkeypatch, tmp_path)
    jobs = Jobs()
    jobs.submit({"settings": {"steps": 2}})
    wait_for((control / "observed").exists)
    jobs.submit({"settings": {"steps": 3}})
    started = time.monotonic()
    jobs.close()
    assert time.monotonic() - started < 5
    assert all(r["status"] == "cancelled" for r in jobs.history())
    assert artifacts() == {}
    assert not jobs.processes
    with pytest.raises(ValueError, match="shutting down"):
        jobs.submit({})


def test_cancellation_wins_before_artifact_publication(monkeypatch):
    from studio.history import artifact_digest

    ready, release = threading.Event(), threading.Event()

    def pause_before_commit(record):
        ready.set()
        assert release.wait(30)
        return artifact_digest(record)

    monkeypatch.setattr("studio.jobs.artifact_digest", pause_before_commit)
    jobs = Jobs()
    try:
        row = jobs.submit({"settings": {"steps": 0}})
        assert ready.wait(30)
        assert jobs.cancel(row["id"])["status"] == "cancelled"
    finally:
        release.set()
        jobs.close()
    assert artifacts() == {}
    assert jobs.history()[0]["status"] == "cancelled"


def test_abrupt_parent_exit_stops_worker(monkeypatch, tmp_path):
    import os
    import subprocess

    control = use_probe(monkeypatch, tmp_path)
    script = """
import sys, time
import studio.jobs
studio.jobs.WORKER_COMMAND = [sys.executable, '-m', 'studio.tests.worker_probe']
jobs = studio.jobs.Jobs()
jobs.submit({'settings': {'steps': 2}})
time.sleep(120)
"""
    parent = subprocess.Popen([sys.executable, "-c", script])
    try:
        wait_for((control / "observed").exists)
        worker_pid = int((control / "observed").read_text())
        parent.kill()
        parent.wait(timeout=5)

        def worker_stopped():
            try:
                os.kill(worker_pid, 0)
            except ProcessLookupError:
                return True
            # Linux may retain an orphan zombie until init reaps it.
            stat = Path(f"/proc/{worker_pid}/stat")
            return stat.exists() and stat.read_text().split()[2] == "Z"

        wait_for(worker_stopped, timeout=10)
        restarted = Jobs()
        try:
            assert restarted.history()[0]["status"] == "failed"
            assert artifacts() == {}
            assert not list(restarted.directory.glob("worker-*"))
        finally:
            restarted.close()
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait()

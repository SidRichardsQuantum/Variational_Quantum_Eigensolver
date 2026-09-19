"""Progress observers report computed samples without changing solver semantics."""

import json

import pytest

from common.paths import results_dir
from vqe import run_vqe, run_adapt_vqe


@pytest.mark.parametrize(
    "runner, options",
    [
        (run_vqe, {"steps": 2}),
        (run_adapt_vqe, {"max_ops": 1, "inner_steps": 2}),
    ],
)
def test_progress_matches_results_and_preserves_cache(runner, options):
    events = []
    result = runner(plot=False, progress_callback=events.append, **options)
    assert result["cache_hit"] is False
    if runner is run_vqe:
        assert [e["iteration"] for e in events] == [0, 1, 2]
        assert [e["energy"] for e in events] == result["energies"]
    else:
        outer = [e for e in events if e["phase"] == "outer_completed"]
        assert [e["energy"] for e in outer] == result["energies"]
        for index, trajectory in enumerate(result["inner_energies"]):
            inner = [
                e
                for e in events
                if e["phase"] == "inner_optimization" and e["outer_iteration"] == index
            ]
            assert [e["energy"] for e in inner] == trajectory
            assert [e["iteration"] for e in inner] == list(range(len(trajectory)))
    paths = list(results_dir("vqe").glob("*.json"))
    assert len(paths) == 1
    before = paths[0].read_bytes()
    cache_events = []
    cached = runner(plot=False, progress_callback=cache_events.append, **options)
    assert cached["cache_hit"] is True
    assert len(cache_events) == 1
    assert cache_events[0]["phase"] == "cache_hit"
    assert cache_events[0]["energy"] == result["energy"]
    assert paths[0].read_bytes() == before
    fresh = runner(plot=False, force=True, **options)
    assert fresh["energies"] == result["energies"]
    assert list(results_dir("vqe").glob("*.json")) == paths


@pytest.mark.parametrize(
    "runner, options", [(run_vqe, {"steps": 0}), (run_adapt_vqe, {"max_ops": 0})]
)
def test_callback_is_detached_and_errors_propagate(runner, options):
    def mutate(event):
        event["energy"] = 999

    result = runner(plot=False, progress_callback=mutate, **options)
    assert result["energy"] != 999

    def fail(event):
        raise RuntimeError("observer failed")

    with pytest.raises(RuntimeError, match="observer failed"):
        runner(plot=False, seed=7, progress_callback=fail, **options)
    assert len(list(results_dir("vqe").glob("*.json"))) == 1


def test_adapt_legacy_cache_remains_reusable_without_invented_compute_time():
    original = run_adapt_vqe(max_ops=0, plot=False)
    path = next(results_dir("vqe").glob("*_adapt.json"))
    record = json.loads(path.read_text())
    for key in ("runtime_s", "compute_runtime_s", "cache_hit"):
        record["result"].pop(key, None)
    path.write_text(json.dumps(record))
    before = path.read_bytes()
    cached = run_adapt_vqe(max_ops=0, plot=False)
    assert cached["cache_hit"] is True
    assert cached["runtime_s"] >= 0
    assert "compute_runtime_s" not in cached
    assert cached["energy"] == original["energy"]
    assert path.read_bytes() == before

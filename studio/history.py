"""Read-only discovery of authoritative VQE records and derived submission metadata."""

from datetime import datetime, timezone

from common.paths import data_root, results_dir
from common.persist import read_json
from qite import io_utils as qite_io
from vqe import io_utils as vqe_io

from .adapter import json_bytes


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def artifacts():
    records = {}
    for path in list(results_dir("vqe").glob("*.json")) + list(
        results_dir("qite").glob("*.json")
    ):
        try:
            if path.is_symlink():
                continue
            record = read_json(path)
            json_bytes(record)  # Exclude malformed/non-finite scientific artifacts.
            cfg, result = record["config"], record["result"]
            # VQE-family methods share a directory. Accept only the standard VQE
            # filename and result shape, with the current scientific signature.
            method = (
                "varqite"
                if path.parent.name == "qite"
                else ("adapt_vqe" if "adapt_pool" in cfg else "vqe")
            )
            io = qite_io if method == "varqite" else vqe_io
            signature = io.run_signature(cfg)
            prefix = io.make_filename_prefix(
                cfg,
                noisy=bool(cfg.get("noise")),
                seed=cfg["seed"],
                hash_str=signature,
                algo="varqite" if method == "varqite" else "vqe",
            )
            if method == "adapt_vqe":
                prefix += "_adapt"
            required = {"energy", "energies", "final_params"}
            required |= (
                {"inner_energies", "selected_operators", "max_gradients"}
                if method == "adapt_vqe"
                else (
                    {"varqite", "final_params_shape"}
                    if method == "varqite"
                    else {"params_history"}
                )
            )
            if path.stem != prefix or not required <= result.keys():
                continue
            records[path.name] = {
                "id": path.stem,
                "status": (
                    "failed"
                    if result.get("termination", {}).get("reason")
                    == "numerical_failure"
                    else "completed"
                ),
                "method": method,
                "artifact": str(path.relative_to(data_root())),
                "signature": signature,
                "resolved_config": cfg,
                "result": result,
                "artifact_digest": artifact_digest(record),
                "timestamp": datetime.fromtimestamp(
                    path.stat().st_mtime, timezone.utc
                ).isoformat(),
                "timestamp_source": "artifact modification time",
                "config": None,
            }
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return records


def artifact_digest(record):
    """Detect external replacement of a referenced cache artifact."""
    import hashlib

    return hashlib.sha256(json_bytes(record)).hexdigest()


def find_artifact(experiment, result):
    """Associate the returned result with its existing artifact without re-resolving chemistry."""
    from vqe.optimizer import get_optimizer_stepsize

    p, s = experiment["problem"], experiment["settings"]
    matches = []
    for name, row in artifacts().items():
        c, r = row["resolved_config"], row["result"]
        if row["method"] != experiment["method"]:
            continue
        if result.get("config") is not None:
            # ADAPT returns its authoritative resolved config; no reconstruction.
            if c == result.get("config") and all(
                r.get(k) == result.get(k)
                for k in (
                    "energies",
                    "final_params",
                    "selected_operators",
                    "termination",
                )
            ):
                matches.append(name)
            continue
        opt = c["optimizer"]
        if (
            c.get("molecule") == p["molecule"]
            and c.get("mapping") == p["mapping"]
            and c.get("basis") == p["basis"]
            and c.get("seed") == s["seed"]
            and c.get("ansatz") == s["ansatz"]
            and not c.get("noise")
            and opt["name"] == s["optimizer"]
            and opt["iterations_planned"] == s["steps"]
            and opt["stepsize"]
            == (
                get_optimizer_stepsize(s["optimizer"])
                if s["stepsize"] is None
                else s["stepsize"]
            )
            and all(
                r.get(k) == result.get(k)
                for k in ("energies", "final_params", "compute_runtime_s")
            )
        ):
            matches.append(name)
    if len(matches) != 1:
        raise RuntimeError(
            "Could not uniquely associate the solver result with its authoritative artifact"
        )
    return matches[0]

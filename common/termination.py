"""Finite, serializable outcomes for fixed-budget and optionally stopped solvers."""

import numpy as np


def stopping_config(steps, energy_tol=None, patience=1):
    if isinstance(steps, bool) or int(steps) != steps or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    if isinstance(patience, bool) or int(patience) != patience or patience < 1:
        raise ValueError("patience must be a positive integer")
    if energy_tol is not None and (not np.isfinite(energy_tol) or energy_tol < 0):
        raise ValueError("energy_tol must be finite and non-negative")
    return {"energy_tol": energy_tol, "patience": int(patience)}


def supplied_parameters(initial_params, template):
    values = np.asarray(initial_params, dtype=float)
    if values.shape != np.shape(template) or not np.all(np.isfinite(values)):
        raise ValueError(
            "initial_params must be finite and match the ansatz parameter shape"
        )
    return values


def optimize(
    params,
    energy_fn,
    update,
    *,
    steps,
    energy_tol=None,
    patience=1,
    progress_callback=None,
):
    """Keep only valid iterates; failures report the last valid state, never NaN."""
    stopping_config(steps, energy_tol, patience)
    energies: list[float] = []
    history: list[list[float]] = []
    reason, diagnostic, streak, attempted = "budget_exhausted", None, 0, 0
    message = None

    def sample(p):
        if not np.all(np.isfinite(p)):
            raise FloatingPointError("Non-finite parameters")
        e = float(energy_fn(p))
        if not np.isfinite(e):
            raise FloatingPointError("Non-finite energy")
        return e

    def report():
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "optimization",
                    "iteration": len(energies) - 1,
                    "total_iterations": int(steps),
                    "energy": energies[-1],
                }
            )

    try:
        energies.append(sample(params))
        history.append(np.asarray(params, dtype=float).ravel().tolist())
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        reason, message = "numerical_failure", str(exc)
    if energies:
        report()
        for _ in range(int(steps)):
            attempted += 1
            try:
                candidate = update(params)
                energy = sample(candidate)
            except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
                reason, message = "numerical_failure", str(exc)
                break
            params = candidate
            diagnostic = abs(energy - energies[-1])
            energies.append(energy)
            history.append(np.asarray(params, dtype=float).ravel().tolist())
            report()
            streak = (
                streak + 1 if energy_tol is not None and diagnostic <= energy_tol else 0
            )
            if energy_tol is not None and streak >= patience:
                reason = "tolerance_satisfied"
                break
    termination = {
        "reason": reason,
        "criterion": "absolute_energy_change" if energy_tol is not None else None,
        "threshold": energy_tol,
        "diagnostic": diagnostic,
        "patience": int(patience),
        "updates": max(0, len(energies) - 1),
        "attempted_updates": attempted,
        "budget": int(steps),
    }
    if message is not None:
        termination["message"] = message
    return params, energies, history, termination

# vqe/ssvqe.py
from __future__ import annotations
import os, json
import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem

from .hamiltonian import build_hamiltonian
from .optimizer import get_optimizer
from .io_utils import (
    IMG_DIR, ensure_dirs, make_run_config_dict,
    run_signature, save_run_record,
)

ensure_dirs()


def _uccsd_excitations(symbols, coordinates, basis, qubits):
    """Return (electrons, hf_state, singles, doubles, n_params) for UCCSD on this orbital space."""
    # Build a Molecule to get electron count; support older PL versions too
    try:
        mol = qchem.Molecule(symbols, coordinates, charge=+1 if "H3" in "".join(symbols) else 0, basis=basis)
    except TypeError:
        mol = qchem.Molecule(symbols, coordinates, charge=+1 if "H3" in "".join(symbols) else 0)

    electrons = mol.n_electrons
    singles, doubles = qchem.excitations(electrons, qubits)
    # Make tuples to be robust across PL versions
    singles = [tuple(x) for x in singles]
    doubles = [tuple(x) for x in doubles]
    hf = qchem.hf_state(electrons, qubits)
    n_params = len(singles) + len(doubles)
    return electrons, hf, singles, doubles, n_params


def _make_uccsd_state_circuit(hf_state, singles, doubles):
    """Factory: returns a function state_circuit(params, wires) that prepares the UCCSD state."""
    def state_circuit(params, wires):
        qml.BasisState(hf_state, wires=wires)
        n_singles = len(singles)
        # Apply singles then doubles with the provided parameter vector
        for i, s in enumerate(singles):
            qml.SingleExcitation(params[i], wires=s)
        for j, d in enumerate(doubles):
            qml.DoubleExcitation(params[n_singles + j], wires=d)
    return state_circuit


def run_ssvqe(
    molecule: str = "H3+",
    optimizer_name: str = "Adam",
    steps: int = 100,
    stepsize: float = 0.4,
    penalty_weight: float = 10.0,
    seed: int = 0,
    plot: bool = True,
    force: bool = False,
    symbols=None,
    coordinates=None,
    basis: str = "sto-3g",
):
    """
    Two-state SSVQE (ground + first excited) with an orthogonality penalty.

    Returns
    -------
    dict with keys:
      - 'E0_list', 'E1_list' : per-iteration energies
      - 'final_params'       : concatenated parameters [theta0..., theta1...]
      - 'config'             : the run configuration used (for reproducibility)
    """
    np.random.seed(seed)
    ensure_dirs()

    # --- Build Hamiltonian & molecular info (uses your standard builder) ---
    if symbols is None or coordinates is None:
        H, qubits, symbols, coordinates, basis = build_hamiltonian(molecule)
        # charge is handled inside build_hamiltonian; we only need electrons below
    else:
        H, qubits = qml.qchem.molecular_hamiltonian(
            symbols, coordinates, charge=+1 if molecule.upper() == "H3+" else 0, basis=basis, unit="angstrom"
        )

    electrons, hf, singles, doubles, n_params = _uccsd_excitations(symbols, coordinates, basis, qubits)
    state_circuit = _make_uccsd_state_circuit(hf, singles, doubles)

    # --- Config + cache signature (consistent with the rest of your package) ---
    ansatz_desc = "SSVQE(UCCSD) two-state, orthogonality penalty"
    cfg = make_run_config_dict(
        symbols=symbols,
        coordinates=coordinates,
        basis=basis,
        ansatz_desc=ansatz_desc,
        optimizer_name=optimizer_name,
        stepsize=stepsize,
        max_iterations=steps,
        seed=seed,
        noisy=False,
        depolarizing_prob=0.0,
        amplitude_damping_prob=0.0,
    )
    cfg["penalty_weight"] = float(penalty_weight)
    sig = run_signature(cfg)
    prefix = f"{molecule.replace('+','plus')}_SSVQE_{optimizer_name}_s{seed}__{sig}"
    result_path = os.path.join("results", f"{prefix}.json")

    if not force and os.path.exists(result_path):
        print(f"📂 Using cached SSVQE result: {result_path}")
        with open(result_path, "r") as f:
            record = json.load(f)
        return record["result"]

    # --- Device & QNodes ---
    dev = qml.device("default.qubit", wires=qubits)

    def _apply_state(params):
        state_circuit(params, wires=range(qubits))

    @qml.qnode(dev)
    def energy_qnode(params):
        _apply_state(params)
        return qml.expval(H)

    # Overlap <psi(p0) | psi(p1)> via adjoint trick -> pick |0...0> prob
    def _overlap00(p0, p1):
        @qml.qnode(dev)
        def _ov(p0, p1):
            _apply_state(p0)
            qml.adjoint(_apply_state)(p1)
            return qml.probs(wires=range(qubits))
        return _ov(p0, p1)[0]

    # --- Cost & optimization ---
    params = np.zeros(2 * n_params, requires_grad=True)
    opt = get_optimizer(optimizer_name, stepsize=stepsize)

    E0_list, E1_list = [], []

    def cost(theta):
        p0, p1 = theta[:n_params], theta[n_params:]
        E0 = energy_qnode(p0)
        E1 = energy_qnode(p1)
        penalty = penalty_weight * _overlap00(p0, p1)
        return E0 + E1 + penalty

    for _ in range(steps):
        try:
            params, _ = opt.step_and_cost(cost, params)
        except AttributeError:
            params = opt.step(cost, params)
        p0, p1 = params[:n_params], params[n_params:]
        E0_list.append(float(energy_qnode(p0)))
        E1_list.append(float(energy_qnode(p1)))

    # --- Persist + optional plot ---
    record = {
        "config": cfg,
        "result": {
            "E0_list": E0_list,
            "E1_list": E1_list,
            "final_params": [float(x) for x in params],
        },
    }
    save_run_record(prefix, record)

    if plot:
        try:
            from .visualize import plot_ssvqe_convergence
            plot_ssvqe_convergence(molecule, E0_list, E1_list, optimizer_name=optimizer_name)
        except Exception as e:
            print(f"⚠️ Plotting failed (non-fatal): {e}")

    return record["result"]

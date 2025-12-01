def test_qubit_hamiltonian_builds_for_new_molecules():
    from vqe_qpe_common.molecules import get_molecule_config
    from vqe_qpe_common.hamiltonian import build_hamiltonian

    for name in ["BeH2", "H4", "HeH+"]:
        cfg = get_molecule_config(name)
        H, n_qubits, hf_state = build_hamiltonian(
            symbols=cfg["symbols"],
            coordinates=cfg["coordinates"],
            charge=cfg["charge"],
            basis=cfg["basis"],
        )
        assert n_qubits > 0
        assert len(H) > 0
        coeffs, ops = H.terms()
        assert len(coeffs) == len(ops)
        assert len(coeffs) > 0
        assert len(hf_state) == n_qubits


def test_hamiltonian_term_structure():
    from vqe_qpe_common.molecules import get_molecule_config
    from vqe_qpe_common.hamiltonian import build_hamiltonian

    cfg = get_molecule_config("BeH2")
    H, n_qubits, hf_state = build_hamiltonian(**cfg)

    coeffs, ops = H.terms()
    assert len(coeffs) == len(ops)
    assert all(hasattr(op, "wires") for op in ops)


def test_hamiltonian_non_empty():
    from vqe_qpe_common.molecules import get_molecule_config
    from vqe_qpe_common.hamiltonian import build_hamiltonian

    for name in ["H2", "H3+", "LiH", "BeH2", "HeH+"]:
        cfg = get_molecule_config(name)
        H, n_qubits, hf_state = build_hamiltonian(**cfg)
        assert len(H) > 0  # modern PennyLane Hamiltonian uses len(H)


def test_hf_state_matches_qubits():
    from vqe_qpe_common.molecules import get_molecule_config
    from vqe_qpe_common.hamiltonian import build_hamiltonian

    cfg = get_molecule_config("H2")
    H, nq, hf = build_hamiltonian(**cfg)
    assert len(hf) == nq

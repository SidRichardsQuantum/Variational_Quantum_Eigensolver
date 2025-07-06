# Theory

## Variational Principle

The variational principle states that for any trial wavefunction $|\psi⟩$, the expectation value of the Hamiltonian is an upper bound to the true ground state energy:
```
E₀ ≤ ⟨ψ|H|ψ⟩
```
where:
- $E_0$ is the ground state energy
- $|\psi⟩$ is any normalized wavefunction
- $H$ is the Hamiltonian

## Hartree-Fock

The Hartree-Fock state is written as the tensor product of a qubit state $\phi$ for every electron orbital:
```
|HF⟩ = |φ₁φ₂...φₙ⟩
```
For LiH with 4 electrons in 12 orbitals, the HF reference state is ```|111100000000⟩```, which describes electrons occupying the four lowest energy orbital states.

## VQE Algorithm

The VQE algorithm consists of:
1. **State Preparation**: Prepare parameterized quantum state |ψ(θ)⟩
2. **Measurement**: Measure expectation value ⟨ψ(θ)|H|ψ(θ)⟩
3. **Optimization**: Classically optimize parameters θ to minimize energy
4. **Iteration**: Repeat until convergence

## Ansatz Construction

- **Single Excitations** for orbital relaxation
- **Double Excitations** for electron-electron correlation effects

## Implementation Parameters

### LiH
- **Bond Length**: 1.6Å
- **Electrons**: 4 total electrons
- **Qubits**: 12 qubits required
- **Ansatz**: Double excitation gates only (72 parameters)
- **Optimizer**: Gradient Descent with 0.1 step size
- **Iterations**: 50 optimization steps

### H₂O
- **Geometry**: 104.5° bond angle between the hydrogens about the oxygen
- **Electrons**: 10 total electrons  
- **Qubits**: 14 qubits required
- **Ansatz**: Single + Double excitations (UCCSD)
- **Optimizer**: Adam with 0.1 step size
- **Iterations**: 50 optimization steps

## References

- [VQE](https://en.wikipedia.org/wiki/Variational_quantum_eigensolver)
- [Hartree-Fock Method](https://en.wikipedia.org/wiki/Hartree–Fock_method)

---

📘 Author: Sid Richards (SidRichardsQuantum)

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="20" /> LinkedIn: https://www.linkedin.com/in/sid-richards-21374b30b/

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

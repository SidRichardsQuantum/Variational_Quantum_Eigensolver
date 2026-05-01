# Theory Overview

The theory material is split into focused pages so equations, method families,
and implementation assumptions are easier to scan.

```{math}
:label: eq-variational-principle
E_0 \le \langle \psi | H | \psi \rangle
```

The variational algorithms in this repository start from the bound in
{eq}`eq-variational-principle`: a parameterized circuit is useful when it can
reach low-energy states of the shared qubit Hamiltonian.

```{math}
:label: eq-vqe-objective
E(\theta) = \langle \psi(\theta) | H | \psi(\theta) \rangle
```

VQE minimizes {eq}`eq-vqe-objective`, while QPE estimates phases of time
evolution and VarQITE/VarQRTE project imaginary-time or real-time dynamics onto
an ansatz manifold.

```{toctree}
:maxdepth: 1

Chemistry and Background <../theory/chemistry>
VQE, Ansatzes, and ADAPT <../theory/vqe>
Excited-State Methods <../theory/excited_states>
Quantum Phase Estimation <../theory/qpe>
QITE and QRTE <../theory/qite_qrte>
Noise Models <../theory/noise>
References <../theory/references>
```

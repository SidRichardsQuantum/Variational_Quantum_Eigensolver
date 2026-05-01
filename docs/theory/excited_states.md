# Excited-State Methods

Excited-state workflows either build reduced eigenproblems around a converged
reference state or solve for excited states directly with variational penalties.

```{math}
:label: eq-vqd-loss
L_n =
\langle \psi_n|H|\psi_n\rangle
+
\beta \sum_{k<n}
|\langle \psi_k|\psi_n\rangle|^2
```

The VQD workflow uses the deflated objective in {eq}`eq-vqd-loss` to discourage
collapse back to lower-energy states.

```{include} ../../THEORY.md
:start-after: "# Excited-State Methods"
:end-before: "# ADAPT-VQE"
:heading-offset: 1
```

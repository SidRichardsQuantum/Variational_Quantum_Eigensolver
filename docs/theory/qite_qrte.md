# QITE and QRTE

Imaginary-time evolution suppresses high-energy components, while real-time
evolution tracks dynamics of a prepared state.

```{math}
:label: eq-imaginary-time
|\psi(\tau)\rangle = e^{-H\tau}|\psi(0)\rangle
```

The VarQITE implementation approximates {eq}`eq-imaginary-time` through a
projected variational update.

```{math}
:label: eq-qrte-linear-system
A(\theta)\dot{\theta} = -C(\theta)
```

VarQRTE solves the projected linear system in {eq}`eq-qrte-linear-system` to
advance parameters in real time.

```{include} ../../THEORY.md
:start-after: "# Quantum Imaginary Time Evolution"
:end-before: "# Noise Models"
:heading-offset: 1
```

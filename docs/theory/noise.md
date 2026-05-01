# Noise Models

Noise studies use PennyLane mixed-state simulation to evaluate robustness under
consistent channels across supported workflows.

```{math}
:label: eq-depolarizing-channel
\mathcal{E}(\rho) =
(1-p)\rho
+
\frac{p}{3}
(X\rho X + Y\rho Y + Z\rho Z)
```

The depolarizing channel in {eq}`eq-depolarizing-channel` is used as an
isotropic benchmark noise model.

```{include} ../../THEORY.md
:start-after: "# Noise Models"
:end-before: "# References"
:heading-offset: 1
```

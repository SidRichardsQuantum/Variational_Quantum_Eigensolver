# Results

The project generates comprehensive visualizations for molecular systems.
Both visualizations demonstrate the dominance of the Hartree-Fock reference state with correlation corrections from excitations.
Basis state indices are converted from binary to decimal for shorter/clearer axis-labeling.

---

## 📚 Table of Contents
- [H₂ (Optimiser Comparison)](#h₂-optimiser-comparison)
- [H₂ (Ansätze Comparison)](#h₂-ansätze-comparison)
- [H₃⁺ Excitation Comparison](#h₃⁺-excitation-comparison)
- [LiH](#lih)
- [Optimal LiH Length](#optimal-lih-length)
- [H₂O](#h₂o)
- [Optimal H₂O Angle](#optimal-h₂o-angle)

---

## H₂ (Optimiser Comparison)

### Set Up

- **Bond Length**: $0.7414 Å$
- **Hartree-Fock Energy**: $-0.88842304 Ha$
- **Convergence**: $50$ iterations

### Visualization

Every optimiser with step size $0.2$, successfully converts at ground state energies:

```
Adam:
Final ground state energy = -0.89801978 Ha

GradientDescent:
Final ground state energy = -0.89805304 Ha

Nesterov:
Final ground state energy = -0.89805302 Ha

Adagrad:
Final ground state energy = -0.89805304 Ha

Momentum:
Final ground state energy = -0.89801009 Ha

SPSA:
Final ground state energy = -0.89566722 Ha
```

![H₂ Optimizer Comparison](notebooks/images/H2_optimiser_comparison.png)

The best optimizer for this dihydrogen example is the Gradient Descent.
Using this, the ground state is found:


```
Ground state of H₂:
|ψ⟩ = -0.0585|0011> + 0.9983|1100>
```

![H₂ Ground State](notebooks/images/H2_ground_state.png)

## H₂ (Ansätze Comparison)

### Set Up

- **Bond Length**: $0.7414 Å$
- **Optimizer**: `AdamOptimizer` with step size $0.2$
- **Iterations**: $40$
- **Ansätze Compared**: `UCCSD`, `RY-CZ`, `Minimal`

### Visualization

The following ansätze were tested in a noiseless simulation:

- **UCCSD**: A chemically motivated circuit including single and double excitations.
- **$R_Y-C_Z$**: A hardware-efficient structure using rotation and $C_Z$ entanglement layers.
- **Minimal**: A single-parameter ansätze tailored for H₂.

All three ansätze successfully converged to near ground-state energies within $40$ iterations:

```
UCCSD:
Final energy = -0.87975978 Ha

RY-CZ:
Final energy = -0.87936449 Ha

Minimal:
Final energy = -0.84822983 Ha
```

![H₂ Ansätze Comparison](notebooks/images/H2_Ansatz_Comparison.png)

Although all ansätze reach similar energy minima, UCCSD and RY-CZ converge slightly faster while **Minimal** shows mild oscillations mid-convergence.

## H₃⁺ Excitation Comparison

### Set Up

- **Molecular Geometry**: Equilateral triangle ($1.0 Å$ side length)
- **Charge**: $+1$
- **Electrons**: 2
- **Optimizer**: `Adam` with step size $0.2$
- **Iterations**: $50$ per excitation type
- **Excitations Compared**: Single, Double, Both (UCCSD)

### Visualization

The simulation compares three ansätze types in a noiseless VQE run. Final ground state energies:

```
Single excitations only:
Final energy = -1.24811821 Ha

Double excitations only:
Final energy = -1.25027788 Ha

Single + Double (UCCSD):
Final energy = -1.25028914 Ha
```

![H₃⁺ Excitation Comparison](images/H3+_Excitation_Comparison.png)

The best convergence and lowest energy are achieved when both single and double excitations are used, consistent with the expected benefits of the full UCCSD ansätze.

The wavefunctions reveal a dominant contribution from the Hartree-Fock reference state, with notable amplitudes in correlated excited states. Example from UCCSD:

```
|ψ⟩ = 0.9813|110000⟩ - 0.0806|100010⟩ - 0.0773|100100⟩ + 0.0667|010001⟩
    - 0.0577|010100⟩ + 0.0481|110011⟩ + 0.0332|110100⟩ + 0.0226|100001⟩
```

This decomposition showcases the entanglement and correlation introduced by higher-order excitations. The Hartree-Fock state $|110000⟩$ is again dominant, but its amplitude is reduced relative to smaller molecules due to increased multi-reference character.

A quantum circuit diagram for the UCCSD ansätze is below:

![H₃⁺ Circuit Diagram](images/H3+_UCCSD_Circuit.png)

## LiH

### Set Up

- **Bond Length**: $1.6 Å$
- **Hartree-Fock Energy**: $-7.66194677 Ha$
- **Convergence**: $50$ iterations

### Visualization

`GradientDescentOptimizer` with step-size $0.1$ successfully converges at ground state energy $-7.67957954 Ha$:

![LiH Convergence](notebooks/images/LiH_convergence.png)

The calculated wavefunction for the ground state of LiH is:

```
|ψ⟩ = 0.9930|111100000000⟩ - 0.0969|110000000011⟩ 
    - 0.0334|110000001100⟩ - 0.0334|110000110000⟩ 
    - 0.0317|110001000010⟩ + 0.0317|110010000001⟩ 
    - 0.0123|110011000000⟩
```

The Hartree-Fock state $|111100000000⟩$ is the most dominant.

![LiH Ground State](notebooks/images/LiH_ground_state.png)

## Optimal LiH Length

The Gradient Descent Optimizer was used to scan over a range of bond-lengths between the Li and H atoms.
$20$ maximum iterations and a stepsize of $0.8$ were used, over $10$ bond-lengths in the range $[1.1, 2.0] Å$.
Plot output from `LiH_Bond_Length.ipynb`:

![Optimal Length](notebooks/images/LiH_Optimal_Bond_Length.png)

```
Optimal bond length: 1.60 Å
Minimum ground state energy: -5.59357194 Ha
```

## H₂O

### Set Up

- **Bond Lengths**: $0.910922 Å$
- **Molecular Geometry**: Bent structure ($104.5°$ bond angle)
- **Hartree-Fock Energy**: $-72.86837737 Ha$
- **Convergence**: $50$ iterations with Adam optimizer

### Visualization

`AdamOptimizer`  with step-size $0.1$ successfully converges at ground state energy $-72.87712785 Ha$:

![H₂O Convergence](notebooks/images/H2O_convergence.png)

The calculated wavefunction for the ground state of water is:

```
|ψ⟩ = 0.9979|11111111110000⟩ - 0.0323|11110011110011⟩
    - 0.0244|11111100110011⟩ - 0.0211|11111111001100⟩
    + 0.0171|11100111110110⟩ - 0.0160|11001111111100⟩
    + 0.0156|11011011111001⟩ - 0.0105|11110011111100⟩
```

The Hartree-Fock state $|11111111110000⟩$ is the most dominant.

![H₂O Ground State](notebooks/images/H2O_ground_state.png)

## Optimal H₂O Angle

The Adam optimizer was used to find the angle between the two hydrogens in water.
$10$ maximum iterations and a stepsize of $0.2$ were used, over $5$ bond-angles in the range $[100, 109]°$.
Plot output from `H2O_Bond_Angle.ipynb`:

![Optimal H₂O Angle](notebooks/images/Water_Optimal_Angle.png)

```
Minimum energy: -71.539353 Ha
Optimal angle: 104.50°
```

These values are very close to the true ground state energy ($\approx -75 Ha$) and bond-angle ($\approx 104.5°$) of water.
[Chemical bonding of water](https://en.wikipedia.org/wiki/Chemical_bonding_of_water)
[Ground-state energy estimation of the water molecule on a trapped ion quantum computer](https://arxiv.org/abs/1902.10171)

---

📘 Author: Sid Richards (SidRichardsQuantum)

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="20" /> LinkedIn: https://www.linkedin.com/in/sid-richards-21374b30b/

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

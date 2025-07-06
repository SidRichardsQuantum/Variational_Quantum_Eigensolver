# Results

## LiH Ground State Energy
- **Bond Length**: 1.6 Å
- **Hartree-Fock Energy**: -7.66194677 Ha
- **VQE Final Energy**: -7.67957954 Ha
- **Energy Improvement**: 0.01763277 Ha (0.48 eV)
- **Convergence**: 50 iterations
- **Dominant State**: |111100000000⟩ with amplitude 0.9930

### LiH Ground State
```
|ψ⟩ = 0.9930|111100000000⟩ - 0.0969|110000000011⟩ 
    - 0.0334|110000001100⟩ - 0.0334|110000110000⟩ 
    - 0.0317|110001000010⟩ + 0.0317|110010000001⟩ 
    - 0.0123|110011000000⟩
```

## H₂O Ground State Energy
- **Molecular Geometry**: Bent structure (104.5° bond angle)
- **Hartree-Fock Energy**: -72.86837737 Ha
- **VQE Final Energy**: -72.87712785 Ha
- **Energy Improvement**: 0.00875048 Ha (0.24 eV)
- **Convergence**: 50 iterations with Adam optimizer
- **Dominant State**: |11111111110000⟩ with amplitude 0.9979

### H₂O Ground State
```
|ψ⟩ = 0.9979|11111111110000⟩ - 0.0323|11110011110011⟩
    - 0.0244|11111100110011⟩ - 0.0211|11111111001100⟩
    + 0.0171|11100111110110⟩ - 0.0160|11001111111100⟩
    + 0.0156|11011011111001⟩ - 0.0105|11110011111100⟩
```

## Visualizations

The project generates comprehensive visualizations for both molecular systems.
Basis state indices are converted from binary to decimal for shorter/clearer axis-labeling.

### LiH Visualizations
1. **Energy Convergence Plot**: Shows VQE optimization progress over 50 iterations
2. **Ground State Amplitudes**: Bar plot of significant quantum state components

![LiH Ground State](notebooks/images/LiH_ground_state.png)

### H₂O Visualizations  
1. **Energy Convergence Plot**: Shows Adam optimizer convergence behavior
2. **Ground State Amplitudes**: Bar plot showing basis state contributions

![H2O Ground State](notebooks/images/H2O_ground_state.png)

Both visualizations demonstrate the dominance of the Hartree-Fock reference state with correlation corrections from excitations.

---

📘 Author: Sid Richards (SidRichardsQuantum)

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="20" /> LinkedIn: https://www.linkedin.com/in/sid-richards-21374b30b/

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# Theory

## Key Parameters in the Implementations

### LiH Implementation
- **Bond Length**: 1.6 Å (Li-H distance)
- **Electrons**: 4 total electrons
- **Qubits**: 12 qubits required
- **Ansatz**: Double excitation gates only (72 parameters)
- **Optimizer**: Gradient Descent with 0.1 step size
- **Iterations**: 50 optimization steps

### H₂O Implementation
- **Geometry**: Bent structure with 104.5° bond angle
- **Electrons**: 10 total electrons  
- **Qubits**: 14 qubits required
- **Ansatz**: Single + Double excitations (UCCSD)
- **Optimizer**: Adam with 0.1 step size
- **Iterations**: 50 optimization steps

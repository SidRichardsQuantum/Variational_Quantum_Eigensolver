# Notebook Gallery

The notebooks are organized by workflow role: quick introductions, method
demonstrations, benchmark evidence, calibration, and model-Hamiltonian studies.
They are package-client examples unless noted otherwise.

## Start Here

```{raw} html
<div class="doc-card-grid">
  <article class="doc-card">
    <h3>VQE vs QPE from scratch</h3>
    <p>Conceptual H2 comparison before moving into package APIs.</p>
    <a href="https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/blob/main/notebooks/getting_started/01_vqe_vs_qpe_from_scratch_h2.ipynb">Open notebook</a>
  </article>
  <article class="doc-card">
    <h3>Getting started with VQE</h3>
    <p>Basic package-client VQE run for H2.</p>
    <a href="https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/blob/main/notebooks/getting_started/02_getting_started_vqe_h2.ipynb">Open notebook</a>
  </article>
  <article class="doc-card">
    <h3>Getting started with QITE</h3>
    <p>Projected imaginary-time relaxation for H2.</p>
    <a href="https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/blob/main/notebooks/getting_started/07_getting_started_qite_h2.ipynb">Open notebook</a>
  </article>
  <article class="doc-card">
    <h3>Getting started with QRTE</h3>
    <p>Prepared-state real-time dynamics on H2.</p>
    <a href="https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/blob/main/notebooks/getting_started/11_getting_started_qrte_h2.ipynb">Open notebook</a>
  </article>
</div>
```

## Getting-Started Sequence

| Notebook | Focus |
| --- | --- |
| [`01_vqe_vs_qpe_from_scratch_h2.ipynb`](https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/blob/main/notebooks/getting_started/01_vqe_vs_qpe_from_scratch_h2.ipynb) | conceptual VQE/QPE comparison |
| [`02_getting_started_vqe_h2.ipynb`](https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/blob/main/notebooks/getting_started/02_getting_started_vqe_h2.ipynb) | basic VQE API |
| [`03_ansatz_comparison_h2.ipynb`](https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/blob/main/notebooks/getting_started/03_ansatz_comparison_h2.ipynb) | ansatz comparison |
| [`04_optimizer_comparison_h2.ipynb`](https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/blob/main/notebooks/getting_started/04_optimizer_comparison_h2.ipynb) | optimizer comparison |
| [`05_excited_states_h2.ipynb`](https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/blob/main/notebooks/getting_started/05_excited_states_h2.ipynb) | excited-state overview |
| [`06_getting_started_qpe_h2.ipynb`](https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/blob/main/notebooks/getting_started/06_getting_started_qpe_h2.ipynb) | QPE API |
| [`07_getting_started_qite_h2.ipynb`](https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/blob/main/notebooks/getting_started/07_getting_started_qite_h2.ipynb) | VarQITE API |
| [`08_geometry_override_h2.ipynb`](https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/blob/main/notebooks/getting_started/08_geometry_override_h2.ipynb) | explicit geometry inputs |
| [`09_bond_scan_h2.ipynb`](https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/blob/main/notebooks/getting_started/09_bond_scan_h2.ipynb) | bond scan workflow |
| [`10_adapt_vqe_h3plus.ipynb`](https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/blob/main/notebooks/getting_started/10_adapt_vqe_h3plus.ipynb) | ADAPT-VQE on H3+ |
| [`11_getting_started_qrte_h2.ipynb`](https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/blob/main/notebooks/getting_started/11_getting_started_qrte_h2.ipynb) | VarQRTE API |

## Method Demonstrations

| Group | Notebooks |
| --- | --- |
| VQE excited states on H2 | [`QSE`](https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/blob/main/notebooks/vqe/H2/QSE.ipynb), [`EOM_QSE`](https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/blob/main/notebooks/vqe/H2/EOM_QSE.ipynb), [`LR_VQE`](https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/blob/main/notebooks/vqe/H2/LR_VQE.ipynb), [`EOM_VQE`](https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/blob/main/notebooks/vqe/H2/EOM_VQE.ipynb), [`SSVQE`](https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/blob/main/notebooks/vqe/H2/SSVQE.ipynb), [`VQD`](https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/blob/main/notebooks/vqe/H2/VQD.ipynb) |
| Geometry | [`H2O bond angle`](https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/blob/main/notebooks/vqe/H2O/Bond_Angle.ipynb) |
| Projected dynamics | [`H2 real time`](https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/blob/main/notebooks/qite/H2/Real_Time.ipynb) |

## Benchmark Families

| Family | Primary notebooks |
| --- | --- |
| VQE H2 benchmarks | ansatz, mapping, noise, optimizer, SSVQE, and VQD comparisons under `notebooks/benchmarks/vqe/H2/` |
| H3+ VQE benchmarks | noiseless and noisy UCC ansatz comparisons under `notebooks/benchmarks/vqe/H3plus/` |
| QPE calibration | noisy QPE, calibration sweep, and decision map under `notebooks/benchmarks/qpe/H2/` |
| Cross-method comparisons | H2, LiH, scaling, low-qubit VQE, registry coverage, and ionization-energy studies under `notebooks/benchmarks/comparisons/` |
| Non-molecule Hamiltonians | TFIM, Heisenberg, and SSH expert-mode benchmarks under `notebooks/benchmarks/non_molecule/` |
| Default calibration | VQE, VarQITE, and QPE default-calibration notebooks under `notebooks/benchmarks/defaults/` |
| QRTE validation | exact-vs-VarQRTE H2 benchmark under `notebooks/benchmarks/qite/H2/` |

## Recommended Reading Order

1. `getting_started/01_vqe_vs_qpe_from_scratch_h2.ipynb`
2. `getting_started/02_getting_started_vqe_h2.ipynb`
3. `getting_started/07_getting_started_qite_h2.ipynb`
4. `getting_started/11_getting_started_qrte_h2.ipynb`
5. `notebooks/vqe/H2/` excited-state notebooks
6. `notebooks/benchmarks/defaults/` calibration notebooks
7. `notebooks/benchmarks/comparisons/` cross-method notebooks

The full source tree is available in the
[`notebooks/`](https://github.com/SidRichardsQuantum/Variational_Quantum_Eigensolver/tree/main/notebooks)
directory.

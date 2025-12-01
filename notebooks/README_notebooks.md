# 📘 VQE & QPE Notebooks

This directory contains curated Jupyter notebooks demonstrating the full workflow of the **Variational Quantum Eigensolver (VQE)** and the initial Quantum Phase Estimation (QPE) pipeline using the packaged code in `vqe/`, `qpe/`, and `vqe_qpe_common/`.

All notebooks are now aligned with the updated modular package structure and the reproducible result-caching system.

For theory background and recommended reading order:

- **[THEORY.md](../THEORY.md)** — essential mathematical background  
- **[USAGE.md](../USAGE.md)** — command-line tools, package entrypoints  
- **[README.md](../README.md)** — top-level project overview  

---

# Directory Overview

```
notebooks/
├── README_notebooks.md   ← this file
├── RESULTS.md            ← consolidated notebook outputs
│
├── vqe/                  
│   ├── H2/
│   ├── H2O/
│   ├── H3plus/
│   └── LiH/
│
└── qpe/                  
    ├── H2/
    └── qpe_utils.py
```

---

# ⚛️ VQE Notebook Collection

## **H₂ — Benchmark Molecule**
📁 `notebooks/vqe/H2/`

Minimal-qubit molecule used to demonstrate:

- Noiseless vs noisy VQE  
- Optimizer comparison  
- Ansatz comparison  
- Geometry scans  
- Reproducibility

---

## **H₃⁺ — Excitations, Mappings, and SSVQE**
📁 `notebooks/vqe/H3plus/`

Includes:

- UCCSD ground state  
- Mapping comparisons  
- SSVQE  
- Noise studies

---

## **H₂O — Geometry & UCCSD**
📁 `notebooks/vqe/H2O/`

Includes:

- Noiseless UCCSD  
- Bond-angle scan  
- Amplitude visualisation  

---

## **LiH — Bond Length Scan**
📁 `notebooks/vqe/LiH/`

Includes:

- UCCSD  
- Bond-length energy curve  
- Ground-state amplitudes  

---

# QPE Notebooks

📁 `notebooks/qpe/H2/`

Initial QPE examples for H₂ only.

---

# Recommended Reading Order

1. **H₂ (VQE)**
2. **LiH / H₂O scans**
3. **H₃⁺ mapping & SSVQE**
4. **H₂ (QPE)**

---

# Reproducibility

Results written to:

```
data/vqe/results/
data/vqe/images/
data/qpe/results/
data/qpe/images/
```

---

📘 Author: Sid Richards (SidRichardsQuantum)

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" width="20" /> LinkedIn: https://www.linkedin.com/in/sid-richards-21374b30b/

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

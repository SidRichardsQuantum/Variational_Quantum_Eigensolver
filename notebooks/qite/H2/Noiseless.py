# %% [markdown]
# # ⚛️ H₂ QITE — **Noiseless** (Pure Package Client)
# 
# This notebook runs **VarQITE (McLachlan's variational principle)** for **H₂**
# using only the packaged API:
# 
# ```python
# from qite.core import run_qite
# ```
# 
# It mirrors the VQE/QPE notebook ergonomics:
# - **Cached** runs saved to `results/qite/`
# - **Plots** saved to `images/qite/`
# 
# ---
# 
# ## What we compute
# 
# - Molecule: **H₂** (STO-3G, equilibrium registry geometry)
# - Ansatz: **UCCSD** (reusing the VQE ansatz library)
# - Update rule: **VarQITE (McLachlan)** (statevector / noiseless)
# - Device: `default.qubit`
# 
# Outputs (from `run_qite`):
# - `energy` (final energy)
# - `energies` (convergence trace)
# - `final_state_real`, `final_state_imag`
# - `num_qubits`

# %% [markdown]
# ## 🔧 Imports & Configuration

# %%
from qite.core import run_qite
from qite.io_utils import ensure_dirs

ensure_dirs()

molecule = "H2"
seed = 0

ansatz_name = "UCCSD"
mapping = "jordan_wigner"

steps = 100
dtau = 0.2

print("Molecule:", molecule)
print("Ansatz:", ansatz_name)
print("Mapping:", mapping)
print("Steps:", steps)
print("dtau:", dtau)
print("Seed:", seed)

# %% [markdown]
# ## 🚀 Run VarQITE (Cached)
# 
# Note:
# If you previously ran this notebook before VarQITE was implemented, you may
# have cached results from a temporary fallback. The first run below uses
# `force=True` to ensure we compute a fresh VarQITE result.

# %%
result = run_qite(
    molecule=molecule,
    seed=int(seed),
    steps=int(steps),
    dtau=float(dtau),
    ansatz_name=str(ansatz_name),
    mapping=str(mapping),
    noisy=False,
    plot=True,
    force=True,   # one clean run after engine changes
    show=True,
)

result

# %% [markdown]
# ## 📌 Summary

# %%
print("\nH₂ VarQITE — Noiseless Summary\n")

E0 = float(result["energies"][0])
E_last = float(result["energies"][-1])
n_qubits = int(result["num_qubits"])

print(f"Num qubits:        {n_qubits}")
print(f"Initial energy:    {E0:.8f} Ha")
print(f"Final energy:      {E_last:.8f} Ha")
print(f"ΔE (final-init):   {E_last - E0:+.8f} Ha")

# Optional quick sanity check (ballpark; depends on exact geometry details)
E_ref = -1.137
print(f"Reference (ballpark): {E_ref:.3f} Ha")
print(f"ΔE (final-ref):       {E_last - E_ref:+.6f} Ha")

# %% [markdown]
# ## 🔁 Noiseless dtau sweep
# 
# Each run is cached independently. After the one forced run above, the sweep
# can use `force=False`.

# %%
dtau_list = [0.05, 0.1, 0.2, 0.3]
energies = []

for dt in dtau_list:
    r = run_qite(
        molecule=molecule,
        seed=int(seed),
        steps=int(steps),
        dtau=float(dt),
        ansatz_name=str(ansatz_name),
        mapping=str(mapping),
        noisy=False,
        plot=False,
        force=False,
        show=False,
    )
    energies.append(float(r["energy"]))

print("\nEnergy vs dtau (noiseless)\n")
for dt, E in zip(dtau_list, energies):
    print(f"dtau={dt:>4.2f}  E={E: .8f} Ha")

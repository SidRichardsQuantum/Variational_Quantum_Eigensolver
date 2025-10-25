import os
import matplotlib.pyplot as plt

def plot_convergence(energies, molecule="H2", save_dir="images"):
    """Plot and save energy convergence."""
    plt.figure(figsize=(6, 4))
    plt.plot(energies, "o-", label="Energy (Hartree)")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title(f"VQE Convergence: {molecule}")
    plt.legend()
    plt.grid(True)

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{molecule}_VQE_Convergence.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"\nSaved convergence plot to: {path}\n")

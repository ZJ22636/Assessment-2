import matplotlib.pyplot as plt
import numpy as np

#path to the folder containing the data files
folder = "Results/Physics_Check/MCS2000_T0.65_Size50/"
versions = ["Original", "NumPy", "Numba", "NumbaParallel"]
runs = [1,2,3]

temp = 0.65

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

for c, version in enumerate(versions):
    colour = f"C{c}"
    for run in runs:
        data = np.loadtxt(f"{folder}{version}/MCS2000_T0.65_Size50_V{run}.txt")
        label = version if run == 1 else None #one legend entry per version
        ax1.plot(data[:, 0], data[:, 2], color=colour, alpha=0.7, label=label)
        ax2.plot(data[:, 0], data[:,3], color=colour, alpha=0.7, label=label)
    

ax1.set_xlabel("MCS")
ax1.set_ylabel("Reduced Energy U/ε")
ax1.set_title(f"Reduced Energy vs MCS at T={temp}")
ax1.legend()

ax2.set_xlabel("MCS")
ax2.set_ylabel("Order Parameter, S")
ax2.set_title(f"Order Parameter vs MCS at T={temp}")
ax2.legend()

plt.tight_layout()
plt.savefig(folder + "EquilibriumPlot_all_versions.png", dpi=300)
plt.show()
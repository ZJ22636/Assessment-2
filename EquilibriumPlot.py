import matplotlib.pyplot as plt
import numpy as np

#path to the folder containing the data files
folder = "Results/MCS_5000_T0.65_Size_50/"
#name of data files to be plotted
"""files= ["MCS1000_T0.65_Size50_V1.txt", 
        "MCS1000_T0.65_Size50_V2.txt", 
        "MCS1000_T0.65_Size50_V3.txt",
        "MCS1000_T0.65_Size50_V4.txt",]"""
files = ["MCS5000_T0.65_Size50_V1.txt"]
temp = 0.65

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

for i, name in enumerate(files):
    data = np.loadtxt(folder + name)
    ax1.plot(data[:, 0], data[:, 2], label=f"V{i+1}")
    ax2.plot(data[:, 0], data[:, 3], label=f"V{i+1}")

ax1.set_xlabel("MCS")
ax1.set_ylabel("Reduced Energy U/ε")
ax1.set_title(f"Reduced Energy vs MCS at T={temp}")
ax1.legend()

ax2.set_xlabel("MCS")
ax2.set_ylabel("Order Parameter, S")
ax2.set_title(f"Order Parameter vs MCS at T={temp}")
ax2.legend()

plt.tight_layout()
plt.savefig(folder + "EquilibriumPlot.png", dpi=300)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

p = np.linspace(0, 1, 1001)

# BEC capacity
C_BEC = 1 - p

H = np.zeros_like(p)
mask = (p > 0) & (p < 1)

H[mask] = (
    -p[mask] * np.log2(p[mask])
    -(1 - p[mask]) * np.log2(1 - p[mask])
)

C_BSC = 1 - H

answer = "The"

plt.plot(p, C_BEC, label="BEC")
plt.plot(p, C_BSC, label="BSC")

plt.xlabel("Probability p")
plt.ylabel("Capacity (bits/channel use)")
plt.title("BEC vs BSC Channel Capacity")
plt.legend()
plt.grid(True)
plt.savefig("q4E.png")

"""
The BEC capacity decreases linearly correlated with pE. The BSC capacity is nonlinear reaching zero at p = 0.5, and it returns to 1 bit at p = 1. This is because every bit is flipped deterministically allowing it to be recovered. THE BEC does not recover capacity as pE approaches 1.
"""


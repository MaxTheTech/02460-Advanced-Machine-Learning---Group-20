import numpy as np
import matplotlib.pyplot as plt
import os
results = {}
path = "02460-Advanced-Machine-Learning---Group-20/Mini Project 2"
for d, exp in [(1, f"{path}/expB_d1"), (2, f"{path}/expB_d2"), (3, f"{path}/expB_d3")]:
    results[d] = {
        "geo": np.load(f"{exp}/cov_geo.npy").mean(),
        "euc": np.load(f"{exp}/cov_euc.npy").mean(),
    }

decoders = [1, 2, 3]
plt.plot(decoders, [results[d]["geo"] for d in decoders], label="Geodesic")
plt.plot(decoders, [results[d]["euc"] for d in decoders], label="Euclidean")
plt.xlabel("Num decoders"); plt.ylabel("Mean CoV"); plt.legend()
plt.xticks(decoders) 
plt.savefig("cov_vs_decoders.png")
import os
import numpy as np
import matplotlib.pyplot as plt

def save_output(X_real, X_2D, tag="real"):
    os.makedirs("results", exist_ok=True)

    # Save raw trajectory
    np.save(f"results/{tag}.npy", X_real)
    np.savetxt(f"results/{tag}.csv", X_real, delimiter=",")

    # Save static plot
    plt.plot(X_2D[:, 0], X_2D[:, 1])
    plt.title(f"Trajectory ({tag})")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True)
    plt.savefig(f"results/{tag}_2Dplot.png")
    plt.close()

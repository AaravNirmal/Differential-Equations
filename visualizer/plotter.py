import matplotlib.pyplot as plt

def plot_2d(X_2D):
    plt.plot(X_2D[:, 0], X_2D[:, 1])
    plt.title("Trajectory (Reduced to 2D)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True)
    plt.show()

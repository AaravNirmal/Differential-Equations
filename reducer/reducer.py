from sklearn.decomposition import PCA
import numpy as np

def reduce_to_2d(X):
    n_features = X.shape[1]
    
    if n_features < 2:
        # No reduction needed — just pad with zeros or duplicate
        X_2D = np.hstack([X, np.zeros((X.shape[0], 1))])
        print("⚠️ Not enough dimensions for PCA. Using pseudo-2D fallback.")
        return X_2D

    return PCA(n_components=2).fit_transform(X)

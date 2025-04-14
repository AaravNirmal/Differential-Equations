from sklearn.decomposition import PCA

def reduce_to_2d(X):
    return PCA(n_components=2).fit_transform(X)

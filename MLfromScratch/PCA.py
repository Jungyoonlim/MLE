import numpy as np 


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    # Unsupervised Learning Method     
    def fit(self, X):
        # mean centering -- subtract the mean from X 
        self.mean = np.mean(X, axis=0)
        X = X - self.mean 

        # Calculate Cov(X,X), functions need samples as columns 
        cov = np.cov(X.T)

        # Calculate E-vectors and E-values of the covariance matrix
        evectors, evalues = np.linalg.eig(cov)

        # e-vectors v = [:, i] column vector, transpose this for easier calc
        evectors = evectors.T

        # sort e-vectors
        idxs = np.argsort(evalues)[::-1]
        evalues = evalues[idxs]
        evectors = evectors[idxs]

        # Choose first k e-vectors and that will be the new k dims
        self.components = evectors[:self.n_components]

    def transform(self, X):
        # projects data 
        X = X - self.mean 
        return np.dot(X, self.components.T)

# Testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn import datasets

    data = datasets.load_iris()
    X = data.data
    y = data.target

    pca = PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print("Shape of X:", X.shape)
    print("Shape of transformed X:", X_projected.shape)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    plt.scatter(
        x1, x2, c=y, edgecolors="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    plt.show()

import numpy as np
 
def euclidean_distance(x1, x2): 
    return np. 


    
class kmeans:
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps 

        # list of sample indices for each cluster 
        self.clusters = [[] for _ in range(self.K)]

        # the centers (mean vector) for each cluster 
        self.centroids = []

    # no fit method, no y b/c unsupervised learning. 
    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape 

        # Initialize 
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.K[idx] for idx in random_sample_idxs]

        # Optimize Clusters
        for _ in range(self.max_iters):
            self.clusters = self._create_clusters(self.centroids)

    def _get_cluster_labels(self, clusters): pass

    def _create_clusters(self, centroids): pass

    def _get_centroids(self, clusters): pass

    def _is_converged(self, centroids_old, centroids): pass

    def plot(self): pass


if __name__ == "__main__":

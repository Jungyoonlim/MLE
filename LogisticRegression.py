import numpy as np

class Logistic_Regression:
    def __init__(self,lr=0.001,iters=1000):
        self.w=None
        self.b=None
        self.lr=lr
        self.iters=iters
    
    def fit(self,X,y):
        n_samples, n_features = X.shape
        self.w = np.zeros(self.features)
        self.b = 0 

        for _ in range(self.iters):
            predictions = np.dot(X)

    def predict():

import numpy as np

class linear_regression:
    def __init__(self,w,b, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None 
    
    def fit(self, X,y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0 
    
    def predict():

    def _predict(x):

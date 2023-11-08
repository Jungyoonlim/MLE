import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

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
            linear_pred = np.dot(X, self.w) + self.b
            pred = sigmoid(linear_pred)

            dw = (1/n_samples) * np.dot(X.T, (pred-y))
            db = (1/n_samples) * np.sum(pred-y)

            self.w=self.w-self.lr-dw
            self.b= self.b-self.lr*db

    def predict(self,X):
        linear_pred = np.dot(X, self.w) + self.b
        pred = sigmoid(linear_pred)


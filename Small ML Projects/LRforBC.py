import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification 

"""
Logistic Regression for Binary Classification 
- Implement LR to classify data points into 0 or 1 
"""

# Generate datasets
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1, flip_y=0.1, random_state=42)

# Logistic Regression 
class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.weights=None
        self.bias=None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y): 
        n_samples, _ = X.shape
        # shape[1] is # of columns in the array // number of features that each data point has 
        self.weights = np.zeros(X.shape[1])
        self.bias = 0 
        # gradient descent 
        for _ in range(self.n_iters):
            # for each iteration, need to update weights and biases
            # np.dot(weights * X) + bias ?
            model = np.dot(self.weights, X) + self.bias
            y_predicted = self._sigmoid(model)
            # need to compute the griadent of loss wrt weight and bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db 

    def predict(self, X): 
        # return binary predictions 
        linear_pred = np.dot(self.weights, X) + self.bias 
        y_pred = self._sigmoid(linear_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred 

if __name__ == "__main__":
    lr = LogisticRegression(lr=0.01, n_iters=1000)
    lr.fit(X,y)
    predictions = lr.predict(X)
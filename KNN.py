import numpy as np

def euclid(x1,x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN:
    def __init__(self, k=3):
        self.k=k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions 
    
    def _predict(self, x):
        # compute the distance for each 
        distances = [euclid(x,x_train) for x_train in self.X_train]

        # get the closest x 
        
        
        # return a prediction 
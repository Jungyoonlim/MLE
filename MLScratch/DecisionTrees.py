import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature 
        self.threshold = threshold
        self.left = left 
        self.right = right 
        self.value = None 


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root=None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree()
    
    def _grow_tree(self,X,y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = np.unique(y)

        # check the stopping criteria 
        if (depth>=self.max_depth or n_labels==1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=)

        # find the best split 

        # create child nodes 

    def _most_commmon_label(self,y):
        


    def predict(): 
import numpy as np
from collections import Counter 

class Node:
    def __init__(self, left=None, right=None, feature=None, threshold=None, *, value=None):
        self.left=left
        self.right=right
        self.feature = feature
        self.threshold=threshold
        self.value=value 
    
    def _is_leaf_node(self): return self.value is not None

class DecTree:
    def __init__(self, min_split=2, max_depth=1000, n_features=None):
        self.min_split=min_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root = None 

    def fit(self, X, y): 
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features) 
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
        
            

    def _best_split(self, X, y, feature_idxs):
        best_gain=-1
        split_idx, split_threshold = None, None

        for feat_idx in feature_idxs:
            X_column = X[:, feat_idx]
            threshold = np.unique(X_column)

        for thr in threshold:
            # calculate information gain 
            gain = self._information_gain(y, X_column, thr)

        if gain > best_gain:
            gain = best_gain
            split_idx = feat_idx
            split_threshold = thr 

        return split_idx, split_threshold 
    
    def _information_gain(): pass
        # parent entropy 
        # IG = E(parent) - [weighted average] * E(children)

    
    def _split(self, X_column, split_thresh): pass 

    # Entropy = -SUM(p(X))*log_2(p(X)), p(X) = #x/n 
    def _entropy(self,y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])

    def _most_common_label(self, y): 
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value 

    def predict(self, X): return np.array([self._traverse_tree(x) for x in X])
    
    def _traverse_tree(self, x, node): 
        if node._is_leaf_node(): return node.value
        if x[node.feature] <= node.threshold: 
            #left 
            return self._traverse_tree(x, node.left)
        #right 
        return self._traverse_tree(x, node.right )

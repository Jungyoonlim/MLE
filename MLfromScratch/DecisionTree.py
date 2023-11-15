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

        # Stopping Criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_split):
            leaf_value = self._most_common_label(y)
            return Node(value = leaf_value)
        
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # best split
        best_thresh, best_feature = self._best_split(X, y, feat_idxs)

        if best_feature is None: return Node(value=self._most_common_label(y))

        # child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :],y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feature_idxs):
        best_gain = -1
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
    
    def _information_gain(self, y, X_column, threshold): 
        # parent entropy 
        # IG = E(parent) - [weighted average] * E(children)
        parent_entropy = self._entropy(y)

        # Create children
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0: return 0

        # Calculate the weighted avg. entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        # Calculate the IG
        info_gain = parent_entropy - child_entropy
        return info_gain
    
    def _split(self, X_column, split_thresh): 
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column >= split_thresh).flatten()
        return left_idxs, right_idxs 

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

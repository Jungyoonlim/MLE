import numpy as np
import pandas as pd 
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

"""
A model that can classify text docs into one of several categories
based on their content. Use '20 Newsgroups' dataset - a collection of 20k 
newspaper docs, partitioned across 20 different newsgroups.
"""

# dataset
categories = fetch_20newsgroups(subset='all', categories=None, remove=('headers', 'footers', 'quotes'))
texts = categories.data
targets = categories.target

# data processing
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5, max_features=10000)
X = vectorizer.fit_transform(texts)
y = targets

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

class LogReg:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.weights=None
        self.bias=None 

    # sigmoid helper function
    def _sigmoid(self, z):
        return 1/(1+ np.exp(-z))
    
    def fit(self, X, y):
        n_samples, _ = X.shape
        weights = np.zeros(X.shape[1])
        bias = 0

        # gradient descent 
        for _ in range(self.n_iters):
            # weights * X + bias 
            model = np.dot(self.weights, X) + self.bias 
            # putting the model into the sigmoid fn
            y_pred = self._sigmoid(model)
            # gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)


    def predict(self, X):
        predictions = 
import numpy as np 

class SVM:
    def __init__(self, n_iters=1000, lr=0.01, lambda_param=0.01):
        self.n_iters = n_iters
        self.lr = lr
        self.lambda_param = lambda_param
        self.weight = None 
        self.bias = None 

    def fit(self, X, y): 
        n_samples, n_features = X.shape 

        y_ = np.where(y <= 0, -1, 1)

        # Initial Weights
        self.weight = np.zeros(n_features)
        self.bias = 0 

        """
        Update Rule
        
        ** a = alpha (learning rate) **

        if y_i * f(x) >= 1: 
            # here f(x) is x_i * weight - bias 
            w = w - lr * dw = w - lr * 2 * lambda * w
            b = b - lr * db = b 

        else:
            w = w - lr * dw = w - lr * (2 * lambda * w - y_i * x_i)
            b = b - lr * db = b - lr * y_i 
        """
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weight) - self.bias) >= 1
                if condition: 
                    self.weight -= self.lr * (2 * self.lambda_param * self.weight)
                else:
                    self.weight -= self.lr * (2 * self.lambda_param * self.weight - np.dot(x_i, y_[idx]))
                    self.bias -= self.lr * y_[idx]


    def predict(self, X): 
        approx = np.dot(X, self.weight) - self.bias 
        return np.sign(approx)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    X, y = datasets.make_blobs(
        n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state = 40
    ) 
    y = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    clf = SVM()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy 
    
    print("SVM classification accuracy", accuracy(y_test, predictions))

    def visualize_svm():
        def get_hyperplane_Value(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plt.scatter(X[:,0], X[:,1], marker="o", c=y)

        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amin(X[:, 0])

        x1_1 = get_hyperplane_Value(x0_1, clf.w, clf.b, 0)
        x1_2 = get_hyperplane_Value(x0_2, clf.w, clf.b, 0)

        x1_1_m = get_hyperplane_Value(x0_1, clf.w, clf.b, -1)
        x1_2_m = get_hyperplane_Value(x0_2, clf.w, clf.b, -1)

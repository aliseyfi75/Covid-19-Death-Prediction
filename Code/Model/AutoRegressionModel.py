from numpy.linalg import solve
import numpy as np


def gradient_descent(X, y, theta, alpha=0.0005, num_iters=1000):
    '''Gradient descent for linear regression'''
    # Initialisation of useful values
    m = np.size(y)

    for i in range(num_iters):
        # Cost and intermediate values for each iteration

        # Grad function in vectorized form
        h = X @ theta
        gradient = (1 / m) * (X.T @ (h - y))
        theta = theta - alpha * gradient
    return theta


def gradient_descent_reg(X, y, theta, alpha=0.0005, lamda=10, num_iters=1000):
    '''Gradient descent for ridge regression'''
    # Initialisation of useful values
    m = np.size(y)

    for i in range(num_iters):
        # Hypothesis function
        h = np.dot(X, theta)

        # Grad function in vectorized form
        theta = theta - alpha * (1 / m) * ((X.T @ (h - y)) + lamda * theta)

    return theta


def gradient_descent_las(X, y, theta, alpha=0.0005, lamda=10, num_iters=1000):
    '''Gradient descent for ridge regression'''
    # Initialisation of useful values
    m = np.size(y)
    J_history = np.zeros(num_iters)
    theta_0_hist, theta_1_hist = [], []  # Used for three D plot

    for i in range(num_iters):
        # Hypothesis function
        h = np.dot(X, theta)

        # Grad function in vectorized form
        theta = theta - alpha * (1 / m) * ((X.T @ (h - y)) + lamda)

    return theta


# Least Squares with a bias added
class AutoRegressionModel:

    def __init__(self, k, bias=False, method="normal", alpha=0.0005, num_iters=1000, lamda=10):
        self.k = k
        self.bias = bias
        self.method = method
        self.alpha = alpha
        self.num_iters = num_iters
        self.lamda = lamda

    def fit(self, X):
        Z, y = self.__ZY_creator(X)
        if self.method=="normal":
            self.w = solve(Z.T @ Z, Z.T @ y)
        elif self.method=="normal-L2":
            self.w = solve(Z.T @ Z + self.lamda*np.eye(Z.shape[1], dtype=float), Z.T @ y)
        elif self.method=="grad":
            w = np.zeros((Z.shape[1],1))
            w = gradient_descent(Z, y, w, alpha=self.alpha, num_iters=self.num_iters)
            self.w = w
        elif self.method=="grad-L2":
            w = np.zeros((Z.shape[1],1))
            w = gradient_descent_reg(Z, y, w, alpha=self.alpha, num_iters=self.num_iters, lamda=self.lamda)
            self.w = w
        elif self.method=="grad-L1":
            w = np.zeros((Z.shape[1],1))
            w = gradient_descent_las(Z, y, w, alpha=self.alpha, num_iters=self.num_iters, lamda=self.lamda)
            self.w = w

    def predict(self, X, num):
        Z = X[-self.k*num:]
        if self.bias:
            Z = np.insert(Z, 0, 1)
        return Z @ self.w

    def __ZY_creator(self, X):
        k = self.k

        num_samples = X[0].shape[0]

        temp_Z = []
        for x in X:
            temp_X = []
            for i in range(num_samples - k):
                # Slice a window of features
                temp_X.append(x[i:i + k])
            temp_Z.append(np.vstack(temp_X))

        y = []
        for i in range(num_samples - k):
            # Slice a window of features
            y.append(X[0][i + k])

        Z = np.array(np.hstack(temp_Z))
        y = np.array(y).reshape(-1, 1)

        n, d = Z.shape

        if self.bias:
            bias = np.ones((n, 1))
            Z = np.append(bias, Z, axis=1)

        return Z, y

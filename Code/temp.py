from numpy.linalg import solve
import numpy as np


def gradient_descent(X, y, theta, alpha=0.0005, num_iters=1000):
    '''Gradient descent for linear regression'''
    # Initialisation of useful values
    m = np.size(y)

    for i in range(num_iters):
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
        self.Z = None

    def fit(self, X):
        Z, y = self.__ZY_creator(X)
        self.Z = Z
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

    def predict(self, X):
        Z = X[-self.k:]
        if self.bias:
            Z = np.insert(Z, 0, 1)
        return Z @ self.w

    def __ZY_creator(self, X):
        k = self.k

        num_samples = X.shape[0]

        temp_X = []
        y = []
        for i in range(num_samples - k):
            # Slice a window of features
            temp_X.append(X[i:i + k])
            y.append(X[i + k])

        Z = np.vstack(temp_X)
        y = np.array(y).reshape(-1, 1)

        n, d = Z.shape

        if self.bias:
            bias = np.ones((n, 1))
            Z = np.append(bias, Z, axis=1)

        return Z, y






################################################## 




from Model.AutoRegressionModel import AutoRegressionModel as arm

errs = []
for lag in range(1, 40):    
    model = arm(k=lag, bias=True)
    model.fit(deaths)
    predictions = pd.DataFrame({
        "deaths": validation_df_ca['deaths'].values, 
        "deaths_pred": model.predict(205, 205+10)
    })
    err1 = np.sqrt(np.mean((predictions['deaths'] - predictions['deaths_pred']).values ** 2))
    err2 = np.sqrt(np.mean((res.predict(205, 205+19) - second_val_df['deaths'].values) ** 2))
    err3 = np.sqrt(np.mean((res.predict(205+11, 205+19) - second_val_df['deaths'].values[-9:]) ** 2))
    errs.append((lag, err1, err2, err3))
    
pd.DataFrame(errs, columns=['lag', 'mse 11', 'mse 20', 'mse 11-20']).plot\
    .line(x='lag', y=['mse 11', 'mse 20', 'mse 11-20'], figsize=(15, 7), grid=True)



    ########################################################


    
from AutoReg_MultiVar import AutoRegressionModelMultiVar as armv

errs = []

i = 0
for begining_ind in range(70, 180):
    data = new_ca_train_df[begining_ind:]
    deaths = data['deaths'].values
    cases = data['cases'].values
    for k_lag in range(1, 20):
        for seq_case_count in range(20):
            for case_lag in range(5, 30):
                lag_indices = list(np.arange(case_lag, min(case_lag+seq_case_count, 30)))
                model = armv(k=k_lag, bias=True)
                if len(lag_indices) == 0:
                    model.fit(deaths, np.empty(0), np.empty(0))
                else:
                    model.fit(deaths, [cases], [lag_indices])
                error = calculate_error(ca_val_df['deaths'][-5:].values, model.predict(len(data), len(data)+4))
                errs.append((begining_ind, k_lag, seq_case_count, case_lag, error))
                i += 1
                if i % 1000 == 0:
                    print("i is:", i)
error_df = pd.DataFrame(errs, columns=['begining_index', 'k_lag', 'case_sequence_count', 'case_lag', 'error'])
    
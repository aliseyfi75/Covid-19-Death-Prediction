from numpy.linalg import solve
import numpy as np
# from AutoRegressionModel import gradient_descent_las, gradient_descent_reg, gradient_descent


# Least Squares with a bias added
class AutoRegressionModelMultiVar():

    def __init__(self, k, bias=False, method="normal", alpha=0.0005, num_iters=1000, lamda=10):
        self.k = k
        self.bias = bias
        self.method = method
        self.alpha = alpha
        self.num_iters = num_iters
        self.lamda = lamda

    def fit(self, X, S, input_lags):
        Z, y = self.__ZY_creator_multiVar(X, S, input_lags)
        if self.method == "normal":
            self.w = solve(Z.T @ Z, Z.T @ y)
#         elif self.method == "normal-L2":
#             self.w = solve(Z.T @ Z + self.lamda * np.eye(Z.shape[1], dtype=float), Z.T @ y)
#         elif self.method == "grad":
#             w = np.zeros((Z.shape[1], 1))
#             w = gradient_descent(Z, y, w, alpha=self.alpha, num_iters=self.num_iters)
#             self.w = w
#         elif self.method == "grad-L2":
#             w = np.zeros((Z.shape[1], 1))
#             w = gradient_descent_reg(Z, y, w, alpha=self.alpha, num_iters=self.num_iters, lamda=self.lamda)
#             self.w = w
#         elif self.method == "grad-L1":
#             w = np.zeros((Z.shape[1], 1))
#             w = gradient_descent_las(Z, y, w, alpha=self.alpha, num_iters=self.num_iters, lamda=self.lamda)
#             self.w = w

    def predict(self, start, end):
        max_lag = self.max_lag
        Z = self.Z
        w = self.w
        y = self.y
        k = self.k
        S = self.S
        input_lags = self.input_lags

        ti = end - max_lag
        bi = start - max_lag

        if ti < Z.shape[0]:
            return Z[ti, :] @ w

        vector = Z[-1, :]
        lp = y[-1]

        result = []

        for i in range(Z.shape[0], ti+1):
            vector = np.append(1, np.append(vector[2:k+1], lp))

            temp_row = []
            for ind, s in enumerate(S):
                temp_S_row = []
                for lag_index in input_lags[ind]:
                    temp_S_row = np.append(temp_S_row, s[i + max_lag - lag_index])

                temp_row = np.append(temp_row, temp_S_row)

            temp_row = temp_row.flatten() if len(temp_row) > 0 else np.empty(0)
            vector = np.append(vector, temp_row)

            lp = vector @ w

            if i >= bi:
                result.append(lp[0])

        return result

    def __ZY_creator_multiVar(self, X, S, input_lags):
        k = self.k

        num_samples = X.shape[0]
        raveled_lags = np.concatenate(input_lags) if len(input_lags) > 0 else input_lags
        max_lag = max(np.append(0, raveled_lags).max(), k)
        temp_X = []
        y = []
        for i in range(num_samples - max_lag):
            # Slice a window of features
            temp_X.append(X[i+max_lag-k:i + max_lag])
            y.append(X[i + max_lag])

        final_X = np.vstack(temp_X)
        y = np.array(y).reshape(-1, 1)

        temp_Z = []
        for i in range(final_X.shape[0]):
            temp_row = []
            for ind, s in enumerate(S):
                temp_S_row = []
                for lag_index in input_lags[ind]:
                    temp_S_row = np.append(temp_S_row, s[i+max_lag-lag_index])

                temp_row = np.append(temp_row, temp_S_row)

            temp_row = temp_row.flatten() if len(temp_row) > 0 else np.empty(0)
            temp_Z = np.append(temp_Z, temp_row)

        temp_Z = temp_Z.reshape(final_X.shape[0], -1)

        temp_Z = np.column_stack((final_X, temp_Z))

        n, d = temp_Z.shape

        Z = temp_Z
        if self.bias:
            bias = np.ones((n, 1))
            Z = np.append(bias, temp_Z, axis=1)

        self.Z = Z
        self.y = y
        self.max_lag = max_lag
        self.S = S
        self.input_lags = input_lags

        return Z, y

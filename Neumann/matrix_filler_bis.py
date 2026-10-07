import numpy as np


# a_hat is the matrix when integrathing d(phi)
def fill_Ax(X, a_hat):

    n = len(X)
    # Initiating matrix of size n+1 with zeros
    Ax = np.zeros((n, n))

    for k in range(n - 1):
        hk = X[k] - X[k + 1]
        # Compute the constant C_xk
        C_xk: float = 1.0 / hk
        # Fill the 2x2 matrix that appears on the diagonal
        Ax[k : k + 2, k : k + 2] += C_xk * a_hat

    return Ax


# b_hat is the matrix when integrating phi
def fill_Bx(X, b_hat):

    n = len(X)
    # Initiating matrix of size n+1 with zeros
    Bx = np.zeros((n, n))

    for k in range(n - 1):
        # Loop through every rectangle
        hk = X[k] - X[k + 1]
        # Compute the constant C_xk
        C_xk: float = hk
        # Fill the 2x2 matrix
        Bx[k : k + 2, k : k + 2] += C_xk * b_hat

    return Bx


# a_hat is the matrix when integrathing d(phi)
def fill_Ay(Y, a_hat):

    m = len(Y)
    Ay = np.zeros((m, m))

    for k in range(m - 1):
        hk = Y[k] - Y[k + 1]
        C_yk: float = 1.0 / hk
        Ay[k : k + 2, k : k + 2] += C_yk * a_hat

    return Ay


# b_hat is the matrix when integrating phi
def fill_By(Y, b_hat):

    m = len(Y)
    By = np.zeros((m, m))

    for k in range(m - 1):

        hk = Y[k] - Y[k + 1]

        C_xk: float = hk

        By[k : k + 2, k : k + 2] += C_xk * b_hat

    return By

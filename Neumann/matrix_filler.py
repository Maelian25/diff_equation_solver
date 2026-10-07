import numpy as np


# a_hat is the matrix when integrathing d(phi)
def fill_Ax(X, a_hat):

    n = len(X) - 1
    # Initiating matrix of size n+1 with zeros
    Ax = np.zeros((n + 1, n + 1))

    for k in range(n):
        # Loop through every rectangle
        i0, i1 = k, k + 1
        # Compute the constant C_xk
        hk = X[i1] - X[i0]
        C_xk: float = 1.0 / hk

        indexes = [i0, i1]
        # Loop through alpha and beta
        for alpha in [0, 1]:
            for beta in [0, 1]:
                Ax[indexes[alpha], indexes[beta]] += C_xk * a_hat[alpha, beta]

    return Ax


# b_hat is the matrix when integrating phi
def fill_Bx(X, b_hat):

    n = len(X) - 1
    Bx = np.zeros((n + 1, n + 1))

    for k in range(n):

        i0, i1 = k, k + 1
        hk = X[i1] - X[i0]  # hk here is C_xk^-1

        indexes = [i0, i1]
        for alpha in [0, 1]:
            for beta in [0, 1]:
                Bx[indexes[alpha], indexes[beta]] += hk * b_hat[alpha, beta]

    return Bx


# a_hat is the matrix when integrathing d(phi)
def fill_Ay(Y, a_hat):

    m = len(Y) - 1
    Ay = np.zeros((m + 1, m + 1))

    for k in range(m):
        j0, j1 = k, k + 1
        hk = Y[j1] - Y[j0]
        C_yk: float = 1.0 / hk

        indexes = [j0, j1]
        for alpha in [0, 1]:
            for beta in [0, 1]:
                Ay[indexes[alpha], indexes[beta]] += C_yk * a_hat[alpha, beta]

    return Ay


# b_hat is the matrix when integrating phi
def fill_By(Y, b_hat):

    m = len(Y) - 1
    By = np.zeros((m + 1, m + 1))

    for k in range(m):

        j0, j1 = k, k + 1
        hk = Y[j1] - Y[j0]

        indexes = [j0, j1]
        for alpha in [0, 1]:
            for beta in [0, 1]:
                By[indexes[alpha], indexes[beta]] += hk * b_hat[alpha, beta]

    return By

import numpy as np
import math


def matrix_invert(M):
    # We take a matrix as input given (here its Bx, By)
    return np.linalg.inv(M)


def M_tilde(B, A):
    # We compute for either x or y
    b_inv = matrix_invert(B)
    return b_inv @ A


def diag_M(A, sort=False):
    # Matrix diagonalisation
    # No real need to sort the eigen_values increasingly
    eigen_values, eigen_vectors = np.linalg.eig(A)

    # eigen values/vectors might not be real bc of approximation
    # (ususally this methods adds a residual complex number)
    eigen_values = np.real(eigen_values)
    eigen_vectors = np.real(eigen_vectors)

    # we can also sort the eigen values (and eigen vectors)
    if sort:
        # get the new indexes that would sort the vector
        new_indexes = np.argsort(eigen_values)

        # We now sort both eigen_values and eigen
        eigen_values = eigen_values[new_indexes]
        # Change all columns according to the new indexes
        eigen_vectors = eigen_vectors[:, new_indexes]

    # print(f"The shape of eigen values is : {eigen_values.shape}")
    # print(f"The shape of eigen vectors is : {eigen_vectors.shape}")

    return eigen_values, eigen_vectors


def exact_solution(x, y):
    # compute f(x,y) = cos(x)*cos(y)
    return math.cos(x) * math.cos(y)


def exact_solution_2(x, y):
    # compute f(x,y) = cos(x)*cos(y)
    return math.sin(x) * math.sin(y)


def second_member(x, y):
    # compute second member f(x,y) = 2 * cos(x)*cos(y)
    return 2 * math.cos(x) * math.cos(y)


def second_member_2(x, y):
    # compute second member f(x,y) = 2 * cos(x)*cos(y)
    return 2 * math.sin(x) * math.sin(y)


def V(X, Y):
    # X and Y are vectors
    V = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            V[i][j] = exact_solution(X[i], Y[j])

    return V


def V_2(X, Y):
    # X and Y are vectors
    V = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            V[i][j] = exact_solution_2(X[i], Y[j])

    return V


def F(X, Y):
    # X and Y are vectors
    F = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            F[i][j] = second_member(X[i], Y[j])
    return F


def F_2(X, Y):
    # X and Y are vectors
    F = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            F[i][j] = second_member_2(X[i], Y[j])
    return F


def edge_conditions(
    X,
    Y,
):
    left_lim, right_lim = X[0], X[-1]
    down_lim, up_lim = Y[0], Y[-1]

    sol = np.zeros((len(X), len(Y)))

    sol[:, 0] = np.cos(X) * np.cos(down_lim)
    sol[:, -1] = np.cos(X) * np.cos(up_lim)

    sol[0, :] = np.cos(left_lim) * np.cos(Y)
    sol[-1, :] = np.cos(right_lim) * np.cos(Y)

    return sol


def edge_conditions_2(
    X,
    Y,
):

    sol = np.zeros((len(X), len(Y)))

    sol[:, 0] = 0
    sol[:, -1] = 0

    sol[0, :] = 0
    sol[-1, :] = 0

    return sol

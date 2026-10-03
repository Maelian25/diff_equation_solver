import numpy as np

import time

import matplotlib.pyplot as plt

from utils import M_tilde, diag_M, matrix_invert, V, F
from matrix_filler_bis import fill_Ax, fill_Ay, fill_Bx, fill_By
from matrix_creation import a_hat, b_hat
from mesh_generation import mesh_generation

if __name__ == "__main__":
    n_min = 10
    n_max = 100

    delta_x_evo = []
    log_e_evo = []
    min_eigen_value_P = []
    min_eigen_value_Q = []

    start = time.time()

    for k in range(n_min, n_max, 2):

        X, Y = mesh_generation(k)

        Ax = fill_Ax(X, a_hat)
        Ay = fill_Ay(Y, a_hat)

        Bx = fill_Bx(X, b_hat)
        By = fill_By(Y, b_hat)

        Ax_tilde = M_tilde(A=Ax, B=Bx)
        Ay_tilde = M_tilde(A=Ay, B=By)

        eigen_values_P, P = diag_M(Ax_tilde)
        eigen_values_Q, Q = diag_M(Ay_tilde)

        P_inv = matrix_invert(P)
        Q_inv = matrix_invert(Q)

        # print(f"Q_inv shape {Q_inv.shape}")

        F_hat_first = F(X, Y) @ Q_inv.T
        F_hat = P_inv @ F_hat_first

        U_hat_list = [
            (
                F_hat[i, j] / (eigen_values_P[i] + eigen_values_Q[j])
                if eigen_values_P[i] > 1e-7 and eigen_values_Q[j] > 1e-7
                else 0
            )
            for i in range(F_hat.shape[0])
            for j in range(F_hat.shape[1])
        ]  # This produce a list that needs to be reshaped

        U_hat = np.array(U_hat_list).reshape(F_hat.shape)

        U_first = P @ U_hat
        U = U_first @ Q.T

        E = U - V(X, Y)

        log_e = np.log(np.max(abs(E)) + 1e-7)

        delta_X = np.diff(X)
        log_delta_X = np.log(delta_X)
        delta_x_evo.append(log_delta_X[0])

        log_e_evo.append(log_e)

        # print("valeur max :", np.max(np.abs(E)))

        min_eigen_value_P.append(np.min(np.abs(eigen_values_P)))
        min_eigen_value_Q.append(np.min(np.abs(eigen_values_Q)))

    end = time.time()

    # print("Size of log_e_evo", log_e_evo.__len__())
    # print("Size of log_delta_x", delta_x_evo.__len__())

    slope, x_origin = np.polyfit(delta_x_evo, log_e_evo, 1)

    print(f"Convergence order : {slope:.2f}")

    plt.figure(figsize=(8, 6))
    plt.plot(delta_x_evo, log_e_evo, "bo-")
    plt.xlabel("log(delta x)")
    plt.ylabel("log(Max error)")
    plt.title("Log E fn of log Delta X")
    plt.grid(True, which="both", ls="--")
    plt.savefig("log_e_fn_delta_x.pdf", format="pdf")
    plt.show()

    x_axis = list(range(n_min, n_max, 2))

    plt.figure(figsize=(8, 6))
    plt.plot(x_axis, min_eigen_value_P, "ro-", label="Min Eigenvalue P (Ax_tilde)")
    plt.plot(x_axis, min_eigen_value_Q, "go-", label="Min Eigenvalue Q (Ay_tilde)")
    plt.xlabel("Iteration")
    plt.ylabel("Min value of eigen value")
    plt.title("Evo of the min eigen value fn of delta x")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig("min_eigen_value_graph.pdf", format="pdf")
    plt.show()

    y_regression = slope * np.array(delta_x_evo) + x_origin

    # 3. Tracé des points et de la droite
    plt.figure(figsize=(8, 6))
    plt.plot(delta_x_evo, log_e_evo, "bo-", label="Measured error")
    plt.plot(delta_x_evo, y_regression, "r--", label="Linear approximation")

    plt.xlabel("log(delta x)")
    plt.ylabel("log(Max error)")
    plt.title("Convergence and method order")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig("linear_regression_plot.pdf", format="pdf")
    plt.show()

    print(f"Time to compute {n_max-n_min} iterations : {end-start:.2f} s")

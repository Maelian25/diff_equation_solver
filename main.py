import argparse

import numpy as np

import time

import os

import matplotlib.pyplot as plt

from utils import M_tilde, diag_M, matrix_invert, V, F
from matrix_filler_bis import fill_Ax, fill_Ay, fill_Bx, fill_By
from matrix_creation import a_hat, b_hat
from mesh_generation import mesh_generation


def solve(n, m):
    # Solve for mesh (nxm)
    X, Y = mesh_generation(n, m, plot=False)

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

    return {
        "Solution": U,
        "X": X,
        "Y": Y,
        "eigen_values_P": eigen_values_P,
        "eigen_values_Q": eigen_values_Q,
    }


def plot_solution(n, m, problem_name, save=True):

    plt.close()
    s = solve(n, m)

    XX, YY = np.meshgrid(s["X"], s["Y"], indexing="ij")

    fig = plt.figure(figsize=(16, 5))

    # Numerical solution in 3D
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    surf = ax1.plot_surface(XX, YY, s["Solution"], cmap="viridis")
    ax1.set_title(f"Numerical Solution (n={n}, m={m})")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    fig.colorbar(surf, ax=ax1, shrink=0.6, pad=0.1)

    # Exact solution in 3D
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    surf2 = ax2.plot_surface(XX, YY, V(s["X"], s["Y"]), cmap="viridis")
    ax2.set_title("Exact Solution")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    fig.colorbar(surf2, ax=ax2, shrink=0.6, pad=0.1)

    # Absolute error map
    ax3 = fig.add_subplot(1, 3, 3)
    cf = ax3.contourf(
        XX, YY, np.abs(s["Solution"] - V(s["X"], s["Y"])), levels=30, cmap="magma"
    )
    ax3.set_title(
        f"|U - V|  (max = {np.abs(s["Solution"] - V(s["X"], s["Y"])).max():.2e})"
    )
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_aspect("equal")
    fig.colorbar(cf, ax=ax3)

    plt.tight_layout()
    if save:
        plt.savefig(f"./results/{problem_name}_solution_plot.pdf", format="pdf")
    plt.show()


if __name__ == "__main__":
    # we used now a square mesh, so n=m
    # still handles cases where n != m

    # Handles argument so that code don't change between calls
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--save", type=bool, default=False, help="Whether we save or not"
    )
    parser.add_argument(
        "-pb",
        "--problem_name",
        type=str,
        help="Name of the question/problem name",
    )
    args = parser.parse_args()

    # Make sure the folder exists
    if not os.path.exists("./results"):
        os.makedirs("./results")

    problem_name = args.problem_name | ""
    n_min = 10
    n_max = 100

    m_min = 10
    m_max = 100

    delta_x_evo = []
    log_e_evo = []
    min_eigen_value_P = []
    min_eigen_value_Q = []

    start = time.time()

    for n, m in zip(range(n_min, n_max, 2), range(m_min, m_max, 2)):
        print(f"Solving for n={n} and m={m}...")
        s = solve(n, m)

        U = s["Solution"]
        X = s["X"]
        Y = s["Y"]
        eigen_values_P = s["eigen_values_P"]
        eigen_values_Q = s["eigen_values_Q"]

        E = U - V(X, Y)
        log_e = np.log(np.max(abs(E)) + 1e-7)
        log_e_evo.append(log_e)

        delta_X = np.diff(X)
        log_delta_X = np.log(delta_X)
        delta_x_evo.append(log_delta_X[0])

        min_eigen_value_P.append(np.min(np.abs(eigen_values_P)))
        min_eigen_value_Q.append(np.min(np.abs(eigen_values_Q)))

    end = time.time()

    slope, x_origin = np.polyfit(delta_x_evo, log_e_evo, 1)
    print(f"Convergence order : {slope:.2f}")

    # Log E fn of Log delta x
    plt.figure(figsize=(8, 6))
    plt.plot(delta_x_evo, log_e_evo, "bo-")
    plt.xlabel("log(delta x)")
    plt.ylabel("log(Max error)")
    plt.title("Log E fn of log Delta X")
    plt.grid(True, which="both", ls="--")
    if args.save:
        plt.savefig(f"./results/{problem_name}_log_e_fn_delta_x.pdf", format="pdf")
    plt.show()

    x_axis = list(range(n_min, n_max, 2))

    # Min eigen value fn of delta x
    plt.figure(figsize=(8, 6))
    plt.plot(x_axis, min_eigen_value_P, "ro-", label="Min Eigenvalue P (Ax_tilde)")
    plt.plot(x_axis, min_eigen_value_Q, "go-", label="Min Eigenvalue Q (Ay_tilde)")
    plt.xlabel("Iteration")
    plt.ylabel("Min value of eigen value")
    plt.title("Evo of the min eigen value fn of delta x")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    if args.save:
        plt.savefig(f"./results/{problem_name}_min_eigen_value_graph.pdf", format="pdf")
    plt.show()

    y_regression = slope * np.array(delta_x_evo) + x_origin

    # Linear approx of measured error
    plt.figure(figsize=(8, 6))
    plt.plot(delta_x_evo, log_e_evo, "bo-", label="Measured error")
    plt.plot(delta_x_evo, y_regression, "r--", label="Linear approximation")
    plt.xlabel("log(delta x)")
    plt.ylabel("log(Max error)")
    plt.title("Convergence and method order")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    if args.save:
        plt.savefig(
            f"./results/{problem_name}_linear_regression_plot.pdf", format="pdf"
        )
    plt.show()

    print(f"Time to compute {n_max-n_min} iterations : {end-start:.2f} s")

    k_plot = 80

    print(f"Now ploting solution for n = {k_plot} and m ={k_plot}...")

    plot_solution(k_plot, k_plot, problem_name, args.save)

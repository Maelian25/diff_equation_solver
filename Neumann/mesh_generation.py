import numpy as np
import math
from matplotlib import pyplot as plt


# Generating mesh for a 2D problem
def mesh_generation(n, m, interval=[2 * math.pi, 2 * math.pi], plot=False):
    # n is now an argument n = 20, it corresponds to the number of cells in x-dir
    # m is now an argument m = 20, it corresponds to the number of cells in y-dir
    # if not evenly spaced

    evenly_spaced = m == n

    Lx = interval[0]  # x-dir
    Ly = interval[1]  # y-dir

    x_list = np.zeros(n + 1)
    y_list = np.zeros(m + 1)

    if evenly_spaced:
        m = n
        # if evenly spaced, then the following is enough
        x_list = np.linspace(0, Lx, n + 1)  # x-coordinates of the mesh
        y_list = np.linspace(0, Ly, m + 1)  # y-coordinates of the mesh

    else:
        # if they are not evenly spaced
        # these could be lists of deltas through the domain, but for now
        # we will just use the same as above
        delta_x = Lx / n
        delta_y = Ly / m

        for i in range(n):
            x_list[i + 1] = x_list[i] + delta_x

        for j in range(m):
            y_list[j + 1] = y_list[j] + delta_y

    if plot:
        # Not showing for coding purposes most of the time
        # red dots
        for i in range(n + 1):
            for j in range(m + 1):
                plt.plot(x_list[i], y_list[j], "ro")

        # vertical and horizontal lines
        for i in range(n + 1):
            # plottting [y0, yn] at [x, x]
            # using b- (plotting only between two points)
            plt.plot([x_list[i], x_list[i]], [y_list[0], y_list[-1]], "b-")

        for j in range(m + 1):
            # plottting [x0, xn] at [y, y]
            # using b- (plotting only between two points)
            plt.plot([x_list[0], x_list[-1]], [y_list[j], y_list[j]], "b-")

        plt.title("Rectangular Mesh")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")

        plt.savefig("./results/new_mesh.pdf", format="pdf")

        plt.show()
    return x_list, y_list

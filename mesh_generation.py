import numpy as np
import math
from matplotlib import pyplot as plt


# Size of the mesh
def mesh_generation(n):
    Lx = 2 * math.pi  # x-dir
    Ly = Lx  # y-dir

    # n is now an argument n = 20  # number of cells in x-dir
    m = n  # number of cells in y-dir

    evenly_spaced = True  # if the mesh is evenly spaced or not

    x_list = np.zeros(n + 1)
    y_list = np.zeros(m + 1)

    if evenly_spaced:
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

    # plt.savefig("new_mesh.pdf", format="pdf")

    # Not showing for coding purposes
    # plt.show()
    return x_list, y_list

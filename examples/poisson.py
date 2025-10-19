# Solver for the Poisson equation

import argparse

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
import numpy as np
import muGrid
from muGrid.Solvers import conjugate_gradients

try:
    from mpi4py import MPI

    comm = muGrid.Communicator(MPI.COMM_WORLD)
except ImportError:
    comm = muGrid.Communicator()

from NuMPI.Testing.Subdivision import suggest_subdivisions

parser = argparse.ArgumentParser(
    prog="Poisson", description="Solve the Poisson equation"
)
parser.add_argument("-n", "--nb-grid-pts", default="32,32")
args = parser.parse_args()

nb_grid_pts = [int(x) for x in args.nb_grid_pts.split(",")]

s = suggest_subdivisions(len(nb_grid_pts), comm.size)

decomposition = muGrid.CartesianDecomposition(comm, nb_grid_pts, s, (1, 1), (1, 1))
fc = decomposition.collection
grid_spacing = 1 / np.array(nb_grid_pts)  # Grid spacing

stencil = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # FD-stencil for the Laplacian
laplace = muGrid.ConvolutionOperator([-1, -1], stencil)

x, y = decomposition.coords  # Domain-local coords for each pixel

rhs = fc.real_field("rhs")
solution = fc.real_field("solution")

rhs.p = (1 + np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)) ** 10
rhs.p -= np.mean(rhs.p)


def callback(it, x, r, p):
    """
    Callback function to print the current solution, residual, and search direction.
    """
    print(f"{it:5} {np.dot(r.ravel(), r.ravel()):.5}")


def hessp(x, Ax):
    """
    Function to compute the product of the Hessian matrix with a vector.
    The Hessian is represented by the convolution operator.
    """
    decomposition.communicate_ghosts(x)
    laplace.apply(x, Ax)
    # We need the minus sign because the Laplace operator is negative
    # definite, but the conjugate-gradients solver assumes a
    # positive-definite operator.
    Ax.s /= -np.mean(grid_spacing) ** 2  # Scale by grid spacing
    return Ax


conjugate_gradients(
    comm,
    fc,
    hessp,  # linear operator
    rhs,
    solution,
    tol=1e-6,
    callback=callback,
    maxiter=1000,
)

if plt is not None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(rhs.p)
    ax2.imshow(solution.p)
    plt.show()

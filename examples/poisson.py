# Solver for the Poisson equation

import argparse
import time

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
import numpy as np

import muGrid
from muGrid import real_field
from muGrid.Solvers import conjugate_gradients

try:
    from mpi4py import MPI

    comm = muGrid.Communicator(MPI.COMM_WORLD)
except ImportError:
    comm = muGrid.Communicator()

from NuMPI.Testing.Subdivision import suggest_subdivisions

parser = argparse.ArgumentParser(
    prog="Poisson",
    description="Solve the Poisson equation"
)

parser.add_argument(
    "-n", "--nb-grid-pts",
    default=[32, 32],
    type=lambda s: [int(x) for x in s.split(",")],
    help="Grid points as nx,ny (default: 32,32)"
)

_memory_locations = {
    "host": muGrid.GlobalFieldCollection.MemoryLocation.Host,
    "device": muGrid.GlobalFieldCollection.MemoryLocation.Device,
}

parser.add_argument(
    "-m", "--memory",
    choices=_memory_locations,
    default="host",
    help="Memory space for allocation (default: host)"
)

parser.add_argument(
    "-i", "--maxiter",
    type=int,
    default=1000,
    help="Maximum number of CG iterations (default: 1000)"
)

parser.add_argument(
    "-p", "--plot",
    action="store_true",
    help="Show plot of RHS and solution (default: off)"
)

args = parser.parse_args()

if args.memory == "host":
    import numpy as arr
else:
    import cupy as arr

args.memory = _memory_locations[args.memory]

s = suggest_subdivisions(len(args.nb_grid_pts), comm.size)

decomposition = muGrid.CartesianDecomposition(comm, args.nb_grid_pts, s, (1, 1), (1, 1),
                                              memory_location=args.memory)
grid_spacing = 1 / np.array(args.nb_grid_pts)  # Grid spacing

stencil = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # FD-stencil for the Laplacian
laplace = muGrid.ConvolutionOperator([-1, -1], stencil)

x, y = decomposition.coords  # Domain-local coords for each pixel

# Create fields using the helper function - works directly with CartesianDecomposition
rhs = real_field(decomposition, "rhs")
solution = real_field(decomposition, "solution")

rhs.p[...] = arr.asarray((1 + np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)) ** 10)
rhs.p[...] -= arr.mean(rhs.p)


def callback(it, x, r, p):
    """
    Callback function to print the current solution, residual, and search direction.
    """
    print(f"{it:5} {arr.dot(r.ravel(), r.ravel()):.5}")


def hessp(x, Ax):
    """
    Function to compute the product of the Hessian matrix with a vector.
    The Hessian is represented by the convolution operator.
    """
    decomposition.communicate_ghosts(x._cpp)
    laplace.apply(x._cpp, Ax._cpp)
    # We need the minus sign because the Laplace operator is negative
    # definite, but the conjugate-gradients solver assumes a
    # positive-definite operator.
    Ax.s[...] /= -np.mean(grid_spacing) ** 2  # Scale by grid spacing
    return Ax


start_time = time.perf_counter()
conjugate_gradients(
    comm,
    decomposition.collection,
    hessp,  # linear operator
    rhs._cpp,  # Pass the underlying C++ field
    solution._cpp,
    tol=1e-6,
    callback=callback,
    maxiter=args.maxiter,
)
elapsed_time = time.perf_counter() - start_time
print(f"CG solver completed in {elapsed_time:.4f} seconds")

if args.plot:
    if plt is None:
        print("Warning: matplotlib not available, cannot show plot")
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(rhs.p)
        ax1.set_title("RHS")
        ax2.imshow(solution.p)
        ax2.set_title("Solution")
        plt.tight_layout()
        plt.show()

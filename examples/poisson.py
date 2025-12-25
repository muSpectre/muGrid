# Solver for the Poisson equation

import argparse

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
    help="Grid points as nx,ny or nx,ny,nz (default: 32,32)"
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

parser.add_argument(
    "-s", "--stencil",
    choices=["generic", "hardcoded"],
    default="generic",
    help="Stencil implementation: 'generic' (sparse convolution) or "
         "'hardcoded' (optimized Laplace operator) (default: generic)"
)

args = parser.parse_args()

if args.memory == "host":
    import numpy as arr
else:
    import cupy as arr

args.memory = _memory_locations[args.memory]

dim = len(args.nb_grid_pts)
if dim not in (2, 3):
    raise ValueError("Only 2D and 3D grids are supported")

s = suggest_subdivisions(dim, comm.size)

# Set up ghost layers for the stencil (1 layer in each direction)
left_ghosts = (1,) * dim
right_ghosts = (1,) * dim

decomposition = muGrid.CartesianDecomposition(comm, args.nb_grid_pts, s,
                                              left_ghosts, right_ghosts,
                                              memory_location=args.memory)
grid_spacing = 1 / np.array(args.nb_grid_pts)  # Grid spacing

# FD-stencil for the Laplacian
if dim == 2:
    nb_stencil_pts = 5
else:
    nb_stencil_pts = 7

# Scaling factor for the Laplacian (we need the minus sign because
# the Laplace operator is negative definite, but the CG solver
# assumes a positive-definite operator)
laplace_scale = -1.0 / np.mean(grid_spacing) ** 2

# Create the Laplace operator based on the selected implementation
if args.stencil == "generic":
    # Generic sparse convolution operator
    if dim == 2:
        # 5-point stencil for 2D
        stencil = laplace_scale * np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        stencil_offset = [-1, -1]
    else:
        # 7-point stencil for 3D
        stencil = np.zeros((3, 3, 3))
        stencil[1, 1, 0] = 1  # z-1
        stencil[1, 0, 1] = 1  # y-1
        stencil[0, 1, 1] = 1  # x-1
        stencil[1, 1, 1] = -6  # center
        stencil[2, 1, 1] = 1  # x+1
        stencil[1, 2, 1] = 1  # y+1
        stencil[1, 1, 2] = 1  # z+1
        stencil *= laplace_scale
        stencil_offset = [-1, -1, -1]
    laplace = muGrid.ConvolutionOperator(stencil_offset, stencil)
    stencil_name = "Generic sparse convolution"
else:
    # Hard-coded optimized Laplace operator (for benchmarking)
    # Pass scale factor to fold in grid spacing and positive-definiteness
    laplace = muGrid.LaplaceOperator(dim, laplace_scale)
    stencil_name = "Hard-coded Laplace operator"

coords = decomposition.coords  # Domain-local coords for each pixel

# Create fields using the helper function - works directly with CartesianDecomposition
rhs = real_field(decomposition, "rhs")
solution = real_field(decomposition, "solution")

# Set up RHS with a smooth function
if dim == 2:
    x, y = coords
    rhs.p[...] = arr.asarray((1 + np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)) ** 10)
else:
    x, y, z = coords
    rhs.p[...] = arr.asarray((1 + np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) *
                              np.cos(2 * np.pi * z)) ** 10)
rhs.p[...] -= arr.mean(rhs.p)

# Performance counters
nb_grid_pts_total = np.prod(args.nb_grid_pts)
nb_hessp_calls = 0

# Create global timer for hierarchical timing
timer = muGrid.Timer()


def callback(it, x, r, p):
    """
    Callback function to print the current solution, residual, and search direction.
    """
    print(f"{it:5} {arr.dot(r.ravel(), r.ravel()):.5}")


def hessp(x, Ax):
    """
    Function to compute the product of the Hessian matrix with a vector.
    The Hessian is represented by the convolution operator.
    The scale factor (grid spacing and sign) is already folded into the operator.
    """
    global nb_hessp_calls
    with timer("hessp"):
        nb_hessp_calls += 1
        with timer("communicate_ghosts"):
            decomposition.communicate_ghosts(x._cpp)
        with timer("apply"):
            laplace.apply(x._cpp, Ax._cpp)
    return Ax


with timer("conjugate_gradients"):
    try:
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
        print("CG converged.")
    except RuntimeError:
        print("CG did not converge.")

elapsed_time = timer.get_time("conjugate_gradients")

# Performance metrics
print(f"\n{'='*60}")
print("Performance Summary")
print(f"{'='*60}")
print(f"Grid size: {' x '.join(map(str, args.nb_grid_pts))} = "
      f"{nb_grid_pts_total:,} points")
print(f"Dimensions: {dim}D")
print(f"Stencil implementation: {stencil_name}")
print(f"Stencil points: {nb_stencil_pts}")
print(f"CG iterations (hessp calls): {nb_hessp_calls}")
print(f"Total time: {elapsed_time:.4f} seconds")

# Memory throughput estimate for the convolution operation:l
# - Read: nb_stencil_pts values per grid point (stencil neighborhood)
# - Write: 1 value per grid point
# Each value is 8 bytes (double precision)
bytes_per_hessp = nb_grid_pts_total * (nb_stencil_pts + 1) * 8  # bytes
total_bytes = nb_hessp_calls * bytes_per_hessp
memory_throughput = total_bytes / elapsed_time

print("\nMemory throughput (estimated):")
print(f"  Bytes per hessp call: {bytes_per_hessp / 1e6:.2f} MB")
print(f"  Total bytes transferred: {total_bytes / 1e9:.2f} GB")
print(f"  Throughput: {memory_throughput / 1e9:.2f} GB/s")

# FLOPS estimate for the convolution operation:
# - nb_stencil_pts multiplications and nb_stencil_pts-1 additions per grid point
# - Plus 1 division for scaling (counted as 1 FLOP)
# Total: 2 * nb_stencil_pts FLOPs per grid point (approx)
flops_per_hessp = nb_grid_pts_total * (2 * nb_stencil_pts)
total_flops = nb_hessp_calls * flops_per_hessp
flops_rate = total_flops / elapsed_time

print("\nFLOPS (estimated for convolution only):")
print(f"  FLOPs per hessp call: {flops_per_hessp / 1e6:.2f} MFLOP")
print(f"  Total FLOPs: {total_flops / 1e9:.2f} GFLOP")
print(f"  FLOP rate: {flops_rate / 1e9:.2f} GFLOP/s")

# Arithmetic intensity (FLOPs per byte)
arithmetic_intensity = total_flops / total_bytes
print(f"\nArithmetic intensity: {arithmetic_intensity:.3f} FLOP/byte")
print(f"{'='*60}")

# Print hierarchical timing breakdown
timer.print_summary()

if args.plot:
    if dim == 3:
        print("Warning: Plotting not supported for 3D grids")
    elif plt is None:
        print("Warning: matplotlib not available, cannot show plot")
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(rhs.p)
        ax1.set_title("RHS")
        ax2.imshow(solution.p)
        ax2.set_title("Solution")
        plt.tight_layout()
        plt.show()

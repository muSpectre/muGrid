# Solver for the Poisson equation

import argparse
import json

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

parser.add_argument(
    "-q", "--quiet",
    action="store_true",
    help="Suppress per-iteration output (default: off)"
)

parser.add_argument(
    "--json",
    action="store_true",
    help="Output results in JSON format (implies --quiet)"
)

parser.add_argument(
    "--papi",
    action="store_true",
    help="Use PAPI hardware counters for performance measurement (requires pypapi)"
)

args = parser.parse_args()

# JSON implies quiet mode
if args.json:
    args.quiet = True

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

# Create fields using the decomposition's method API
rhs = decomposition.real_field("rhs")
solution = decomposition.real_field("solution")

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

# Create global timer for hierarchical timing
# PAPI is only available on host (CPU), not on device (GPU)
use_papi = args.papi and args.memory == _memory_locations["host"]
if args.papi and args.memory != _memory_locations["host"]:
    if not args.quiet:
        print("Warning: PAPI not available for device memory (GPU). Using estimates only.")
timer = muGrid.Timer(use_papi=use_papi)


def callback(it, x, r, p):
    """
    Callback function to print the current solution, residual, and search direction.
    """
    if not args.quiet:
        print(f"{it:5} {arr.dot(r.ravel(), r.ravel()):.5}")


def hessp(x, Ax):
    """
    Function to compute the product of the Hessian matrix with a vector.
    The Hessian is represented by the convolution operator.
    The scale factor (grid spacing and sign) is already folded into the operator.
    """
    with timer("hessp"):
        with timer("communicate_ghosts"):
            decomposition.communicate_ghosts(x)
        with timer("apply"):
            laplace.apply(x, Ax)
    return Ax


converged = False
with timer("conjugate_gradients"):
    try:
        conjugate_gradients(
            comm,
            decomposition,
            hessp,  # linear operator
            rhs,
            solution,
            tol=1e-6,
            callback=callback,
            maxiter=args.maxiter,
        )
        converged = True
        if not args.quiet:
            print("CG converged.")
    except RuntimeError:
        if not args.quiet:
            print("CG did not converge.")

elapsed_time = timer.get_time("conjugate_gradients")

# Performance metrics calculations
# Get number of hessp calls from timer
nb_hessp_calls = timer.get_calls("hessp")

# Memory throughput estimate for the convolution operation:
# - Read: nb_stencil_pts values per grid point (stencil neighborhood)
# - Write: 1 value per grid point
# Each value is 8 bytes (double precision)
bytes_per_hessp = nb_grid_pts_total * (nb_stencil_pts + 1) * 8  # bytes
total_bytes = nb_hessp_calls * bytes_per_hessp
memory_throughput = total_bytes / elapsed_time if elapsed_time > 0 else 0

# FLOPS estimate for the convolution operation:
# - nb_stencil_pts multiplications and nb_stencil_pts-1 additions per grid point
# - Plus 1 division for scaling (counted as 1 FLOP)
# Total: 2 * nb_stencil_pts FLOPs per grid point (approx)
flops_per_hessp = nb_grid_pts_total * (2 * nb_stencil_pts)
total_flops = nb_hessp_calls * flops_per_hessp
flops_rate = total_flops / elapsed_time if elapsed_time > 0 else 0

# Arithmetic intensity (FLOPs per byte)
arithmetic_intensity = total_flops / total_bytes if total_bytes > 0 else 0

# Get apply time from timer
apply_time = timer.get_time("conjugate_gradients/hessp/apply")
apply_throughput = total_bytes / apply_time if apply_time > 0 else 0
apply_flops_rate = total_flops / apply_time if apply_time > 0 else 0

if args.json:
    # JSON output (convert numpy types to Python types for JSON serialization)
    # Timer's to_dict() includes PAPI data when available
    results = {
        "config": {
            "nb_grid_pts": [int(x) for x in args.nb_grid_pts],
            "nb_grid_pts_total": int(nb_grid_pts_total),
            "dimensions": int(dim),
            "stencil": args.stencil,
            "stencil_name": stencil_name,
            "nb_stencil_pts": int(nb_stencil_pts),
            "memory": "host" if args.memory == _memory_locations["host"] else "device",
            "maxiter": int(args.maxiter),
        },
        "results": {
            "converged": converged,
            "iterations": int(nb_hessp_calls),
            "total_time_seconds": float(elapsed_time),
            "bytes_per_iteration": int(bytes_per_hessp),
            "total_bytes": int(total_bytes),
            "memory_throughput_GBps": float(memory_throughput / 1e9),
            "flops_per_iteration_estimated": int(flops_per_hessp),
            "total_flops_estimated": int(total_flops),
            "flops_rate_GFLOPs_estimated": float(flops_rate / 1e9),
            "arithmetic_intensity": float(arithmetic_intensity),
            "apply_time_seconds": float(apply_time),
            "apply_throughput_GBps": float(apply_throughput / 1e9),
            "apply_flops_rate_GFLOPs_estimated": float(apply_flops_rate / 1e9),
        },
        "timing": timer.to_dict(),
    }
    print(json.dumps(results, indent=2))
else:
    # Text output
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

    print("\nMemory throughput (estimated):")
    print(f"  Bytes per hessp call: {bytes_per_hessp / 1e6:.2f} MB")
    print(f"  Total bytes transferred: {total_bytes / 1e9:.2f} GB")
    print(f"  Throughput: {memory_throughput / 1e9:.2f} GB/s")

    print("\nFLOPS (estimated for convolution only):")
    print(f"  FLOPs per hessp call: {flops_per_hessp / 1e6:.2f} MFLOP")
    print(f"  Total FLOPs: {total_flops / 1e9:.2f} GFLOP")
    print(f"  FLOP rate: {flops_rate / 1e9:.2f} GFLOP/s")

    print(f"\nArithmetic intensity: {arithmetic_intensity:.3f} FLOP/byte")
    print(f"{'='*60}")

    # Print hierarchical timing breakdown (includes PAPI data when enabled)
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

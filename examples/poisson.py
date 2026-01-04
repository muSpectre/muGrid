# Solver for the Poisson equation

import argparse
import json

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
import numpy as np

import muGrid
from muGrid import parprint
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

parser.add_argument(
    "-n",
    "--nb-grid-pts",
    default=[32, 32],
    type=lambda s: [int(x) for x in s.split(",")],
    help="Grid points as nx,ny or nx,ny,nz (default: 32,32)",
)

_devices = {
    "cpu": muGrid.Device.cpu(),
    "gpu": muGrid.Device.gpu(),  # Auto-detect CUDA or ROCm
}

parser.add_argument(
    "-d",
    "--device",
    choices=_devices,
    default="cpu",
    help="Device for computation: 'cpu' or 'gpu' (auto-detect CUDA/ROCm) "
    "(default: cpu)",
)

parser.add_argument(
    "-i",
    "--maxiter",
    type=int,
    default=1000,
    help="Maximum number of CG iterations (default: 1000)",
)

parser.add_argument(
    "-p",
    "--plot",
    action="store_true",
    help="Show plot of RHS and solution (default: off)",
)

parser.add_argument(
    "-k",
    "--kernel",
    choices=["generic", "hardcoded"],
    default="hardcoded",
    help="Kernel implementation: 'generic' (sparse convolution) or "
    "'hardcoded' (optimized Laplace operator) (default: hardcoded)",
)

parser.add_argument(
    "-q",
    "--quiet",
    action="store_true",
    help="Suppress per-iteration output (default: off)",
)

parser.add_argument(
    "--json",
    action="store_true",
    help="Output results in JSON format (implies --quiet)",
)

args = parser.parse_args()

# JSON implies quiet mode
if args.json:
    args.quiet = True

if args.device == "cpu":
    import numpy as arr
else:
    import cupy as arr

device = _devices[args.device]

dim = len(args.nb_grid_pts)
if dim not in (2, 3):
    raise ValueError("Only 2D and 3D grids are supported")

s = suggest_subdivisions(dim, comm.size)

# Set up ghost layers for the stencil (1 layer in each direction)
left_ghosts = (1,) * dim
right_ghosts = (1,) * dim

decomposition = muGrid.CartesianDecomposition(
    comm, args.nb_grid_pts, s, left_ghosts, right_ghosts, device=device
)
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
if args.kernel == "generic":
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
    laplace = muGrid.GenericLinearOperator(stencil_offset, stencil)
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
    rhs.p[...] = arr.asarray(
        (1 + np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y) * np.cos(2 * np.pi * z))
        ** 10
    )
rhs.p[...] -= arr.mean(rhs.p)

# Performance counters
nb_grid_pts_total = np.prod(args.nb_grid_pts)

# Create global timer for hierarchical timing
timer = muGrid.Timer()


def callback(iteration, state):
    """
    Callback function to print the iteration and squared residual norm.
    """
    if not args.quiet:
        parprint(f"{iteration:5} {state['rr']:.5}", comm=comm)


def hessp(x, Ax):
    """
    Hessian-vector product function for CG solver.

    This function computes the product of the Hessian matrix with a vector.
    """
    with timer("communicate_ghosts"):
        decomposition.communicate_ghosts(x)
    with timer("apply"):
        laplace.apply(x, Ax)


converged = False
with timer("conjugate_gradients"):
    try:
        conjugate_gradients(
            comm,
            decomposition,
            rhs,
            solution,
            hessp=hessp,
            tol=1e-6,
            callback=callback,
            maxiter=args.maxiter,
            timer=timer,
        )
        converged = True
        if not args.quiet:
            parprint("CG converged.", comm=comm)
    except RuntimeError:
        if not args.quiet:
            parprint("CG did not converge.", comm=comm)

elapsed_time = timer.get_time("conjugate_gradients")

# Performance metrics calculations
# Get number of hessp calls from timer (= number of CG iterations)
nb_iterations = timer.get_calls("conjugate_gradients/iteration/hessp")

# Lattice updates per second (LUPS)
# One "lattice update" = one grid point processed in one CG iteration
total_lattice_updates = nb_grid_pts_total * nb_iterations
lups = total_lattice_updates / elapsed_time if elapsed_time > 0 else 0

# Memory and FLOPS estimates per CG iteration:
# Each value is 8 bytes (double precision)
#
# Per CG iteration (excluding ghost communication):
#   hessp (apply):         read nb_stencil_pts, write 1, FLOPs 2*nb_stencil_pts
#   dot_pAp:               read 2 (p, Ap),       FLOPs 2 (mul + add)
#   update_x (axpy):       read 2, write 1,      FLOPs 2
#   update_r (axpy_norm_sq): read 2, write 1,    FLOPs 4 (fused axpy + norm)
#   update_p (axpby):      read 2, write 1,      FLOPs 2 (mul + mul + add)
#
# The fused axpy_norm_sq saves 1 read compared to separate axpy + norm_sq,
# because the norm is computed during the write pass without re-reading y.
#
# Total reads:  nb_stencil_pts + 2 + 2 + 2 + 2 = nb_stencil_pts + 8
# Total writes: 1 + 0 + 1 + 1 + 1 = 4
# Total FLOPs:  2*nb_stencil_pts + 2 + 2 + 4 + 2 = 2*nb_stencil_pts + 10

reads_per_iteration = nb_stencil_pts + 8  # values read per grid point
writes_per_iteration = 4  # values written per grid point
flops_per_iteration = 2 * nb_stencil_pts + 10  # FLOPs per grid point

bytes_per_iteration = (
    nb_grid_pts_total * (reads_per_iteration + writes_per_iteration) * 8
)
total_bytes = nb_iterations * bytes_per_iteration
memory_throughput = total_bytes / elapsed_time if elapsed_time > 0 else 0

flops_per_cg_iteration = nb_grid_pts_total * flops_per_iteration
total_flops = nb_iterations * flops_per_cg_iteration
flops_rate = total_flops / elapsed_time if elapsed_time > 0 else 0

# Arithmetic intensity (FLOPs per byte)
arithmetic_intensity = flops_per_iteration / (
    (reads_per_iteration + writes_per_iteration) * 8
)

# Breakdown: hessp (apply) only
bytes_per_hessp = nb_grid_pts_total * (nb_stencil_pts + 1) * 8
flops_per_hessp = nb_grid_pts_total * (2 * nb_stencil_pts)
apply_time = timer.get_time("conjugate_gradients/iteration/hessp/apply")
apply_lups = total_lattice_updates / apply_time if apply_time > 0 else 0
apply_throughput = (
    (nb_iterations * bytes_per_hessp) / apply_time if apply_time > 0 else 0
)
apply_flops_rate = (
    (nb_iterations * flops_per_hessp) / apply_time if apply_time > 0 else 0
)

if args.json:
    # JSON output (convert numpy types to Python types for JSON serialization)
    # Timer's to_dict() includes PAPI data when available
    results = {
        "config": {
            "nb_grid_pts": [int(x) for x in args.nb_grid_pts],
            "nb_grid_pts_total": int(nb_grid_pts_total),
            "dimensions": int(dim),
            "kernel": args.kernel,
            "stencil_name": stencil_name,
            "nb_stencil_pts": int(nb_stencil_pts),
            "device": device.device_string,
            "maxiter": int(args.maxiter),
        },
        "results": {
            "converged": converged,
            "iterations": int(nb_iterations),
            "total_time_seconds": float(elapsed_time),
            "total_lattice_updates": int(total_lattice_updates),
            "MLUPS": float(lups / 1e6),
            "GLUPS": float(lups / 1e9),
            "reads_per_grid_point": int(reads_per_iteration),
            "writes_per_grid_point": int(writes_per_iteration),
            "bytes_per_iteration": int(bytes_per_iteration),
            "total_bytes": int(total_bytes),
            "memory_throughput_GBps": float(memory_throughput / 1e9),
            "flops_per_grid_point": int(flops_per_iteration),
            "flops_per_iteration": int(flops_per_cg_iteration),
            "total_flops": int(total_flops),
            "flops_rate_GFLOPs": float(flops_rate / 1e9),
            "arithmetic_intensity": float(arithmetic_intensity),
            "apply_time_seconds": float(apply_time),
            "apply_MLUPS": float(apply_lups / 1e6),
            "apply_throughput_GBps": float(apply_throughput / 1e9),
            "apply_flops_rate_GFLOPs": float(apply_flops_rate / 1e9),
        },
        "timing": timer.to_dict(),
    }
    parprint(json.dumps(results, indent=2), comm=comm)
else:
    # Text output
    parprint(f"\n{'='*60}", comm=comm)
    parprint("Performance Summary", comm=comm)
    parprint(f"{'='*60}", comm=comm)
    parprint(
        f"Grid size: {' x '.join(map(str, args.nb_grid_pts))} = "
        f"{nb_grid_pts_total:,} points",
        comm=comm,
    )
    parprint(f"Dimensions: {dim}D", comm=comm)
    parprint(f"Device: {device.device_string}", comm=comm)
    parprint(f"Stencil implementation: {stencil_name}", comm=comm)
    parprint(f"Stencil points: {nb_stencil_pts}", comm=comm)
    parprint(f"CG iterations: {nb_iterations}", comm=comm)
    parprint(f"Total time: {elapsed_time:.4f} seconds", comm=comm)

    parprint("\nLattice updates per second:", comm=comm)
    parprint(f"  Total lattice updates: {total_lattice_updates:,}", comm=comm)
    parprint(f"  LUPS: {lups / 1e6:.2f} MLUPS ({lups / 1e9:.4f} GLUPS)", comm=comm)

    parprint("\nMemory traffic per CG iteration (estimated):", comm=comm)
    parprint(
        f"  Per grid point: {reads_per_iteration} reads + "
        f"{writes_per_iteration} writes "
        f"= {(reads_per_iteration + writes_per_iteration) * 8} bytes",
        comm=comm,
    )
    parprint(f"    hessp:    {nb_stencil_pts} reads, 1 write", comm=comm)
    parprint("    dot_pAp:  2 reads", comm=comm)
    parprint("    update_x: 2 reads, 1 write", comm=comm)
    parprint("    update_r: 2 reads, 1 write (fused axpy_norm_sq)", comm=comm)
    parprint("    update_p: 2 reads, 1 write", comm=comm)
    parprint(f"  Per iteration: {bytes_per_iteration / 1e6:.2f} MB", comm=comm)
    parprint(f"  Total: {total_bytes / 1e9:.2f} GB", comm=comm)
    parprint(f"  Throughput: {memory_throughput / 1e9:.2f} GB/s", comm=comm)

    parprint("\nFLOPs per CG iteration (estimated):", comm=comm)
    parprint(f"  Per grid point: {flops_per_iteration} FLOPs", comm=comm)
    parprint(f"    hessp:    {2 * nb_stencil_pts} FLOPs", comm=comm)
    parprint("    dot_pAp:  2 FLOPs", comm=comm)
    parprint("    update_x: 2 FLOPs", comm=comm)
    parprint("    update_r: 4 FLOPs (fused axpy_norm_sq)", comm=comm)
    parprint("    update_p: 2 FLOPs", comm=comm)
    parprint(f"  Per iteration: {flops_per_cg_iteration / 1e6:.2f} MFLOP", comm=comm)
    parprint(f"  Total: {total_flops / 1e9:.2f} GFLOP", comm=comm)
    parprint(f"  FLOP rate: {flops_rate / 1e9:.2f} GFLOP/s", comm=comm)

    parprint(f"\nArithmetic intensity: {arithmetic_intensity:.3f} FLOP/byte", comm=comm)
    parprint(f"{'='*60}", comm=comm)

    # Print hierarchical timing breakdown (includes PAPI data when enabled)
    timer.print_summary()

if args.plot:
    if dim == 3:
        parprint("Warning: Plotting not supported for 3D grids", comm=comm)
    elif plt is None:
        parprint("Warning: matplotlib not available, cannot show plot", comm=comm)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(rhs.p)
        ax1.set_title("RHS")
        ax2.imshow(solution.p)
        ax2.set_title("Solution")
        plt.tight_layout()
        plt.show()

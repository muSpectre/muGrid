"""
Micromechanical homogenization example.

This example computes the effective elastic properties of a periodic volume
element with material microstructure using the linear FEM gradient operator.

The governing equation is the Lippmann-Schwinger equation for linear elasticity:
    K u = -B^T : C : E_macro
where:
    K = B^T : C : B is the stiffness operator
    B is the FEM gradient operator (strain from displacement)
    B^T is the divergence operator (equilibrium)
    C is the material stiffness tensor at each quadrature point
    E_macro is the applied macroscopic strain

After solving, the homogenized stress is computed as:
    Σ = (1/|Ω|) ∫ C : (E_macro + ε(u)) dΩ

For a 2D problem, we compute all 3 independent stiffness components (xx, yy, xy).
For a 3D problem, we compute all 6 independent stiffness components (xx, yy, zz, yz, xz, xy).
"""

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


# Voigt notation mappings
# 2D: [xx, yy, xy] -> indices 0, 1, 2
# 3D: [xx, yy, zz, yz, xz, xy] -> indices 0, 1, 2, 3, 4, 5

def voigt_index_2d(i, j):
    """Convert 2D tensor indices to Voigt notation index."""
    if i == j:
        return i
    return 2


def voigt_index_3d(i, j):
    """Convert 3D tensor indices to Voigt notation index.

    Voigt ordering: xx=0, yy=1, zz=2, yz=3, xz=4, xy=5
    """
    if i == j:
        return i
    # Off-diagonal: map (i,j) to Voigt index
    # (1,2) or (2,1) -> yz -> 3
    # (0,2) or (2,0) -> xz -> 4
    # (0,1) or (1,0) -> xy -> 5
    pair = tuple(sorted([i, j]))
    mapping = {(1, 2): 3, (0, 2): 4, (0, 1): 5}
    return mapping[pair]


def voigt_index(dim, i, j):
    """Convert tensor indices to Voigt notation index for given dimension."""
    if dim == 2:
        return voigt_index_2d(i, j)
    else:
        return voigt_index_3d(i, j)


def isotropic_stiffness_2d(E, nu):
    """
    Create 2D plane strain isotropic stiffness tensor in Voigt notation.
    Returns C[3, 3] where [xx, yy, xy] ordering is used.
    """
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    C = np.zeros((3, 3))
    C[0, 0] = lam + 2 * mu  # C_xxxx
    C[1, 1] = lam + 2 * mu  # C_yyyy
    C[2, 2] = mu  # C_xyxy
    C[0, 1] = C[1, 0] = lam  # C_xxyy
    return C


def isotropic_stiffness_3d(E, nu):
    """
    Create 3D isotropic stiffness tensor in Voigt notation.
    Returns C[6, 6] where [xx, yy, zz, yz, xz, xy] ordering is used.
    """
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    C = np.zeros((6, 6))
    # Diagonal: normal components
    C[0, 0] = lam + 2 * mu  # C_xxxx
    C[1, 1] = lam + 2 * mu  # C_yyyy
    C[2, 2] = lam + 2 * mu  # C_zzzz
    # Diagonal: shear components
    C[3, 3] = mu  # C_yzyz
    C[4, 4] = mu  # C_xzxz
    C[5, 5] = mu  # C_xyxy
    # Off-diagonal: coupling between normal components
    C[0, 1] = C[1, 0] = lam  # C_xxyy
    C[0, 2] = C[2, 0] = lam  # C_xxzz
    C[1, 2] = C[2, 1] = lam  # C_yyzz
    return C


def isotropic_stiffness(dim, E, nu):
    """Create isotropic stiffness tensor for given dimension."""
    if dim == 2:
        return isotropic_stiffness_2d(E, nu)
    else:
        return isotropic_stiffness_3d(E, nu)


def create_microstructure(coords, inclusion_type="single", inclusion_radius=0.25):
    """
    Create a phase field (0=matrix, 1=inclusion) based on coordinates.
    """
    dim = len(coords)

    if inclusion_type == "single":
        center = [0.5] * dim
        r2 = sum((c - center[i]) ** 2 for i, c in enumerate(coords))
        phase = np.where(r2 < inclusion_radius**2, 1.0, 0.0)
    elif inclusion_type == "checkerboard":
        if dim == 2:
            x, y = coords
            phase = ((x > 0.5).astype(float) + (y > 0.5).astype(float)) % 2
        else:
            x, y, z = coords
            phase = (
                (x > 0.5).astype(float)
                + (y > 0.5).astype(float)
                + (z > 0.5).astype(float)
            ) % 2
    elif inclusion_type == "random":
        np.random.seed(42)
        shape = coords[0].shape
        phase = (np.random.random(shape) < 0.3).astype(float)
    else:
        raise ValueError(f"Unknown inclusion type: {inclusion_type}")

    return phase


parser = argparse.ArgumentParser(
    prog="Homogenization",
    description="Compute effective elastic properties via FEM-based homogenization",
)

parser.add_argument(
    "-n",
    "--nb-grid-pts",
    default=[16, 16],
    type=lambda s: [int(x) for x in s.split(",")],
    help="Grid points as nx,ny or nx,ny,nz (default: 16,16)",
)

_memory_locations = {
    "host": muGrid.GlobalFieldCollection.MemoryLocation.Host,
    "device": muGrid.GlobalFieldCollection.MemoryLocation.Device,
}

parser.add_argument(
    "-m",
    "--memory",
    choices=_memory_locations,
    default="host",
    help="Memory space for allocation (default: host)",
)

parser.add_argument(
    "-i",
    "--maxiter",
    type=int,
    default=500,
    help="Maximum number of CG iterations (default: 500)",
)

parser.add_argument(
    "-t",
    "--tol",
    type=float,
    default=1e-6,
    help="CG convergence tolerance (default: 1e-6)",
)

parser.add_argument(
    "--E-matrix",
    type=float,
    default=1.0,
    help="Young's modulus of matrix (default: 1.0)",
)

parser.add_argument(
    "--E-inclusion",
    type=float,
    default=10.0,
    help="Young's modulus of inclusion (default: 10.0)",
)

parser.add_argument(
    "--nu",
    type=float,
    default=0.3,
    help="Poisson's ratio (same for both phases, default: 0.3)",
)

parser.add_argument(
    "--inclusion-type",
    choices=["single", "checkerboard", "random"],
    default="single",
    help="Type of inclusion pattern (default: single)",
)

parser.add_argument(
    "--inclusion-radius",
    type=float,
    default=0.25,
    help="Radius of inclusion for 'single' type (default: 0.25)",
)

parser.add_argument(
    "-p",
    "--plot",
    action="store_true",
    help="Show plot of microstructure and stress fields (default: off, 2D only)",
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

parser.add_argument(
    "--papi",
    action="store_true",
    help="Use PAPI hardware counters for performance measurement (requires pypapi)",
)

args = parser.parse_args()

# JSON implies quiet mode
if args.json:
    args.quiet = True

# Select array library based on memory location
if args.memory == "host":
    import numpy as arr
else:
    import cupy as arr

memory_location = _memory_locations[args.memory]

# Parse grid dimensions
dim = len(args.nb_grid_pts)
if dim not in (2, 3):
    raise ValueError("Only 2D and 3D grids are supported")

# Physical domain size (unit cell)
domain_size = np.ones(dim)
grid_spacing = domain_size / np.array(args.nb_grid_pts)

# Number of Voigt components
nb_voigt = 3 if dim == 2 else 6  # 2D: xx, yy, xy; 3D: xx, yy, zz, yz, xz, xy

# Voigt component labels for output
if dim == 2:
    voigt_labels = ["xx", "yy", "xy"]
else:
    voigt_labels = ["xx", "yy", "zz", "yz", "xz", "xy"]

# Create the FEM gradient operator first to get accurate quadrature info
gradient_op = muGrid.FEMGradientOperator(dim, list(grid_spacing))

# Number of quadrature points and nodal points from the operator
nb_quad = gradient_op.nb_quad_pts
nb_nodes = gradient_op.nb_nodal_pts

# Quadrature weights (area/volume of each element)
quad_weights = np.array(gradient_op.get_quadrature_weights())

# Create Cartesian decomposition for ghost handling.
# The FEM gradient kernel requires ghosts for accessing neighbor nodes.
# CartesianDecomposition handles ghost communication for both serial and MPI parallel execution.
#
# We use ghosts on BOTH sides (left and right). This approach means:
# - Ghost elements on the left boundary provide contributions to interior nodes directly
# - Ghost elements on the right boundary provide contributions to interior nodes directly
# - No ghost reduction is needed since interior nodes receive all contributions directly
# - The trade-off is slightly more memory and computation in ghost regions
decomposition = muGrid.CartesianDecomposition(
    comm,
    args.nb_grid_pts,
    nb_subdivisions=(1,) * dim,  # Serial execution: single subdivision
    nb_ghosts_left=(1,) * dim,
    nb_ghosts_right=(1,) * dim,
    nb_sub_pts={"quad": nb_quad},  # Use actual quad count from operator
    memory_location=memory_location,
)

# Get local grid dimensions
grid_shape = tuple(args.nb_grid_pts)

# Get coordinates for microstructure generation
coord_arrays = []
for d in range(dim):
    coord_arrays.append(
        np.linspace(0, 1, args.nb_grid_pts[d], endpoint=False)
        + 0.5 / args.nb_grid_pts[d]
    )
coords = np.meshgrid(*coord_arrays, indexing="ij")

# Create the microstructure (phase field)
phase = create_microstructure(coords, args.inclusion_type, args.inclusion_radius)

# Create the material stiffness tensor at each pixel
C_matrix = isotropic_stiffness(dim, args.E_matrix, args.nu)
C_inclusion = isotropic_stiffness(dim, args.E_inclusion, args.nu)

if comm.rank == 0 and not args.quiet:
    print(f"Grid size: {args.nb_grid_pts}")
    print(f"Dimensions: {dim}D")
    print(f"Grid spacing: {grid_spacing}")
    print(f"Memory location: {args.memory}")
    print(f"Number of quadrature points per pixel: {nb_quad}")
    print(f"Number of nodal points per pixel: {nb_nodes}")
    print(f"Quadrature weights: {quad_weights}")
    print(f"Inclusion volume fraction: {np.mean(phase):.4f}")
    print()

# Create muGrid fields for gradient operations.
# The FEM gradient operator now supports multi-component fields:
# - Input: vector field with (dim,) components -> shape (dim, sub_pts, pixels)
# - Output: tensor field with (dim, dim) components at quad pts
#   (dim input components × dim operators)
#
# Tensor fields for gradient/stress at quadrature points:
grad_u = decomposition.real_field("grad_u", (dim, dim), "quad")  # displacement gradient tensor
stress_field = decomposition.real_field("stress_field", (dim, dim), "quad")  # stress tensor

# Vector fields for CG solver (displacement and force vectors)
u_field = decomposition.real_field("u_field", (dim,))
f_field = decomposition.real_field("f_field", (dim,))
rhs_field = decomposition.real_field("rhs_field", (dim,))

# Material stiffness at each quadrature point [voigt, voigt, quad, *grid_shape]
# Create on host first, then convert to device array if needed
C_field_shape = (nb_voigt, nb_voigt, nb_quad) + grid_shape
C_field_np = np.zeros(C_field_shape)
for q in range(nb_quad):
    for i in range(nb_voigt):
        for j in range(nb_voigt):
            C_field_np[i, j, q] = C_matrix[i, j] * (1 - phase) + C_inclusion[i, j] * phase

# Convert to device array if using GPU
C_field = arr.asarray(C_field_np)

# Create global timer for hierarchical timing
# PAPI is only available on host (CPU), not on device (GPU)
use_papi = args.papi and args.memory == "host"
if args.papi and args.memory != "host":
    if comm.rank == 0 and not args.quiet:
        print("Warning: PAPI not available for device memory (GPU). Using estimates only.")
timer = muGrid.Timer(use_papi=use_papi)

# Performance counters
nb_grid_pts_total = np.prod(args.nb_grid_pts)


def compute_strain(u_vec, strain_out):
    """
    Compute strain from displacement field.

    The gradient operator directly handles vector input:
    - Input: vector field u with (dim,) components
    - Output: tensor field ∂u_i/∂x_j with (dim, dim) components
    - Strain: ε_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)

    Parameters
    ----------
    u_vec : Field
        Vector displacement field with shape (dim, nb_nodes, pixels)
    strain_out : ndarray
        Output strain array with shape (dim, dim, quad, *grid_shape)
    """
    with timer("communicate_ghosts"):
        # Fill ghost values for periodic BC
        decomposition.communicate_ghosts(u_vec)

    with timer("gradient_apply"):
        # Compute gradient tensor: grad_u.s[i, j, ...] = ∂u_i/∂x_j
        gradient_op.apply(u_vec, grad_u)

    # Compute symmetric strain: ε_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
    grad = grad_u.s
    strain_out[...] = 0.5 * (grad + grad.swapaxes(0, 1))


def strain_to_voigt(strain):
    """
    Convert strain tensor to Voigt notation.

    Parameters
    ----------
    strain : ndarray
        Strain tensor with shape (dim, dim, quad, *grid_shape)

    Returns
    -------
    eps_voigt : ndarray
        Strain in Voigt notation with shape (nb_voigt, quad, *grid_shape)
    """
    voigt_shape = (nb_voigt, nb_quad) + grid_shape
    eps_voigt = arr.zeros(voigt_shape)

    if dim == 2:
        eps_voigt[0, ...] = strain[0, 0, ...]  # exx
        eps_voigt[1, ...] = strain[1, 1, ...]  # eyy
        eps_voigt[2, ...] = 2 * strain[0, 1, ...]  # 2*exy (engineering shear)
    else:  # dim == 3
        eps_voigt[0, ...] = strain[0, 0, ...]  # exx
        eps_voigt[1, ...] = strain[1, 1, ...]  # eyy
        eps_voigt[2, ...] = strain[2, 2, ...]  # ezz
        eps_voigt[3, ...] = 2 * strain[1, 2, ...]  # 2*eyz (engineering shear)
        eps_voigt[4, ...] = 2 * strain[0, 2, ...]  # 2*exz (engineering shear)
        eps_voigt[5, ...] = 2 * strain[0, 1, ...]  # 2*exy (engineering shear)

    return eps_voigt


def voigt_to_stress(sig_voigt, stress):
    """
    Convert stress from Voigt notation to tensor.

    Parameters
    ----------
    sig_voigt : ndarray
        Stress in Voigt notation with shape (nb_voigt, quad, *grid_shape)
    stress : ndarray
        Output stress tensor with shape (dim, dim, quad, *grid_shape)
    """
    if dim == 2:
        stress[0, 0, ...] = sig_voigt[0, ...]  # sxx
        stress[1, 1, ...] = sig_voigt[1, ...]  # syy
        stress[0, 1, ...] = sig_voigt[2, ...]  # sxy
        stress[1, 0, ...] = sig_voigt[2, ...]  # syx
    else:  # dim == 3
        stress[0, 0, ...] = sig_voigt[0, ...]  # sxx
        stress[1, 1, ...] = sig_voigt[1, ...]  # syy
        stress[2, 2, ...] = sig_voigt[2, ...]  # szz
        stress[1, 2, ...] = sig_voigt[3, ...]  # syz
        stress[2, 1, ...] = sig_voigt[3, ...]  # szy
        stress[0, 2, ...] = sig_voigt[4, ...]  # sxz
        stress[2, 0, ...] = sig_voigt[4, ...]  # szx
        stress[0, 1, ...] = sig_voigt[5, ...]  # sxy
        stress[1, 0, ...] = sig_voigt[5, ...]  # syx


def compute_stress(strain, stress, C):
    """
    Compute stress from strain using Voigt notation.

    Parameters
    ----------
    strain : ndarray
        Strain tensor with shape (dim, dim, quad, *grid_shape)
    stress : ndarray
        Output stress tensor with shape (dim, dim, quad, *grid_shape)
    C : ndarray
        Material stiffness in Voigt notation (nb_voigt, nb_voigt, quad, *grid_shape)
    """
    with timer("compute_stress"):
        # Convert strain to Voigt notation
        eps_voigt = strain_to_voigt(strain)

        # Compute stress in Voigt: sig = C @ eps
        if args.memory == "host":
            sig_voigt = np.einsum("ijq...,jq...->iq...", C, eps_voigt)
        else:
            # CuPy einsum
            sig_voigt = arr.einsum("ijq...,jq...->iq...", C, eps_voigt)

        # Convert back to tensor
        voigt_to_stress(sig_voigt, stress)


def compute_divergence(stress, f_vec):
    """
    Compute divergence of stress tensor.

    The transpose of the gradient operator directly handles tensor input:
    - Input: stress tensor with (dim, dim) components
    - Output: force vector with (dim,) components
    - f_i = Σ_j ∂σ_ij/∂x_j (divergence of stress)

    With two-sided ghosts:
    1. We fill ghost pixel stresses via communicate_ghosts (periodic copies)
    2. The transpose reads stress from ALL pixels (interior + ghost)
    3. Ghost elements contribute directly to interior nodes
    4. No ghost reduction needed

    Parameters
    ----------
    stress : ndarray
        Stress tensor with shape (dim, dim, quad, *grid_shape)
    f_vec : Field
        Output vector force field with shape (dim, nb_nodes, *grid_shape)
    """
    # Copy stress to field and fill ghost values
    stress_field.s[...] = stress
    with timer("communicate_ghosts"):
        decomposition.communicate_ghosts(stress_field)

    with timer("gradient_transpose"):
        # Apply transpose (divergence) with quadrature weights
        # The transpose sums over operators (j direction) for each input component (i)
        f_vec.pg[...] = 0.0
        gradient_op.transpose(stress_field, f_vec, list(quad_weights))


# Temporary arrays for strain and stress [dim, dim, quad, *grid_shape]
strain_shape = (dim, dim, nb_quad) + grid_shape
strain_arr = arr.zeros(strain_shape)
stress_arr = arr.zeros(strain_shape)


def apply_stiffness(u_in, f_out):
    """
    Apply K = B^T C B to displacement vector.

    Parameters
    ----------
    u_in : Field
        Input displacement field with (dim, nb_nodes) components
    f_out : Field
        Output force field with (dim, nb_nodes) components (modified in place)
    """
    with timer("apply_stiffness"):
        # Compute strain eps = B * u
        compute_strain(u_in, strain_arr)

        # Compute stress sig = C : eps
        compute_stress(strain_arr, stress_arr, C_field)

        # Compute force f = B^T * sig
        compute_divergence(stress_arr, f_out)


def compute_rhs(E_macro, rhs_out):
    """
    Compute RHS: f = -B^T C E_macro

    Parameters
    ----------
    E_macro : ndarray
        Macroscopic strain tensor [dim, dim]
    rhs_out : Field
        Output field for RHS (modified in place)
    """
    # Create uniform strain field from macroscopic strain
    eps_macro = arr.zeros(strain_shape)
    for i in range(dim):
        for j in range(dim):
            eps_macro[i, j, ...] = E_macro[i, j]

    # Compute stress from macroscopic strain
    sig_macro = arr.zeros_like(eps_macro)
    compute_stress(eps_macro, sig_macro, C_field)

    # Compute divergence (with negative sign for RHS)
    compute_divergence(sig_macro, rhs_out)
    rhs_out.s[...] *= -1.0


# Storage for homogenized stiffness
C_eff = np.zeros((nb_voigt, nb_voigt))

# Macroscopic strain cases
if dim == 2:
    strain_cases = [
        (0, 0),  # xx
        (1, 1),  # yy
        (0, 1),  # xy
    ]
else:  # dim == 3
    strain_cases = [
        (0, 0),  # xx
        (1, 1),  # yy
        (2, 2),  # zz
        (1, 2),  # yz
        (0, 2),  # xz
        (0, 1),  # xy
    ]

if comm.rank == 0 and not args.quiet:
    print("=" * 60)
    print("Computing homogenized stiffness tensor")
    print("=" * 60)

total_iterations = 0

with timer("total_solve"):
    for case_idx, (i, j) in enumerate(strain_cases):
        # Create unit macroscopic strain
        E_macro = arr.zeros((dim, dim))
        E_macro[i, j] = 1.0
        if i != j:
            E_macro[j, i] = 1.0  # Symmetric

        voigt_col = voigt_index(dim, i, j)

        if comm.rank == 0 and not args.quiet:
            print(f"\nCase {case_idx + 1}: E_macro[{i},{j}] = 1")

        # Compute RHS into rhs_field
        compute_rhs(E_macro, rhs_field)

        # Initialize displacement to zero
        u_field.s[...] = 0.0

        # CG callback
        iteration_count = [0]  # Use list to allow modification in nested function

        def callback(it, x, r, p):
            iteration_count[0] = it
            if not args.quiet:
                r_flat = r.ravel()
                res_norm = float(arr.sqrt(comm.sum(arr.dot(r_flat, r_flat))))
                if comm.rank == 0 and it % 10 == 0:
                    print(f"  CG iteration {it}: |r| = {res_norm:.6e}")

        # Solve K u = f using conjugate_gradients from Solvers.py
        with timer(f"cg_case_{case_idx}"):
            try:
                conjugate_gradients(
                    comm,
                    decomposition,
                    apply_stiffness,
                    rhs_field,
                    u_field,
                    tol=args.tol,
                    callback=callback,
                    maxiter=args.maxiter,
                )
                converged = True
            except RuntimeError as e:
                if "did not converge" in str(e):
                    converged = False
                else:
                    raise

        total_iterations += iteration_count[0]

        if comm.rank == 0 and not args.quiet:
            if converged:
                print(f"  CG converged in {iteration_count[0]} iterations")
            else:
                print(f"  CG did not converge after {args.maxiter} iterations")

        # Compute strain from solution
        compute_strain(u_field, strain_arr)

        # Add macroscopic strain to get total strain
        for ii in range(dim):
            for jj in range(dim):
                strain_arr[ii, jj, ...] += E_macro[ii, jj]

        # Compute stress from total strain
        compute_stress(strain_arr, stress_arr, C_field)

        # Compute average stress (homogenized stress)
        # Σ_kl = (1/V) ∫ σ_kl dV = (1/V) Σ_q w_q * σ_kl(q)
        sig_avg = np.zeros((dim, dim))
        for k in range(dim):
            for L in range(dim):
                local_sum = 0.0
                for q in range(nb_quad):
                    if args.memory == "host":
                        local_sum += quad_weights[q] * np.sum(stress_arr[k, L, q, ...])
                    else:
                        local_sum += quad_weights[q] * float(arr.sum(stress_arr[k, L, q, ...]))
                sig_avg[k, L] = comm.sum(local_sum)

        # Normalize by total volume
        total_volume = np.prod(domain_size)
        sig_avg /= total_volume

        # Store in homogenized stiffness (column voigt_col)
        for k in range(dim):
            for L in range(dim):
                voigt_row = voigt_index(dim, k, L)
                # Only store unique components (upper triangle in tensor, all in Voigt)
                if k <= L:
                    C_eff[voigt_row, voigt_col] = sig_avg[k, L]

        if comm.rank == 0 and not args.quiet:
            if dim == 2:
                print(
                    f"  Average stress: xx={sig_avg[0, 0]:.6f}, "
                    f"yy={sig_avg[1, 1]:.6f}, xy={sig_avg[0, 1]:.6f}"
                )
            else:
                print(
                    f"  Average stress: xx={sig_avg[0, 0]:.6f}, yy={sig_avg[1, 1]:.6f}, "
                    f"zz={sig_avg[2, 2]:.6f}"
                )
                print(
                    f"                  yz={sig_avg[1, 2]:.6f}, xz={sig_avg[0, 2]:.6f}, "
                    f"xy={sig_avg[0, 1]:.6f}"
                )

# Get timing information
elapsed_time = timer.get_time("total_solve")
nb_stiffness_calls = timer.get_calls("apply_stiffness")
apply_stiffness_time = timer.get_time("total_solve/apply_stiffness") if nb_stiffness_calls > 0 else 0

# Performance metrics
# Memory throughput estimate:
# - FEM gradient: reads neighbor values, writes gradient
# - Stress computation: reads strain, writes stress
# - Divergence: reads stress, writes force
# Approximate: 2 * dim * dim * nb_quad + dim values per grid point, 8 bytes each
bytes_per_call = nb_grid_pts_total * (2 * dim * dim * nb_quad + dim) * 8
total_bytes = nb_stiffness_calls * bytes_per_call
memory_throughput = total_bytes / elapsed_time if elapsed_time > 0 else 0

# FLOPS estimate (rough)
# - Gradient: ~dim * dim * nb_stencil_pts * 2 FLOPs per grid point
# - Stress: ~nb_voigt * nb_voigt * nb_quad FLOPs per grid point
# - Divergence: similar to gradient
flops_per_call = nb_grid_pts_total * (dim * dim * 10 + nb_voigt * nb_voigt * nb_quad * 2)
total_flops = nb_stiffness_calls * flops_per_call
flops_rate = total_flops / elapsed_time if elapsed_time > 0 else 0

# Analytical bounds
v_f = float(np.mean(phase))  # Volume fraction of inclusion
E_m, E_i = args.E_matrix, args.E_inclusion
nu = args.nu
E_voigt = v_f * E_i + (1 - v_f) * E_m
E_reuss = 1.0 / (v_f / E_i + (1 - v_f) / E_m)
E_eff_approx = C_eff[0, 0] * (1 - nu**2)

if args.json and comm.rank == 0:
    # JSON output
    # Timer's to_dict() includes PAPI data when available
    results = {
        "config": {
            "nb_grid_pts": [int(x) for x in args.nb_grid_pts],
            "nb_grid_pts_total": int(nb_grid_pts_total),
            "dimensions": int(dim),
            "memory": args.memory,
            "maxiter": int(args.maxiter),
            "tolerance": float(args.tol),
            "E_matrix": float(args.E_matrix),
            "E_inclusion": float(args.E_inclusion),
            "nu": float(args.nu),
            "inclusion_type": args.inclusion_type,
            "inclusion_radius": float(args.inclusion_radius),
            "volume_fraction": float(v_f),
        },
        "results": {
            "total_cg_iterations": int(total_iterations),
            "stiffness_calls": int(nb_stiffness_calls),
            "total_time_seconds": float(elapsed_time),
            "apply_stiffness_time_seconds": float(apply_stiffness_time),
            "bytes_per_call": int(bytes_per_call),
            "total_bytes": int(total_bytes),
            "memory_throughput_GBps": float(memory_throughput / 1e9),
            "flops_per_call_estimated": int(flops_per_call),
            "total_flops_estimated": int(total_flops),
            "flops_rate_GFLOPs_estimated": float(flops_rate / 1e9),
            "C_eff": [[float(C_eff[i, j]) for j in range(nb_voigt)] for i in range(nb_voigt)],
            "E_effective_approx": float(E_eff_approx),
            "E_voigt_bound": float(E_voigt),
            "E_reuss_bound": float(E_reuss),
        },
        "timing": timer.to_dict(),
    }
    print(json.dumps(results, indent=2))
elif comm.rank == 0:
    # Text output
    print("\n" + "=" * 60)
    print("Homogenized stiffness tensor (Voigt notation)")
    print("=" * 60)

    # Print header
    header = "      " + "".join(f"{lbl:>10}" for lbl in voigt_labels)
    print(header)

    # Print matrix
    for i, row_label in enumerate(voigt_labels):
        row = f"{row_label:>4}  " + "".join(f"{C_eff[i, j]:10.6f}" for j in range(nb_voigt))
        print(row)

    print("\n" + "=" * 60)
    print("Comparison with analytical bounds")
    print("=" * 60)
    print(f"Volume fraction of inclusion: {v_f:.4f}")
    print(f"Voigt bound (upper): E = {E_voigt:.4f}")
    print(f"Reuss bound (lower): E = {E_reuss:.4f}")
    print(f"Effective E (approx from C_xxxx): E ≈ {E_eff_approx:.4f}")

    print("\n" + "=" * 60)
    print("Performance Summary")
    print("=" * 60)
    print(f"Grid size: {' x '.join(map(str, args.nb_grid_pts))} = {nb_grid_pts_total:,} points")
    print(f"Dimensions: {dim}D")
    print(f"Memory location: {args.memory}")
    print(f"Total CG iterations: {total_iterations}")
    print(f"Stiffness operator calls: {nb_stiffness_calls}")
    print(f"Total time: {elapsed_time:.4f} seconds")

    print("\nMemory throughput (estimated):")
    print(f"  Bytes per stiffness call: {bytes_per_call / 1e6:.2f} MB")
    print(f"  Total bytes transferred: {total_bytes / 1e9:.2f} GB")
    print(f"  Throughput: {memory_throughput / 1e9:.2f} GB/s")

    print("\nFLOPS (estimated):")
    print(f"  FLOPs per stiffness call: {flops_per_call / 1e6:.2f} MFLOP")
    print(f"  Total FLOPs: {total_flops / 1e9:.2f} GFLOP")
    print(f"  FLOP rate: {flops_rate / 1e9:.2f} GFLOP/s")
    print("=" * 60)

    # Print hierarchical timing breakdown (includes PAPI data when enabled)
    timer.print_summary()

# Optional plotting (2D only)
if args.plot and comm.rank == 0:
    if dim == 3:
        print("Warning: Plotting not supported for 3D grids")
    elif plt is None:
        print("Warning: matplotlib not available, cannot show plot")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Microstructure
        ax = axes[0]
        im = ax.imshow(phase.T, origin="lower", cmap="viridis")
        ax.set_title("Microstructure (0=matrix, 1=inclusion)")
        plt.colorbar(im, ax=ax)

        # Stress xx from last load case (convert to numpy if on device)
        ax = axes[1]
        if args.memory == "host":
            sig_xx_avg = np.mean(stress_arr[0, 0, ...], axis=0)
        else:
            sig_xx_avg = np.mean(arr.asnumpy(stress_arr[0, 0, ...]), axis=0)
        im = ax.imshow(sig_xx_avg.T, origin="lower", cmap="RdBu_r")
        ax.set_title(r"$\sigma_{xx}$ (last load case)")
        plt.colorbar(im, ax=ax)

        # Displacement magnitude from last load case
        ax = axes[2]
        if args.memory == "host":
            u_mag = np.sqrt(u_field.s[0, 0, ...] ** 2 + u_field.s[1, 0, ...] ** 2)
        else:
            u_s = arr.asnumpy(u_field.s)
            u_mag = np.sqrt(u_s[0, 0, ...] ** 2 + u_s[1, 0, ...] ** 2)
        im = ax.imshow(u_mag.T, origin="lower", cmap="viridis")
        ax.set_title("|u| (last load case)")
        plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.show()

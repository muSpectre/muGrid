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

For a 2D problem, we compute all 3 independent stiffness components (xx, yy, xy)
by applying unit strains in each direction.
"""

import argparse

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

import muGrid
from muGrid.Solvers import conjugate_gradients

try:
    from mpi4py import MPI

    comm = muGrid.Communicator(MPI.COMM_WORLD)
except ImportError:
    comm = muGrid.Communicator()


def voigt_index_2d(i, j):
    """Convert 2D tensor indices to Voigt notation index."""
    if i == j:
        return i
    return 2


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
    help="Grid points as nx,ny (default: 16,16)",
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
    help="Show plot of microstructure and stress fields (default: off)",
)

parser.add_argument(
    "-q",
    "--quiet",
    action="store_true",
    help="Suppress per-iteration output (default: off)",
)

args = parser.parse_args()

# Parse grid dimensions
dim = len(args.nb_grid_pts)
if dim != 2:
    raise ValueError("Only 2D grids are supported in this example")

# Physical domain size (unit cell)
domain_size = np.ones(dim)
grid_spacing = domain_size / np.array(args.nb_grid_pts)

# Number of Voigt components
nb_voigt = 3  # 2D: xx, yy, xy

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
    nb_sub_pts={"quad": 2},  # 2 quadrature points (triangles) per pixel
)

# Get local grid dimensions
nx, ny = args.nb_grid_pts

# Get coordinates for microstructure generation
x = np.linspace(0, 1, args.nb_grid_pts[0], endpoint=False) + 0.5 / args.nb_grid_pts[0]
y = np.linspace(0, 1, args.nb_grid_pts[1], endpoint=False) + 0.5 / args.nb_grid_pts[1]
coords = np.meshgrid(x, y, indexing="ij")

# Create the microstructure (phase field)
phase = create_microstructure(coords, args.inclusion_type, args.inclusion_radius)

# Create the material stiffness tensor at each pixel
C_matrix = isotropic_stiffness_2d(args.E_matrix, args.nu)
C_inclusion = isotropic_stiffness_2d(args.E_inclusion, args.nu)

# Create the FEM gradient operator
gradient_op = muGrid.FEMGradientOperator(dim, list(grid_spacing))

# Number of quadrature points and nodal points
nb_quad = gradient_op.nb_quad_pts  # 2 for 2D triangles
nb_nodes = gradient_op.nb_nodal_pts  # 1 for continuous FEM

# Quadrature weights (area of each triangle)
quad_weights = np.array(gradient_op.get_quadrature_weights())

if comm.rank == 0 and not args.quiet:
    print(f"Grid size: {args.nb_grid_pts}")
    print(f"Grid spacing: {grid_spacing}")
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

# Material stiffness at each quadrature point [voigt, voigt, quad, nx, ny]
C_field = np.zeros((nb_voigt, nb_voigt, nb_quad, nx, ny))
for q in range(nb_quad):
    for i in range(nb_voigt):
        for j in range(nb_voigt):
            C_field[i, j, q] = C_matrix[i, j] * (1 - phase) + C_inclusion[i, j] * phase


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
        Output strain array with shape (dim, dim, quad, pixels)
    """
    # Fill ghost values for periodic BC
    decomposition.communicate_ghosts(u_vec)

    # Compute gradient tensor: grad_u.s[i, j, ...] = ∂u_i/∂x_j
    gradient_op.apply(u_vec, grad_u)

    # Compute symmetric strain: ε_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
    grad = grad_u.s
    strain_out[...] = 0.5 * (grad + grad.swapaxes(0, 1))


def compute_stress(strain, stress, C):
    """
    Compute stress from strain using Voigt notation.

    Parameters
    ----------
    strain : ndarray
        Strain tensor with shape (dim, dim, quad, pixels)
    stress : ndarray
        Output stress tensor with shape (dim, dim, quad, pixels)
    C : ndarray
        Material stiffness in Voigt notation (nb_voigt, nb_voigt, quad, pixels)
    """
    # Convert strain to Voigt: [voigt, quad, nx, ny]
    eps_voigt = np.zeros((nb_voigt, nb_quad, nx, ny))
    eps_voigt[0, ...] = strain[0, 0, ...]  # exx
    eps_voigt[1, ...] = strain[1, 1, ...]  # eyy
    eps_voigt[2, ...] = 2 * strain[0, 1, ...]  # 2*exy (engineering shear)

    # Compute stress in Voigt: sig = C @ eps
    sig_voigt = np.einsum("ijq...,jq...->iq...", C, eps_voigt)

    # Convert back to tensor
    stress[0, 0, ...] = sig_voigt[0, ...]  # sxx
    stress[1, 1, ...] = sig_voigt[1, ...]  # syy
    stress[0, 1, ...] = sig_voigt[2, ...]  # sxy
    stress[1, 0, ...] = sig_voigt[2, ...]  # syx


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
        Stress tensor with shape (dim, dim, quad, pixels)
    f_vec : Field
        Output vector force field with shape (dim, nb_nodes, pixels)
    """
    # Copy stress to field and fill ghost values
    stress_field.s[...] = stress
    decomposition.communicate_ghosts(stress_field)

    # Apply transpose (divergence) with quadrature weights
    # The transpose sums over operators (j direction) for each input component (i)
    f_vec.pg[...] = 0.0
    gradient_op.transpose(stress_field, f_vec, list(quad_weights))


# Temporary arrays for strain and stress [dim, dim, quad, nx, ny]
strain_arr = np.zeros((dim, dim, nb_quad, nx, ny))
stress_arr = np.zeros((dim, dim, nb_quad, nx, ny))


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
    eps_macro = np.zeros((dim, dim, nb_quad, nx, ny))
    for i in range(dim):
        for j in range(dim):
            eps_macro[i, j, ...] = E_macro[i, j]

    # Compute stress from macroscopic strain
    sig_macro = np.zeros_like(eps_macro)
    compute_stress(eps_macro, sig_macro, C_field)

    # Compute divergence (with negative sign for RHS)
    compute_divergence(sig_macro, rhs_out)
    rhs_out.s[...] *= -1.0


# Storage for homogenized stiffness
C_eff = np.zeros((nb_voigt, nb_voigt))

# Macroscopic strain cases for 2D
strain_cases = [
    (0, 0),  # xx
    (1, 1),  # yy
    (0, 1),  # xy
]

if comm.rank == 0 and not args.quiet:
    print("=" * 60)
    print("Computing homogenized stiffness tensor")
    print("=" * 60)

for case_idx, (i, j) in enumerate(strain_cases):
    # Create unit macroscopic strain
    E_macro = np.zeros((dim, dim))
    E_macro[i, j] = 1.0
    if i != j:
        E_macro[j, i] = 1.0  # Symmetric

    voigt_col = voigt_index_2d(i, j)

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
            res_norm = np.sqrt(comm.sum(np.dot(r.ravel(), r.ravel())))
            if comm.rank == 0 and it % 10 == 0:
                print(f"  CG iteration {it}: |r| = {res_norm:.6e}")

    # Solve K u = f using conjugate_gradients from Solvers.py
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
                local_sum += quad_weights[q] * np.sum(stress_arr[k, L, q, ...])
            sig_avg[k, L] = comm.sum(local_sum)

    # Normalize by total volume
    total_volume = np.prod(domain_size)
    sig_avg /= total_volume

    # Store in homogenized stiffness (column voigt_col)
    C_eff[0, voigt_col] = sig_avg[0, 0]  # Sigma_xx
    C_eff[1, voigt_col] = sig_avg[1, 1]  # Sigma_yy
    C_eff[2, voigt_col] = sig_avg[0, 1]  # Sigma_xy

    if comm.rank == 0 and not args.quiet:
        print(
            f"  Average stress: xx={sig_avg[0, 0]:.6f}, "
            f"yy={sig_avg[1, 1]:.6f}, xy={sig_avg[0, 1]:.6f}"
        )

# Print results
if comm.rank == 0:
    print("\n" + "=" * 60)
    print("Homogenized stiffness tensor (Voigt notation)")
    print("=" * 60)
    print("\n       xx         yy         xy")
    print(f"xx  {C_eff[0, 0]:10.6f} {C_eff[0, 1]:10.6f} {C_eff[0, 2]:10.6f}")
    print(f"yy  {C_eff[1, 0]:10.6f} {C_eff[1, 1]:10.6f} {C_eff[1, 2]:10.6f}")
    print(f"xy  {C_eff[2, 0]:10.6f} {C_eff[2, 1]:10.6f} {C_eff[2, 2]:10.6f}")

    # Compare to analytical bounds
    v_f = np.mean(phase)  # Volume fraction of inclusion
    E_m, E_i = args.E_matrix, args.E_inclusion
    nu = args.nu

    # Voigt (upper) bound: parallel
    E_voigt = v_f * E_i + (1 - v_f) * E_m
    # Reuss (lower) bound: series
    E_reuss = 1.0 / (v_f / E_i + (1 - v_f) / E_m)

    # Effective modulus from C_eff (plane strain)
    E_eff_approx = C_eff[0, 0] * (1 - nu**2)

    print("\n" + "=" * 60)
    print("Comparison with analytical bounds")
    print("=" * 60)
    print(f"Volume fraction of inclusion: {v_f:.4f}")
    print(f"Voigt bound (upper): E = {E_voigt:.4f}")
    print(f"Reuss bound (lower): E = {E_reuss:.4f}")
    print(f"Effective E (approx from C_xxxx): E ≈ {E_eff_approx:.4f}")
    print("=" * 60)

# Optional plotting
if args.plot and comm.rank == 0:
    if plt is None:
        print("Warning: matplotlib not available, cannot show plot")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Microstructure
        ax = axes[0]
        im = ax.imshow(phase.T, origin="lower", cmap="viridis")
        ax.set_title("Microstructure (0=matrix, 1=inclusion)")
        plt.colorbar(im, ax=ax)

        # Stress xx from last load case
        ax = axes[1]
        sig_xx_avg = np.mean(stress_arr[0, 0, ...], axis=0)
        im = ax.imshow(sig_xx_avg.T, origin="lower", cmap="RdBu_r")
        ax.set_title(r"$\sigma_{xx}$ (last load case)")
        plt.colorbar(im, ax=ax)

        # Displacement magnitude from last load case
        ax = axes[2]
        u_mag = np.sqrt(u_field.s[0, 0, ...] ** 2 + u_field.s[1, 0, ...] ** 2)
        im = ax.imshow(u_mag.T, origin="lower", cmap="viridis")
        ax.set_title("|u| (last load case)")
        plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.show()

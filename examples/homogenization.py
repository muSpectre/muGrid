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
from muGrid import real_field

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


def simple_cg(hessp, b, x0, tol=1e-6, maxiter=1000, callback=None):
    """
    Simple conjugate gradient solver for Ax = b.

    Parameters
    ----------
    hessp : callable
        Function computing A @ x, takes (x, Ax) and fills Ax in place
    b : ndarray
        Right-hand side
    x0 : ndarray
        Initial guess (modified in place)
    tol : float
        Convergence tolerance (on residual norm squared)
    maxiter : int
        Maximum iterations
    callback : callable
        Called as callback(iteration, x, r) after each iteration

    Returns
    -------
    x : ndarray
        Solution (same as x0)
    converged : bool
        Whether the solver converged
    iterations : int
        Number of iterations performed
    """
    tol_sq = tol * tol
    x = x0
    Ax = np.zeros_like(x)

    # Initial residual: r = b - A @ x
    hessp(x, Ax)
    r = b - Ax
    p = r.copy()

    rr = comm.sum(np.dot(r.ravel(), r.ravel()))

    if callback:
        callback(0, x, r)

    if rr < tol_sq:
        return x, True, 0

    for iteration in range(maxiter):
        # Compute A @ p
        hessp(p, Ax)

        pAp = comm.sum(np.dot(p.ravel(), Ax.ravel()))
        if pAp <= 0:
            raise RuntimeError("Matrix is not positive definite")

        alpha = rr / pAp
        x += alpha * p
        r -= alpha * Ax

        if callback:
            callback(iteration + 1, x, r)

        next_rr = comm.sum(np.dot(r.ravel(), r.ravel()))
        if next_rr < tol_sq:
            return x, True, iteration + 1

        beta = next_rr / rr
        rr = next_rr
        p = r + beta * p

    return x, False, maxiter


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

# Note: For serial execution, we use GlobalFieldCollection with right-only ghosts.
# The FEM gradient kernel requires ghosts for accessing neighbor nodes.
# For MPI parallel execution, additional ghost communication would be needed.
if comm.size > 1:
    raise NotImplementedError(
        "MPI parallel homogenization not yet supported. "
        "Please run with a single process."
    )

# Create field collection with right ghosts only (for FEM stencil)
fc = muGrid.GlobalFieldCollection(
    args.nb_grid_pts,
    sub_pts={"quad": 2},  # 2 quadrature points (triangles) per pixel
    nb_ghosts_right=(1,) * dim,
)

# Get local grid dimensions
local_shape = args.nb_grid_pts
nx, ny = local_shape

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

# Create muGrid fields for gradient operations
# These are scalar fields (one component at a time)
u_nodal = real_field(fc, "u_nodal", nb_nodes)
grad_u = real_field(fc, "grad_u", dim, "quad")
f_nodal = real_field(fc, "f_nodal", nb_nodes)
stress_field = real_field(fc, "stress_field", dim, "quad")

# Material stiffness at each quadrature point [voigt, voigt, quad, nx, ny]
C_field = np.zeros((nb_voigt, nb_voigt, nb_quad, nx, ny))
for q in range(nb_quad):
    for i in range(nb_voigt):
        for j in range(nb_voigt):
            C_field[i, j, q] = C_matrix[i, j] * (1 - phase) + C_inclusion[i, j] * phase

# Temporary arrays for strain and stress [dim, dim, quad, nx, ny]
strain_arr = np.zeros((dim, dim, nb_quad, nx, ny))
stress_arr = np.zeros((dim, dim, nb_quad, nx, ny))


def compute_strain(u_arr, strain):
    """
    Compute strain from displacement array.
    u_arr has shape [dim, nb_nodes, nx, ny]
    strain has shape [dim, dim, nb_quad, nx, ny]
    """
    strain[...] = 0.0

    for i in range(dim):
        # Copy displacement component to muGrid field
        u_nodal.p[...] = u_arr[i, ...]

        # Set up ghost values (periodic BC via wrap)
        u_nodal.pg[...] = np.pad(u_nodal.p, ((0, 0), (0, 1), (0, 1)), mode="wrap")

        # Compute gradient of u_i
        gradient_op.apply(u_nodal._cpp, grad_u._cpp)

        # Add to strain (symmetric part)
        for j in range(dim):
            strain[i, j, ...] += 0.5 * grad_u.s[j, ...]
            if i != j:
                strain[j, i, ...] += 0.5 * grad_u.s[j, ...]


def compute_stress(strain, stress, C):
    """
    Compute stress from strain using Voigt notation.
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


def reduce_ghosts(field):
    """
    Add ghost contributions back to the main domain for periodic BC.

    For periodic boundary conditions, the transpose operation writes
    contributions to the ghost region that need to be accumulated back
    to the corresponding nodes in the main domain.
    """
    # field.pg has shape [nb_sub, nx+nb_ghosts, ny+nb_ghosts]
    # Ghost at [:, nx:, :ny] should add to [:, 0:nb_ghosts, :]
    # Ghost at [:, :nx, ny:] should add to [:, :, 0:nb_ghosts]
    nb_ghosts_x = field.pg.shape[-2] - field.p.shape[-2]
    nb_ghosts_y = field.pg.shape[-1] - field.p.shape[-1]
    local_nx = field.p.shape[-2]
    local_ny = field.p.shape[-1]

    # Add right ghost to left
    field.p[..., :nb_ghosts_x, :] += field.pg[..., local_nx:, :local_ny]
    # Add top ghost to bottom
    field.p[..., :, :nb_ghosts_y] += field.pg[..., :local_nx, local_ny:]
    # Add corner ghost
    field.p[..., :nb_ghosts_x, :nb_ghosts_y] += field.pg[..., local_nx:, local_ny:]


def compute_divergence(stress, f_arr):
    """
    Compute divergence of stress.
    f_arr has shape [dim, nb_nodes, nx, ny]
    """
    f_arr[...] = 0.0

    for i in range(dim):
        # Prepare stress row for divergence: sigma_i: = [sigma_ix, sigma_iy]
        for j in range(dim):
            stress_field.s[j, ...] = stress[i, j, ...]

        # Apply transpose (divergence) with quadrature weights
        f_nodal.pg[...] = 0.0  # Clear including ghosts
        gradient_op.transpose(stress_field._cpp, f_nodal._cpp, list(quad_weights))

        # Reduce ghost contributions for periodic BC
        reduce_ghosts(f_nodal)

        # Copy result
        f_arr[i, ...] = f_nodal.p[...]


def apply_stiffness(u_flat, f_flat):
    """
    Apply K = B^T C B to displacement vector.
    """
    # Reshape flat arrays to structured form
    u_arr = u_flat.reshape(dim, nb_nodes, nx, ny)
    f_arr = f_flat.reshape(dim, nb_nodes, nx, ny)

    # Compute strain eps = B * u
    compute_strain(u_arr, strain_arr)

    # Compute stress sig = C : eps
    compute_stress(strain_arr, stress_arr, C_field)

    # Compute force f = B^T * sig
    compute_divergence(stress_arr, f_arr)


def compute_rhs(E_macro):
    """
    Compute RHS: f = -B^T C E_macro
    Returns flat array [dim * nb_nodes * nx * ny]
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
    f_arr = np.zeros((dim, nb_nodes, nx, ny))
    compute_divergence(sig_macro, f_arr)
    f_arr *= -1.0

    return f_arr.ravel()


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

    # Compute RHS
    rhs = compute_rhs(E_macro)

    # Initialize displacement to zero
    u_flat = np.zeros(dim * nb_nodes * nx * ny)

    # CG callback
    def callback(it, x, r):
        if not args.quiet:
            res_norm = np.sqrt(comm.sum(np.dot(r.ravel(), r.ravel())))
            if comm.rank == 0 and it % 10 == 0:
                print(f"  CG iteration {it}: |r| = {res_norm:.6e}")

    # Solve K u = f using simple CG
    u_flat, converged, iterations = simple_cg(
        apply_stiffness,
        rhs,
        u_flat,
        tol=args.tol,
        callback=callback,
        maxiter=args.maxiter,
    )

    if comm.rank == 0 and not args.quiet:
        if converged:
            print(f"  CG converged in {iterations} iterations")
        else:
            print(f"  CG did not converge after {args.maxiter} iterations")

    # Reshape displacement
    u_arr = u_flat.reshape(dim, nb_nodes, nx, ny)

    # Compute total strain = E_macro + eps(u)
    compute_strain(u_arr, strain_arr)
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
        u_arr = u_flat.reshape(dim, nb_nodes, nx, ny)
        u_mag = np.sqrt(u_arr[0, 0, ...] ** 2 + u_arr[1, 0, ...] ** 2)
        im = ax.imshow(u_mag.T, origin="lower", cmap="viridis")
        ax.set_title("|u| (last load case)")
        plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.show()

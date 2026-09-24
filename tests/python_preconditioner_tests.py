"""
Tests for muGrid.Preconditioners: generic preconditioner interface and
FFT (spectral) preconditioning of a finite-difference Laplace solve.
"""

import numpy as np
import pytest
from conftest import (
    create_device,
    get_array_module,
    get_test_devices,
    skip_if_gpu_unavailable,
)

import muGrid
from muGrid.Preconditioners import (
    BlockFourierPreconditioner,
    FourierPreconditioner,
    GreenJacobiPreconditioner,
    IdentityPreconditioner,
    JacobiPreconditioner,
    make_green_jacobi_preconditioner,
    make_reference_stiffness_preconditioner,
)
from muGrid.Solvers import conjugate_gradients


def make_engine(comm, nb_grid_pts):
    """FFT engine that doubles as ghosted decomposition for the FD stencil."""
    nb_ghosts = (1,) * len(nb_grid_pts)
    return muGrid.FFTEngine(
        nb_grid_pts, comm, nb_ghosts_left=nb_ghosts, nb_ghosts_right=nb_ghosts
    )


def fd_laplace_hessp(engine, grid_spacing):
    """Matrix-free application of minus the FD Laplacian (positive definite)."""
    stencil = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    laplace = muGrid.GenericLinearOperator([-1, -1], stencil)

    def hessp(x_field, Ax_field):
        engine.communicate_ghosts(x_field)
        laplace.apply(x_field, Ax_field)
        Ax_field.s[...] /= -grid_spacing**2
    return hessp


def inverse_fd_laplace_kernel(grid_spacing):
    """Exact inverse symbol of minus the FD Laplacian; zero mode projected."""

    def kernel(engine):
        q = engine.fftfreq  # [dim, *local_fourier_shape]
        denom = (4 * np.sin(np.pi * q) ** 2 / grid_spacing**2).sum(axis=0)
        return np.where(denom > 0, 1 / np.where(denom > 0, denom, 1), 0.0)

    return kernel


def discrete_reference_solution(nb_grid_pts, rhs_global, grid_spacing):
    """Exact solution of the discrete system -L u / h^2 = rhs (zero-mean)."""
    nx, ny = nb_grid_pts
    qx = np.fft.rfftfreq(nx)[:, np.newaxis]
    qy = np.fft.fftfreq(ny)[np.newaxis, :]
    denom = (
        4 * np.sin(np.pi * qx) ** 2 + 4 * np.sin(np.pi * qy) ** 2
    ) / grid_spacing**2
    kernel = np.where(denom > 0, 1 / np.where(denom > 0, denom, 1), 0.0)
    rhs_hat = np.fft.rfftn(rhs_global.T).T
    return np.fft.irfftn(
        (rhs_hat * kernel).T, s=(ny, nx), axes=(0, 1)
    ).T


def global_rhs(nb_grid_pts):
    """Deterministic zero-mean right-hand side exciting (nearly) all Fourier
    modes, so that unpreconditioned CG is genuinely iterative (CG converges
    in as many iterations as there are distinct excited eigenvalues)."""
    rng = np.random.default_rng(42)
    rhs = rng.standard_normal(nb_grid_pts)
    return rhs - rhs.mean()


def run_poisson_cg(comm, engine, prec, rtol=1e-8, maxiter=200):
    """Solve the FD Poisson problem; return (local solution, iterations)."""
    (nx, ny) = nb_grid_pts = tuple(engine.nb_domain_grid_pts)
    assert nx == ny, "test assumes square grid (isotropic spacing)"
    grid_spacing = 1 / nx

    rhs = engine.real_space_field("rhs")
    solution = engine.real_space_field("solution")
    ox, oy = engine.subdomain_locations
    lx, ly = engine.nb_subdomain_grid_pts
    rhs.p[...] = global_rhs(nb_grid_pts)[ox : ox + lx, oy : oy + ly]
    solution.p[...] = 0

    iterations = []

    def callback(iteration, state):
        iterations.append(iteration)

    conjugate_gradients(
        comm,
        engine.real_space_collection,
        rhs,
        solution,
        hessp=fd_laplace_hessp(engine, grid_spacing),
        prec=prec,
        rtol=rtol,
        maxiter=maxiter,
        callback=callback,
    )
    return solution, max(iterations)


def local_reference(engine):
    """Exact discrete solution restricted to this rank's subdomain."""
    (nx, ny) = tuple(engine.nb_domain_grid_pts)
    grid_spacing = 1 / nx
    ref = discrete_reference_solution(
        (nx, ny), global_rhs((nx, ny)), grid_spacing
    )
    ox, oy = engine.subdomain_locations
    lx, ly = engine.nb_subdomain_grid_pts
    return ref[ox : ox + lx, oy : oy + ly]


def test_fourier_preconditioned_poisson(comm):
    """FFT-preconditioned CG with the exact inverse symbol converges in O(1)
    iterations and reproduces the exact discrete solution."""
    engine = make_engine(comm, (32, 32))
    grid_spacing = 1 / 32

    prec = FourierPreconditioner(
        engine, inverse_fd_laplace_kernel(grid_spacing)
    )
    solution, iterations = run_poisson_cg(comm, engine, prec)

    # The kernel is the exact inverse of the discrete operator, so CG is a
    # direct solve: one iteration plus roundoff.
    assert iterations <= 3

    np.testing.assert_allclose(
        solution.p, local_reference(engine), atol=1e-10
    )


def test_unpreconditioned_baseline(comm):
    """Unpreconditioned CG reaches the same solution but needs many more
    iterations; this pins the speedup the preconditioner provides."""
    engine = make_engine(comm, (32, 32))
    solution, iterations = run_poisson_cg(comm, engine, prec=None)

    assert iterations > 10  # multimode rhs: genuinely iterative

    np.testing.assert_allclose(
        solution.p, local_reference(engine), atol=1e-8
    )


def test_identity_preconditioner_matches_unpreconditioned(comm):
    """IdentityPreconditioner reproduces the unpreconditioned iteration."""
    engine = make_engine(comm, (16, 16))
    _, iterations_none = run_poisson_cg(comm, engine, prec=None)

    engine2 = make_engine(comm, (16, 16))
    _, iterations_id = run_poisson_cg(
        comm, engine2, prec=IdentityPreconditioner()
    )
    assert iterations_id == iterations_none


def test_fourier_preconditioner_multicomponent(comm):
    """The spectral kernel broadcasts over field components."""
    engine = make_engine(comm, (16, 16))
    grid_spacing = 1 / 16

    prec = FourierPreconditioner(
        engine, inverse_fd_laplace_kernel(grid_spacing)
    )
    r = engine.real_space_field("residual", components=(2,))
    z = engine.real_space_field("preconditioned", components=(2,))

    x, y = engine.coords
    r.p[0] = np.sin(2 * np.pi * x)
    r.p[1] = np.cos(4 * np.pi * y)
    prec(r, z)

    # Each component is preconditioned independently with the same kernel:
    # for a pure mode q, z = r / lambda(q) with the FD eigenvalue lambda.
    lam = lambda fx, fy: (  # noqa: E731
        4 * np.sin(np.pi * fx) ** 2 + 4 * np.sin(np.pi * fy) ** 2
    ) / grid_spacing**2
    np.testing.assert_allclose(
        z.p[0], np.sin(2 * np.pi * x) / lam(1 / 16, 0), atol=1e-12
    )
    np.testing.assert_allclose(
        z.p[1], np.cos(4 * np.pi * y) / lam(0, 2 / 16), atol=1e-12
    )


def test_block_fourier_matches_scalar(comm):
    """Diagonal blocks reproduce the scalar FourierPreconditioner.

    BlockFourierPreconditioner with blocks ``k(q)·I`` must act identically to
    FourierPreconditioner with the scalar kernel ``k(q)`` on every component.
    (FourierPreconditioner folds the inverse-transform normalisation in itself,
    so the block version is given ``k(q)·normalisation`` on the diagonal.)
    """
    engine = make_engine(comm, (16, 16))
    grid_spacing = 1 / 16
    n = 2
    kernel = inverse_fd_laplace_kernel(grid_spacing)(engine)

    scalar = FourierPreconditioner(engine, kernel)

    fourier_shape = tuple(engine.nb_fourier_subdomain_grid_pts)
    blocks = np.zeros((n, n) + fourier_shape, dtype=complex)
    for i in range(n):
        blocks[i, i] = kernel * engine.normalisation
    block = BlockFourierPreconditioner(engine, blocks)

    r = engine.real_space_field("r", components=(n,))
    z_scalar = engine.real_space_field("z_scalar", components=(n,))
    z_block = engine.real_space_field("z_block", components=(n,))
    x, y = engine.coords
    r.p[0] = np.sin(2 * np.pi * x) + 0.5 * np.cos(4 * np.pi * y)
    r.p[1] = np.cos(2 * np.pi * y)

    scalar(r, z_scalar)
    block(r, z_block)
    np.testing.assert_allclose(
        np.asarray(z_block.p), np.asarray(z_scalar.p), atol=1e-12
    )


def test_block_fourier_shape_validation(comm):
    """Blocks whose Fourier shape mismatches the engine are rejected."""
    engine = make_engine(comm, (16, 16))
    with pytest.raises(ValueError):
        BlockFourierPreconditioner(engine, np.ones((2, 2, 3, 3)))


def test_reference_stiffness_preconditioner_is_exact_inverse(comm):
    """The assembled preconditioner is the exact inverse of its operator.

    For a block-circulant operator A (here a componentwise minus-FD-Laplacian on
    an n-component field), `make_reference_stiffness_preconditioner` with the
    action of A builds M⁻¹ = A⁺. Applying it to ``b = A x`` for a zero-mean x
    must recover x (the rigid-body / zero-frequency mode is projected out).
    """
    engine = make_engine(comm, (16, 16))
    grid_spacing = 1 / 16
    n = 2
    laplace = muGrid.GenericLinearOperator(
        [-1, -1], np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    )

    def apply_operator(u, Au):
        engine.communicate_ghosts(u)
        laplace.apply(u, Au)
        Au.s[...] /= -grid_spacing**2

    prec = make_reference_stiffness_preconditioner(engine, apply_operator, n)

    x = engine.real_space_field("x", components=(n,))
    b = engine.real_space_field("b", components=(n,))
    z = engine.real_space_field("z", components=(n,))
    # Build a deterministic global field (identical on every rank, so the
    # global mean below is consistent), make it zero-mean, then assign this
    # rank's subdomain slice. Mirrors run_poisson_cg/global_rhs; assigning the
    # whole global array would break under MPI decomposition, where x.p[c] is
    # only the local subdomain.
    (nx, ny) = tuple(engine.nb_domain_grid_pts)
    ox, oy = engine.subdomain_locations
    lx, ly = engine.nb_subdomain_grid_pts
    rng = np.random.default_rng(0)
    for c in range(n):
        xc = rng.standard_normal((nx, ny))
        xc -= xc.mean()  # zero-mean (orthogonal to the rigid-body mode)
        x.p[c] = xc[ox : ox + lx, oy : oy + ly]

    apply_operator(x, b)  # b = A x
    prec(b, z)            # z = A⁺ b = x (zero mode projected out)

    for c in range(n):
        xc = np.asarray(x.p[c])
        zc = np.asarray(z.p[c])
        np.testing.assert_allclose(zc, xc, atol=1e-10)


def test_reference_stiffness_preconditioner_frees_scratch(comm):
    """Regression guard: the impulse-response scratch must be released after
    the symbol is assembled, so it does not persist (uselessly) through the
    whole solve. Only the preconditioner's own Fourier work field should
    remain. This is a memory optimization; if it regresses these fields would
    silently linger in the engine's collections."""
    name = "ref-prec-free-test"
    engine = make_engine(comm, (16, 16))
    grid_spacing = 1 / 16
    n = 2
    laplace = muGrid.GenericLinearOperator(
        [-1, -1], np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    )

    def apply_operator(u, Au):
        engine.communicate_ghosts(u)
        laplace.apply(u, Au)
        Au.s[...] /= -grid_spacing**2

    make_reference_stiffness_preconditioner(engine, apply_operator, n, name=name)

    rsc = engine.real_space_collection
    fsc = engine.fourier_space_collection
    assert not rsc.field_exists(f"{name}-impulse")
    assert not rsc.field_exists(f"{name}-column")
    assert not fsc.field_exists(f"{name}-column-hat")
    # the per-iteration work buffer must still be there (needed by apply)
    assert fsc.field_exists(f"{name}-work")


def test_block_fourier_hermitian_compressed_storage(comm):
    """Regression guard: a Hermitian symbol (the reference-stiffness case) is
    stored as its triangle (n real diagonals + n(n-1)/2 complex off-diagonals),
    not the dense n×n complex block. This halves the symbol's resident memory;
    if the detection regresses the dense block would be kept."""
    engine = make_engine(comm, (16, 16))
    grid_spacing = 1 / 16
    n = 2
    laplace = muGrid.GenericLinearOperator(
        [-1, -1], np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    )

    def apply_operator(u, Au):
        engine.communicate_ghosts(u)
        laplace.apply(u, Au)
        Au.s[...] /= -grid_spacing**2

    prec = make_reference_stiffness_preconditioner(engine, apply_operator, n)
    assert prec._hermitian is True
    assert prec._blocks is None  # dense block not retained
    assert sorted(prec._off.keys()) == [(0, 1)]  # only the upper triangle
    assert prec._diag.dtype == prec._diag.real.dtype  # diagonals stored real


def test_block_fourier_non_hermitian_keeps_dense(comm):
    """A non-Hermitian block set must fall back to dense storage and still
    apply correctly (z_i = Σ_j M_ij r_j), so the optimization never corrupts
    the general case."""
    engine = make_engine(comm, (8, 8))
    n = 2
    fourier = tuple(engine.nb_fourier_subdomain_grid_pts)
    rng = np.random.default_rng(0)
    # A deterministic, clearly non-Hermitian block field.
    blocks = (rng.standard_normal((n, n) + fourier)
              + 1j * rng.standard_normal((n, n) + fourier))
    prec = BlockFourierPreconditioner(engine, blocks)
    assert prec._hermitian is False
    assert prec._blocks is not None

    # Apply must equal the dense einsum reference, mode by mode.
    r = engine.real_space_field("r", components=(n,))
    z = engine.real_space_field("z", components=(n,))
    rng2 = np.random.default_rng(1)
    r.p[...] = rng2.standard_normal(r.p.shape)
    # Reference: FFT, dense per-mode multiply, IFFT.
    work = engine.fourier_space_field("ref-work", components=(n,))
    engine.fft(r, work)
    ref = np.einsum("ij...,js...->is...", blocks, np.asarray(work.s))
    work.s[...] = ref
    z_ref = engine.real_space_field("z_ref", components=(n,))
    engine.ifft(work, z_ref)
    prec.apply(r, z)
    np.testing.assert_allclose(np.asarray(z.p), np.asarray(z_ref.p), atol=1e-12)


def test_kernel_shape_validation(comm):
    """A kernel that does not match the local Fourier subdomain is rejected."""
    engine = make_engine(comm, (16, 16))
    # Derive a shape that cannot coincide with any rank's local Fourier
    # subdomain (one larger in every axis). A hard-coded shape such as (3, 3)
    # can equal a real subdomain under MPI decomposition, in which case the
    # constructor legitimately accepts it and the expected error is not raised.
    bad_shape = tuple(s + 1 for s in engine.nb_fourier_subdomain_grid_pts)
    with pytest.raises(ValueError, match="Fourier subdomain"):
        FourierPreconditioner(engine, np.ones(bad_shape))


def test_jacobi_apply(comm):
    """JacobiPreconditioner divides by the diagonal, elementwise."""
    engine = make_engine(comm, (16, 16))
    x, y = engine.coords

    diag = engine.real_space_field("diagonal")
    diag.p[...] = 1 + x + 2 * y

    r = engine.real_space_field("residual")
    z = engine.real_space_field("preconditioned")
    r.p[...] = np.sin(2 * np.pi * x)

    prec = JacobiPreconditioner(diag)
    prec(r, z)
    np.testing.assert_allclose(z.p, r.p / diag.p, atol=1e-15)

    # The diagonal was copied at construction
    diag.p[...] = 1e3
    prec(r, z)
    np.testing.assert_allclose(z.p, r.p / (1 + x + 2 * y), atol=1e-15)

    # Scalar diagonal
    JacobiPreconditioner(2.0)(r, z)
    np.testing.assert_allclose(z.p, r.p / 2, atol=1e-15)

    # Singular diagonal is rejected
    with pytest.raises(ValueError, match="non-singular"):
        JacobiPreconditioner(0.0)


def screened_poisson_cg(comm, engine, prec, rtol=1e-8, maxiter=1000):
    """Solve (-FD-Laplacian / h^2 + c(x)) u = b with strongly varying c."""
    (nx, ny) = nb_grid_pts = tuple(engine.nb_domain_grid_pts)
    grid_spacing = 1 / nx
    laplace_hessp = fd_laplace_hessp(engine, grid_spacing)

    x, y = engine.coords
    # Screening coefficient varying over six orders of magnitude
    c = 1 + 1e6 * (np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)) ** 2

    def hessp(x_field, Ax_field):
        laplace_hessp(x_field, Ax_field)
        Ax_field.s[...] += c * x_field.s

    rhs = engine.real_space_field("rhs")
    solution = engine.real_space_field("solution")
    ox, oy = engine.subdomain_locations
    lx, ly = engine.nb_subdomain_grid_pts
    rhs.p[...] = global_rhs(nb_grid_pts)[ox : ox + lx, oy : oy + ly]
    solution.p[...] = 0

    iterations = []
    conjugate_gradients(
        comm,
        engine.real_space_collection,
        rhs,
        solution,
        hessp=hessp,
        prec=prec,
        rtol=rtol,
        maxiter=maxiter,
        callback=lambda it, state: iterations.append(it),
    )
    return solution, max(iterations)


def test_jacobi_preconditioned_screened_poisson(comm):
    """For a heterogeneous screened Poisson problem, Jacobi preconditioning
    with the operator diagonal substantially reduces CG iterations."""
    nb_grid_pts = (32, 32)
    grid_spacing = 1 / nb_grid_pts[0]

    engine = make_engine(comm, nb_grid_pts)
    _, iterations_plain = screened_poisson_cg(comm, engine, prec=None)
    solution_plain = engine.real_space_field("solution").p.copy()

    engine2 = make_engine(comm, nb_grid_pts)
    x, y = engine2.coords
    diag = engine2.real_space_field("diagonal")
    # Exact diagonal of hessp: 4/h^2 from the stencil center plus c(x)
    diag.p[...] = 4 / grid_spacing**2 + 1 + 1e6 * (
        np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
    ) ** 2
    solution_jacobi, iterations_jacobi = screened_poisson_cg(
        comm, engine2, prec=JacobiPreconditioner(diag)
    )

    # Both converged to the same solution ...
    np.testing.assert_allclose(solution_jacobi.p, solution_plain, atol=1e-7)
    # ... but Jacobi equilibrates the heterogeneous coefficient
    assert iterations_jacobi < iterations_plain / 2


def to_host(array):
    """Return a host copy of a numpy or cupy array."""
    return array.get() if hasattr(array, "get") else np.array(array)


@pytest.mark.parametrize("device", get_test_devices())
def test_jacobi_apply_devices(comm, device):
    """JacobiPreconditioner applies D^-1 through field kernels on the
    device the solver fields live on (this exercises the field-valued
    linalg.scal on host and GPU)."""
    skip_if_gpu_unavailable(device)
    xp = get_array_module(device)
    engine = muGrid.FFTEngine((16, 16), comm, device=create_device(device))
    x, y = engine.coords

    diag = 1 + x + 2 * y  # spatial-only diagonal, host values

    r = engine.real_space_field("residual")
    z = engine.real_space_field("preconditioned")
    r_values = np.sin(2 * np.pi * x)
    r.p[...] = xp.asarray(r_values)

    prec = JacobiPreconditioner(diag)
    prec(r, z)
    np.testing.assert_allclose(to_host(z.p), r_values / diag, atol=1e-15)

    # Scalar diagonal stays on the device, too
    JacobiPreconditioner(2.0)(r, z)
    np.testing.assert_allclose(to_host(z.p), r_values / 2, atol=1e-15)


@pytest.mark.parametrize("device", get_test_devices())
def test_jacobi_apply_per_component(comm, device):
    """A per-component diagonal is applied elementwise (no broadcast)."""
    skip_if_gpu_unavailable(device)
    xp = get_array_module(device)
    engine = muGrid.FFTEngine((16, 16), comm, device=create_device(device))
    x, y = engine.coords

    r = engine.real_space_field("residual", components=(2,))
    z = engine.real_space_field("preconditioned", components=(2,))
    r_values = np.stack([np.sin(2 * np.pi * x), np.cos(2 * np.pi * y)])
    r.p[...] = xp.asarray(r_values)

    diag = np.stack([1 + x, 2 + y]).reshape(r.s.shape)
    prec = JacobiPreconditioner(diag)
    prec(r, z)
    np.testing.assert_allclose(
        to_host(z.s), r_values.reshape(r.s.shape) / diag, atol=1e-15
    )


@pytest.mark.parametrize("device", get_test_devices())
def test_jacobi_screened_poisson_devices(comm, device):
    """Jacobi-preconditioned CG for the heterogeneous screened Poisson
    problem runs end-to-end on the device and reduces iterations."""
    skip_if_gpu_unavailable(device)
    xp = get_array_module(device)
    nb_grid_pts = (32, 32)
    grid_spacing = 1 / nb_grid_pts[0]

    # Positive-definite -Laplacian/h^2 via the hardcoded stencil operator
    laplace = muGrid.LaplaceOperator(2, -1.0 / grid_spacing**2)

    def make_device_engine():
        return muGrid.FFTEngine(
            nb_grid_pts, comm, ghosts=laplace, device=create_device(device)
        )

    def solve(prec):
        engine = make_device_engine()
        x, y = engine.coords
        c = xp.asarray(
            1 + 1e6 * (np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)) ** 2
        )

        def hessp(x_field, Ax_field):
            engine.communicate_ghosts(x_field)
            laplace.apply(x_field, Ax_field)
            Ax_field.s[...] += c * x_field.s

        rhs = engine.real_space_field("rhs")
        solution = engine.real_space_field("solution")
        ox, oy = engine.subdomain_locations
        lx, ly = engine.nb_subdomain_grid_pts
        rhs.p[...] = xp.asarray(
            global_rhs(nb_grid_pts)[ox : ox + lx, oy : oy + ly]
        )
        iterations = []
        conjugate_gradients(
            comm,
            engine.real_space_collection,
            rhs,
            solution,
            hessp=hessp,
            prec=prec,
            rtol=1e-8,
            maxiter=1000,
            callback=lambda it, state: iterations.append(it),
        )
        return to_host(solution.p), max(iterations), engine

    solution_plain, iterations_plain, _ = solve(prec=None)

    engine = make_device_engine()
    x, y = engine.coords
    diag = (
        4 / grid_spacing**2
        + 1
        + 1e6 * (np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)) ** 2
    )
    solution_jacobi, iterations_jacobi, _ = solve(
        prec=JacobiPreconditioner(diag)
    )

    np.testing.assert_allclose(solution_jacobi, solution_plain, atol=1e-7)
    assert iterations_jacobi < iterations_plain / 2


@pytest.mark.parametrize("device", get_test_devices())
def test_fourier_preconditioned_poisson_devices(comm, device):
    """The spectral preconditioner runs on the device the solver fields
    live on (exercises the complex field-valued linalg.scal and the
    device FFT path) and still acts as a direct solve."""
    skip_if_gpu_unavailable(device)
    xp = get_array_module(device)
    nb_grid_pts = (32, 32)
    grid_spacing = 1 / nb_grid_pts[0]

    laplace = muGrid.LaplaceOperator(2, -1.0 / grid_spacing**2)
    engine = muGrid.FFTEngine(
        nb_grid_pts, comm, ghosts=laplace, device=create_device(device)
    )
    prec = FourierPreconditioner(
        engine, inverse_fd_laplace_kernel(grid_spacing)
    )

    def hessp(x_field, Ax_field):
        engine.communicate_ghosts(x_field)
        laplace.apply(x_field, Ax_field)

    rhs = engine.real_space_field("rhs")
    solution = engine.real_space_field("solution")
    ox, oy = engine.subdomain_locations
    lx, ly = engine.nb_subdomain_grid_pts
    rhs.p[...] = xp.asarray(global_rhs(nb_grid_pts)[ox : ox + lx, oy : oy + ly])

    iterations = []
    conjugate_gradients(
        comm,
        engine.real_space_collection,
        rhs,
        solution,
        hessp=hessp,
        prec=prec,
        rtol=1e-8,
        maxiter=200,
        callback=lambda it, state: iterations.append(it),
    )
    assert max(iterations) <= 3
    np.testing.assert_allclose(
        to_host(solution.p), local_reference(engine), atol=1e-10
    )

    # The spectral kernel broadcasts over components on the device, too:
    # identical per-component inputs give identical per-component outputs.
    r1 = engine.real_space_field("residual")
    z1 = engine.real_space_field("preconditioned")
    r2 = engine.real_space_field("residual2", components=(2,))
    z2 = engine.real_space_field("preconditioned2", components=(2,))
    r1.p[...] = xp.asarray(np.sin(2 * np.pi * np.array(engine.coords)[0]))
    r2.p[0] = r1.p
    r2.p[1] = r1.p
    prec(r1, z1)
    prec(r2, z2)
    np.testing.assert_allclose(to_host(z2.p[0]), to_host(z1.p), atol=1e-12)
    np.testing.assert_allclose(to_host(z2.p[1]), to_host(z1.p), atol=1e-12)


@pytest.mark.parametrize("device", get_test_devices())
def test_jacobi_broadcast_multicomponent(comm, device):
    """A spatial-only diagonal broadcasts over the components of a
    multi-component residual."""
    skip_if_gpu_unavailable(device)
    xp = get_array_module(device)
    engine = muGrid.FFTEngine((16, 16), comm, device=create_device(device))
    x, y = engine.coords

    r = engine.real_space_field("residual", components=(2,))
    z = engine.real_space_field("preconditioned", components=(2,))
    r_values = np.stack([np.sin(2 * np.pi * x), np.cos(2 * np.pi * y)])
    r.p[...] = xp.asarray(r_values)

    diag = 1 + x + 2 * y  # spatial-only: shared across components
    prec = JacobiPreconditioner(diag)
    prec(r, z)
    np.testing.assert_allclose(to_host(z.p), r_values / diag, atol=1e-15)


# =============================================================================
# Green-Jacobi (J-FFT) preconditioner
# =============================================================================


def _elasticity_reference_precond(comm, engine, n):
    """A 2-component block-circulant reference operator (componentwise minus-FD-
    Laplacian) and its exact Green preconditioner — a lightweight stand-in for
    the reference stiffness that keeps these class-level tests self-contained."""
    grid_spacing = 1 / tuple(engine.nb_domain_grid_pts)[0]
    laplace = muGrid.GenericLinearOperator(
        [-1, -1], np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    )

    def apply_operator(u, Au):
        engine.communicate_ghosts(u)
        laplace.apply(u, Au)
        Au.s[...] /= -grid_spacing**2

    green = make_reference_stiffness_preconditioner(engine, apply_operator, n)
    return apply_operator, green


def test_green_jacobi_identity_diagonal(comm):
    """With a unit diagonal, J^{1/2} = 1 and Green-Jacobi reduces exactly to the
    inner Green preconditioner — and warns about it, since a constant diagonal
    means the Jacobi part is doing nothing."""
    n = 2
    engine = make_engine(comm, (16, 16))
    _, green = _elasticity_reference_precond(comm, engine, n)

    diag = engine.real_space_field("diag", components=(n,))
    diag.p[...] = 1.0
    with pytest.warns(RuntimeWarning, match="constant"):
        gj = GreenJacobiPreconditioner(
            green, diag, communicator=engine.communicator
        )

    r = engine.real_space_field("r", components=(n,))
    z_green = engine.real_space_field("z_green", components=(n,))
    z_gj = engine.real_space_field("z_gj", components=(n,))
    rng = np.random.default_rng(3)
    r.p[...] = rng.standard_normal(r.p.shape)

    green.apply(r, z_green)
    gj.apply(r, z_gj)
    np.testing.assert_allclose(
        np.asarray(z_gj.p), np.asarray(z_green.p), atol=1e-12
    )


def test_green_jacobi_all_void_diagonal_warns(comm):
    """An entirely void diagonal (typically: material fields never filled
    before assembly) silently reduces Green-Jacobi to plain Green — this must
    warn, and the apply must stay finite."""
    n = 2
    engine = make_engine(comm, (16, 16))
    _, green = _elasticity_reference_precond(comm, engine, n)

    diag = engine.real_space_field("diag", components=(n,))
    diag.p[...] = 0.0
    with pytest.warns(RuntimeWarning, match="entirely void"):
        gj = GreenJacobiPreconditioner(
            green, diag, communicator=engine.communicator
        )

    r = engine.real_space_field("r", components=(n,))
    z = engine.real_space_field("z", components=(n,))
    r.p[...] = 1.0
    gj.apply(r, z)
    assert np.all(np.isfinite(np.asarray(z.p)))


def test_green_jacobi_void_diagonal_is_finite(comm):
    """Void entries in the diagonal (<= void_tol) must not produce NaN/Inf: they
    are treated as J^{1/2} = 1 (those DOFs carry no stiffness). A partial void
    with otherwise constant entries is legitimate (two-phase with voids) and
    must NOT trigger the degenerate-diagonal warning — also under MPI, where
    only one rank owns the void patch (exercises the collective verdict)."""
    import warnings

    n = 2
    engine = make_engine(comm, (16, 16))
    _, green = _elasticity_reference_precond(comm, engine, n)

    diag = engine.real_space_field("diag", components=(n,))
    diag.p[...] = 1.0
    ox, oy = engine.subdomain_locations
    lx, ly = engine.nb_subdomain_grid_pts
    x = np.arange(ox, ox + lx)
    y = np.arange(oy, oy + ly)
    void = (x[:, np.newaxis] < 4) & (y[np.newaxis, :] < 4)  # global [:4, :4]
    diag.p[:, void] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        gj = GreenJacobiPreconditioner(
            green, diag, communicator=engine.communicator
        )

    r = engine.real_space_field("r", components=(n,))
    z = engine.real_space_field("z", components=(n,))
    r.p[...] = 1.0
    gj.apply(r, z)
    assert np.all(np.isfinite(np.asarray(z.p)))


def test_green_jacobi_elasticity_homogenization(comm):
    """End-to-end demonstration of J-FFT (the paper's headline result): an
    FFT-accelerated FE elasticity homogenization with a smooth, high-contrast
    material. The plain Green (reference-material) preconditioner degrades badly
    as the contrast grows, while the Green-Jacobi preconditioner assembled by
    make_green_jacobi_preconditioner — Green scaled by the Jacobi diagonal from
    the fused assemble_diagonal kernel — stays fast, converging to the same
    displacement in far fewer iterations."""
    n = 2
    nb_grid_pts = (32, 32)
    grid_spacing = [1 / nb_grid_pts[0], 1 / nb_grid_pts[1]]
    op = muGrid.IsotropicStiffnessOperator2D(grid_spacing, muGrid.FEMElement.q1)

    def build():
        # Node and material fields must share one collection so their
        # stencil-computable regions match; the FFT engine's real-space
        # collection is r2c-padded, so create the material there too (mirrors
        # examples/homogenization.py) and fill ghosts via communicate_ghosts.
        engine = muGrid.FFTEngine(
            nb_grid_pts, comm, nb_ghosts_left=(1, 1), nb_ghosts_right=(1, 1)
        )
        fc = engine.real_space_collection
        lam = fc.real_field("lambda")
        mu = fc.real_field("mu")
        x, y = engine.coords  # local, in [0, 1)
        contrast = 1 + 1e4 * (np.sin(np.pi * x) ** 2 * np.sin(np.pi * y) ** 2)
        lam.p[...] = contrast
        mu.p[...] = contrast
        engine.communicate_ghosts(lam)
        engine.communicate_ghosts(mu)
        return engine, lam, mu

    def hessp_factory(engine, lam, mu):
        def hessp(u, Au):
            engine.communicate_ghosts(u)
            op.apply(u, lam, mu, Au)
        return hessp

    def rhs_field(engine, lam, mu):
        # RHS = -div(C : E_macro) for a shear macro strain.
        E = [0.0, 0.5, 0.5, 0.0]
        f = engine.real_space_field("rhs", components=(n,))
        op.apply_macro_rhs(lam, mu, E, f)
        f.s[...] *= -1.0
        return f

    def solve(use_jacobi):
        engine, lam, mu = build()
        # Global (MPI-reduced) reference Lamé means, so the reference stiffness
        # — and hence the assembled Green symbol — is identical on every rank.
        n_global = comm.sum(int(lam.p.size))
        lam_ref = comm.sum(float(np.asarray(lam.p).sum())) / n_global
        mu_ref = comm.sum(float(np.asarray(mu.p).sum())) / n_global
        if use_jacobi:
            prec = make_green_jacobi_preconditioner(
                engine, op, lam, mu, n,
                reference_lambda=lam_ref, reference_mu=mu_ref,
            )
        else:

            def apply_ref(u, f):
                engine.communicate_ghosts(u)
                op.apply_uniform(u, lam_ref, mu_ref, f)

            prec = make_reference_stiffness_preconditioner(engine, apply_ref, n)

        rhs = rhs_field(engine, lam, mu)
        sol = engine.real_space_field("solution", components=(n,))
        sol.p[...] = 0.0
        iterations = []
        conjugate_gradients(
            comm, engine.real_space_collection, rhs, sol,
            hessp=hessp_factory(engine, lam, mu), prec=prec,
            rtol=1e-8, maxiter=2000,
            callback=lambda it, state: iterations.append(it),
        )
        return to_host(sol.p), max(iterations)

    sol_green, it_green = solve(use_jacobi=False)
    sol_gj, it_gj = solve(use_jacobi=True)

    # Same solution (both solve the same SPD system to the same tolerance)...
    np.testing.assert_allclose(sol_gj, sol_green, atol=1e-5)
    # ... but Green-Jacobi converges dramatically faster on smooth high
    # contrast (empirically ~20 vs several hundred iterations here).
    assert it_gj < it_green / 3
    assert it_gj < 60


def _laminate_setup(comm, n):
    """Engine, stiffness operator and (initially uniform) Lamé fields for the
    paper's laminate experiment (Ladecký et al., Sec. 4.1.1)."""
    op = muGrid.IsotropicStiffnessOperator2D(
        [1 / n, 1 / n], muGrid.FEMElement.q1
    )
    engine = muGrid.FFTEngine(
        (n, n), comm, nb_ghosts_left=(1, 1), nb_ghosts_right=(1, 1)
    )
    fc = engine.real_space_collection
    lam = fc.real_field("lambda")
    mu = fc.real_field("mu")
    lam.p[...] = 1.0
    mu.p[...] = 1.0
    engine.communicate_ghosts(lam)
    engine.communicate_ghosts(mu)
    return engine, op, lam, mu


def _laminate_density(engine, n, chi):
    """Smooth pixel-wise linear density ramp from 1/chi to 1 along x."""
    x = np.asarray(engine.coords)[0]
    j = np.floor(x * n) / n
    return 1 / chi + (1 - 1 / chi) * j / (1 - 1 / n)


def _laminate_solve(comm, engine, op, lam, mu, prec):
    """PCG iteration count for the shear-loaded laminate problem."""
    s = 1 / np.sqrt(2)
    E = [1.0, s, s, 1.0]
    fc = engine.real_space_collection
    rhs = engine.real_space_field("rhs", components=(2,))
    op.apply_macro_rhs(lam, mu, E, rhs)
    rhs.s[...] *= -1.0
    sol = engine.real_space_field("solution", components=(2,))
    sol.p[...] = 0.0

    def hessp(u, Au):
        engine.communicate_ghosts(u)
        op.apply(u, lam, mu, Au)

    iterations = []
    conjugate_gradients(
        comm, fc, rhs, sol, hessp=hessp, prec=prec,
        rtol=1e-5, maxiter=2000,
        callback=lambda it, state: iterations.append(it),
    )
    fc.pop_field("rhs")
    fc.pop_field("solution")
    return max(iterations)


def test_green_jacobi_refresh_after_material_update(comm):
    """The silent stale-diagonal trap: a Green-Jacobi
    preconditioner built while the material is still uniform (it warns), then
    used after an in-place material update WITHOUT refresh(), converges no
    faster than plain Green; after refresh() it recovers the J-FFT speedup."""
    n, chi = 64, 1e4
    engine, op, lam, mu = _laminate_setup(comm, n)

    # Built too early: the diagonal is constant -> warns.
    with pytest.warns(RuntimeWarning, match="constant"):
        prec = make_green_jacobi_preconditioner(
            engine, op, lam, mu, 2, reference_lambda=1.0, reference_mu=1.0
        )

    # In-place material update to the smooth high-contrast ramp.
    rho = _laminate_density(engine, n, chi)
    lam.p[...] = rho
    mu.p[...] = rho
    engine.communicate_ghosts(lam)
    engine.communicate_ghosts(mu)

    it_stale = _laminate_solve(comm, engine, op, lam, mu, prec)
    prec.refresh()
    it_fresh = _laminate_solve(comm, engine, op, lam, mu, prec)

    # Empirically (serial): stale = 41 (exactly the plain-Green count),
    # fresh = 7.
    assert it_fresh * 2 < it_stale
    assert it_fresh < 20


def test_green_jacobi_default_reference_mpi_deterministic(comm):
    """The default reference Lamé parameters are global (MPI-reduced) means:
    identical on every rank and equal to the mean over the whole domain,
    regardless of the decomposition."""
    n, chi = 32, 1e2
    engine, op, lam, mu = _laminate_setup(comm, n)
    rho = _laminate_density(engine, n, chi)
    lam.p[...] = 2.0 * rho
    mu.p[...] = rho
    engine.communicate_ghosts(lam)
    engine.communicate_ghosts(mu)

    prec = make_green_jacobi_preconditioner(engine, op, lam, mu, 2)

    # Expected global mean, computed redundantly on every rank from the
    # global formula (pixel values j/n, j = 0 .. n-1).
    j = np.arange(n) / n
    rho_global = 1 / chi + (1 - 1 / chi) * j / (1 - 1 / n)
    expected_mu = rho_global.mean()
    np.testing.assert_allclose(prec._reference_mu, expected_mu, rtol=1e-12)
    np.testing.assert_allclose(
        prec._reference_lambda, 2.0 * expected_mu, rtol=1e-12
    )

    # Bitwise identical across ranks (Allreduce returns the same value
    # everywhere).
    assert comm.max(prec._reference_mu) == prec._reference_mu
    assert comm.max(-prec._reference_mu) == -prec._reference_mu


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_reference_stiffness_symbol_is_slab_invariant(comm, monkeypatch):
    """Inverting the symbol one slab of Fourier modes at a time must give the
    same preconditioner as inverting it in one go.

    The slab loop bounds the double-precision working set during assembly --
    at 512^3 the symbol alone is 9.7 GB in complex128, and the previous
    mask-indexed form held four or five arrays that size at once. At any
    resolution these tests can afford, the whole symbol fits in one slab, so
    the multi-slab path is only reachable by shrinking the budget. The
    per-mode inversion is independent between modes, so the result must not
    depend on where the slab boundaries fall -- including a boundary that
    separates the singular q = 0 block from the rest.
    """
    import muGrid.Preconditioners as P

    engine = make_engine(comm, (16, 16))
    grid_spacing = 1 / 16
    n = 2
    laplace = muGrid.GenericLinearOperator(
        [-1, -1], np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    )

    def apply_operator(u, Au):
        engine.communicate_ghosts(u)
        laplace.apply(u, Au)
        Au.s[...] /= -grid_spacing**2

    def symbol(name):
        prec = make_reference_stiffness_preconditioner(
            engine, apply_operator, n, name=name)
        if prec._blocks is not None:
            return np.asarray(prec._blocks)
        parts = [np.asarray(prec._diag)]
        parts += [np.asarray(prec._off[k]) for k in sorted(prec._off)]
        return np.concatenate([p.ravel() for p in parts])

    one_slab = symbol("slab-whole")
    # One Fourier index per slab: the most boundaries possible, and it puts
    # q = 0 alone in the first slab.
    monkeypatch.setattr(P, "SYMBOL_INVERSION_SLAB_BYTES", 1)
    many_slabs = symbol("slab-split")

    np.testing.assert_array_equal(one_slab, many_slabs)


@pytest.mark.parametrize("dim,n", [(2, 16), (3, 12)])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_analytic_reference_symbol_matches_impulse_assembly(comm, dim, n, dtype):
    """Naming the operator must give the same preconditioner as probing it.

    The uniform reference operator is a ``3^dim`` stencil -- 243 numbers in 3D
    -- so its symbol is the closed-form sum ``Σ_d S[d] exp(-i q·d)``. The
    impulse route recovers exactly the same information the expensive way: one
    impulse response per component on the *full* grid, each followed by a
    full-grid FFT, then a dense ``n × n`` symbol held at 4.5 GB at 512³.

    This is the gate on the whole analytic path. Getting ``q`` wrong is silent
    -- muGrid's engine makes **axis 0** the half-complex one, while the hybrid
    preconditioner's ``numpy.fft.rfftn`` semantics halve the **last** of the
    axes it transforms -- and a wrong convention still produces a
    plausible-looking symbol. Comparing the applied results catches it.
    """
    engine = make_engine(comm, (n,) * dim)
    spacing = (1.0 / n,) * dim
    lam, mu = 1.3, 0.7
    element = muGrid.FEMElement.q1
    op = (muGrid.IsotropicStiffnessOperator2D if dim == 2
          else muGrid.IsotropicStiffnessOperator3D)(list(spacing), element)

    def apply_ref(u, f):
        engine.communicate_ghosts(u)
        op.apply_uniform(u, lam, mu, f)

    tag = f"{dim}d-{np.dtype(dtype).name}"
    impulse = make_reference_stiffness_preconditioner(
        engine, apply_ref, dim, dtype=dtype, name=f"imp-{tag}")
    analytic = make_reference_stiffness_preconditioner(
        engine, nb_components=dim, element=element, grid_spacing=spacing,
        lambda_ref=lam, mu_ref=mu, dtype=dtype, name=f"ana-{tag}")

    r = engine.real_space_field(f"r-{tag}", components=(dim,), dtype=dtype)
    z_imp = engine.real_space_field(f"zi-{tag}", components=(dim,), dtype=dtype)
    z_ana = engine.real_space_field(f"za-{tag}", components=(dim,), dtype=dtype)
    rng = np.random.default_rng(0)
    r.p[...] = rng.standard_normal(r.p.shape).astype(dtype)

    impulse(r, z_imp)
    analytic(r, z_ana)

    a = np.asarray(z_imp.p).astype(np.float64)
    b = np.asarray(z_ana.p).astype(np.float64)
    scale = np.abs(a).max()
    # float32 fields round the symbol and the transforms; float64 should agree
    # to assembly round-off.
    tol = 1e-5 if np.dtype(dtype) == np.dtype(np.float32) else 1e-12
    assert np.abs(a - b).max() / scale < tol


@pytest.mark.parametrize("dim", [2, 3])
def test_analytic_reference_needs_a_complete_description(comm, dim):
    """Half a description is an error, not a silent fallback to the slow path."""
    engine = make_engine(comm, (8,) * dim)
    with pytest.raises(ValueError, match="grid_spacing"):
        make_reference_stiffness_preconditioner(
            engine, nb_components=dim, element=muGrid.FEMElement.q1,
            lambda_ref=1.3, name=f"incomplete-{dim}")
    with pytest.raises(ValueError, match="apply_reference_stiffness"):
        make_reference_stiffness_preconditioner(
            engine, nb_components=dim, name=f"nothing-{dim}")


@pytest.mark.parametrize("dim,n", [(2, 16), (3, 10)])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_evaluated_symbol_matches_stored_symbol(comm, dim, n, dtype):
    """Evaluating the symbol per mode must give the same operator as storing it.

    ``AnalyticReferencePreconditioner`` keeps the ``3^dim`` stencil -- 243
    numbers in 3D -- and rebuilds ``K(q)``, inverts it and applies it in
    registers, where ``BlockFourierPreconditioner`` holds ``n²`` complex values
    per Fourier mode (2.3 GB at 512³ in single precision). Measured at 128³:
    build peak 604 -> 51 MB, total peak 803 -> 258 MB, apply 59 -> 75 ms.

    This is the gate on the C++ kernel, and the thing it most easily gets wrong
    is the mode ordering: the kernel indexes modes with **axis 0 fastest**,
    matching the Fourier field's Fortran-ordered buffer, and a C-order reading
    of the same buffer produces a wrong but entirely plausible result.
    """
    from muGrid.Preconditioners import (
        AnalyticReferencePreconditioner,
        reference_stencil,
    )

    engine = make_engine(comm, (n,) * dim)
    spacing = (1.0 / n,) * dim
    lam, mu = 1.3, 0.7
    element = muGrid.FEMElement.q1
    op = (muGrid.IsotropicStiffnessOperator2D if dim == 2
          else muGrid.IsotropicStiffnessOperator3D)(list(spacing), element)

    def apply_ref(u, f):
        engine.communicate_ghosts(u)
        op.apply_uniform(u, lam, mu, f)

    tag = f"{dim}d-{np.dtype(dtype).name}"
    stored = make_reference_stiffness_preconditioner(
        engine, apply_ref, dim, dtype=dtype, name=f"stored-{tag}")
    evaluated = AnalyticReferencePreconditioner(
        engine, reference_stencil(dim, spacing, element, lam, mu),
        dtype=dtype, name=f"eval-{tag}")

    r = engine.real_space_field(f"er-{tag}", components=(dim,), dtype=dtype)
    z_s = engine.real_space_field(f"ezs-{tag}", components=(dim,), dtype=dtype)
    z_e = engine.real_space_field(f"eze-{tag}", components=(dim,), dtype=dtype)
    rng = np.random.default_rng(0)
    r.p[...] = rng.standard_normal(r.p.shape).astype(dtype)

    stored(r, z_s)
    evaluated(r, z_e)

    a = np.asarray(z_s.p).astype(np.float64)
    b = np.asarray(z_e.p).astype(np.float64)
    tol = 1e-5 if np.dtype(dtype) == np.dtype(np.float32) else 1e-12
    assert np.abs(a - b).max() / np.abs(a).max() < tol


@pytest.mark.parametrize("dim", [2, 3])
def test_evaluated_symbol_stores_only_the_stencil(comm, dim):
    """The point of the class is what it does *not* keep: no array it owns may
    scale with the number of Fourier modes."""
    from muGrid.Preconditioners import (
        AnalyticReferencePreconditioner,
        reference_stencil,
    )

    n = 12
    engine = make_engine(comm, (n,) * dim)
    spacing = (1.0 / n,) * dim
    prec = AnalyticReferencePreconditioner(
        engine, reference_stencil(dim, spacing, muGrid.FEMElement.q1, 1.3, 0.7),
        name=f"small-{dim}")

    fourier_shape = tuple(engine.nb_fourier_subdomain_grid_pts)
    assert prec._stencil.size == 3 ** dim * dim * dim
    # Frequency tables are per axis, not per mode -- the per-mode form would be
    # 809 MB at 512^3, most of what not storing the symbol saves. Stated
    # structurally rather than as a size comparison: under MPI a subdomain can
    # hold fewer modes than the axis lengths sum to, which says nothing about
    # how the storage scales.
    assert [q.size for q in prec._q] == list(fourier_shape)
    # Everything it owns is O(stencil + sum of axis lengths), never
    # O(product). The saving is asymptotic, not universal: 243 numbers is a
    # fixed cost that only pays off once a subdomain holds many modes, and an
    # MPI rank with a dozen of them is better served by the stored symbol.
    stored_scalars = prec._stencil.size + sum(q.size for q in prec._q)
    assert stored_scalars == 3 ** dim * dim * dim + sum(fourier_shape)


@pytest.mark.parametrize("dim,shape", [
    (2, (16, 16)), (3, (10, 10, 10)),
    # axis 0 halves to two modes, so a thread block spans more lines than it
    # keeps in shared memory and every thread builds its own hoisted sums
    (2, (2, 64)), (3, (3, 16, 16)),
])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_evaluated_symbol_on_device_matches_stored_on_host(comm, dim, shape,
                                                           dtype):
    """The device kernel must reproduce the stored symbol applied on the host.

    What it can get wrong silently is the layout: the device Fourier buffer is
    structure-of-arrays (component stride ``nb_modes``, mode stride 1) where
    the host one is array-of-structures, and the kernel's thread blocks share
    the hoisted sums over axes 1 and 2 per axis-0 line. The short-axis-0 grids
    take the path where a block spans too many lines to share them.
    """
    skip_if_gpu_unavailable("gpu")
    import cupy

    from muGrid.Preconditioners import AnalyticReferencePreconditioner

    # Isotropic even where the grid is not: a 32:1 element is ill-conditioned
    # enough to push single precision past the tolerance on its own.
    spacing = (1.0 / shape[-1],) * dim
    lam, mu = 1.3, 0.7
    element = muGrid.FEMElement.q1
    tag = f"{dim}d-{shape[0]}-{np.dtype(dtype).name}"

    host = muGrid.FFTEngine(shape, comm)
    device = muGrid.FFTEngine(shape, comm, device=create_device("gpu"))
    stored = make_reference_stiffness_preconditioner(
        host, nb_components=dim, dtype=dtype, element=element,
        grid_spacing=spacing, lambda_ref=lam, mu_ref=mu,
        evaluate_symbol=False, name=f"stored-{tag}")
    evaluated = make_reference_stiffness_preconditioner(
        device, nb_components=dim, dtype=dtype, element=element,
        grid_spacing=spacing, lambda_ref=lam, mu_ref=mu,
        name=f"eval-{tag}")
    assert isinstance(stored, BlockFourierPreconditioner)
    assert isinstance(evaluated, AnalyticReferencePreconditioner)

    r_h = host.real_space_field(f"r-{tag}", components=(dim,), dtype=dtype)
    z_h = host.real_space_field(f"z-{tag}", components=(dim,), dtype=dtype)
    r_d = device.real_space_field(f"r-{tag}", components=(dim,), dtype=dtype)
    z_d = device.real_space_field(f"z-{tag}", components=(dim,), dtype=dtype)
    x = np.random.default_rng(0).standard_normal(r_h.p.shape).astype(dtype)
    r_h.p[...] = x
    r_d.p[...] = cupy.asarray(x)

    stored(r_h, z_h)
    evaluated(r_d, z_d)

    a = np.asarray(z_h.p).astype(np.float64)
    b = cupy.asnumpy(z_d.p).astype(np.float64)
    tol = 1e-5 if np.dtype(dtype) == np.dtype(np.float32) else 1e-12
    assert np.abs(a - b).max() / np.abs(a).max() < tol


@pytest.mark.parametrize("device", get_test_devices())
def test_reference_factory_chooses_storage_by_device(comm, device):
    """By default the analytic route evaluates the symbol on a device and
    stores it on the host; ``evaluate_symbol`` overrides either way."""
    skip_if_gpu_unavailable(device)
    from muGrid.Preconditioners import AnalyticReferencePreconditioner

    dim, n = 3, 8
    engine = muGrid.FFTEngine((n,) * dim, comm, device=create_device(device))
    kw = dict(nb_components=dim, element=muGrid.FEMElement.q1,
              grid_spacing=(1.0 / n,) * dim, lambda_ref=1.3, mu_ref=0.7)

    default = make_reference_stiffness_preconditioner(
        engine, name=f"default-{device}", **kw)
    expected = (BlockFourierPreconditioner if device == "cpu"
                else AnalyticReferencePreconditioner)
    assert isinstance(default, expected)

    forced = make_reference_stiffness_preconditioner(
        engine, name=f"forced-{device}", evaluate_symbol=device == "cpu",
        **kw)
    assert not isinstance(forced, expected)

    with pytest.raises(ValueError, match="analytic route"):
        make_reference_stiffness_preconditioner(
            engine, lambda u, f: None, dim, evaluate_symbol=True,
            name=f"opaque-{device}")


def test_evaluated_symbol_rejects_a_malformed_stencil(comm):
    from muGrid.Preconditioners import AnalyticReferencePreconditioner

    engine = make_engine(comm, (8, 8))
    with pytest.raises(ValueError, match="stencil must have shape"):
        AnalyticReferencePreconditioner(engine, np.zeros((3, 3, 2)),
                                        name="bad-stencil")

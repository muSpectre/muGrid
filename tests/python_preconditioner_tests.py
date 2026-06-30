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
    IdentityPreconditioner,
    JacobiPreconditioner,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

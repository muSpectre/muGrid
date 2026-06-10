"""
Tests for muGrid.Preconditioners: generic preconditioner interface and
FFT (spectral) preconditioning of a finite-difference Laplace solve.
"""

import numpy as np
import pytest

import muGrid
from muGrid.Preconditioners import (
    FourierPreconditioner,
    IdentityPreconditioner,
    JacobiPreconditioner,
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


def run_poisson_cg(comm, engine, prec, tol=1e-8, maxiter=200):
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
        tol=tol,
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


def test_kernel_shape_validation(comm):
    """A kernel that does not match the local Fourier subdomain is rejected."""
    engine = make_engine(comm, (16, 16))
    with pytest.raises(ValueError, match="Fourier subdomain"):
        FourierPreconditioner(engine, np.ones((3, 3)))


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


def screened_poisson_cg(comm, engine, prec, tol=1e-8, maxiter=1000):
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
        tol=tol,
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

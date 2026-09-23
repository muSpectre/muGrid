import warnings

import numpy as np
import pytest

from muGrid import Communicator, GlobalFieldCollection
from muGrid.Solvers import conjugate_gradients, conjugate_gradients_pipelined


@pytest.mark.skipif(Communicator().size > 1, reason="Test only works in serial")
@pytest.mark.parametrize(
    "A,b,x0",
    [
        ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [1, 2, 3], [0, 0, 0]),
        ([[1, 1, 0], [1, 2, 0], [0, 0, 1]], [1, 2, 3], [0, 0, 0]),
        ([[4, 1], [1, 3]], [1, 2], [2, 1]),
    ],
)
def test_conjugate_gradients(A, b, x0):
    comm = Communicator()

    fc = GlobalFieldCollection((len(b),))

    A = np.array(A)

    def hessp(x, Ax):
        Ax.p[...] = A @ x.p
        return Ax

    solution = fc.real_field("solution")
    solution.p[...] = np.array(x0)
    rhs = fc.real_field("rhs")
    rhs.p[...] = np.array(b)

    conjugate_gradients(
        comm, fc, rhs, solution, hessp=hessp, rtol=1e-8, maxiter=10
    )

    np.testing.assert_allclose(solution.p, np.linalg.solve(A, b), atol=1e-6)


@pytest.mark.skipif(Communicator().size > 1, reason="Test only works in serial")
@pytest.mark.parametrize(
    "A,b,x0",
    [
        ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [1, 2, 3], [0, 0, 0]),
        ([[1, 1, 0], [1, 2, 0], [0, 0, 1]], [1, 2, 3], [0, 0, 0]),
        ([[4, 1], [1, 3]], [1, 2], [2, 1]),
    ],
)
def test_conjugate_gradients_pipelined(A, b, x0):
    comm = Communicator()

    fc = GlobalFieldCollection((len(b),))

    A = np.array(A)

    def hessp(x, Ax):
        Ax.p[...] = A @ x.p
        return Ax

    solution = fc.real_field("solution")
    solution.p[...] = np.array(x0)
    rhs = fc.real_field("rhs")
    rhs.p[...] = np.array(b)

    conjugate_gradients_pipelined(
        comm, fc, rhs, solution, hessp=hessp, rtol=1e-8, maxiter=10
    )

    np.testing.assert_allclose(solution.p, np.linalg.solve(A, b), atol=1e-6)


@pytest.mark.skipif(Communicator().size > 1, reason="Test only works in serial")
def test_conjugate_gradients_deprecated_tol():
    """`tol` keeps working as an absolute tolerance but warns."""
    comm = Communicator()

    fc = GlobalFieldCollection((2,))
    A = np.array([[4, 1], [1, 3]])

    def hessp(x, Ax):
        Ax.p[...] = A @ x.p
        return Ax

    solution = fc.real_field("solution")
    solution.p[...] = 0.0
    rhs = fc.real_field("rhs")
    rhs.p[...] = np.array([1.0, 2.0])

    with pytest.warns(DeprecationWarning):
        conjugate_gradients(
            comm, fc, rhs, solution, hessp=hessp, tol=1e-6, maxiter=10
        )

    np.testing.assert_allclose(
        solution.p, np.linalg.solve(A, [1.0, 2.0]), atol=1e-6
    )


serial_only = pytest.mark.skipif(
    Communicator().size > 1, reason="Test only works in serial"
)


def _scaled_identity_problem(dtype, scale, rhs=1e7, n=(8, 8, 8)):
    """``A = scale * I`` on a ``dtype`` vector field with a uniform ``rhs``.

    Condition number 1, so CG must solve it in a single iteration at any
    precision. The knob that matters is the *magnitude* of the inner
    products: ``pAp = scale * rhs^2 * N``, which can leave the float32 range
    while every individual field value stays comfortably inside it.
    """
    fc = GlobalFieldCollection(n)
    b = fc.real_field("rhs", (3,), dtype=dtype)
    b.p[...] = rhs
    x = fc.real_field("solution", (3,), dtype=dtype)
    x.set_zero()

    def hessp(src, dst):
        dst.p[...] = dtype(scale) * src.p

    return fc, b, x, hessp


@serial_only
@pytest.mark.parametrize(
    "solver", [conjugate_gradients, conjugate_gradients_pipelined]
)
def test_float32_large_rhs_does_not_stall(solver):
    """A float32 solve of a perfectly conditioned system must not stall.

    ``A = I`` and ``b = 1e19``: every field value is far inside the float32
    range (max 3.4e38), but the inner product ``|b|^2 * N`` is 1.5e41 and is
    not. Regression test for the reduction result being narrowed back to
    float32 on return -- ``pAp`` came out as inf, ``alpha = rz/pAp`` became
    exactly 0, and the solve ran to maxiter without ever moving ``x``,
    reported as a generic "did not converge" with a bit-identical residual
    every iteration. The float64 solve converges in one iteration, and so
    must the float32 one.

    The right-hand side rather than the operator carries the magnitude on
    purpose: the pipelined variant applies ``A`` twice per iteration, so a
    large operator norm would overflow the *fields* before the reduction,
    which is a real dynamic-range limit rather than the bug under test.
    """
    comm = Communicator()
    fc, b, x, hessp = _scaled_identity_problem(np.float32, 1.0, rhs=1e19)
    solver(comm, fc, b, x, hessp=hessp, rtol=1e-6, maxiter=25)
    np.testing.assert_allclose(np.asarray(x.p), 1e19, rtol=1e-5)


@serial_only
@pytest.mark.parametrize(
    "solver", [conjugate_gradients, conjugate_gradients_pipelined]
)
def test_singular_operator_raises_convergence_error(solver):
    """A zero operator must raise ConvergenceError, not ZeroDivisionError.

    The pipelined variant used to divide by an ``alpha_prev`` of zero and let
    a bare ZeroDivisionError escape from inside the recurrence, past its own
    ConvergenceError contract -- so callers catching ConvergenceError (as
    muTopOpt does) saw an unhandled traceback instead of a failed solve.
    """
    from muGrid.Solvers import ConvergenceError

    comm = Communicator()
    fc, b, x, hessp = _scaled_identity_problem(np.float32, 0.0)
    with pytest.raises(ConvergenceError, match="curvature"):
        solver(comm, fc, b, x, hessp=hessp, rtol=1e-6, maxiter=25)


@serial_only
@pytest.mark.parametrize(
    "solver", [conjugate_gradients, conjugate_gradients_pipelined]
)
def test_float32_rtol_below_floor_warns(solver):
    """Asking a float32 solve for a tolerance below float32 eps is
    unsatisfiable by construction, and should say so rather than silently
    running to maxiter."""
    comm = Communicator()
    fc, b, x, hessp = _scaled_identity_problem(np.float32, 1.0)
    with pytest.warns(RuntimeWarning, match="float32 solve accuracy floor"):
        solver(comm, fc, b, x, hessp=hessp, rtol=1e-9, maxiter=5)


@serial_only
@pytest.mark.parametrize(
    "solver", [conjugate_gradients, conjugate_gradients_pipelined]
)
def test_float64_rtol_below_float32_floor_is_silent(solver):
    """The float32 floor must not leak into double-precision solves, where a
    1e-9 tolerance is entirely reasonable."""
    comm = Communicator()
    fc, b, x, hessp = _scaled_identity_problem(np.float64, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        solver(comm, fc, b, x, hessp=hessp, rtol=1e-9, maxiter=5)

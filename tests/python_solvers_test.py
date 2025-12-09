import numpy as np
import pytest

from muGrid import Communicator, GlobalFieldCollection, wrap_field
from muGrid.Solvers import conjugate_gradients


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
        x_wrap = wrap_field(x)
        Ax_wrap = wrap_field(Ax)
        Ax_wrap.p[...] = A @ x_wrap.p
        return Ax

    solution_cpp = fc.real_field("solution")
    solution = wrap_field(solution_cpp)
    solution.p[...] = np.array(x0)
    rhs_cpp = fc.real_field("rhs")
    rhs = wrap_field(rhs_cpp)
    rhs.p[...] = np.array(b)

    conjugate_gradients(
        comm, fc, hessp, rhs_cpp, solution_cpp, tol=1e-6, maxiter=10  # linear operator
    )

    np.testing.assert_allclose(solution.p, np.linalg.solve(A, b), atol=1e-6)

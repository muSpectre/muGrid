import numpy as np
import pytest

from muGrid import Communicator, GlobalFieldCollection
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
        Ax.p[...] = A @ x.p
        return Ax

    solution = fc.real_field("solution")
    solution.p[...] = np.array(x0)
    rhs = fc.real_field("rhs")
    rhs.p[...] = np.array(b)

    conjugate_gradients(
        comm, fc, hessp, rhs, solution, tol=1e-6, maxiter=10
    )

    np.testing.assert_allclose(solution.p, np.linalg.solve(A, b), atol=1e-6)

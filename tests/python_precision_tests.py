"""
Phase E tests: the single-precision (float32) field-creation ``dtype=`` API,
the ``Field.dtype`` accessor, and an end-to-end float32 CG solve that matches
the float64 result.
"""
import numpy as np
import pytest

from muGrid import Communicator, GlobalFieldCollection
from muGrid.Solvers import conjugate_gradients

serial_only = pytest.mark.skipif(
    Communicator().size > 1, reason="Test only works in serial"
)


@serial_only
@pytest.mark.parametrize(
    "kind,dtype,expected",
    [
        ("real", np.float64, np.float64),
        ("real", np.float32, np.float32),
        ("complex", np.complex128, np.complex128),
        ("complex", np.complex64, np.complex64),
    ],
)
def test_dtype_field_creation(kind, dtype, expected):
    fc = GlobalFieldCollection((4, 4))
    make = fc.real_field if kind == "real" else fc.complex_field
    f = make("f", (2,), dtype=dtype)
    assert f.dtype == np.dtype(expected)
    assert f.s.dtype == np.dtype(expected)


@serial_only
def test_dtype_default_is_double():
    fc = GlobalFieldCollection((4, 4))
    assert fc.real_field("r").dtype == np.dtype(np.float64)
    assert fc.complex_field("c").dtype == np.dtype(np.complex128)


@serial_only
@pytest.mark.parametrize(
    "kind,dtype", [("real", np.int32), ("complex", np.float32)]
)
def test_dtype_rejects_unsupported(kind, dtype):
    fc = GlobalFieldCollection((4, 4))
    make = fc.real_field if kind == "real" else fc.complex_field
    with pytest.raises(ValueError):
        make("bad", (2,), dtype=dtype)


@serial_only
def test_float32_field_get_or_create():
    """A second request for the same float32 field returns the existing one
    (get-or-create), matching the double-precision convenience semantics."""
    fc = GlobalFieldCollection((4, 4))
    a = fc.real_field("x", (2,), dtype=np.float32)
    b = fc.real_field("x", (2,), dtype=np.float32)
    a.s[...] = 1.5
    # Same underlying buffer => the second handle sees the first's writes.
    assert np.allclose(np.asarray(b.s), 1.5)


@serial_only
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_conjugate_gradients_dtype(dtype):
    """CG solves the system in the precision of the right-hand side and the
    work fields it allocates inherit that precision."""
    comm = Communicator()
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])

    fc = GlobalFieldCollection((2,))

    def hessp(x, Ax):
        Ax.p[...] = (A @ np.asarray(x.p)).astype(dtype)
        return Ax

    solution = fc.real_field("solution", dtype=dtype)
    solution.p[...] = 0.0
    rhs = fc.real_field("rhs", dtype=dtype)
    rhs.p[...] = b.astype(dtype)

    conjugate_gradients(comm, fc, rhs, solution, hessp=hessp,
                        rtol=1e-6, maxiter=20)

    atol = 1e-6 if dtype == np.float64 else 1e-4
    np.testing.assert_allclose(
        np.asarray(solution.p).ravel(), np.linalg.solve(A, b), atol=atol
    )


@serial_only
def test_float32_cg_matches_float64():
    """Same well-conditioned system solved in both precisions agrees."""
    comm = Communicator()
    A = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    b = np.array([1.0, 2.0, -1.0])

    def solve(dtype):
        fc = GlobalFieldCollection((3,))

        def hessp(x, Ax):
            Ax.p[...] = (A @ np.asarray(x.p)).astype(dtype)
            return Ax

        sol = fc.real_field("sol", dtype=dtype)
        sol.p[...] = 0.0
        rhs = fc.real_field("rhs", dtype=dtype)
        rhs.p[...] = b.astype(dtype)
        conjugate_gradients(comm, fc, rhs, sol, hessp=hessp,
                            rtol=1e-6, maxiter=50)
        return np.asarray(sol.p).ravel().astype(np.float64)

    x64 = solve(np.float64)
    x32 = solve(np.float32)
    np.testing.assert_allclose(x32, x64, atol=1e-4)

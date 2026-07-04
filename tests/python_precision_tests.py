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


@serial_only
@pytest.mark.parametrize("ghosts", [0, 1])
def test_float32_reductions_accumulate_in_double(ghosts):
    """vecdot / norm_sq / axpy_norm_sq on float32 fields accumulate in double
    precision internally (only the final scalar is narrowed to float32).

    A running float32 accumulation of N comparable values loses
    O(N * ulp(sum)) — for the value 0.1 (inexact in binary) over ~1M entries
    this is a relative error of 1e-4..1e-2 depending on lane count, which is
    what inflated float32 CG termination decisions. Double accumulation
    leaves only the final O(eps_f32) narrowing. Both reduction code paths are
    exercised: the ghost-free Eigen path and the ghost-skipping interior
    loop."""
    from muGrid import linalg

    n = (1024, 1024)
    g = (ghosts, ghosts)
    fc = GlobalFieldCollection(n, nb_ghosts_left=g, nb_ghosts_right=g)
    x = fc.real_field("x", dtype=np.float32)
    x.s[...] = np.float32(0.1)

    nb_interior = n[0] * n[1]
    # Reference: exact double sum of the float32-representable value 0.1f.
    exact = float(np.float64(np.float32(0.1)) ** 2 * nb_interior)

    rel = abs(float(linalg.norm_sq(x)) - exact) / exact
    assert rel < 1e-6, f"norm_sq relative error {rel:.2e}"

    y = fc.real_field("y", dtype=np.float32)
    y.s[...] = np.float32(0.1)
    rel = abs(float(linalg.vecdot(x, y)) - exact) / exact
    assert rel < 1e-6, f"vecdot relative error {rel:.2e}"

    # axpy_norm_sq(0, x, y) leaves y unchanged and returns ||y||^2.
    rel = abs(float(linalg.axpy_norm_sq(0.0, x, y)) - exact) / exact
    assert rel < 1e-6, f"axpy_norm_sq relative error {rel:.2e}"

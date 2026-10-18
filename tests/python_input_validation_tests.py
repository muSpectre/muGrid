"""
Input validation and out-of-bounds access (audit findings L1-L3, O2, M2-M4).

Each case used to read or write past a buffer, return a wrong value, or
deadlock an MPI run, without an error.
"""

import numpy as np
import pytest

import muGrid
from muGrid import _muGrid, linalg

# --- L1: cross / leray_project check sub-points ---


def test_cross_rejects_a_sub_point_mismatch():
    """The number of three-vectors is taken from one operand; with another
    one having fewer sub-points, the kernel read and wrote past its end."""
    fc = muGrid.GlobalFieldCollection((4, 4), sub_pts={"quad": 2})
    a = fc.real_field("a", (3,), "quad")
    b = fc.real_field("b", (3,), "quad")
    out = fc.real_field("out", (3,))
    with pytest.raises(RuntimeError, match="sub-points"):
        linalg.cross(a, b, out)
    linalg.cross(a, b, fc.real_field("out-quad", (3,), "quad"))  # matching


def test_leray_project_rejects_a_sub_point_mismatch():
    fc = muGrid.GlobalFieldCollection((4, 4), sub_pts={"quad": 2})
    k = fc.real_field("k", (3,), "quad")
    invk = fc.real_field("invk", (3,), "quad")
    N = fc.complex_field("N", (3,))
    out = fc.complex_field("out", (3,))
    with pytest.raises(RuntimeError, match="sub-points"):
        linalg.leray_project(k, invk, N, out)


# --- L2: the float32 pipelined_cg_dots validates its operands ---


def test_pipelined_cg_dots_float32_rejects_mismatched_operands():
    fc = muGrid.GlobalFieldCollection((4, 4))
    r = fc.real_field("r", (2,), dtype=np.float32)
    u = fc.real_field("u", (2,), dtype=np.float32)
    w = fc.real_field("w", (3,), dtype=np.float32)
    with pytest.raises(RuntimeError, match="components"):
        linalg.pipelined_cg_dots(r, u, w)
    other = muGrid.GlobalFieldCollection((4, 4))
    w = other.real_field("w", (2,), dtype=np.float32)
    with pytest.raises(RuntimeError, match="same collection"):
        linalg.pipelined_cg_dots(r, u, w)


# --- L3: flat-buffer kernels refuse padded pixel storage ---


def _padded_collection():
    """A 4 x 4 grid whose second axis has stride 8: half of the buffer is
    padding between the pixel columns."""
    return _muGrid.GlobalFieldCollection(
        [4, 4], [4, 4], [0, 0], [1, 8], {})


def test_flat_buffer_kernels_refuse_padded_storage():
    fc = _padded_collection()
    a = fc.register_real_field("a", 3)
    b = fc.register_real_field("b", 3)
    out = fc.register_real_field("out", 3)
    alpha = fc.register_real_field("alpha", 1)
    with pytest.raises(RuntimeError, match="contiguous"):
        linalg.cross(a, b, out)
    with pytest.raises(RuntimeError, match="contiguous"):
        linalg.scal(alpha, a)


# --- O2: transpose weights must have one entry per quadrature point ---


@pytest.mark.parametrize("make_op", ["fem", "generic"])
def test_transpose_rejects_weights_of_the_wrong_length(make_op):
    """The kernels index the weights by quadrature point: a short vector was
    read past its end."""
    if make_op == "fem":
        op = muGrid.FEMGradientOperator(2, [0.1, 0.1])
    else:
        op = muGrid.GenericLinearOperator([0, 0], np.ones((2, 2, 2, 2)))
    nq = op.nb_quad_pts
    fc = muGrid.GlobalFieldCollection((6, 6), sub_pts={"quad": nq},
                                      nb_ghosts_left=(1, 1),
                                      nb_ghosts_right=(1, 1))
    nodal = fc.real_field("nodal")
    quad = fc.real_field("quad", (op.nb_output_components,), "quad")
    op.transpose(quad, nodal, [1.0] * nq)  # right length is fine
    for wrong in ([1.0] * (nq - 1), [1.0] * (nq + 1)):
        with pytest.raises(RuntimeError, match="one entry per quadrature"):
            op.transpose(quad, nodal, wrong)


# --- M2: gather rejects uneven rows on every rank ---


def test_gather_rejects_uneven_rows_collectively(comm):
    """Ranks with different row counts used to be accepted in release
    builds (the check was an assert) and the result was scrambled."""
    if comm.size == 1:
        pytest.skip("needs several ranks")
    rows = 2 if comm.rank == 0 else 3
    with pytest.raises(RuntimeError, match="same number of rows"):
        comm.gather(np.ones((rows, 2), order="F"))
    # empty ranks are allowed
    rows = 0 if comm.rank == 0 else 3
    result = comm.gather(np.ones((rows, 2), order="F"))
    assert result.shape == (3, 2 * (comm.size - 1))


# --- M3: integer min/max with an empty rank ---


def test_integer_min_max_tolerate_empty_ranks(comm):
    """An empty rank converted +-inf to an integer, raised on that rank
    alone, and left the others waiting in the Allreduce."""
    if comm.rank == 0:
        values = np.array([], dtype=np.int64)
    else:
        values = np.array([comm.rank, 10 * comm.rank], dtype=np.int64)
    lo = comm.reduce_min(values)
    hi = comm.reduce_max(values)
    if comm.size == 1:
        assert lo == np.iinfo(np.int64).max and hi == np.iinfo(np.int64).min
    else:
        assert lo == 1 and hi == 10 * (comm.size - 1)


# --- M4: sums keep complex parts and large integers ---


def test_sums_keep_imaginary_parts_and_large_integers(comm):
    big = 2**60 + 1  # not representable in double
    assert comm.sum(big) == comm.size * big
    assert comm.sum(np.int64(big)) == comm.size * big
    assert comm.sum(1 + 2j) == comm.size * (1 + 2j)
    assert comm.reduce_sum(np.array([1 + 2j, 3 - 1j])) == \
        comm.size * (4 + 1j)
    assert comm.reduce_sum(np.array([big], dtype=np.int64)) == \
        comm.size * big


# --- L6: the squared norm of a complex field is real ---


def test_complex_norm_sq_has_no_imaginary_part():
    """Summed conj products could leave a rounding imaginary part."""
    fc = muGrid.GlobalFieldCollection((7, 5), nb_ghosts_left=(1, 1),
                                      nb_ghosts_right=(1, 1))
    z = fc.complex_field("z", (3,))
    rng = np.random.default_rng(4)
    z.p[...] = rng.standard_normal(z.p.shape) + 1j * rng.standard_normal(
        z.p.shape)
    n = linalg.norm_sq(z)
    assert np.imag(n) == 0
    assert np.isclose(np.real(n), np.sum(np.abs(np.asarray(z.p)) ** 2))


# --- S8: float32 Jacobi, pipelined residual=, multigrid MPI guard ---


def _diagonal_system(dtype, n=(8, 8)):
    fc = muGrid.GlobalFieldCollection(n)
    d = np.linspace(1.0, 10.0, int(np.prod(n))).reshape(n)
    b = fc.real_field("b", dtype=dtype)
    x = fc.real_field("x", dtype=dtype)
    b.p[...] = 1.0

    def hessp(src, dst):
        dst.p[...] = (d * np.asarray(src.p)).astype(dtype)

    return fc, b, x, hessp, d


@pytest.mark.skipif(muGrid.Communicator().size > 1, reason="serial")
def test_jacobi_preconditioner_in_a_float32_solve():
    """The inverse diagonal used to be float64 always, so a float32 solve
    raised in linalg.scal."""
    from muGrid.Preconditioners import JacobiPreconditioner
    from muGrid.Solvers import conjugate_gradients

    fc, b, x, hessp, d = _diagonal_system(np.float32)
    prec = JacobiPreconditioner(d)
    conjugate_gradients(muGrid.Communicator(), fc, b, x, hessp=hessp,
                        prec=prec, rtol=1e-5, maxiter=20)
    np.testing.assert_allclose(np.asarray(x.p), 1.0 / d, rtol=1e-5)


@pytest.mark.skipif(muGrid.Communicator().size > 1, reason="serial")
def test_pipelined_cg_returns_the_residual():
    from muGrid.Solvers import conjugate_gradients_pipelined

    fc, b, x, hessp, d = _diagonal_system(np.float64)
    r = fc.real_field("r")
    conjugate_gradients_pipelined(muGrid.Communicator(), fc, b, x,
                                  hessp=hessp, rtol=1e-8, maxiter=200,
                                  residual=r)
    np.testing.assert_allclose(np.asarray(r.p),
                               1.0 - d * np.asarray(x.p), atol=1e-12)


def test_decomposition_default_subdivisions(comm):
    """Leaving out nb_subdivisions used to raise (the default was zeros)."""
    decomposition = muGrid.CartesianDecomposition(comm, [12, 8])
    assert int(np.prod(decomposition.nb_subdivisions)) == comm.size
    if comm.size > 1:
        from muGrid.Preconditioners import MultigridReferencePreconditioner

        # A parallel decomposition without a communicator argument used to
        # slip past the serial-only guard
        with pytest.raises(NotImplementedError):
            MultigridReferencePreconditioner(decomposition, (1.0, 1.0),
                                             1.0, 1.0)


# --- O9 / O7: float32 FEM wrapper; fourier() of non-scalar operators ---


def test_fem_gradient_wrapper_in_single_precision():
    op = muGrid.FEMGradientOperator(2, [0.5, 0.5], dtype=np.float32)
    fc = muGrid.GlobalFieldCollection((6, 6), sub_pts={"quad": op.nb_quad_pts},
                                      nb_ghosts_left=(1, 1),
                                      nb_ghosts_right=(1, 1))
    u = fc.real_field("u", dtype=np.float32)
    g = fc.real_field("g", (2,), "quad", dtype=np.float32)
    u.pg[...] = np.arange(8, dtype=np.float32)[:, None]  # u = x / h, slope 2
    op.apply(u, g)
    np.testing.assert_allclose(np.asarray(g.s)[0, :, :-1, :-1], 2.0, rtol=1e-6)


def test_fourier_refuses_non_scalar_operators():
    """Summing a gradient's components into one symbol mixed them."""
    op = muGrid.GenericLinearOperator([0, 0], np.ones((2, 1, 2, 2)))
    with pytest.raises(RuntimeError, match="scalar operators"):
        op.fourier(np.array([0.1, 0.2]))

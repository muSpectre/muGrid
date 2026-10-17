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

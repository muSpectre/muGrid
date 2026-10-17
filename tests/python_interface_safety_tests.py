"""
Silent precision and data loss at interfaces (audit findings S7, O3, M5).

Each of these used to return a plausible but wrong result without an error:
an in-place kernel writing into a converted temporary, an operator dropping
small coefficients by an absolute threshold, a NetCDF read converting types.
"""

import numpy as np
import pytest

import muGrid
from muGrid import linalg

serial_only = pytest.mark.skipif(
    muGrid.Communicator().size > 1, reason="test is serial")


# --- S7: in-place kernels must reject outputs they cannot write in place ---


def _green_symbol_args(n=4):
    from muGrid.Preconditioners import reference_stencil

    stencil = reference_stencil(2, (1.0 / n,) * 2, muGrid.FEMElement.q1,
                                1.3, 0.7).ravel()
    q = [np.arange(n // 2 + 1) / n, np.fft.fftfreq(n)]
    shape = [len(qd) for qd in q]
    return stencil, q, shape, int(np.prod(shape))


def test_green_symbol_writes_in_place():
    stencil, q, shape, nb_modes = _green_symbol_args()
    field = np.ones(2 * nb_modes, dtype=np.complex64)
    linalg.apply_green_symbol_2d_f32(field, 1, 2, stencil, q, shape, 1.0)
    assert not np.all(field == 1)  # the caller's array was modified


@pytest.mark.parametrize("bad", ["dtype", "strided"])
def test_green_symbol_rejects_an_output_it_would_copy(bad):
    """A complex128 or non-contiguous array used to be converted into a
    temporary that took the result, leaving the caller's array unchanged."""
    stencil, q, shape, nb_modes = _green_symbol_args()
    if bad == "dtype":
        field = np.ones(2 * nb_modes, dtype=np.complex128)
    else:
        field = np.ones(4 * nb_modes, dtype=np.complex64)[::2]
    before = field.copy()
    with pytest.raises(TypeError):
        linalg.apply_green_symbol_2d_f32(field, 1, 2, stencil, q, shape, 1.0)
    np.testing.assert_array_equal(field, before)


def test_green_symbol_rejects_read_only_and_out_of_range_strides():
    stencil, q, shape, nb_modes = _green_symbol_args()
    field = np.ones(2 * nb_modes, dtype=np.complex64)
    field.flags.writeable = False
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        linalg.apply_green_symbol_2d_f32(field, 1, 2, stencil, q, shape, 1.0)
    field = np.ones(2 * nb_modes, dtype=np.complex64)
    with pytest.raises(RuntimeError, match="beyond the field"):
        # component stride of a structure-of-arrays layout twice too large
        linalg.apply_green_symbol_2d_f32(field, 2 * nb_modes, 1, stencil, q,
                                         shape, 1.0)


def _block_thomas_args(nz=5, nb_modes=3, nb_head=2, nb_exc=1):
    c = np.complex128
    rng = np.random.default_rng(0)
    eye = np.broadcast_to(np.eye(2, dtype=c), (nb_modes, 2, 2))
    return dict(
        rhs=rng.standard_normal((nz, nb_modes, 2)).astype(c),
        head=np.broadcast_to(eye, (nb_head, nb_modes, 2, 2)).copy(),
        exc=np.broadcast_to(np.eye(2, dtype=c),
                            (nz, nb_exc, 2, 2)).copy(),
        exc_index=np.array([-1, 0, -1][:nb_modes], dtype=np.int32),
        A0=np.zeros((nb_modes, 2, 2), dtype=c),
        A2=np.zeros((nb_modes, 2, 2), dtype=c),
        y=np.zeros((nz, nb_modes, 2), dtype=c),
        out=np.zeros((nz, nb_modes, 2), dtype=c),
    )


def test_block_thomas_writes_in_place():
    args = _block_thomas_args()
    linalg.block_thomas_2d(**args)
    # Zero coupling and identity diagonal inverses: out == rhs
    np.testing.assert_allclose(args["out"], args["rhs"])


def test_block_thomas_rejects_an_output_it_would_copy():
    args = _block_thomas_args()
    args["out"] = args["out"].astype(np.complex64)
    with pytest.raises(TypeError):
        linalg.block_thomas_2d(**args)


@pytest.mark.parametrize("bad", ["slot", "no_head", "coupling"])
def test_block_thomas_rejects_inconsistent_shapes(bad):
    """These read outside the arrays they were given."""
    args = _block_thomas_args()
    if bad == "slot":
        args["exc_index"][1] = 1  # exc has one slot
    elif bad == "no_head":
        args["head"] = args["head"][:0]  # a negative slot read plane -1
    else:
        args["A2"] = args["A2"][:2]
    with pytest.raises(RuntimeError):
        linalg.block_thomas_2d(**args)


# --- O3: small coefficients are not dropped by an absolute threshold ---


@pytest.mark.parametrize("scale", [1.0, 1e-15, 1e-30])
def test_generic_operator_keeps_coefficients_at_any_scale(scale):
    """The identity times 1e-15 used to apply as zero: coefficients below an
    absolute 1e-14 were dropped, so an operator in SI units at the
    micrometre scale vanished."""
    op = muGrid.GenericLinearOperator([0, 0], np.full((1, 1, 1, 1), scale))
    fc = muGrid.GlobalFieldCollection((4, 4), sub_pts={"quad": 1})
    u = fc.real_field("u")
    out = fc.real_field("out", (1,), "quad")
    u.p[...] = np.arange(16.0).reshape(u.p.shape)
    op.apply(u, out)
    np.testing.assert_allclose(np.asarray(out.p).ravel(),
                               scale * np.asarray(u.p).ravel(), rtol=1e-14)


# --- M5: NetCDF reads refuse a type they would convert ---


@serial_only
def test_netcdf_read_refuses_a_different_stored_type(tmp_path):
    """A float64 variable read into a float32 field came back rounded
    without an error. Widening reads (float32 into float64) stay allowed;
    test_FileIONetCDF_dtype_conversion covers them."""
    comm = muGrid.Communicator()
    fname = str(tmp_path / "m5.nc")
    fc = muGrid.GlobalFieldCollection((4, 4))
    f = fc.real_field("u")
    f.p[...] = 1.0 / 3.0
    writer = muGrid.FileIONetCDF(fname, muGrid.OpenMode.Write, comm)
    writer.register_field_collection(fc)
    writer.append_frame().write()
    writer.close()

    same = muGrid.GlobalFieldCollection((4, 4))
    g = same.real_field("u")
    reader = muGrid.FileIONetCDF(fname, muGrid.OpenMode.Read, comm)
    reader.register_field_collection(same)
    reader.read(0)
    reader.close()
    np.testing.assert_array_equal(np.asarray(g.p), 1.0 / 3.0)

    narrow = muGrid.GlobalFieldCollection((4, 4))
    narrow.real_field("u", dtype=np.float32)
    reader = muGrid.FileIONetCDF(fname, muGrid.OpenMode.Read, comm)
    with pytest.raises(Exception, match="stored as NC_DOUBLE"):
        reader.register_field_collection(narrow)

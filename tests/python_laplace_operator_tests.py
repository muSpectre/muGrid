#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file    python_laplace_operator_tests.py

@author  Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date    05 Jun 2026

@brief   Functional tests for the hard-coded LaplaceOperator (2D/3D)

Copyright © 2026 Lars Pastewka

µGrid is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µGrid is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with µGrid; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.

Additional permission under GNU GPL version 3 section 7

If you modify this Program, or any covered work, by linking or combining it
with proprietary FFT implementations or numerical libraries, containing parts
covered by the terms of those libraries' licenses, the licensors of this
Program grant you additional permission to convey the resulting work.

------------------------------------------------------------------------------

These tests exercise the *hard-coded* ``LaplaceOperator`` (``LaplaceOperator2D``
/ ``LaplaceOperator3D`` in C++), which uses an optimized fixed stencil rather
than the generic convolution code path. The tests are functional rather than
smoke tests: each one checks the numerically correct result against either an
analytical reference or an independent implementation
(``GenericLinearOperator`` with the equivalent finite-difference stencil).

Definitions used throughout:

  * The 2D discrete Laplacian is the 5-point stencil
        [[0,  1, 0],
         [1, -4, 1],
         [0,  1, 0]]
    with offset (-1, -1).
  * The 3D discrete Laplacian is the 7-point stencil: centre -6, the six
    face-neighbours +1, offset (-1, -1, -1).
  * For a discrete plane wave  f(x) = cos(2π (Σ_d p_d x_d / N_d))  sampled on an
    integer grid that is periodic with period N_d, the stencils above have f as
    an exact eigenfunction with eigenvalue
        λ = Σ_d (2 cos(2π p_d / N_d)) - 2·dim .
    This follows from cos(a(x±1)) summing to 2 cos(a) cos(ax) so the odd (sin)
    parts cancel. We use this as the analytical ground truth.
"""

import numpy as np
import pytest
from conftest import (
    create_device,
    get_test_devices,
    skip_if_gpu_unavailable,
)

import muGrid

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# 2D and 3D finite-difference Laplacian stencils, used to build an independent
# reference operator (GenericLinearOperator).
STENCIL_2D = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])


def _stencil_3d():
    stencil = np.zeros((3, 3, 3))
    stencil[1, 1, 1] = -6.0
    for axis in range(3):
        for shift in (0, 2):
            idx = [1, 1, 1]
            idx[axis] = shift
            stencil[tuple(idx)] = 1.0
    return stencil


def _make_collection(nb_grid_pts):
    """Global collection with one ghost layer on each side (for the 3x3[x3]
    centred Laplace stencil)."""
    nb_ghosts = tuple(1 for _ in nb_grid_pts)
    return (
        muGrid.GlobalFieldCollection(
            list(nb_grid_pts),
            nb_ghosts_left=nb_ghosts,
            nb_ghosts_right=nb_ghosts,
        ),
        nb_ghosts,
    )


def _set_periodic(field, values, nb_ghosts):
    """Write ``values`` (physical-pixel shaped) into ``field`` and fill the ghost
    layers assuming periodic boundary conditions."""
    field.p[...] = values
    field.pg[...] = np.pad(field.p, tuple(zip(nb_ghosts, nb_ghosts)), mode="wrap")


def _coords(nb_grid_pts):
    axes = [np.arange(n) for n in nb_grid_pts]
    return np.meshgrid(*axes, indexing="ij")


def _plane_wave(nb_grid_pts, frequencies):
    """Discrete plane wave cos(2π Σ p_d x_d / N_d) and its analytical Laplacian
    eigenvalue."""
    coords = _coords(nb_grid_pts)
    phase = sum(
        freq * coord / n
        for freq, coord, n in zip(frequencies, coords, nb_grid_pts)
    )
    field = np.cos(2.0 * np.pi * phase)
    eigenvalue = sum(
        2.0 * np.cos(2.0 * np.pi * freq / n)
        for freq, n in zip(frequencies, nb_grid_pts)
    ) - 2.0 * len(nb_grid_pts)
    return field, eigenvalue


def _apply(op, collection, nb_ghosts, input_values):
    """Convenience: set a periodic input field, apply ``op``, return output
    physical pixels."""
    ifield = collection.real_field("in")
    ofield = collection.real_field("out")
    _set_periodic(ifield, input_values, nb_ghosts)
    op.apply(ifield, ofield)
    return np.array(ofield.p)


# ===========================================================================
# 2D functional tests
# ===========================================================================


class TestLaplaceOperator2D:
    def test_constant_field_is_annihilated(self):
        """Laplacian of a constant field is exactly zero (stencil sums to 0)."""
        nb_grid_pts = (6, 5)
        fc, nb_ghosts = _make_collection(nb_grid_pts)
        op = muGrid.LaplaceOperator(2)
        out = _apply(op, fc, nb_ghosts, np.full(nb_grid_pts, 3.7))
        np.testing.assert_allclose(out, 0.0, atol=1e-12)

    def test_impulse_response_is_the_stencil(self):
        """A unit impulse on a 3x3 periodic grid reproduces the 5-point stencil
        (the Laplacian is symmetric, so the impulse response equals the
        stencil)."""
        nb_grid_pts = (3, 3)
        fc, nb_ghosts = _make_collection(nb_grid_pts)
        impulse = np.zeros(nb_grid_pts)
        impulse[1, 1] = 1.0
        op = muGrid.LaplaceOperator(2)
        out = _apply(op, fc, nb_ghosts, impulse)
        np.testing.assert_allclose(out, STENCIL_2D)

    @pytest.mark.parametrize("frequencies", [(1, 0), (0, 1), (1, 2), (3, 2)])
    def test_plane_wave_eigenfunction(self, frequencies):
        """The operator reproduces the analytical eigenvalue on a discrete plane
        wave: L f = (2cosθx + 2cosθy - 4) f."""
        nb_grid_pts = (8, 6)
        fc, nb_ghosts = _make_collection(nb_grid_pts)
        field, eigenvalue = _plane_wave(nb_grid_pts, frequencies)
        op = muGrid.LaplaceOperator(2)
        out = _apply(op, fc, nb_ghosts, field)
        np.testing.assert_allclose(out, eigenvalue * field, atol=1e-12)

    def test_matches_generic_convolution_operator(self):
        """The hard-coded operator must agree bit-for-(numerically)-bit with the
        generic convolution operator built from the same stencil."""
        nb_grid_pts = (7, 9)
        rng = np.random.default_rng(12345)
        values = rng.standard_normal(nb_grid_pts)

        fc_hard, nb_ghosts = _make_collection(nb_grid_pts)
        out_hard = _apply(muGrid.LaplaceOperator(2), fc_hard, nb_ghosts, values)

        fc_gen, _ = _make_collection(nb_grid_pts)
        gen = muGrid.GenericLinearOperator([-1, -1], STENCIL_2D)
        out_gen = _apply(gen, fc_gen, nb_ghosts, values)

        np.testing.assert_allclose(out_hard, out_gen, atol=1e-13)

    @pytest.mark.parametrize("scale", [-1.0, 0.5, 2.0, 3.3])
    def test_scale_factor(self, scale):
        """Output scales linearly with the ``scale`` constructor argument."""
        nb_grid_pts = (6, 6)
        rng = np.random.default_rng(7)
        values = rng.standard_normal(nb_grid_pts)

        fc1, nb_ghosts = _make_collection(nb_grid_pts)
        unscaled = _apply(muGrid.LaplaceOperator(2, scale=1.0), fc1, nb_ghosts, values)

        fc2, _ = _make_collection(nb_grid_pts)
        scaled = _apply(muGrid.LaplaceOperator(2, scale=scale), fc2, nb_ghosts, values)

        np.testing.assert_allclose(scaled, scale * unscaled, atol=1e-13)

    def test_linearity(self):
        """L(a·u + b·v) == a·L(u) + b·L(v)."""
        nb_grid_pts = (8, 8)
        rng = np.random.default_rng(99)
        u = rng.standard_normal(nb_grid_pts)
        v = rng.standard_normal(nb_grid_pts)
        a, b = 2.5, -1.75
        op = muGrid.LaplaceOperator(2)

        fc_u, nb_ghosts = _make_collection(nb_grid_pts)
        lu = _apply(op, fc_u, nb_ghosts, u)
        fc_v, _ = _make_collection(nb_grid_pts)
        lv = _apply(op, fc_v, nb_ghosts, v)
        fc_comb, _ = _make_collection(nb_grid_pts)
        l_comb = _apply(op, fc_comb, nb_ghosts, a * u + b * v)

        np.testing.assert_allclose(l_comb, a * lu + b * lv, atol=1e-12)

    def test_self_adjoint(self):
        """The discrete Laplacian is symmetric: <u, L v> == <L u, v> under
        periodic boundary conditions."""
        nb_grid_pts = (9, 7)
        rng = np.random.default_rng(2024)
        u = rng.standard_normal(nb_grid_pts)
        v = rng.standard_normal(nb_grid_pts)
        op = muGrid.LaplaceOperator(2)

        fc_u, nb_ghosts = _make_collection(nb_grid_pts)
        lu = _apply(op, fc_u, nb_ghosts, u)
        fc_v, _ = _make_collection(nb_grid_pts)
        lv = _apply(op, fc_v, nb_ghosts, v)

        np.testing.assert_allclose(np.sum(u * lv), np.sum(v * lu), rtol=1e-12)

    def test_transpose_equals_apply(self):
        """For the self-adjoint Laplacian, transpose() must equal apply()."""
        nb_grid_pts = (6, 7)
        rng = np.random.default_rng(555)
        values = rng.standard_normal(nb_grid_pts)
        op = muGrid.LaplaceOperator(2)

        fc, nb_ghosts = _make_collection(nb_grid_pts)
        ifield = fc.real_field("in")
        applied = fc.real_field("applied")
        transposed = fc.real_field("transposed")
        _set_periodic(ifield, values, nb_ghosts)

        op.apply(ifield, applied)
        op.transpose(ifield, transposed)

        np.testing.assert_allclose(transposed.p, applied.p, atol=1e-14)

    @pytest.mark.parametrize("alpha", [1.0, -2.0, 0.25])
    def test_apply_increment(self, alpha):
        """apply_increment computes out += alpha · L(in) without clobbering the
        existing contents of the output field."""
        nb_grid_pts = (7, 6)
        rng = np.random.default_rng(31415)
        values = rng.standard_normal(nb_grid_pts)
        base = rng.standard_normal(nb_grid_pts)
        op = muGrid.LaplaceOperator(2)

        fc, nb_ghosts = _make_collection(nb_grid_pts)
        # Reference Laplacian via plain apply.
        ref = _apply(op, fc, nb_ghosts, values)

        fc2, _ = _make_collection(nb_grid_pts)
        ifield = fc2.real_field("in")
        ofield = fc2.real_field("out")
        _set_periodic(ifield, values, nb_ghosts)
        ofield.p[...] = base
        op.apply_increment(ifield, alpha, ofield)

        np.testing.assert_allclose(ofield.p, base + alpha * ref, atol=1e-12)


# ===========================================================================
# 3D functional tests
# ===========================================================================


class TestLaplaceOperator3D:
    def test_constant_field_is_annihilated(self):
        nb_grid_pts = (4, 5, 6)
        fc, nb_ghosts = _make_collection(nb_grid_pts)
        op = muGrid.LaplaceOperator(3)
        out = _apply(op, fc, nb_ghosts, np.full(nb_grid_pts, -2.25))
        np.testing.assert_allclose(out, 0.0, atol=1e-12)

    @pytest.mark.parametrize("frequencies", [(1, 0, 0), (0, 1, 1), (2, 1, 3)])
    def test_plane_wave_eigenfunction(self, frequencies):
        """L f = (2cosθx + 2cosθy + 2cosθz - 6) f for a discrete 3D plane
        wave."""
        nb_grid_pts = (6, 8, 5)
        fc, nb_ghosts = _make_collection(nb_grid_pts)
        field, eigenvalue = _plane_wave(nb_grid_pts, frequencies)
        op = muGrid.LaplaceOperator(3)
        out = _apply(op, fc, nb_ghosts, field)
        np.testing.assert_allclose(out, eigenvalue * field, atol=1e-12)

    def test_matches_generic_convolution_operator(self):
        nb_grid_pts = (5, 6, 4)
        rng = np.random.default_rng(20260605)
        values = rng.standard_normal(nb_grid_pts)

        fc_hard, nb_ghosts = _make_collection(nb_grid_pts)
        out_hard = _apply(muGrid.LaplaceOperator(3), fc_hard, nb_ghosts, values)

        fc_gen, _ = _make_collection(nb_grid_pts)
        gen = muGrid.GenericLinearOperator([-1, -1, -1], _stencil_3d())
        out_gen = _apply(gen, fc_gen, nb_ghosts, values)

        np.testing.assert_allclose(out_hard, out_gen, atol=1e-13)

    @pytest.mark.parametrize("scale", [-1.0, 0.5, 2.0])
    def test_scale_factor(self, scale):
        nb_grid_pts = (5, 5, 5)
        rng = np.random.default_rng(8)
        values = rng.standard_normal(nb_grid_pts)

        fc1, nb_ghosts = _make_collection(nb_grid_pts)
        unscaled = _apply(muGrid.LaplaceOperator(3, scale=1.0), fc1, nb_ghosts, values)

        fc2, _ = _make_collection(nb_grid_pts)
        scaled = _apply(muGrid.LaplaceOperator(3, scale=scale), fc2, nb_ghosts, values)

        np.testing.assert_allclose(scaled, scale * unscaled, atol=1e-13)

    def test_self_adjoint(self):
        nb_grid_pts = (5, 6, 7)
        rng = np.random.default_rng(11)
        u = rng.standard_normal(nb_grid_pts)
        v = rng.standard_normal(nb_grid_pts)
        op = muGrid.LaplaceOperator(3)

        fc_u, nb_ghosts = _make_collection(nb_grid_pts)
        lu = _apply(op, fc_u, nb_ghosts, u)
        fc_v, _ = _make_collection(nb_grid_pts)
        lv = _apply(op, fc_v, nb_ghosts, v)

        np.testing.assert_allclose(np.sum(u * lv), np.sum(v * lu), rtol=1e-12)


# ===========================================================================
# Input validation
# ===========================================================================


def test_invalid_spatial_dimension():
    """Only 2D and 3D Laplace operators exist."""
    with pytest.raises(ValueError):
        muGrid.LaplaceOperator(4)


def test_rejects_multicomponent_field():
    """The Laplace kernels address one scalar per pixel; a multi-component
    field would silently mix components with spatial neighbours, so it must
    be rejected."""
    fc, _ = _make_collection((5, 4))
    op = muGrid.LaplaceOperator(2)
    inp = fc.real_field("in", (2,))
    out = fc.real_field("out", (2,))
    with pytest.raises(RuntimeError):
        op.apply(inp, out)


def test_rejects_multi_subpoint_field():
    """Same as above for sub-points: only one degree of freedom per pixel is
    supported."""
    fc = muGrid.GlobalFieldCollection(
        [5, 4],
        nb_ghosts_left=(1, 1),
        nb_ghosts_right=(1, 1),
        sub_pts={"quad": 2},
    )
    op = muGrid.LaplaceOperator(2)
    inp = fc.real_field("in", (), "quad")
    out = fc.real_field("out", (), "quad")
    with pytest.raises(RuntimeError):
        op.apply(inp, out)


def test_rejects_transpose_weights():
    """The Laplacian has no quadrature points; passing weights to
    transpose() must raise instead of being silently ignored."""
    fc, _ = _make_collection((5, 4))
    op = muGrid.LaplaceOperator(2)
    inp = fc.real_field("in")
    out = fc.real_field("out")
    # no weights (or empty weights) is fine
    op.transpose(inp, out)
    with pytest.raises(RuntimeError):
        op.transpose(inp, out, [1.0])


# ===========================================================================
# Anisotropic grid spacing
# ===========================================================================


def _anisotropic_stencil(grid_spacing, scale):
    """Finite-difference Laplacian of an anisotropic grid as a 3^dim stencil:
    scale / h_d^2 on the two neighbours along axis d, minus twice the sum of
    those on the centre."""
    dim = len(grid_spacing)
    stencil = np.zeros((3,) * dim)
    centre = (1,) * dim
    for axis, h in enumerate(grid_spacing):
        w = scale / h**2
        for shift in (0, 2):
            idx = list(centre)
            idx[axis] = shift
            stencil[tuple(idx)] = w
        stencil[centre] -= 2.0 * w
    return stencil


@pytest.mark.parametrize("grid_spacing,scale", [
    ((0.5, 2.0), -1.0), ((0.1, 0.1), 1.0), ((1.5, 0.25), 2.0),
    ((0.5, 2.0, 0.25), -1.0), ((0.2, 0.2, 0.2), 1.0), ((3.0, 1.0, 0.5), -0.5),
])
def test_anisotropic_matches_generic_operator(grid_spacing, scale):
    """With a grid spacing the hard-coded operator is the anisotropic
    finite-difference Laplacian, which the generic operator built from the
    per-axis stencil reproduces independently (it used to be unavailable:
    one scale served every axis)."""
    dim = len(grid_spacing)
    nb_grid_pts = (7, 9, 5)[:dim]
    rng = np.random.default_rng(42)
    values = rng.standard_normal(nb_grid_pts)

    fc_hard, nb_ghosts = _make_collection(nb_grid_pts)
    op = muGrid.LaplaceOperator(dim, scale, grid_spacing=grid_spacing)
    out_hard = _apply(op, fc_hard, nb_ghosts, values)

    fc_gen, _ = _make_collection(nb_grid_pts)
    gen = muGrid.GenericLinearOperator(
        [-1] * dim, _anisotropic_stencil(grid_spacing, scale))
    out_gen = _apply(gen, fc_gen, nb_ghosts, values)

    scale_ref = np.abs(out_gen).max()
    np.testing.assert_allclose(out_hard, out_gen, atol=1e-13 * scale_ref)
    # the reported coefficients are the same stencil
    np.testing.assert_allclose(
        np.asarray(op.coefficients).reshape((3,) * dim),
        _anisotropic_stencil(grid_spacing, scale), atol=1e-13 * scale_ref)
    np.testing.assert_allclose(op.grid_spacing, grid_spacing)
    assert op.scale == scale


@pytest.mark.parametrize("dim", [2, 3])
def test_anisotropic_plane_wave_eigenvalue(dim):
    """A discrete plane wave is an eigenfunction with eigenvalue
    sum_d 2 (cos(2 pi p_d / N_d) - 1) / h_d^2."""
    nb_grid_pts = (8, 6, 10)[:dim]
    grid_spacing = (0.5, 1.5, 0.25)[:dim]
    frequencies = (1, 2, 3)[:dim]
    fc, nb_ghosts = _make_collection(nb_grid_pts)
    field, _ = _plane_wave(nb_grid_pts, frequencies)
    eigenvalue = sum(
        2.0 * (np.cos(2.0 * np.pi * p / n) - 1.0) / h**2
        for p, n, h in zip(frequencies, nb_grid_pts, grid_spacing))
    op = muGrid.LaplaceOperator(dim, grid_spacing=grid_spacing)
    out = _apply(op, fc, nb_ghosts, field)
    np.testing.assert_allclose(out, eigenvalue * field,
                               atol=1e-12 * abs(eigenvalue))


def test_unit_spacing_equals_scale_only():
    """grid_spacing=(1, ..., 1) is the scale-only operator, so existing
    callers that fold 1/h^2 into the scale are unchanged."""
    nb_grid_pts = (6, 5, 4)
    rng = np.random.default_rng(3)
    values = rng.standard_normal(nb_grid_pts)
    fc_a, nb_ghosts = _make_collection(nb_grid_pts)
    fc_b, _ = _make_collection(nb_grid_pts)
    a = _apply(muGrid.LaplaceOperator(3, -2.5), fc_a, nb_ghosts, values)
    b = _apply(muGrid.LaplaceOperator(3, -2.5, grid_spacing=(1.0,) * 3),
               fc_b, nb_ghosts, values)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(muGrid.LaplaceOperator(3, -2.5).grid_spacing,
                               [1.0, 1.0, 1.0])


@pytest.mark.parametrize("dim", [2, 3])
def test_anisotropic_is_self_adjoint(dim):
    """<L u, v> == <u, L v> on the periodic grid, and transpose == apply."""
    nb_grid_pts = (6, 7, 5)[:dim]
    grid_spacing = (0.5, 2.0, 0.75)[:dim]
    rng = np.random.default_rng(11)
    u = rng.standard_normal(nb_grid_pts)
    v = rng.standard_normal(nb_grid_pts)
    op = muGrid.LaplaceOperator(dim, -1.0, grid_spacing=grid_spacing)
    fc, nb_ghosts = _make_collection(nb_grid_pts)
    Lu = _apply(op, fc, nb_ghosts, u)
    fc2, _ = _make_collection(nb_grid_pts)
    Lv = _apply(op, fc2, nb_ghosts, v)
    assert np.isclose(np.vdot(Lu, v), np.vdot(u, Lv), rtol=1e-12)
    fc3, _ = _make_collection(nb_grid_pts)
    ifield = fc3.real_field("in")
    ofield = fc3.real_field("out")
    _set_periodic(ifield, u, nb_ghosts)
    op.transpose(ifield, ofield)
    np.testing.assert_array_equal(np.asarray(ofield.p), Lu)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("device", get_test_devices())
def test_anisotropic_on_device_and_in_single_precision(device, dtype, dim):
    """The device kernels and the float32 instantiations take the per-axis
    weights too; compared against the host float64 result."""
    skip_if_gpu_unavailable(device)
    nb_grid_pts = (12, 9, 7)[:dim]
    grid_spacing = (0.5, 2.0, 0.25)[:dim]
    rng = np.random.default_rng(5)
    values = rng.standard_normal(nb_grid_pts)
    fc_ref, nb_ghosts = _make_collection(nb_grid_pts)
    op = muGrid.LaplaceOperator(dim, -1.0, grid_spacing=grid_spacing)
    ref = _apply(op, fc_ref, nb_ghosts, values)

    device_obj = create_device(device)
    kwargs = {"device": device_obj} if device_obj else {}
    fc = muGrid.GlobalFieldCollection(
        list(nb_grid_pts), nb_ghosts_left=nb_ghosts,
        nb_ghosts_right=nb_ghosts, **kwargs)
    ifield = fc.real_field("in", dtype=dtype)
    ofield = fc.real_field("out", dtype=dtype)
    padded = np.pad(values, tuple(zip(nb_ghosts, nb_ghosts)), mode="wrap")
    if device_obj:
        import cupy

        ifield.pg[...] = cupy.asarray(padded.astype(dtype))
    else:
        ifield.pg[...] = padded.astype(dtype)
    op.apply(ifield, ofield)
    out = ofield.p
    out = out.get() if hasattr(out, "get") else np.asarray(out)
    rtol = 1e-5 if dtype == np.float32 else 1e-13
    np.testing.assert_allclose(out, ref, rtol=rtol, atol=rtol * np.abs(ref).max())
    # increment adds alpha times the anisotropic Laplacian
    op.apply_increment(ifield, 0.5, ofield)
    out = ofield.p
    out = out.get() if hasattr(out, "get") else np.asarray(out)
    np.testing.assert_allclose(out, 1.5 * ref, rtol=rtol,
                               atol=rtol * np.abs(ref).max())


@pytest.mark.parametrize("bad", [(0.5,), (0.5, 1.0, 2.0), (0.5, 0.0),
                                 (0.5, -1.0), (0.5, float("nan"))])
def test_rejects_a_malformed_grid_spacing(bad):
    """One positive spacing per axis, or the constructor raises."""
    with pytest.raises(RuntimeError):
        muGrid.LaplaceOperator(2, grid_spacing=bad)

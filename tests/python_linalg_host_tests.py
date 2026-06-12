#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file    python_linalg_host_tests.py

@author  Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date    06 Jun 2026

@brief   Functional tests for the host (CPU) linalg operations

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

These tests exercise the CPU implementations in ``src/libmugrid/linalg/
linalg_host.cc``. The existing ``python_linalg_tests.py`` only covers the Real
``norm_sq``/``vecdot``/``axpy_norm_sq`` reductions in 2D; here we add:

  * the update operations ``axpy``, ``scal``, ``axpby`` and ``copy`` (which
    operate on the FULL buffer, ghosts included),
  * the Complex specialisations of every operation (sesquilinear ``vecdot``
    using conj(a)*b, and the real-valued ``norm_sq``),
  * the 1D and 3D ghost-region code paths of the reductions, and
  * the input-validation error branches.

Reductions (``vecdot``/``norm_sq``/``axpy_norm_sq``) are defined over the
INTERIOR only (ghosts excluded) for MPI correctness; updates operate on the
full buffer. We verify both behaviours explicitly by filling the ghost layer
with non-trivial values via the ``.sg`` (sub-point, ghost-inclusive) view and
comparing against numpy references computed on ``.s`` (interior) or ``.sg``
(full) as appropriate.
"""

import numpy as np
import pytest

import muGrid
from muGrid import linalg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collection(nb_grid_pts, nb_ghosts=1):
    g = (nb_ghosts,) * len(nb_grid_pts)
    return muGrid.GlobalFieldCollection(
        list(nb_grid_pts), nb_ghosts_left=g, nb_ghosts_right=g
    )


def _make(fc, name, components=(2,), complex_=False):
    if complex_:
        return fc.complex_field(name, components)
    return fc.real_field(name, components)


def _fill(field, rng, complex_=False):
    """Fill the full buffer (including ghosts) with random values."""
    shape = field.sg.shape
    data = rng.standard_normal(shape)
    if complex_:
        data = data + 1j * rng.standard_normal(shape)
    field.sg[...] = data
    return np.array(field.sg)  # snapshot of the full buffer


@pytest.fixture
def rng():
    return np.random.default_rng(2026_06_06)


DIMS = [(7,), (5, 6), (4, 5, 3)]
DIM_IDS = ["1d", "2d", "3d"]


# ===========================================================================
# Reductions: vecdot / norm_sq  (interior only)  — Real and Complex, 1D/2D/3D
# ===========================================================================

class TestReductions:
    @pytest.mark.parametrize("nb_grid_pts", DIMS, ids=DIM_IDS)
    @pytest.mark.parametrize("complex_", [False, True], ids=["real", "complex"])
    def test_vecdot_excludes_ghosts(self, nb_grid_pts, complex_, rng):
        fc = _collection(nb_grid_pts)
        a = _make(fc, "a", complex_=complex_)
        b = _make(fc, "b", complex_=complex_)
        _fill(a, rng, complex_)
        _fill(b, rng, complex_)

        # Reference over the interior only. vecdot is sesquilinear: conj(a)*b.
        ref = np.sum(np.conj(np.array(a.s)) * np.array(b.s))
        result = linalg.vecdot(a, b)

        if complex_:
            assert result == pytest.approx(ref, rel=1e-12)
        else:
            assert result == pytest.approx(float(ref.real), rel=1e-12)

    @pytest.mark.parametrize("nb_grid_pts", DIMS, ids=DIM_IDS)
    @pytest.mark.parametrize("complex_", [False, True], ids=["real", "complex"])
    def test_norm_sq_excludes_ghosts(self, nb_grid_pts, complex_, rng):
        fc = _collection(nb_grid_pts)
        x = _make(fc, "x", complex_=complex_)
        _fill(x, rng, complex_)

        ref = np.sum(np.abs(np.array(x.s)) ** 2)  # interior only, real-valued
        result = linalg.norm_sq(x)

        # norm_sq returns the real squared magnitude (Complex variant carries a
        # ~0 imaginary part).
        assert complex(result).real == pytest.approx(ref, rel=1e-12)
        assert complex(result).imag == pytest.approx(0.0, abs=1e-9)

    def test_norm_sq_equals_vecdot_self(self, rng):
        fc = _collection((5, 6))
        x = _make(fc, "x")
        _fill(x, rng)
        assert linalg.norm_sq(x) == pytest.approx(linalg.vecdot(x, x), rel=1e-12)

    @pytest.mark.parametrize("nb_grid_pts", DIMS, ids=DIM_IDS)
    def test_norm_sq_asymmetric_ghosts(self, nb_grid_pts, rng):
        # Independent left/right ghost widths exercise the ghost-bound
        # arithmetic in every dimension.
        dim = len(nb_grid_pts)
        left = tuple(range(1, dim + 1))            # (1,), (1,2), (1,2,3)
        right = tuple(range(dim, 0, -1))           # (1,), (2,1), (3,2,1)
        fc = muGrid.GlobalFieldCollection(
            list(nb_grid_pts), nb_ghosts_left=left, nb_ghosts_right=right
        )
        x = _make(fc, "x")
        _fill(x, rng)
        ref = np.sum(np.array(x.s) ** 2)
        assert linalg.norm_sq(x) == pytest.approx(ref, rel=1e-12)


# ===========================================================================
# Updates: axpy / scal / axpby / copy  (FULL buffer, ghosts included)
# ===========================================================================

class TestUpdates:
    @pytest.mark.parametrize("complex_", [False, True], ids=["real", "complex"])
    def test_axpy_full_buffer(self, complex_, rng):
        fc = _collection((5, 6))
        x = _make(fc, "x", complex_=complex_)
        y = _make(fc, "y", complex_=complex_)
        x0 = _fill(x, rng, complex_)
        y0 = _fill(y, rng, complex_)
        alpha = (1.5 - 0.5j) if complex_ else 1.5

        linalg.axpy(alpha, x, y)
        # axpy touches the FULL buffer (ghosts too).
        np.testing.assert_allclose(np.array(y.sg), alpha * x0 + y0, rtol=1e-12)

    @pytest.mark.parametrize("complex_", [False, True], ids=["real", "complex"])
    def test_scal_full_buffer(self, complex_, rng):
        fc = _collection((5, 6))
        x = _make(fc, "x", complex_=complex_)
        x0 = _fill(x, rng, complex_)
        alpha = (-2.0 + 0.25j) if complex_ else -2.0

        linalg.scal(alpha, x)
        np.testing.assert_allclose(np.array(x.sg), alpha * x0, rtol=1e-12)

    @pytest.mark.parametrize("complex_", [False, True], ids=["real", "complex"])
    def test_axpby_full_buffer(self, complex_, rng):
        fc = _collection((5, 6))
        x = _make(fc, "x", complex_=complex_)
        y = _make(fc, "y", complex_=complex_)
        x0 = _fill(x, rng, complex_)
        y0 = _fill(y, rng, complex_)
        alpha = (0.75 + 1j) if complex_ else 0.75
        beta = (-0.5 - 0.25j) if complex_ else -0.5

        linalg.axpby(alpha, x, beta, y)
        np.testing.assert_allclose(
            np.array(y.sg), alpha * x0 + beta * y0, rtol=1e-12
        )

    @pytest.mark.parametrize("complex_", [False, True], ids=["real", "complex"])
    def test_copy_full_buffer(self, complex_, rng):
        fc = _collection((5, 6))
        src = _make(fc, "src", complex_=complex_)
        dst = _make(fc, "dst", complex_=complex_)
        src0 = _fill(src, rng, complex_)
        _fill(dst, rng, complex_)

        linalg.copy(src, dst)
        np.testing.assert_allclose(np.array(dst.sg), src0, rtol=1e-12)

    def test_scal_touches_ghosts(self, rng):
        """Regression guard: scal must scale ghost values too (full buffer)."""
        fc = _collection((4, 4))
        x = _make(fc, "x")
        x0 = _fill(x, rng)
        linalg.scal(3.0, x)
        full = np.array(x.sg)
        interior_slices = tuple(
            slice(o, o + s) for o, s in zip(x._cpp.offsets_s, x._cpp.shape_s)
        )
        # At least one ghost entry exists and was scaled.
        mask = np.ones(full.shape, dtype=bool)
        mask[interior_slices] = False
        assert mask.any()
        np.testing.assert_allclose(full[mask], 3.0 * x0[mask], rtol=1e-12)


# ===========================================================================
# Fused axpy_norm_sq:  y = alpha*x + y (full),  returns ||y||^2 (interior)
# ===========================================================================

class TestAxpyNormSq:
    @pytest.mark.parametrize("nb_grid_pts", DIMS, ids=DIM_IDS)
    @pytest.mark.parametrize("complex_", [False, True], ids=["real", "complex"])
    def test_axpy_norm_sq(self, nb_grid_pts, complex_, rng):
        fc = _collection(nb_grid_pts)
        x = _make(fc, "x", complex_=complex_)
        y = _make(fc, "y", complex_=complex_)
        x0 = _fill(x, rng, complex_)
        y0 = _fill(y, rng, complex_)
        alpha = (0.5 + 0.5j) if complex_ else 0.5

        result = linalg.axpy_norm_sq(alpha, x, y)

        # y is updated on the full buffer ...
        np.testing.assert_allclose(np.array(y.sg), alpha * x0 + y0, rtol=1e-12)
        # ... and the returned norm is over the interior only.
        ref = np.sum(np.abs(np.array(y.s)) ** 2)
        assert complex(result).real == pytest.approx(ref, rel=1e-12)


# ===========================================================================
# Input-validation error branches
# ===========================================================================

class TestErrors:
    def test_vecdot_different_collection(self, rng):
        fc1 = _collection((4, 4))
        fc2 = _collection((4, 4))
        a = _make(fc1, "a")
        b = _make(fc2, "b")
        with pytest.raises(RuntimeError):
            linalg.vecdot(a, b)

    def test_vecdot_component_mismatch(self, rng):
        fc = _collection((4, 4))
        a = fc.real_field("a", (2,))
        b = fc.real_field("b", (3,))
        with pytest.raises(RuntimeError):
            linalg.vecdot(a, b)

    def test_vecdot_subpt_mismatch(self):
        fc = muGrid.GlobalFieldCollection(
            [4, 4], nb_ghosts_left=(1, 1), nb_ghosts_right=(1, 1),
            sub_pts={"quad": 2},
        )
        nodal = fc.real_field("nodal", (1,))
        quad = fc.real_field("quad", (1,), "quad")
        with pytest.raises(RuntimeError):
            linalg.vecdot(nodal, quad)

    def test_axpy_component_mismatch(self):
        fc = _collection((4, 4))
        x = fc.real_field("x", (2,))
        y = fc.real_field("y", (3,))
        with pytest.raises(RuntimeError):
            linalg.axpy(1.0, x, y)

    def test_copy_component_mismatch(self):
        fc = _collection((4, 4))
        src = fc.real_field("src", (2,))
        dst = fc.real_field("dst", (3,))
        with pytest.raises(RuntimeError):
            linalg.copy(src, dst)

    def test_axpby_different_collection(self):
        fc1 = _collection((4, 4))
        fc2 = _collection((4, 4))
        x = fc1.real_field("x", (2,))
        y = fc2.real_field("y", (2,))
        with pytest.raises(RuntimeError):
            linalg.axpby(1.0, x, 1.0, y)


# ===========================================================================
# GPU validation regression tests
# ===========================================================================
#
# These mirror the host component-mismatch tests on the device. The guard is
# memory-safety critical on the GPU: without it the element count `n` is derived
# from the first operand's component count and the kernel indexes BOTH operands
# up to `n`, reading/writing past the shorter device buffer. The validation must
# raise (FieldError -> RuntimeError) before any kernel launch, so these tests
# need a GPU at runtime but not CuPy (no array data is touched).

GPU_AVAILABLE = muGrid.has_gpu


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU backend not compiled in")
class TestGPUValidation:
    def setup_method(self, method):
        if not muGrid.is_gpu_available():
            pytest.skip("No GPU device available at runtime")

    def _device_collection(self):
        return muGrid.GlobalFieldCollection(
            [4, 4], nb_ghosts_left=(1, 1), nb_ghosts_right=(1, 1),
            device=muGrid.Device.gpu(),
        )

    def test_axpy_component_mismatch(self):
        fc = self._device_collection()
        x = fc.real_field("x", (2,))
        y = fc.real_field("y", (3,))
        with pytest.raises(RuntimeError):
            linalg.axpy(1.0, x, y)

    def test_axpby_component_mismatch(self):
        fc = self._device_collection()
        x = fc.real_field("x", (2,))
        y = fc.real_field("y", (3,))
        with pytest.raises(RuntimeError):
            linalg.axpby(1.0, x, 1.0, y)

    def test_copy_component_mismatch(self):
        fc = self._device_collection()
        src = fc.real_field("src", (2,))
        dst = fc.real_field("dst", (3,))
        with pytest.raises(RuntimeError):
            linalg.copy(src, dst)

    def test_axpy_norm_sq_component_mismatch(self):
        fc = self._device_collection()
        x = fc.real_field("x", (2,))
        y = fc.real_field("y", (3,))
        with pytest.raises(RuntimeError):
            linalg.axpy_norm_sq(1.0, x, y)


# ===========================================================================
# Sub-point fields (nb_sub_pts > 1)
# ===========================================================================
#
# Regression coverage for the element-count computation. The total buffer size
# is get_nb_entries() * nb_components, and get_nb_entries() already includes the
# number of sub-points. A formula that multiplies by nb_sub_pts again overruns
# the buffer once nb_sub_pts > 1 (harmless for the scalar/nodal fields the rest
# of the suite uses).

def _subpt_collection(nb_grid_pts, nb_quad=2):
    g = (1,) * len(nb_grid_pts)
    return muGrid.GlobalFieldCollection(
        list(nb_grid_pts), nb_ghosts_left=g, nb_ghosts_right=g,
        sub_pts={"quad": nb_quad},
    )


class TestSubPointsHost:
    def test_norm_sq_subpoints(self, rng):
        fc = _subpt_collection((5, 6))
        x = fc.real_field("x", (2,), "quad")  # 2 components x 2 quad pts per pixel
        _fill(x, rng)
        ref = np.sum(np.abs(np.array(x.s)) ** 2)
        assert complex(linalg.norm_sq(x)).real == pytest.approx(ref, rel=1e-12)

    def test_vecdot_subpoints(self, rng):
        fc = _subpt_collection((5, 6))
        a = fc.real_field("a", (2,), "quad")
        b = fc.real_field("b", (2,), "quad")
        _fill(a, rng)
        _fill(b, rng)
        ref = np.sum(np.array(a.s) * np.array(b.s))
        assert linalg.vecdot(a, b) == pytest.approx(ref, rel=1e-12)

    def test_axpy_subpoints_full_buffer(self, rng):
        fc = _subpt_collection((5, 6))
        x = fc.real_field("x", (2,), "quad")
        y = fc.real_field("y", (2,), "quad")
        x0 = _fill(x, rng)
        y0 = _fill(y, rng)
        linalg.axpy(2.0, x, y)
        np.testing.assert_allclose(np.array(y.sg), 2.0 * x0 + y0, rtol=1e-12)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU backend not compiled in")
class TestGPUSubPoints:
    """The GPU element count had an extra nb_sub_pts factor that overran the
    device buffer for sub-point fields. Compare CPU and GPU on a quad field
    (nb_sub_pts = 2); they must agree and match the analytical value."""

    def setup_method(self, method):
        if not muGrid.is_gpu_available():
            pytest.skip("No GPU device available at runtime")
        from conftest import HAS_CUPY
        if not HAS_CUPY:
            pytest.skip("CuPy not available for GPU array access")

    def _collections(self, nb_grid_pts=(8, 8), nb_quad=2):
        g = (1,) * len(nb_grid_pts)
        cpu = muGrid.GlobalFieldCollection(
            list(nb_grid_pts), nb_ghosts_left=g, nb_ghosts_right=g,
            sub_pts={"quad": nb_quad},
        )
        gpu = muGrid.GlobalFieldCollection(
            list(nb_grid_pts), nb_ghosts_left=g, nb_ghosts_right=g,
            sub_pts={"quad": nb_quad}, device=muGrid.Device.gpu(),
        )
        return cpu, gpu

    def test_norm_sq_subpoints_cpu_gpu_match(self):
        nb = (8, 8)
        nb_components, nb_quad, fill = 2, 2, 3.5
        cpu_fc, gpu_fc = self._collections(nb, nb_quad)
        cpu = cpu_fc.real_field("x", (nb_components,), "quad")
        gpu = gpu_fc.real_field("x", (nb_components,), "quad")
        cpu.s[...] = fill
        gpu.s[...] = fill

        cpu_res = linalg.norm_sq(cpu)
        gpu_res = linalg.norm_sq(gpu)
        assert cpu_res == pytest.approx(gpu_res, rel=1e-10)
        expected = nb[0] * nb[1] * nb_components * nb_quad * fill * fill
        assert cpu_res == pytest.approx(expected, rel=1e-10)

    def test_axpy_subpoints_cpu_gpu_match(self):
        nb = (8, 8)
        nb_components, nb_quad = 2, 2
        alpha, fx, fy = 0.5, 2.0, 3.0
        cpu_fc, gpu_fc = self._collections(nb, nb_quad)
        cx = cpu_fc.real_field("x", (nb_components,), "quad")
        cy = cpu_fc.real_field("y", (nb_components,), "quad")
        gx = gpu_fc.real_field("x", (nb_components,), "quad")
        gy = gpu_fc.real_field("y", (nb_components,), "quad")
        cx.s[...] = fx
        cy.s[...] = fy
        gx.s[...] = fx
        gy.s[...] = fy

        linalg.axpy(alpha, cx, cy)
        linalg.axpy(alpha, gx, gy)
        # Interior norm of the updated y must match between CPU and GPU.
        assert linalg.norm_sq(cy) == pytest.approx(linalg.norm_sq(gy), rel=1e-10)


class TestFieldScal:
    """scal with a field-valued alpha: broadcast and elementwise modes, on
    both storage orders (host collections default to AoS; SoA is what GPU
    collections use and is also constructible on the host)."""

    @pytest.mark.parametrize(
        "storage_order",
        [muGrid.StorageOrder.ArrayOfStructures,
         muGrid.StorageOrder.StructureOfArrays],
    )
    @pytest.mark.parametrize("complex_", [False, True])
    @pytest.mark.parametrize("nb_alpha_components", [1, 2])
    def test_scal_field_alpha(self, rng, storage_order, complex_,
                              nb_alpha_components):
        fc = muGrid.GlobalFieldCollection(
            [7, 5], storage_order=storage_order
        )
        x = _make(fc, "x", components=(2,), complex_=complex_)
        alpha = fc.real_field(
            "alpha", () if nb_alpha_components == 1 else (2,)
        )
        x_before = _fill(x, rng, complex_=complex_)
        alpha_values = _fill(alpha, rng)

        linalg.scal(alpha, x)

        # numpy reference: broadcast a single-component alpha over the
        # components of x, apply a matching alpha elementwise
        expected = x_before * alpha_values
        np.testing.assert_allclose(np.array(x.sg), expected, atol=1e-14)

    def test_scal_field_alpha_validation(self, rng):
        fc = muGrid.GlobalFieldCollection([7, 5])
        other = muGrid.GlobalFieldCollection([7, 5])
        x = _make(fc, "x", components=(2,))
        _fill(x, rng)

        # alpha from a different collection is rejected
        alpha_other = other.real_field("alpha")
        with pytest.raises(RuntimeError, match="same collection"):
            linalg.scal(alpha_other, x)

        # component-count mismatch (neither 1 nor x's count) is rejected
        alpha_bad = fc.real_field("alpha3", (3,))
        with pytest.raises(RuntimeError, match="components"):
            linalg.scal(alpha_bad, x)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

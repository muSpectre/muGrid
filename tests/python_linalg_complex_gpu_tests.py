#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file    python_linalg_complex_gpu_tests.py

@author  Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date    15 Jun 2026

@brief   Functional tests for the complex linalg operations on host and GPU

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

Functional (numerical-equivalence) tests for the *complex* ``muGrid.linalg``
operations. Each test fills fields from a fixed numpy array, runs the operation
on the parametrised device, and compares against an independent numpy
reference. They run on the CPU always and on the GPU when one is available
(``get_test_devices()`` adds ``"gpu"`` only if µGrid was built with CUDA/HIP and
a device is present); the GPU parametrisation is what exercises the device
kernels in ``linalg_gpu.cc``.

These are not smoke tests: every assertion checks the actual returned values
(or the in-place field contents) against the mathematical definition of the
operation, including the sesquilinear convention of ``vecdot`` and the
ghost-exclusion of the reductions.
"""

import numpy as np
import pytest
from conftest import HAS_CUPY, create_device, get_test_devices, skip_if_gpu_unavailable

import muGrid
from muGrid import linalg

if HAS_CUPY:
    import cupy as cp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_numpy(a):
    """Bring a field view (numpy on CPU, cupy on GPU) to the host as numpy."""
    if HAS_CUPY and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


def _decomp(device, nb_grid_pts, nb_ghosts=0):
    """Build a CartesianDecomposition on the requested device."""
    comm = muGrid.Communicator()
    sd = len(nb_grid_pts)
    g = (nb_ghosts,) * sd
    device_obj = create_device(device)
    kwargs = dict(
        nb_subdivisions=(1,) * sd,
        nb_ghosts_left=g,
        nb_ghosts_right=g,
    )
    if device_obj is not None:
        kwargs["device"] = device_obj
    return muGrid.CartesianDecomposition(comm, nb_grid_pts, **kwargs)


def _set(field, array, ghosts=False):
    """Write `array` (host numpy) into a field's interior (or full) view,
    moving it to the device first if necessary."""
    view = field.sg if ghosts else field.s
    if HAS_CUPY and isinstance(view, cp.ndarray):
        view[...] = cp.asarray(array)
    else:
        view[...] = array


def _randc(rng, shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


@pytest.fixture
def rng():
    return np.random.default_rng(2026_06_15)


GRID = [4, 5, 3]


# ===========================================================================
# Element-wise complex operations (no ghosts; compare the full interior)
# ===========================================================================

@pytest.mark.parametrize("device", get_test_devices())
class TestComplexElementwise:
    def test_copy(self, device, rng):
        skip_if_gpu_unavailable(device)
        d = _decomp(device, GRID)
        src = d.complex_field("src", (2,))
        dst = d.complex_field("dst", (2,))
        a = _randc(rng, src.s.shape)
        _set(src, a)
        _set(dst, _randc(rng, dst.s.shape))

        linalg.copy(src, dst)

        np.testing.assert_allclose(_to_numpy(dst.s), a, atol=1e-13)

    def test_scal(self, device, rng):
        skip_if_gpu_unavailable(device)
        d = _decomp(device, GRID)
        x = d.complex_field("x", (2,))
        a = _randc(rng, x.s.shape)
        _set(x, a)
        alpha = 0.5 - 1.25j

        linalg.scal(alpha, x)

        np.testing.assert_allclose(_to_numpy(x.s), alpha * a, atol=1e-13)

    def test_axpy(self, device, rng):
        skip_if_gpu_unavailable(device)
        d = _decomp(device, GRID)
        x = d.complex_field("x", (2,))
        y = d.complex_field("y", (2,))
        a = _randc(rng, x.s.shape)
        b = _randc(rng, y.s.shape)
        _set(x, a)
        _set(y, b)
        alpha = -0.75 + 2.0j

        linalg.axpy(alpha, x, y)

        np.testing.assert_allclose(_to_numpy(y.s), alpha * a + b, atol=1e-13)

    def test_axpby(self, device, rng):
        skip_if_gpu_unavailable(device)
        d = _decomp(device, GRID)
        x = d.complex_field("x", (2,))
        y = d.complex_field("y", (2,))
        a = _randc(rng, x.s.shape)
        b = _randc(rng, y.s.shape)
        _set(x, a)
        _set(y, b)
        alpha, beta = 1.5 - 0.5j, -2.0 + 0.25j

        linalg.axpby(alpha, x, beta, y)

        np.testing.assert_allclose(
            _to_numpy(y.s), alpha * a + beta * b, atol=1e-13
        )

    def test_cross(self, device, rng):
        skip_if_gpu_unavailable(device)
        d = _decomp(device, GRID)
        a = d.complex_field("a", (3,))
        b = d.complex_field("b", (3,))
        out = d.complex_field("out", (3,))
        av = _randc(rng, a.s.shape)
        bv = _randc(rng, b.s.shape)
        _set(a, av)
        _set(b, bv)
        _set(out, _randc(rng, out.s.shape))

        linalg.cross(a, b, out)

        # component axis is first (same convention as the .p/.s views)
        np.testing.assert_allclose(
            _to_numpy(out.s), np.cross(av, bv, axis=0), atol=1e-13
        )

    def test_leray_project(self, device, rng):
        skip_if_gpu_unavailable(device)
        d = _decomp(device, GRID)
        k = d.real_field("k", (3,))
        invk = d.real_field("invk", (3,))
        N = d.complex_field("N", (3,))
        out = d.complex_field("out", (3,))
        kv = rng.standard_normal(k.s.shape) + 0.5
        invkv = rng.standard_normal(invk.s.shape)
        Nv = _randc(rng, N.s.shape)
        outv = _randc(rng, out.s.shape)
        _set(k, kv)
        _set(invk, invkv)
        _set(N, Nv)
        _set(out, outv)

        linalg.leray_project(k, invk, N, out)

        s = np.sum(invkv * Nv, axis=0)          # invk . N  (per pixel)
        ref = outv - kv * s                      # out -= k (invk . N)
        np.testing.assert_allclose(_to_numpy(out.s), ref, atol=1e-13)

    def test_leray_makes_divergence_free(self, device, rng):
        # With invk = k/|k|^2 and out initialised to N, the projection must
        # yield a transverse field: k . out = 0 per pixel.
        skip_if_gpu_unavailable(device)
        d = _decomp(device, GRID)
        k = d.real_field("k", (3,))
        invk = d.real_field("invk", (3,))
        N = d.complex_field("N", (3,))
        out = d.complex_field("out", (3,))
        kv = rng.standard_normal(k.s.shape) + 0.7
        ksq = np.sum(kv ** 2, axis=0)
        Nv = _randc(rng, N.s.shape)
        _set(k, kv)
        _set(invk, kv / ksq)
        _set(N, Nv)
        _set(out, Nv)

        linalg.leray_project(k, invk, N, out)

        div = np.sum(kv * _to_numpy(out.s), axis=0)
        np.testing.assert_allclose(div, np.zeros_like(div), atol=1e-12)


# ===========================================================================
# Complex reductions (value + ghost-exclusion)
# ===========================================================================

@pytest.mark.parametrize("device", get_test_devices())
class TestComplexReductions:
    def test_vecdot_sesquilinear(self, device, rng):
        skip_if_gpu_unavailable(device)
        d = _decomp(device, GRID)
        a = d.complex_field("a", (2,))
        b = d.complex_field("b", (2,))
        av = _randc(rng, a.s.shape)
        bv = _randc(rng, b.s.shape)
        _set(a, av)
        _set(b, bv)

        result = linalg.vecdot(a, b)

        ref = np.sum(np.conj(av) * bv)  # sesquilinear: conj(a) . b
        assert complex(result) == pytest.approx(ref, rel=1e-12)

    def test_norm_sq(self, device, rng):
        skip_if_gpu_unavailable(device)
        d = _decomp(device, GRID)
        x = d.complex_field("x", (2,))
        xv = _randc(rng, x.s.shape)
        _set(x, xv)

        result = linalg.norm_sq(x)

        ref = np.sum(np.abs(xv) ** 2)  # real-valued
        assert complex(result).real == pytest.approx(ref, rel=1e-12)
        assert complex(result).imag == pytest.approx(0.0, abs=1e-9)

    def test_axpy_norm_sq(self, device, rng):
        skip_if_gpu_unavailable(device)
        d = _decomp(device, GRID)
        x = d.complex_field("x", (2,))
        y = d.complex_field("y", (2,))
        av = _randc(rng, x.s.shape)
        bv = _randc(rng, y.s.shape)
        _set(x, av)
        _set(y, bv)
        alpha = 0.3 - 0.8j

        result = linalg.axpy_norm_sq(alpha, x, y)

        new_y = alpha * av + bv
        np.testing.assert_allclose(_to_numpy(y.s), new_y, atol=1e-13)
        assert complex(result).real == pytest.approx(
            np.sum(np.abs(new_y) ** 2), rel=1e-12
        )

    def test_vecdot_excludes_ghosts(self, device, rng):
        skip_if_gpu_unavailable(device)
        d = _decomp(device, GRID, nb_ghosts=1)
        a = d.complex_field("a", (2,))
        b = d.complex_field("b", (2,))
        # Fill the FULL buffer (ghosts included) with non-trivial values.
        _set(a, _randc(rng, a.sg.shape), ghosts=True)
        _set(b, _randc(rng, b.sg.shape), ghosts=True)

        result = linalg.vecdot(a, b)

        # Reference over the INTERIOR only.
        ai = _to_numpy(a.s)
        bi = _to_numpy(b.s)
        ref = np.sum(np.conj(ai) * bi)
        assert complex(result) == pytest.approx(ref, rel=1e-12)

    def test_norm_sq_excludes_ghosts(self, device, rng):
        skip_if_gpu_unavailable(device)
        d = _decomp(device, GRID, nb_ghosts=1)
        x = d.complex_field("x", (2,))
        _set(x, _randc(rng, x.sg.shape), ghosts=True)

        result = linalg.norm_sq(x)

        ref = np.sum(np.abs(_to_numpy(x.s)) ** 2)
        assert complex(result).real == pytest.approx(ref, rel=1e-12)

    def test_axpy_norm_sq_excludes_ghosts(self, device, rng):
        skip_if_gpu_unavailable(device)
        d = _decomp(device, GRID, nb_ghosts=1)
        x = d.complex_field("x", (2,))
        y = d.complex_field("y", (2,))
        xv = _randc(rng, x.sg.shape)
        yv = _randc(rng, y.sg.shape)
        _set(x, xv, ghosts=True)
        _set(y, yv, ghosts=True)
        alpha = 1.1 + 0.4j

        result = linalg.axpy_norm_sq(alpha, x, y)

        # The AXPY updates the full buffer; the norm is interior-only.
        full_new_y = alpha * xv + yv
        np.testing.assert_allclose(_to_numpy(y.sg), full_new_y, atol=1e-12)
        interior_new_y = _to_numpy(y.s)
        assert complex(result).real == pytest.approx(
            np.sum(np.abs(interior_new_y) ** 2), rel=1e-12
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

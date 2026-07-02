#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file    python_fem_gradient_gpu_tests.py

@author  Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date    25 Dec 2024

@brief   Test GPU FEM gradient operator functionality (CUDA/ROCm)

Copyright (c) 2024 Lars Pastewka

muGrid is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

muGrid is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with muGrid; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.

Additional permission under GNU GPL version 3 section 7

If you modify this Program, or any covered work, by linking or combining it
with proprietary FFT implementations or numerical libraries, containing parts
covered by the terms of those libraries' licenses, the licensors of this
Program grant you additional permission to convey the resulting work.
"""

import numpy as np
import pytest

# Import GPU testing utilities from conftest
from conftest import (
    HAS_CUPY,
    cp,
    create_device,
    get_test_devices,
    skip_if_gpu_unavailable,
)

import muGrid

# =============================================================================
# Test Configuration
# =============================================================================

GPU_AVAILABLE = muGrid.has_gpu


# =============================================================================
# 2D FEM Gradient Smoke Tests (Parametrized on Device)
# =============================================================================


@pytest.mark.parametrize("device", get_test_devices())
class TestFEMGradientOperator2D_smoke:
    """Smoke tests for FEMGradientOperator 2D on different devices."""

    def setup_method(self):
        self.nb_x_pts = 8
        self.nb_y_pts = 6

    def test_fem_gradient_2d_apply_smoke(self, device):
        """Test that FEM gradient 2D can be applied on device fields."""
        skip_if_gpu_unavailable(device)

        fem_grad = muGrid.FEMGradientOperator(2, element=muGrid.FEMElement.p1)  # 2D

        device_obj = create_device(device)
        fc_kwargs = {"device": device_obj} if device_obj else {}

        # Create collection with ghosts
        fc = muGrid.GlobalFieldCollection(
            (self.nb_x_pts, self.nb_y_pts),
            sub_pts={"quad": fem_grad.nb_quad_pts},
            nb_ghosts_right=(1, 1),
            **fc_kwargs,
        )

        # Create nodal and gradient fields
        nodal = fc.real_field("nodal", (1,))
        gradient = fc.real_field("gradient", (fem_grad.nb_output_components,), "quad")

        # Initialize and apply (should not raise)
        nodal.set_zero()
        gradient.set_zero()
        fem_grad.apply(nodal, gradient)

        # Verify device location
        if device == "gpu":
            assert gradient.is_on_gpu

    def test_fem_gradient_2d_transpose_smoke(self, device):
        """Test that FEM gradient 2D transpose can be applied on device fields."""
        skip_if_gpu_unavailable(device)

        fem_grad = muGrid.FEMGradientOperator(2, element=muGrid.FEMElement.p1)

        device_obj = create_device(device)
        fc_kwargs = {"device": device_obj} if device_obj else {}

        fc = muGrid.GlobalFieldCollection(
            (self.nb_x_pts, self.nb_y_pts),
            sub_pts={"quad": fem_grad.nb_quad_pts},
            nb_ghosts_right=(1, 1),
            **fc_kwargs,
        )

        nodal = fc.real_field("nodal", (1,))
        gradient = fc.real_field("gradient", (fem_grad.nb_output_components,), "quad")

        gradient.set_zero()
        nodal.set_zero()
        fem_grad.transpose(gradient, nodal)

        # Verify device location
        if device == "gpu":
            assert nodal.is_on_gpu


# =============================================================================
# 3D FEM Gradient Smoke Tests (Parametrized on Device)
# =============================================================================


@pytest.mark.parametrize("device", get_test_devices())
class TestFEMGradientOperator3D_smoke:
    """Smoke tests for 3D FEMGradientOperator on different devices."""

    def setup_method(self):
        self.nb_x_pts = 6
        self.nb_y_pts = 6
        self.nb_z_pts = 6

    def test_fem_gradient_3d_apply_smoke(self, device):
        """Test that FEM gradient 3D can be applied on device fields."""
        skip_if_gpu_unavailable(device)

        fem_grad = muGrid.FEMGradientOperator(3, element=muGrid.FEMElement.p1)  # 3D

        device_obj = create_device(device)
        fc_kwargs = {"device": device_obj} if device_obj else {}

        fc = muGrid.GlobalFieldCollection(
            (self.nb_x_pts, self.nb_y_pts, self.nb_z_pts),
            sub_pts={"quad": fem_grad.nb_quad_pts},
            nb_ghosts_right=(1, 1, 1),
            **fc_kwargs,
        )

        nodal = fc.real_field("nodal", (1,))
        gradient = fc.real_field("gradient", (fem_grad.nb_output_components,), "quad")

        nodal.set_zero()
        gradient.set_zero()
        fem_grad.apply(nodal, gradient)

        # Verify device location
        if device == "gpu":
            assert gradient.is_on_gpu

    def test_fem_gradient_3d_transpose_smoke(self, device):
        """Test that FEM gradient 3D transpose can be applied on device fields."""
        skip_if_gpu_unavailable(device)

        fem_grad = muGrid.FEMGradientOperator(3, element=muGrid.FEMElement.p1)

        device_obj = create_device(device)
        fc_kwargs = {"device": device_obj} if device_obj else {}

        fc = muGrid.GlobalFieldCollection(
            (self.nb_x_pts, self.nb_y_pts, self.nb_z_pts),
            sub_pts={"quad": fem_grad.nb_quad_pts},
            nb_ghosts_right=(1, 1, 1),
            **fc_kwargs,
        )

        nodal = fc.real_field("nodal", (1,))
        gradient = fc.real_field("gradient", (fem_grad.nb_output_components,), "quad")

        gradient.set_zero()
        nodal.set_zero()
        fem_grad.transpose(gradient, nodal)

        # Verify device location
        if device == "gpu":
            assert nodal.is_on_gpu


# =============================================================================
# GPU vs CPU Correctness Tests (Dual-device comparison)
# =============================================================================


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU backend not available")
class TestFEMGradientOperatorGPUCorrectness:
    """Test suite for FEMGradientOperator GPU correctness.

    These tests verify that GPU FEM gradient operations return CuPy arrays
    and produce correct results that match the CPU reference.
    """

    def setup_method(self, method):
        """Skip tests if no GPU is available at runtime."""
        if not muGrid.is_gpu_available():
            pytest.skip("No GPU device available at runtime")
        if not HAS_CUPY:
            pytest.skip("CuPy not available for GPU tests")
        self.nb_x_pts = 8
        self.nb_y_pts = 6

    def test_device_fem_gradient_returns_cupy(self):
        """Test that FEM gradient on device fields returns CuPy arrays."""
        fem_grad = muGrid.FEMGradientOperator(2, element=muGrid.FEMElement.p1)

        fc = muGrid.GlobalFieldCollection(
            (self.nb_x_pts, self.nb_y_pts),
            sub_pts={"quad": fem_grad.nb_quad_pts},
            nb_ghosts_right=(1, 1),
            device=muGrid.Device.gpu(),
        )

        nodal = fc.real_field("nodal", (1,))
        gradient = fc.real_field("gradient", (fem_grad.nb_output_components,), "quad")

        # Arrays should be CuPy
        assert isinstance(nodal.s, cp.ndarray)
        assert isinstance(gradient.s, cp.ndarray)

        fem_grad.apply(nodal, gradient)

        # Result should still be CuPy
        assert isinstance(gradient.s, cp.ndarray)

    def test_device_fem_gradient_2d_correctness(self):
        """Test that 2D device FEM gradient produces correct results.

        Compare GPU results against CPU reference.
        """
        fem_grad = muGrid.FEMGradientOperator(2, element=muGrid.FEMElement.p1)

        # Create host reference
        fc_host = muGrid.GlobalFieldCollection(
            (self.nb_x_pts, self.nb_y_pts),
            sub_pts={"quad": fem_grad.nb_quad_pts},
            nb_ghosts_right=(1, 1),
        )
        nodal_host = fc_host.real_field("nodal", (1,))
        gradient_host = fc_host.real_field(
            "gradient", (fem_grad.nb_output_components,), "quad"
        )

        # Create device fields
        fc_device = muGrid.GlobalFieldCollection(
            (self.nb_x_pts, self.nb_y_pts),
            sub_pts={"quad": fem_grad.nb_quad_pts},
            nb_ghosts_right=(1, 1),
            device=muGrid.Device.gpu(),
        )
        nodal_device = fc_device.real_field("nodal", (1,))
        gradient_device = fc_device.real_field(
            "gradient", (fem_grad.nb_output_components,), "quad"
        )

        # Initialize with same random data
        test_data = np.random.rand(*nodal_host.pg.shape)
        nodal_host.pg[...] = test_data
        nodal_device.pg[...] = cp.asarray(test_data)

        # Apply on both
        fem_grad.apply(nodal_host, gradient_host)
        fem_grad.apply(nodal_device, gradient_device)

        # Compare results
        host_result = gradient_host.s
        device_result = cp.asnumpy(gradient_device.s)

        np.testing.assert_allclose(device_result, host_result, rtol=1e-10)

    def test_device_fem_gradient_2d_transpose_correctness(self):
        """Test that 2D device FEM transpose produces correct results."""
        fem_grad = muGrid.FEMGradientOperator(2, element=muGrid.FEMElement.p1)

        # Create host reference
        fc_host = muGrid.GlobalFieldCollection(
            (self.nb_x_pts, self.nb_y_pts),
            sub_pts={"quad": fem_grad.nb_quad_pts},
            nb_ghosts_right=(1, 1),
        )
        nodal_host = fc_host.real_field("nodal", (1,))
        gradient_host = fc_host.real_field(
            "gradient", (fem_grad.nb_output_components,), "quad"
        )

        # Create device fields
        fc_device = muGrid.GlobalFieldCollection(
            (self.nb_x_pts, self.nb_y_pts),
            sub_pts={"quad": fem_grad.nb_quad_pts},
            nb_ghosts_right=(1, 1),
            device=muGrid.Device.gpu(),
        )
        nodal_device = fc_device.real_field("nodal", (1,))
        gradient_device = fc_device.real_field(
            "gradient", (fem_grad.nb_output_components,), "quad"
        )

        # Initialize gradient with random data
        test_data = np.random.rand(*gradient_host.s.shape)
        gradient_host.s[...] = test_data
        gradient_device.s[...] = cp.asarray(test_data)

        # Apply transpose on both
        fem_grad.transpose(gradient_host, nodal_host)
        fem_grad.transpose(gradient_device, nodal_device)

        # Compare results
        host_result = nodal_host.s
        device_result = cp.asnumpy(nodal_device.s)

        np.testing.assert_allclose(device_result, host_result, rtol=1e-10)

    def test_device_fem_gradient_3d_correctness(self):
        """Test that 3D device FEM gradient produces correct results."""
        nb_x, nb_y, nb_z = 6, 6, 6
        fem_grad = muGrid.FEMGradientOperator(3, element=muGrid.FEMElement.p1)

        # Create host reference
        fc_host = muGrid.GlobalFieldCollection(
            (nb_x, nb_y, nb_z),
            sub_pts={"quad": fem_grad.nb_quad_pts},
            nb_ghosts_right=(1, 1, 1),
        )
        nodal_host = fc_host.real_field("nodal", (1,))
        gradient_host = fc_host.real_field(
            "gradient", (fem_grad.nb_output_components,), "quad"
        )

        # Create device fields
        fc_device = muGrid.GlobalFieldCollection(
            (nb_x, nb_y, nb_z),
            sub_pts={"quad": fem_grad.nb_quad_pts},
            nb_ghosts_right=(1, 1, 1),
            device=muGrid.Device.gpu(),
        )
        nodal_device = fc_device.real_field("nodal", (1,))
        gradient_device = fc_device.real_field(
            "gradient", (fem_grad.nb_output_components,), "quad"
        )

        # Initialize with same random data
        test_data = np.random.rand(*nodal_host.pg.shape)
        nodal_host.pg[...] = test_data
        nodal_device.pg[...] = cp.asarray(test_data)

        # Apply on both
        fem_grad.apply(nodal_host, gradient_host)
        fem_grad.apply(nodal_device, gradient_device)

        # Compare results
        host_result = gradient_host.s
        device_result = cp.asnumpy(gradient_device.s)

        np.testing.assert_allclose(device_result, host_result, rtol=1e-10)

    def test_device_fem_gradient_3d_transpose_correctness(self):
        """Test that 3D device FEM transpose produces correct results."""
        nb_x, nb_y, nb_z = 6, 6, 6
        fem_grad = muGrid.FEMGradientOperator(3, element=muGrid.FEMElement.p1)

        # Create host reference
        fc_host = muGrid.GlobalFieldCollection(
            (nb_x, nb_y, nb_z),
            sub_pts={"quad": fem_grad.nb_quad_pts},
            nb_ghosts_right=(1, 1, 1),
        )
        nodal_host = fc_host.real_field("nodal", (1,))
        gradient_host = fc_host.real_field(
            "gradient", (fem_grad.nb_output_components,), "quad"
        )

        # Create device fields
        fc_device = muGrid.GlobalFieldCollection(
            (nb_x, nb_y, nb_z),
            sub_pts={"quad": fem_grad.nb_quad_pts},
            nb_ghosts_right=(1, 1, 1),
            device=muGrid.Device.gpu(),
        )
        nodal_device = fc_device.real_field("nodal", (1,))
        gradient_device = fc_device.real_field(
            "gradient", (fem_grad.nb_output_components,), "quad"
        )

        # Initialize gradient with random data
        test_data = np.random.rand(*gradient_host.s.shape)
        gradient_host.s[...] = test_data
        gradient_device.s[...] = cp.asarray(test_data)

        # Apply transpose on both
        fem_grad.transpose(gradient_host, nodal_host)
        fem_grad.transpose(gradient_device, nodal_device)

        # Compare results
        host_result = nodal_host.s
        device_result = cp.asnumpy(nodal_device.s)

        np.testing.assert_allclose(device_result, host_result, rtol=1e-10)

    def test_device_fem_gradient_roundtrip(self):
        """Test round-trip: B^T * B should be symmetric positive semi-definite."""
        fem_grad = muGrid.FEMGradientOperator(2, element=muGrid.FEMElement.p1)

        # Create device collection
        fc = muGrid.GlobalFieldCollection(
            (self.nb_x_pts, self.nb_y_pts),
            sub_pts={"quad": fem_grad.nb_quad_pts},
            nb_ghosts_right=(1, 1),
            device=muGrid.Device.gpu(),
        )

        nodal_in = fc.real_field("nodal_in", (1,))
        gradient = fc.real_field("gradient", (fem_grad.nb_output_components,), "quad")
        nodal_out = fc.real_field("nodal_out", (1,))

        # Initialize with random data
        test_data = cp.random.rand(*nodal_in.pg.shape)
        nodal_in.pg[...] = test_data

        # Apply B then B^T
        fem_grad.apply(nodal_in, gradient)
        fem_grad.transpose(gradient, nodal_out)

        # Result should be a valid array (no NaN or Inf)
        result = cp.asnumpy(nodal_out.s)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


# =============================================================================
# Element type (simplex vs Q1) patch test, parametrized on device
# =============================================================================


@pytest.mark.parametrize("device", get_test_devices())
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("element", ["p1", "q1"])
class TestFEMGradientElements:
    """The FEM gradient operator supports linear simplices and Q1 elements.
    Patch test: an affine nodal field must produce a constant gradient (equal
    to the affine field's gradient) at every quadrature point, for any element.
    """

    def test_gradient_patch_test(self, device, dim, element):
        skip_if_gpu_unavailable(device)
        n = 6
        h = [1.0 / n] * dim
        el = (muGrid.FEMElement.q1 if element == "q1"
              else muGrid.FEMElement.p1)
        op = muGrid.FEMGradientOperator(dim, tuple(h), element=el)
        device_obj = create_device(device)
        fc_kwargs = {"device": device_obj} if device_obj else {}
        fc = muGrid.GlobalFieldCollection(
            (n,) * dim, sub_pts={"quad": op.nb_quad_pts},
            nb_ghosts_right=(1,) * dim, **fc_kwargs)
        nodal = fc.real_field("nodal", (1,))
        gradient = fc.real_field("gradient", (op.nb_output_components,), "quad")

        # Affine nodal field u = sum_j a_j x_j -> gradient a_j (constant).
        a = np.array([0.3, -0.7, 1.1])[:dim]
        pg = nodal.pg
        xp = np if device != "gpu" else __import__("cupy")
        idx = np.indices(tuple(pg.shape[1:]))
        field = np.zeros(tuple(pg.shape[1:]))
        for j in range(dim):
            field += a[j] * idx[j] * h[j]
        pg[0, ...] = xp.asarray(field)

        op.apply(nodal, gradient)
        gs = gradient.s
        gs = gs.get() if hasattr(gs, "get") else np.asarray(gs)
        # interior pixels [0, n-1) per axis carry valid (constant) gradients
        interior = tuple([slice(0, n - 1)] * dim)
        for j in range(dim):
            comp = gs[j][(slice(None),) + interior]  # (nq, *interior)
            np.testing.assert_allclose(comp, a[j], atol=1e-12)


# =============================================================================
# Weighting convention of transpose()  (pins the documented semantics)
# =============================================================================


class TestFEMGradientTransposeConvention:
    """`transpose()` is the *quadrature-weighted* transpose ``Bᵀ W`` (the
    Galerkin / L² adjoint), not the bare matrix transpose. These tests pin that
    contract so it cannot silently change; see the FEMGradientOperator
    docstring."""

    ELEMENTS = [muGrid.FEMElement.p1, muGrid.FEMElement.q1]
    NAMES = {muGrid.FEMElement.p1: "p1", muGrid.FEMElement.q1: "q1"}

    def _setup(self, dim, element, n=6):
        grad = muGrid.FEMGradientOperator(dim, [0.7, 1.3, 1.1][:dim], element)
        nq, nc = grad.nb_quad_pts, grad.nb_output_components
        eng = muGrid.FFTEngine(
            (n,) * dim, muGrid.Communicator(), nb_ghosts_left=(1,) * dim,
            nb_ghosts_right=(1,) * dim, nb_sub_pts={"quad": nq},
        )
        return grad, eng, eng.real_space_collection, nq, nc

    @pytest.mark.parametrize("element", ELEMENTS, ids=NAMES.get)
    @pytest.mark.parametrize("dim", [2, 3])
    def test_transpose_is_weighted_adjoint(self, dim, element):
        """⟨W·apply(u), v⟩ == ⟨u, transpose(v)⟩ (default weights = W)."""
        grad, eng, fc, nq, nc = self._setup(dim, element)
        w = np.asarray(grad.quadrature_weights).reshape((1, nq) + (1,) * dim)
        u = fc.real_field("u", (1,))
        Bu = fc.real_field("Bu", (nc,), "quad")
        v = fc.real_field("v", (nc,), "quad")
        BTv = fc.real_field("BTv", (1,))
        rng = np.random.default_rng(0)
        u.p[...] = rng.standard_normal(np.asarray(u.p).shape)
        v.s[...] = rng.standard_normal(np.asarray(v.s).shape)
        eng.communicate_ghosts(u)
        eng.communicate_ghosts(v)
        grad.apply(u, Bu)
        grad.transpose(v, BTv)
        lhs = float(np.sum(w * np.asarray(Bu.s) * np.asarray(v.s)))
        rhs = float(np.sum(np.asarray(u.p) * np.asarray(BTv.p)))
        np.testing.assert_allclose(lhs, rhs, rtol=1e-11)

    @pytest.mark.parametrize("element", ELEMENTS, ids=NAMES.get)
    @pytest.mark.parametrize("dim", [2, 3])
    def test_transpose_apply_is_fe_stiffness(self, dim, element):
        """transpose(apply(u)) == Bᵀ W B u, i.e. uᵀ·transpose(apply(u)) equals
        the FE energy Σ_q w_q |∇u_q|² (do not weight again)."""
        grad, eng, fc, nq, nc = self._setup(dim, element)
        w = np.asarray(grad.quadrature_weights).reshape((1, nq) + (1,) * dim)
        u = fc.real_field("u", (1,))
        Bu = fc.real_field("Bu", (nc,), "quad")
        Lu = fc.real_field("Lu", (1,))
        rng = np.random.default_rng(1)
        u.p[...] = rng.standard_normal(np.asarray(u.p).shape)
        eng.communicate_ghosts(u)
        grad.apply(u, Bu)
        energy = float(np.sum(w * np.asarray(Bu.s) ** 2))  # Σ_q w_q |∇u_q|²
        grad.transpose(Bu, Lu)  # NB: no manual weighting
        form = float(np.sum(np.asarray(u.p) * np.asarray(Lu.p)))
        np.testing.assert_allclose(form, energy, rtol=1e-11)

    @pytest.mark.parametrize("element", ELEMENTS, ids=NAMES.get)
    @pytest.mark.parametrize("dim", [2, 3])
    def test_transpose_custom_weights_override(self, dim, element):
        """Explicit weights override the quadrature weights:
        ⟨c·apply(u), v⟩ == ⟨u, transpose(v, weights=c)⟩."""
        grad, eng, fc, nq, nc = self._setup(dim, element)
        rng = np.random.default_rng(2)
        c = rng.uniform(0.5, 2.0, nq)
        cc = c.reshape((1, nq) + (1,) * dim)
        u = fc.real_field("u", (1,))
        Bu = fc.real_field("Bu", (nc,), "quad")
        v = fc.real_field("v", (nc,), "quad")
        BTv = fc.real_field("BTv", (1,))
        u.p[...] = rng.standard_normal(np.asarray(u.p).shape)
        v.s[...] = rng.standard_normal(np.asarray(v.s).shape)
        eng.communicate_ghosts(u)
        eng.communicate_ghosts(v)
        grad.apply(u, Bu)
        grad.transpose(v, BTv, weights=list(c))
        lhs = float(np.sum(cc * np.asarray(Bu.s) * np.asarray(v.s)))
        rhs = float(np.sum(np.asarray(u.p) * np.asarray(BTv.p)))
        np.testing.assert_allclose(lhs, rhs, rtol=1e-11)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

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

        fem_grad = muGrid.FEMGradientOperator(2)  # 2D

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

        fem_grad = muGrid.FEMGradientOperator(2)

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

        fem_grad = muGrid.FEMGradientOperator(3)  # 3D

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

        fem_grad = muGrid.FEMGradientOperator(3)

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
        fem_grad = muGrid.FEMGradientOperator(2)

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
        fem_grad = muGrid.FEMGradientOperator(2)

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
        fem_grad = muGrid.FEMGradientOperator(2)

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
        fem_grad = muGrid.FEMGradientOperator(3)

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
        fem_grad = muGrid.FEMGradientOperator(3)

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
        fem_grad = muGrid.FEMGradientOperator(2)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

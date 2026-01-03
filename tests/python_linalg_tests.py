#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file     python_linalg_tests.py

@author  Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date    30 Dec 2025

@brief   Test linear algebra operations, especially ghost subtraction on GPU

These tests verify that linalg operations (norm_sq, vecdot, axpy_norm_sq)
correctly exclude ghost regions from their computations. This is essential
for MPI-parallel computations where ghost values are duplicated.

The tests specifically cover the SoA (Structure of Arrays) memory layout
used on GPUs, which requires different memory access patterns than the
AoS (Array of Structures) layout used on CPUs.

Copyright © 2025 Lars Pastewka

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
"""

import numpy as np
import pytest

# Import GPU testing utilities from conftest
from conftest import HAS_CUPY, create_device, get_test_devices, skip_if_gpu_unavailable

import muGrid
from muGrid import linalg

# Use compile-time feature flag for GPU availability
GPU_AVAILABLE = muGrid.has_gpu


# =============================================================================
# Parametrized tests for ghost exclusion (runs on both CPU and GPU)
# =============================================================================


@pytest.mark.parametrize("device", get_test_devices())
class TestLinalgGhostExclusion:
    """Test that linalg operations correctly exclude ghost regions.

    These tests run on both CPU and GPU to verify consistent behavior
    across different memory layouts (AoS for CPU, SoA for GPU).
    """

    def test_norm_sq_excludes_ghosts(self, device):
        """Test that norm_sq correctly excludes ghost regions."""
        skip_if_gpu_unavailable(device)

        nb_grid_pts = [8, 8]
        spatial_dim = 2
        nb_components = 2
        nb_ghosts_left = (1,) * spatial_dim
        nb_ghosts_right = (1,) * spatial_dim

        comm = muGrid.Communicator()
        device_obj = create_device(device)

        if device_obj is None:
            decomposition = muGrid.CartesianDecomposition(
                comm,
                nb_grid_pts,
                nb_subdivisions=(1,) * spatial_dim,
                nb_ghosts_left=nb_ghosts_left,
                nb_ghosts_right=nb_ghosts_right,
            )
        else:
            decomposition = muGrid.CartesianDecomposition(
                comm,
                nb_grid_pts,
                nb_subdivisions=(1,) * spatial_dim,
                nb_ghosts_left=nb_ghosts_left,
                nb_ghosts_right=nb_ghosts_right,
                device=device_obj,
            )

        field = decomposition.real_field("test", (nb_components,))

        # Fill entire field (including ghosts) with 1.0
        field.s[...] = 1.0

        # Expected: only interior pixels should be counted
        expected = nb_grid_pts[0] * nb_grid_pts[1] * nb_components

        result = linalg.norm_sq(field)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_vecdot_excludes_ghosts(self, device):
        """Test that vecdot correctly excludes ghost regions."""
        skip_if_gpu_unavailable(device)

        nb_grid_pts = [8, 8]
        spatial_dim = 2
        nb_components = 2
        nb_ghosts_left = (1,) * spatial_dim
        nb_ghosts_right = (1,) * spatial_dim

        comm = muGrid.Communicator()
        device_obj = create_device(device)

        if device_obj is None:
            decomposition = muGrid.CartesianDecomposition(
                comm,
                nb_grid_pts,
                nb_subdivisions=(1,) * spatial_dim,
                nb_ghosts_left=nb_ghosts_left,
                nb_ghosts_right=nb_ghosts_right,
            )
        else:
            decomposition = muGrid.CartesianDecomposition(
                comm,
                nb_grid_pts,
                nb_subdivisions=(1,) * spatial_dim,
                nb_ghosts_left=nb_ghosts_left,
                nb_ghosts_right=nb_ghosts_right,
                device=device_obj,
            )

        field_a = decomposition.real_field("test_a", (nb_components,))
        field_b = decomposition.real_field("test_b", (nb_components,))

        # Fill with 1.0 and 2.0
        field_a.s[...] = 1.0
        field_b.s[...] = 2.0

        # Expected: interior pixels * nb_components * (1.0 * 2.0)
        expected = nb_grid_pts[0] * nb_grid_pts[1] * nb_components * 2.0

        result = linalg.vecdot(field_a, field_b)
        assert result == pytest.approx(expected, rel=1e-10)

    def test_axpy_norm_sq_excludes_ghosts(self, device):
        """Test that axpy_norm_sq correctly excludes ghost regions."""
        skip_if_gpu_unavailable(device)

        nb_grid_pts = [8, 8]
        spatial_dim = 2
        nb_components = 2
        nb_ghosts_left = (1,) * spatial_dim
        nb_ghosts_right = (1,) * spatial_dim

        comm = muGrid.Communicator()
        device_obj = create_device(device)

        if device_obj is None:
            decomposition = muGrid.CartesianDecomposition(
                comm,
                nb_grid_pts,
                nb_subdivisions=(1,) * spatial_dim,
                nb_ghosts_left=nb_ghosts_left,
                nb_ghosts_right=nb_ghosts_right,
            )
        else:
            decomposition = muGrid.CartesianDecomposition(
                comm,
                nb_grid_pts,
                nb_subdivisions=(1,) * spatial_dim,
                nb_ghosts_left=nb_ghosts_left,
                nb_ghosts_right=nb_ghosts_right,
                device=device_obj,
            )

        field_x = decomposition.real_field("test_x", (nb_components,))
        field_y = decomposition.real_field("test_y", (nb_components,))

        alpha = 0.5
        fill_x = 2.0
        fill_y = 3.0
        field_x.s[...] = fill_x
        field_y.s[...] = fill_y

        # y_new = alpha * x + y = 0.5 * 2 + 3 = 4
        y_new = alpha * fill_x + fill_y
        expected = nb_grid_pts[0] * nb_grid_pts[1] * nb_components * y_new * y_new

        result = linalg.axpy_norm_sq(alpha, field_x, field_y)
        assert result == pytest.approx(expected, rel=1e-10)


# =============================================================================
# CPU vs GPU comparison tests (dual-device tests)
# =============================================================================


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU backend not available")
class TestCPUGPULinalgComparison:
    """Test that CPU and GPU linalg operations produce identical results.

    These tests are critical regression tests for the SoA memory layout fix.
    """

    def setup_method(self, method):
        """Skip tests if no GPU is available at runtime."""
        if not muGrid.is_gpu_available():
            pytest.skip("No GPU device available at runtime")
        if not HAS_CUPY:
            pytest.skip("CuPy not available for GPU tests")
        self.nb_grid_pts = [16, 16]
        self.spatial_dim = 2
        self.nb_components = 2
        self.nb_ghosts_left = (1,) * self.spatial_dim
        self.nb_ghosts_right = (1,) * self.spatial_dim

        self.comm = muGrid.Communicator()

        # CPU decomposition
        self.decomp_cpu = muGrid.CartesianDecomposition(
            self.comm,
            self.nb_grid_pts,
            nb_subdivisions=(1,) * self.spatial_dim,
            nb_ghosts_left=self.nb_ghosts_left,
            nb_ghosts_right=self.nb_ghosts_right,
        )

        # GPU decomposition
        self.device = muGrid.Device.gpu()
        self.decomp_gpu = muGrid.CartesianDecomposition(
            self.comm,
            self.nb_grid_pts,
            nb_subdivisions=(1,) * self.spatial_dim,
            nb_ghosts_left=self.nb_ghosts_left,
            nb_ghosts_right=self.nb_ghosts_right,
            device=self.device,
        )

    def test_norm_sq_cpu_gpu_match(self):
        """Test that CPU and GPU norm_sq produce identical results."""
        cpu_field = self.decomp_cpu.real_field("cpu_field", (self.nb_components,))
        gpu_field = self.decomp_gpu.real_field("gpu_field", (self.nb_components,))

        # Use constant value (works for both layouts)
        fill_value = 3.5
        cpu_field.s[...] = fill_value
        gpu_field.s[...] = fill_value

        cpu_result = linalg.norm_sq(cpu_field)
        gpu_result = linalg.norm_sq(gpu_field)

        assert cpu_result == pytest.approx(gpu_result, rel=1e-10)

        # Also verify expected value
        expected = (
            self.nb_grid_pts[0]
            * self.nb_grid_pts[1]
            * self.nb_components
            * fill_value
            * fill_value
        )
        assert cpu_result == pytest.approx(expected, rel=1e-10)

    def test_vecdot_cpu_gpu_match(self):
        """Test that CPU and GPU vecdot produce identical results."""
        cpu_a = self.decomp_cpu.real_field("cpu_a", (self.nb_components,))
        cpu_b = self.decomp_cpu.real_field("cpu_b", (self.nb_components,))
        gpu_a = self.decomp_gpu.real_field("gpu_a", (self.nb_components,))
        gpu_b = self.decomp_gpu.real_field("gpu_b", (self.nb_components,))

        fill_a = 2.5
        fill_b = 1.5
        cpu_a.s[...] = fill_a
        cpu_b.s[...] = fill_b
        gpu_a.s[...] = fill_a
        gpu_b.s[...] = fill_b

        cpu_result = linalg.vecdot(cpu_a, cpu_b)
        gpu_result = linalg.vecdot(gpu_a, gpu_b)

        assert cpu_result == pytest.approx(gpu_result, rel=1e-10)

        expected = (
            self.nb_grid_pts[0]
            * self.nb_grid_pts[1]
            * self.nb_components
            * fill_a
            * fill_b
        )
        assert cpu_result == pytest.approx(expected, rel=1e-10)

    def test_axpy_norm_sq_cpu_gpu_match(self):
        """Test that CPU and GPU axpy_norm_sq produce identical results."""
        cpu_x = self.decomp_cpu.real_field("cpu_x", (self.nb_components,))
        cpu_y = self.decomp_cpu.real_field("cpu_y", (self.nb_components,))
        gpu_x = self.decomp_gpu.real_field("gpu_x", (self.nb_components,))
        gpu_y = self.decomp_gpu.real_field("gpu_y", (self.nb_components,))

        alpha = 0.5
        fill_x = 2.0
        fill_y = 3.0

        cpu_x.s[...] = fill_x
        cpu_y.s[...] = fill_y
        gpu_x.s[...] = fill_x
        gpu_y.s[...] = fill_y

        cpu_result = linalg.axpy_norm_sq(alpha, cpu_x, cpu_y)
        gpu_result = linalg.axpy_norm_sq(alpha, gpu_x, gpu_y)

        assert cpu_result == pytest.approx(gpu_result, rel=1e-10)

        # y_new = alpha * x + y = 0.5 * 2 + 3 = 4
        y_new = alpha * fill_x + fill_y
        expected = (
            self.nb_grid_pts[0]
            * self.nb_grid_pts[1]
            * self.nb_components
            * y_new
            * y_new
        )
        assert cpu_result == pytest.approx(expected, rel=1e-10)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU backend not available")
class TestAsymmetricGhosts:
    """Test linalg with asymmetric ghost sizes."""

    def setup_method(self, method):
        """Skip tests if no GPU is available at runtime."""
        if not muGrid.is_gpu_available():
            pytest.skip("No GPU device available at runtime")
        if not HAS_CUPY:
            pytest.skip("CuPy not available for GPU tests")

    def test_norm_sq_asymmetric_ghosts(self):
        """Test norm_sq with asymmetric ghost sizes on CPU and GPU."""
        nb_grid_pts = [10, 10]
        spatial_dim = 2
        nb_components = 2
        nb_ghosts_left = (1, 2)  # Asymmetric
        nb_ghosts_right = (2, 1)  # Asymmetric

        comm = muGrid.Communicator()

        # CPU
        decomp_cpu = muGrid.CartesianDecomposition(
            comm,
            nb_grid_pts,
            nb_subdivisions=(1,) * spatial_dim,
            nb_ghosts_left=nb_ghosts_left,
            nb_ghosts_right=nb_ghosts_right,
        )

        # GPU
        device = muGrid.Device.gpu()
        decomp_gpu = muGrid.CartesianDecomposition(
            comm,
            nb_grid_pts,
            nb_subdivisions=(1,) * spatial_dim,
            nb_ghosts_left=nb_ghosts_left,
            nb_ghosts_right=nb_ghosts_right,
            device=device,
        )

        cpu_field = decomp_cpu.real_field("cpu", (nb_components,))
        gpu_field = decomp_gpu.real_field("gpu", (nb_components,))

        cpu_field.s[...] = 1.0
        gpu_field.s[...] = 1.0

        cpu_result = linalg.norm_sq(cpu_field)
        gpu_result = linalg.norm_sq(gpu_field)

        # Results should match
        assert cpu_result == pytest.approx(gpu_result, rel=1e-10)

        # Expected value
        expected = nb_grid_pts[0] * nb_grid_pts[1] * nb_components
        assert cpu_result == pytest.approx(expected, rel=1e-10)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU backend not available")
class TestDeepCopy:
    """Test deep_copy between CPU and GPU fields.

    deep_copy performs automatic layout conversion between AoS (CPU) and SoA
    (GPU) formats, so logical field values are preserved across the copy.
    """

    def setup_method(self, method):
        """Skip tests if no GPU is available at runtime."""
        if not muGrid.is_gpu_available():
            pytest.skip("No GPU device available at runtime")
        if not HAS_CUPY:
            pytest.skip("CuPy not available for GPU tests")
        self.nb_grid_pts = [8, 8]
        self.spatial_dim = 2
        self.nb_components = 3
        self.nb_ghosts_left = (1,) * self.spatial_dim
        self.nb_ghosts_right = (1,) * self.spatial_dim

        self.comm = muGrid.Communicator()

        # CPU decomposition
        self.decomp_cpu = muGrid.CartesianDecomposition(
            self.comm,
            self.nb_grid_pts,
            nb_subdivisions=(1,) * self.spatial_dim,
            nb_ghosts_left=self.nb_ghosts_left,
            nb_ghosts_right=self.nb_ghosts_right,
        )

        # GPU decomposition
        self.device = muGrid.Device.gpu()
        self.decomp_gpu = muGrid.CartesianDecomposition(
            self.comm,
            self.nb_grid_pts,
            nb_subdivisions=(1,) * self.spatial_dim,
            nb_ghosts_left=self.nb_ghosts_left,
            nb_ghosts_right=self.nb_ghosts_right,
            device=self.device,
        )

    def test_deep_copy_cpu_to_gpu_constant_values(self):
        """Test deep_copy from CPU to GPU with constant values."""
        cpu_field = self.decomp_cpu.real_field("cpu", (self.nb_components,))
        gpu_field = self.decomp_gpu.real_field("gpu", (self.nb_components,))

        fill_value = 2.5
        cpu_field.s[...] = fill_value

        # Copy CPU -> GPU (with layout conversion)
        gpu_field._cpp.deep_copy_from(cpu_field._cpp)

        # Both should give identical norm_sq
        cpu_norm = linalg.norm_sq(cpu_field)
        gpu_norm = linalg.norm_sq(gpu_field)

        assert cpu_norm == pytest.approx(gpu_norm, rel=1e-10)

        # Verify expected value
        expected = (
            self.nb_grid_pts[0]
            * self.nb_grid_pts[1]
            * self.nb_components
            * fill_value
            * fill_value
        )
        assert cpu_norm == pytest.approx(expected, rel=1e-10)

    def test_deep_copy_cpu_to_gpu_varying_values(self):
        """Test deep_copy from CPU to GPU with varying values.

        This verifies that layout conversion correctly transposes the data.
        """
        cpu_field = self.decomp_cpu.real_field("cpu", (self.nb_components,))
        gpu_field = self.decomp_gpu.real_field("gpu", (self.nb_components,))

        # Fill CPU field with varying values
        arr = np.asarray(cpu_field.s)
        for i in range(arr.size):
            arr.flat[i] = (i % 17 + 1) * 0.5

        # Compute expected norm_sq from CPU field
        cpu_norm = linalg.norm_sq(cpu_field)

        # Copy CPU -> GPU (with layout conversion)
        gpu_field._cpp.deep_copy_from(cpu_field._cpp)

        # GPU norm_sq should match CPU if layout conversion worked correctly
        gpu_norm = linalg.norm_sq(gpu_field)

        assert cpu_norm == pytest.approx(gpu_norm, rel=1e-10)

    def test_deep_copy_round_trip_cpu_gpu_cpu(self):
        """Test round-trip deep_copy: CPU -> GPU -> CPU preserves values."""
        cpu_field1 = self.decomp_cpu.real_field("cpu1", (self.nb_components,))
        gpu_field = self.decomp_gpu.real_field("gpu", (self.nb_components,))
        cpu_field2 = self.decomp_cpu.real_field("cpu2", (self.nb_components,))

        # Fill with varying values
        arr = np.asarray(cpu_field1.s)
        for i in range(arr.size):
            arr.flat[i] = (i % 13 + 1) * 0.3

        # Round trip: CPU -> GPU -> CPU
        gpu_field._cpp.deep_copy_from(cpu_field1._cpp)
        cpu_field2._cpp.deep_copy_from(gpu_field._cpp)

        # Original and final CPU fields should have identical values
        cpu1_norm = linalg.norm_sq(cpu_field1)
        cpu2_norm = linalg.norm_sq(cpu_field2)

        assert cpu1_norm == pytest.approx(cpu2_norm, rel=1e-10)

        # Verify bytes are identical
        np.testing.assert_array_almost_equal(
            np.asarray(cpu_field1.s).flat, np.asarray(cpu_field2.s).flat, decimal=14
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

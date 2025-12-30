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

import unittest

import numpy as np

import muGrid
from muGrid import linalg

# Try to import CuPy for GPU tests
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

# Use compile-time feature flag for GPU availability
GPU_AVAILABLE = muGrid.has_gpu


class HostLinalgGhostTests(unittest.TestCase):
    """Test linalg operations with ghost regions on host."""

    def setUp(self):
        self.nb_grid_pts = [8, 8]
        self.spatial_dim = 2
        self.nb_components = 2
        self.nb_ghosts_left = (1,) * self.spatial_dim
        self.nb_ghosts_right = (1,) * self.spatial_dim

        self.comm = muGrid.Communicator()
        self.decomposition = muGrid.CartesianDecomposition(
            self.comm,
            self.nb_grid_pts,
            nb_subdivisions=(1,) * self.spatial_dim,
            nb_ghosts_left=self.nb_ghosts_left,
            nb_ghosts_right=self.nb_ghosts_right,
        )

    def test_norm_sq_excludes_ghosts(self):
        """Test that norm_sq correctly excludes ghost regions."""
        field = self.decomposition.real_field("test", (self.nb_components,))

        # Fill entire field (including ghosts) with 1.0
        field.s[...] = 1.0

        # Expected: only interior pixels should be counted
        # Interior = nb_grid_pts[0] * nb_grid_pts[1] pixels
        # Each with nb_components values of 1.0^2 = 1.0
        expected = (
            self.nb_grid_pts[0] * self.nb_grid_pts[1] * self.nb_components
        )

        result = linalg.norm_sq(field)
        self.assertAlmostEqual(result, expected, places=10)

    def test_vecdot_excludes_ghosts(self):
        """Test that vecdot correctly excludes ghost regions."""
        field_a = self.decomposition.real_field("test_a", (self.nb_components,))
        field_b = self.decomposition.real_field("test_b", (self.nb_components,))

        # Fill with 1.0 and 2.0
        field_a.s[...] = 1.0
        field_b.s[...] = 2.0

        # Expected: interior pixels * nb_components * (1.0 * 2.0)
        expected = (
            self.nb_grid_pts[0] * self.nb_grid_pts[1] * self.nb_components * 2.0
        )

        result = linalg.vecdot(field_a, field_b)
        self.assertAlmostEqual(result, expected, places=10)

    def test_axpy_norm_sq_excludes_ghosts(self):
        """Test that axpy_norm_sq correctly excludes ghost regions."""
        field_x = self.decomposition.real_field("test_x", (self.nb_components,))
        field_y = self.decomposition.real_field("test_y", (self.nb_components,))

        alpha = 0.5
        fill_x = 2.0
        fill_y = 3.0
        field_x.s[...] = fill_x
        field_y.s[...] = fill_y

        # y_new = alpha * x + y = 0.5 * 2 + 3 = 4
        y_new = alpha * fill_x + fill_y
        expected = (
            self.nb_grid_pts[0] * self.nb_grid_pts[1] * self.nb_components *
            y_new * y_new
        )

        result = linalg.axpy_norm_sq(alpha, field_x, field_y)
        self.assertAlmostEqual(result, expected, places=10)


@unittest.skipUnless(GPU_AVAILABLE and HAS_CUPY, "GPU support not available")
class GPULinalgGhostTests(unittest.TestCase):
    """Test linalg operations with ghost regions on GPU.

    These tests specifically verify the SoA (Structure of Arrays) memory
    layout ghost subtraction which was fixed in the GPU kernels.
    """

    def setUp(self):
        self.nb_grid_pts = [8, 8]
        self.spatial_dim = 2
        self.nb_components = 2
        self.nb_ghosts_left = (1,) * self.spatial_dim
        self.nb_ghosts_right = (1,) * self.spatial_dim

        self.comm = muGrid.Communicator()
        self.device = muGrid.Device.cuda(0)
        self.decomposition = muGrid.CartesianDecomposition(
            self.comm,
            self.nb_grid_pts,
            nb_subdivisions=(1,) * self.spatial_dim,
            nb_ghosts_left=self.nb_ghosts_left,
            nb_ghosts_right=self.nb_ghosts_right,
            device=self.device,
        )

    def test_norm_sq_gpu_excludes_ghosts(self):
        """Test that GPU norm_sq correctly excludes ghost regions (SoA layout)."""
        field = self.decomposition.real_field("test", (self.nb_components,))

        # Fill entire field with 1.0
        field.s[...] = 1.0

        # Expected: only interior pixels counted
        expected = (
            self.nb_grid_pts[0] * self.nb_grid_pts[1] * self.nb_components
        )

        result = linalg.norm_sq(field)
        self.assertAlmostEqual(result, expected, places=10)

    def test_vecdot_gpu_excludes_ghosts(self):
        """Test that GPU vecdot correctly excludes ghost regions (SoA layout)."""
        field_a = self.decomposition.real_field("test_a", (self.nb_components,))
        field_b = self.decomposition.real_field("test_b", (self.nb_components,))

        field_a.s[...] = 1.0
        field_b.s[...] = 2.0

        expected = (
            self.nb_grid_pts[0] * self.nb_grid_pts[1] * self.nb_components * 2.0
        )

        result = linalg.vecdot(field_a, field_b)
        self.assertAlmostEqual(result, expected, places=10)

    def test_axpy_norm_sq_gpu_excludes_ghosts(self):
        """Test that GPU axpy_norm_sq correctly excludes ghost regions (SoA layout)."""
        field_x = self.decomposition.real_field("test_x", (self.nb_components,))
        field_y = self.decomposition.real_field("test_y", (self.nb_components,))

        alpha = 0.5
        fill_x = 2.0
        fill_y = 3.0
        field_x.s[...] = fill_x
        field_y.s[...] = fill_y

        # y_new = alpha * x + y = 0.5 * 2 + 3 = 4
        y_new = alpha * fill_x + fill_y
        expected = (
            self.nb_grid_pts[0] * self.nb_grid_pts[1] * self.nb_components *
            y_new * y_new
        )

        result = linalg.axpy_norm_sq(alpha, field_x, field_y)
        self.assertAlmostEqual(result, expected, places=10)


@unittest.skipUnless(GPU_AVAILABLE and HAS_CUPY, "GPU support not available")
class CPUGPULinalgComparisonTests(unittest.TestCase):
    """Test that CPU and GPU linalg operations produce identical results.

    These tests are critical regression tests for the SoA memory layout fix.
    """

    def setUp(self):
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
        self.device = muGrid.Device.cuda(0)
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

        self.assertAlmostEqual(cpu_result, gpu_result, places=10)

        # Also verify expected value
        expected = (
            self.nb_grid_pts[0] * self.nb_grid_pts[1] * self.nb_components *
            fill_value * fill_value
        )
        self.assertAlmostEqual(cpu_result, expected, places=10)

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

        self.assertAlmostEqual(cpu_result, gpu_result, places=10)

        expected = (
            self.nb_grid_pts[0] * self.nb_grid_pts[1] * self.nb_components *
            fill_a * fill_b
        )
        self.assertAlmostEqual(cpu_result, expected, places=10)

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

        self.assertAlmostEqual(cpu_result, gpu_result, places=10)

        # y_new = alpha * x + y = 0.5 * 2 + 3 = 4
        y_new = alpha * fill_x + fill_y
        expected = (
            self.nb_grid_pts[0] * self.nb_grid_pts[1] * self.nb_components *
            y_new * y_new
        )
        self.assertAlmostEqual(cpu_result, expected, places=10)


@unittest.skipUnless(GPU_AVAILABLE and HAS_CUPY, "GPU support not available")
class AsymmetricGhostTests(unittest.TestCase):
    """Test linalg with asymmetric ghost sizes."""

    def test_norm_sq_asymmetric_ghosts(self):
        """Test norm_sq with asymmetric ghost sizes on CPU and GPU."""
        nb_grid_pts = [10, 10]
        spatial_dim = 2
        nb_components = 2
        nb_ghosts_left = (1, 2)   # Asymmetric
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
        device = muGrid.Device.cuda(0)
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
        self.assertAlmostEqual(cpu_result, gpu_result, places=10)

        # Expected value
        expected = nb_grid_pts[0] * nb_grid_pts[1] * nb_components
        self.assertAlmostEqual(cpu_result, expected, places=10)


# Note: deep_copy_from method is not exposed in Python bindings.
# deep_copy tests are only available in the C++ test suite (test_linalg.cc).
# If deep_copy functionality is needed from Python in the future, the binding
# would need to be added to language_bindings/python/bind_py_*.cc


if __name__ == "__main__":
    unittest.main()

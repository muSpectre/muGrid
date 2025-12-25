#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file     python_field_gpu_tests.py

@author  Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date    09 Dec 2024

@brief   Test GPU field functionality (CUDA/ROCm)

Copyright © 2024 Lars Pastewka

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

# Try to import CuPy for GPU tests
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


# Use compile-time feature flag for GPU availability
GPU_AVAILABLE = muGrid.has_gpu


class HostFieldDeviceInfoTests(unittest.TestCase):
    """Test device introspection for host fields."""

    def setUp(self):
        self.nb_grid_pts = (8, 8)
        self.fc = muGrid.GlobalFieldCollection(self.nb_grid_pts)

    def test_host_field_device_property(self):
        """Test that host fields report 'cpu' as device."""
        field_cpp = self.fc.register_real_field("test", 3)
        self.assertEqual(field_cpp.device, "cpu")

    def test_host_field_is_on_gpu(self):
        """Test that host fields report is_on_gpu = False."""
        field_cpp = self.fc.register_real_field("test", 3)
        self.assertFalse(field_cpp.is_on_gpu)

    def test_host_field_dlpack_device(self):
        """Test that host fields report correct DLPack device tuple."""
        field_cpp = self.fc.register_real_field("test", 3)
        device_type, device_id = field_cpp.__dlpack_device__()
        # kDLCPU = 1
        self.assertEqual(device_type, 1)
        self.assertEqual(device_id, 0)

    def test_wrapped_field_device_properties(self):
        """Test device properties through Python Field wrapper."""
        field = self.fc.real_field("wrapped_test", (3,))
        self.assertEqual(field.device, "cpu")
        self.assertFalse(field.is_on_gpu)

    def test_host_collection_is_on_device(self):
        """Test that host collection reports is_on_device = False."""
        self.assertFalse(self.fc.is_on_device)

    def test_host_collection_memory_location(self):
        """Test that host collection reports Host memory location."""
        self.assertEqual(self.fc.memory_location,
                         muGrid.GlobalFieldCollection.MemoryLocation.Host)


class HostFieldAccessTests(unittest.TestCase):
    """Test host field memory access."""

    def setUp(self):
        self.nb_grid_pts = (8, 8)
        self.fc = muGrid.GlobalFieldCollection(self.nb_grid_pts)

    def test_host_field_returns_numpy(self):
        """Test that host field accessors return numpy arrays."""
        field = self.fc.real_field("numpy_test", (3,))
        arr = field.s
        self.assertIsInstance(arr, np.ndarray)

    def test_all_field_types_host(self):
        """Test all field types with host memory space."""
        real_f = self.fc.real_field("real_host")
        int_f = self.fc.int_field("int_host")
        uint_f = self.fc.uint_field("uint_host")
        complex_f = self.fc.complex_field("complex_host")

        for f in [real_f, int_f, uint_f, complex_f]:
            self.assertFalse(f.is_on_gpu)
            self.assertEqual(f.device, "cpu")


@unittest.skipUnless(GPU_AVAILABLE, "GPU backend not available")
class DeviceCollectionTests(unittest.TestCase):
    """Test device collection creation (requires GPU backend)."""

    def setUp(self):
        self.nb_grid_pts = (8, 8)
        # Create a device collection
        self.fc = muGrid.GlobalFieldCollection(
            self.nb_grid_pts,
            memory_location=muGrid.GlobalFieldCollection.MemoryLocation.Device
        )

    def test_device_collection_is_on_device(self):
        """Test that device collection reports is_on_device = True."""
        self.assertTrue(self.fc.is_on_device)

    def test_device_collection_memory_location(self):
        """Test that device collection reports Device memory location."""
        self.assertEqual(self.fc.memory_location,
                         muGrid.GlobalFieldCollection.MemoryLocation.Device)

    def test_device_field_is_on_gpu(self):
        """Test that fields in device collection report is_on_gpu = True."""
        field = self.fc.register_real_field("device_test", 3)
        self.assertTrue(field.is_on_gpu)

    def test_device_field_device_property(self):
        """Test that device fields report correct device string."""
        field = self.fc.register_real_field("device_test", 3)
        # Should be "cuda:0" or "rocm:0"
        device = field.device
        self.assertTrue(device.startswith("cuda:") or device.startswith("rocm:"))

    def test_device_field_dlpack_device(self):
        """Test that device fields report correct DLPack device tuple."""
        field = self.fc.register_real_field("device_test", 3)
        device_type, device_id = field.__dlpack_device__()
        # kDLCUDA = 2, kDLROCm = 10
        self.assertIn(device_type, [2, 10])
        self.assertEqual(device_id, 0)

    def test_all_device_field_types(self):
        """Test creation of all device field types."""
        real_f = self.fc.register_real_field("real_device", 2)
        complex_f = self.fc.register_complex_field("complex_device", 2)
        int_f = self.fc.register_int_field("int_device", 2)
        uint_f = self.fc.register_uint_field("uint_device", 2)

        for f in [real_f, complex_f, int_f, uint_f]:
            self.assertTrue(f.is_on_gpu)

    def test_device_field_set_zero(self):
        """Test that set_zero works on device fields."""
        field = self.fc.register_real_field("zero_test", 3)
        # Should not raise
        field.set_zero()


@unittest.skipUnless(GPU_AVAILABLE, "GPU backend not available")
class DeviceFieldFactoryTests(unittest.TestCase):
    """Test device field creation via factory functions."""

    def setUp(self):
        self.nb_grid_pts = (8, 8)
        # Create a device collection
        self.fc = muGrid.GlobalFieldCollection(
            self.nb_grid_pts,
            memory_location=muGrid.GlobalFieldCollection.MemoryLocation.Device
        )

    def test_real_field_on_device_collection(self):
        """Test creating real field on device collection."""
        field = self.fc.real_field("device_real", (3,))
        self.assertTrue(field.is_on_gpu)

    def test_all_field_types_device(self):
        """Test all field factory functions on device collection."""
        real_f = self.fc.real_field("real_dev")
        int_f = self.fc.int_field("int_dev")
        uint_f = self.fc.uint_field("uint_dev")
        complex_f = self.fc.complex_field("complex_dev")

        for f in [real_f, int_f, uint_f, complex_f]:
            self.assertTrue(f.is_on_gpu)


@unittest.skipUnless(GPU_AVAILABLE and HAS_CUPY, "GPU backend or CuPy not available")
class CuPyIntegrationTests(unittest.TestCase):
    """Test CuPy integration for GPU fields."""

    def setUp(self):
        self.nb_grid_pts = (8, 8)
        # Create a device collection
        self.fc = muGrid.GlobalFieldCollection(
            self.nb_grid_pts,
            memory_location=muGrid.GlobalFieldCollection.MemoryLocation.Device
        )

    def test_device_field_returns_cupy_array(self):
        """Test that device field accessors return CuPy arrays."""
        field = self.fc.real_field("cupy_test", (3,))

        # Access should return CuPy array
        arr = field.s
        self.assertIsInstance(arr, cp.ndarray)

    def test_cupy_array_shape(self):
        """Test that CuPy array has correct shape."""
        field = self.fc.real_field("shape_test", (2, 3))

        arr = field.s
        # Shape should be (2, 3, 1) + nb_grid_pts for SubPt layout
        expected_shape = (2, 3, 1) + self.nb_grid_pts
        self.assertEqual(arr.shape, expected_shape)

    def test_cupy_array_writability(self):
        """Test that CuPy arrays from device fields are writable."""
        field = self.fc.real_field("write_test", (2,))

        arr = field.s
        # Fill with values
        arr[:] = 42.0

        # Read back
        cp.testing.assert_array_equal(arr, 42.0)

    def test_cupy_numpy_round_trip(self):
        """Test data transfer between CuPy (device) and NumPy (host)."""
        # Create host collection
        fc_host = muGrid.GlobalFieldCollection(self.nb_grid_pts)

        # Create both host and device fields
        host_field = fc_host.real_field("host", (2,))
        device_field = self.fc.real_field("device", (2,))

        # Initialize host field with test data
        test_data = np.random.rand(*host_field.s.shape).astype(np.float64)
        host_field.s[...] = test_data

        # Copy host to device via CuPy
        device_arr = device_field.s
        device_arr[...] = cp.asarray(host_field.s)

        # Read back from device to host
        result = cp.asnumpy(device_arr)

        # Should match
        np.testing.assert_allclose(result, test_data)

    def test_all_accessors_return_cupy(self):
        """Test that all field accessors (s, sg, p, pg) return CuPy arrays."""
        field = self.fc.real_field("accessor_test", (2,))

        # All accessors should return CuPy arrays
        self.assertIsInstance(field.s, cp.ndarray)
        self.assertIsInstance(field.sg, cp.ndarray)
        self.assertIsInstance(field.p, cp.ndarray)
        self.assertIsInstance(field.pg, cp.ndarray)


@unittest.skipUnless(GPU_AVAILABLE and HAS_CUPY, "GPU backend or CuPy not available")
class MixedHostDeviceTests(unittest.TestCase):
    """Test scenarios with both host and device collections."""

    def setUp(self):
        self.nb_grid_pts = (8, 8)
        # Create host collection
        self.fc_host = muGrid.GlobalFieldCollection(self.nb_grid_pts)
        # Create device collection
        self.fc_device = muGrid.GlobalFieldCollection(
            self.nb_grid_pts,
            memory_location=muGrid.GlobalFieldCollection.MemoryLocation.Device
        )

    def test_host_and_device_collections(self):
        """Test that host and device collections can coexist."""
        host_field = self.fc_host.register_real_field("host", 2)
        device_field = self.fc_device.register_real_field("device", 2)

        self.assertFalse(host_field.is_on_gpu)
        self.assertTrue(device_field.is_on_gpu)

        # Both should be accessible via DLPack
        np.from_dlpack(host_field)
        cp.from_dlpack(device_field)

    def test_different_array_types(self):
        """Test that host returns numpy, device returns cupy."""
        host_field = self.fc_host.real_field("host")
        device_field = self.fc_device.real_field("device")

        host_arr = host_field.s
        device_arr = device_field.s

        self.assertIsInstance(host_arr, np.ndarray)
        self.assertIsInstance(device_arr, cp.ndarray)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file     python_fft_tests.py

@author  Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date    18 Dec 2024

@brief   test FFT engine Python bindings

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
from numpy.testing import assert_allclose, assert_array_equal

import muGrid


class FFTUtilsTest(unittest.TestCase):
    """Test FFT utility functions."""

    def test_get_hermitian_grid_pts_2d(self):
        """Test hermitian grid dimensions for 2D."""
        fourier = muGrid.get_hermitian_grid_pts([8, 10])
        self.assertEqual(fourier[0], 5)  # 8/2 + 1
        self.assertEqual(fourier[1], 10)

    def test_get_hermitian_grid_pts_3d(self):
        """Test hermitian grid dimensions for 3D."""
        fourier = muGrid.get_hermitian_grid_pts([8, 10, 12])
        self.assertEqual(fourier[0], 5)  # 8/2 + 1
        self.assertEqual(fourier[1], 10)
        self.assertEqual(fourier[2], 12)

    def test_fft_normalization(self):
        """Test FFT normalization factor."""
        norm_2d = muGrid.fft_normalization([8, 10])
        self.assertAlmostEqual(norm_2d, 1.0 / 80, places=15)

        norm_3d = muGrid.fft_normalization([8, 10, 12])
        self.assertAlmostEqual(norm_3d, 1.0 / 960, places=15)


class FFTEngineCreationTest(unittest.TestCase):
    """Test FFT engine creation and properties."""

    def test_create_2d_engine(self):
        """Test creating a 2D FFT engine."""
        engine = muGrid.FFTEngine([16, 20])

        # Check real space dimensions
        self.assertEqual(engine.nb_domain_grid_pts[0], 16)
        self.assertEqual(engine.nb_domain_grid_pts[1], 20)

        # Check Fourier space dimensions (r2c transform)
        self.assertEqual(engine.nb_fourier_grid_pts[0], 9)  # 16/2 + 1
        self.assertEqual(engine.nb_fourier_grid_pts[1], 20)

    def test_create_3d_engine(self):
        """Test creating a 3D FFT engine."""
        engine = muGrid.FFTEngine([8, 10, 12])

        self.assertEqual(engine.nb_domain_grid_pts[0], 8)
        self.assertEqual(engine.nb_domain_grid_pts[1], 10)
        self.assertEqual(engine.nb_domain_grid_pts[2], 12)

        self.assertEqual(engine.nb_fourier_grid_pts[0], 5)  # 8/2 + 1
        self.assertEqual(engine.nb_fourier_grid_pts[1], 10)
        self.assertEqual(engine.nb_fourier_grid_pts[2], 12)

    def test_normalization(self):
        """Test normalization factor."""
        engine_2d = muGrid.FFTEngine([8, 10])
        self.assertAlmostEqual(engine_2d.normalisation, 1.0 / 80, places=15)

        engine_3d = muGrid.FFTEngine([8, 10, 12])
        self.assertAlmostEqual(engine_3d.normalisation, 1.0 / 960, places=15)

    def test_backend_name(self):
        """Test backend name is reported."""
        engine = muGrid.FFTEngine([8, 10])
        self.assertEqual(engine.backend_name, "PocketFFT")


class FFTFieldTest(unittest.TestCase):
    """Test FFT field creation and access."""

    def setUp(self):
        self.engine_2d = muGrid.FFTEngine([16, 20])
        self.engine_3d = muGrid.FFTEngine([8, 10, 12])

    def test_create_real_field(self):
        """Test creating real-space field."""
        field = self.engine_2d.real_space_field("test_real")
        self.assertEqual(field.name, "test_real")
        self.assertEqual(field.s.shape, (1, 1, 16, 20))

    def test_create_fourier_field(self):
        """Test creating Fourier-space field."""
        field = self.engine_2d.fourier_space_field("test_fourier")
        self.assertEqual(field.name, "test_fourier")
        self.assertEqual(field.s.shape, (1, 1, 9, 20))  # Half-complex

    def test_field_data_types(self):
        """Test field data types."""
        real_field = self.engine_2d.real_space_field("real")
        fourier_field = self.engine_2d.fourier_space_field("fourier")

        self.assertEqual(real_field.s.dtype, np.float64)
        self.assertEqual(fourier_field.s.dtype, np.complex128)

    def test_field_pixel_layout(self):
        """Test pixel layout access."""
        real_field = self.engine_2d.real_space_field("real_p")

        # SubPt layout: (components, sub_pts, nx, ny)
        self.assertEqual(real_field.s.shape, (1, 1, 16, 20))

        # Pixel layout: (components * sub_pts, nx, ny)
        self.assertEqual(real_field.p.shape, (1, 16, 20))

    def test_field_3d_shapes(self):
        """Test 3D field shapes."""
        real_field = self.engine_3d.real_space_field("real_3d")
        fourier_field = self.engine_3d.fourier_space_field("fourier_3d")

        self.assertEqual(real_field.s.shape, (1, 1, 8, 10, 12))
        self.assertEqual(fourier_field.s.shape, (1, 1, 5, 10, 12))

    def test_field_write_access(self):
        """Test that field data can be written."""
        field = self.engine_2d.real_space_field("writable")
        field.s[:] = 1.0
        self.assertTrue(np.all(field.s == 1.0))

        field.p[:] = 2.0
        self.assertTrue(np.all(field.p == 2.0))


class FFTRoundtripTest(unittest.TestCase):
    """Test FFT forward/inverse roundtrip."""

    def test_2d_roundtrip_sine(self):
        """Test 2D FFT roundtrip with sine wave."""
        engine = muGrid.FFTEngine([16, 20])
        real_field = engine.real_space_field("real")
        fourier_field = engine.fourier_space_field("fourier")

        # Initialize with sine wave
        x = np.arange(16 * 20).reshape(1, 1, 16, 20)
        real_field.s[:] = np.sin(2 * np.pi * x / (16 * 20))
        original = real_field.s.copy()

        # Forward and inverse FFT
        engine.fft(real_field, fourier_field)
        engine.ifft(fourier_field, real_field)

        # Normalize
        real_field.s[:] *= engine.normalisation

        # Check roundtrip
        assert_allclose(real_field.s, original, atol=1e-14)

    def test_2d_roundtrip_random(self):
        """Test 2D FFT roundtrip with random data."""
        engine = muGrid.FFTEngine([16, 20])
        real_field = engine.real_space_field("real")
        fourier_field = engine.fourier_space_field("fourier")

        # Initialize with random data
        np.random.seed(42)
        real_field.s[:] = np.random.randn(1, 1, 16, 20)
        original = real_field.s.copy()

        # Forward and inverse FFT
        engine.fft(real_field, fourier_field)
        engine.ifft(fourier_field, real_field)
        real_field.s[:] *= engine.normalisation

        # Check roundtrip
        assert_allclose(real_field.s, original, atol=1e-14)

    def test_3d_roundtrip(self):
        """Test 3D FFT roundtrip."""
        engine = muGrid.FFTEngine([8, 10, 12])
        real_field = engine.real_space_field("real")
        fourier_field = engine.fourier_space_field("fourier")

        # Initialize
        np.random.seed(42)
        real_field.s[:] = np.random.randn(1, 1, 8, 10, 12)
        original = real_field.s.copy()

        # Forward and inverse FFT
        engine.fft(real_field, fourier_field)
        engine.ifft(fourier_field, real_field)
        real_field.s[:] *= engine.normalisation

        # Check roundtrip
        assert_allclose(real_field.s, original, atol=1e-14)

    def test_fft_dc_component(self):
        """Test that constant field has only DC component."""
        engine = muGrid.FFTEngine([8, 10])
        real_field = engine.real_space_field("real")
        fourier_field = engine.fourier_space_field("fourier")

        # Constant field
        real_field.s[:] = 5.0

        engine.fft(real_field, fourier_field)

        # DC component should be sum = 5 * 80 = 400
        self.assertAlmostEqual(fourier_field.s[0, 0, 0, 0].real, 400.0, places=10)
        self.assertAlmostEqual(fourier_field.s[0, 0, 0, 0].imag, 0.0, places=10)

        # All other components should be zero
        fourier_copy = fourier_field.s.copy()
        fourier_copy[0, 0, 0, 0] = 0
        assert_allclose(fourier_copy, 0, atol=1e-12)


class FFTFrequencyTest(unittest.TestCase):
    """Test FFT frequency and coordinate properties."""

    def test_fftfreq_2d(self):
        """Test 2D fftfreq property matches numpy."""
        nb_grid_pts = [7, 4]
        nx, ny = nb_grid_pts
        engine = muGrid.FFTEngine(nb_grid_pts)

        # Build reference from numpy
        freq_ref = np.array(
            np.meshgrid(*(np.fft.fftfreq(n) for n in nb_grid_pts), indexing="ij")
        )
        # Slice for half-complex (r2c) transform
        freq_ref = freq_ref[:, : nx // 2 + 1, :]

        assert_allclose(engine.fftfreq, freq_ref)

    def test_ifftfreq_2d(self):
        """Test 2D ifftfreq property (integer indices)."""
        nb_grid_pts = [7, 4]
        nx, ny = nb_grid_pts
        engine = muGrid.FFTEngine(nb_grid_pts)

        # Build reference from numpy
        freq_ref = np.array(
            np.meshgrid(
                *(np.fft.fftfreq(n, 1 / n) for n in nb_grid_pts),
                indexing="ij",
            )
        )
        freq_ref = freq_ref[:, : nx // 2 + 1, :]

        assert_allclose(engine.ifftfreq, freq_ref)

    def test_fftfreq_3d(self):
        """Test 3D fftfreq property matches numpy."""
        nb_grid_pts = [6, 4, 5]
        nx, ny, nz = nb_grid_pts
        engine = muGrid.FFTEngine(nb_grid_pts)

        # Build reference from numpy
        freq_ref = np.array(
            np.meshgrid(*(np.fft.fftfreq(n) for n in nb_grid_pts), indexing="ij")
        )
        # Slice for half-complex (r2c) transform
        freq_ref = freq_ref[:, : nx // 2 + 1, :, :]

        assert_allclose(engine.fftfreq, freq_ref)

    def test_coords_2d(self):
        """Test 2D coords property."""
        nb_grid_pts = [7, 4]
        nx, ny = nb_grid_pts
        engine = muGrid.FFTEngine(nb_grid_pts)

        x, y = engine.coords
        xref, yref = np.mgrid[0:nx, 0:ny]

        assert_allclose(x, xref / nx)
        assert_allclose(y, yref / ny)

    def test_coords_3d(self):
        """Test 3D coords property."""
        nb_grid_pts = [7, 4, 5]
        nx, ny, nz = nb_grid_pts
        engine = muGrid.FFTEngine(nb_grid_pts)

        assert_array_equal(engine.coords.shape, [3] + nb_grid_pts)

        x, y, z = engine.coords
        xref, yref, zref = np.mgrid[0:nx, 0:ny, 0:nz]

        assert_allclose(x, xref / nx)
        assert_allclose(y, yref / ny)
        assert_allclose(z, zref / nz)

    def test_icoords_2d(self):
        """Test 2D icoords property (integer indices)."""
        nb_grid_pts = [7, 4]
        nx, ny = nb_grid_pts
        engine = muGrid.FFTEngine(nb_grid_pts)

        x, y = engine.coords
        ix, iy = engine.icoords

        # Integer coords should equal fractional * n
        assert_allclose(ix, x * nx)
        assert_allclose(iy, y * ny)

    def test_icoords_dtype(self):
        """Test that icoords returns integer type."""
        engine = muGrid.FFTEngine([7, 4])
        ix, iy = engine.icoords
        self.assertTrue(np.issubdtype(ix.dtype, np.integer))
        self.assertTrue(np.issubdtype(iy.dtype, np.integer))

    def test_ifftfreq_dtype(self):
        """Test that ifftfreq returns integer type."""
        engine = muGrid.FFTEngine([7, 4])
        iqx, iqy = engine.ifftfreq
        self.assertTrue(np.issubdtype(iqx.dtype, np.integer))
        self.assertTrue(np.issubdtype(iqy.dtype, np.integer))

    def test_frequency_coordinate_consistency(self):
        """Test that setting Fourier modes using fftfreq produces correct result."""
        nb_grid_pts = [7, 4]
        nx, ny = nb_grid_pts
        engine = muGrid.FFTEngine(nb_grid_pts)

        x, y = engine.coords
        qx, qy = engine.fftfreq

        real_field = engine.real_space_field("real")
        fourier_field = engine.fourier_space_field("fourier")

        # Set a single x-direction mode
        fourier_field.s[:] = 0
        fourier_field.s[
            0,
            0,
            np.logical_and(
                np.abs(np.abs(qx) * nx - 1) < 1e-6, np.abs(np.abs(qy) * ny - 0) < 1e-6
            ),
        ] = 0.5

        engine.ifft(fourier_field, real_field)
        # Should produce cos(2*pi*x)
        assert_allclose(real_field.p[0], np.cos(2 * np.pi * x), atol=1e-12)

        # Set a single y-direction mode
        fourier_field.s[:] = 0
        fourier_field.s[
            0,
            0,
            np.logical_and(
                np.abs(np.abs(qx) * nx - 0) < 1e-6, np.abs(np.abs(qy) * ny - 1) < 1e-6
            ),
        ] = 0.5

        engine.ifft(fourier_field, real_field)
        # Should produce cos(2*pi*y)
        assert_allclose(real_field.p[0], np.cos(2 * np.pi * y), atol=1e-12)

    def test_properties_return_tuples(self):
        """Test that dimension properties return tuples."""
        engine = muGrid.FFTEngine([8, 10])

        # These should all be tuples, not DynGridIndex objects
        self.assertIsInstance(engine.nb_fourier_grid_pts, tuple)
        self.assertIsInstance(engine.nb_fourier_subdomain_grid_pts, tuple)
        self.assertIsInstance(engine.fourier_subdomain_locations, tuple)
        self.assertIsInstance(engine.nb_subdomain_grid_pts, tuple)
        self.assertIsInstance(engine.subdomain_locations, tuple)
        self.assertIsInstance(engine.process_grid, tuple)
        self.assertIsInstance(engine.process_coords, tuple)

    def test_spatial_dim_property(self):
        """Test spatial_dim property."""
        engine_2d = muGrid.FFTEngine([8, 10])
        self.assertEqual(engine_2d.spatial_dim, 2)

        engine_3d = muGrid.FFTEngine([8, 10, 12])
        self.assertEqual(engine_3d.spatial_dim, 3)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file     python_mpi_fft_tests.py

@author  Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date    19 Dec 2024

@brief   Test MPI parallel FFT engine Python bindings

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

import numpy as np
import pytest
from numpy.testing import assert_allclose

from muGrid import FFTEngine


def make_subdomain_slices(locations, nb_pts):
    """Create slice objects for extracting subdomain from global array.

    Parameters
    ----------
    locations : tuple
        Starting indices of subdomain
    nb_pts : tuple
        Number of points in subdomain

    Returns
    -------
    tuple of slices
        Slices for each dimension
    """
    return tuple(slice(loc, loc + n) for loc, n in zip(locations, nb_pts))


def make_fourier_slices(locations, nb_pts):
    """Create slice objects for extracting subdomain from global Fourier array.

    Parameters
    ----------
    locations : tuple
        Starting indices of Fourier subdomain
    nb_pts : tuple
        Number of points in Fourier subdomain

    Returns
    -------
    tuple of slices
        Slices for each dimension
    """
    return tuple(slice(loc, loc + n) for loc, n in zip(locations, nb_pts))


class TestMPIFFTEngineConstruction:
    """Test FFT engine construction with MPI."""

    def test_create_engine_with_communicator(self, comm):
        """Test that FFT engine can be created with an MPI communicator."""
        nb_grid_pts = [16 * max(1, comm.size), 20]
        engine = FFTEngine(nb_grid_pts, comm)

        # Check global dimensions
        assert engine.nb_domain_grid_pts[0] == nb_grid_pts[0]
        assert engine.nb_domain_grid_pts[1] == nb_grid_pts[1]

        # Check Fourier dimensions (r2c transform)
        assert engine.nb_fourier_grid_pts[0] == nb_grid_pts[0] // 2 + 1
        assert engine.nb_fourier_grid_pts[1] == nb_grid_pts[1]

    def test_subdomain_decomposition_sums_to_total(self, comm):
        """Test that subdomain sizes sum to total domain size."""
        nb_grid_pts = [16 * max(1, comm.size), 20]
        engine = FFTEngine(nb_grid_pts, comm)

        # Sum subdomain sizes across all ranks
        local_pts = np.prod(engine.nb_subdomain_grid_pts)
        total_pts = comm.sum(local_pts)
        assert total_pts == np.prod(nb_grid_pts)

    def test_subdomain_decomposition_3d(self, comm):
        """Test 3D subdomain decomposition."""
        nb_grid_pts = [8 * max(1, comm.size), 10, 12]
        engine = FFTEngine(nb_grid_pts, comm)

        # Sum subdomain sizes across all ranks
        local_pts = np.prod(engine.nb_subdomain_grid_pts)
        total_pts = comm.sum(local_pts)
        assert total_pts == np.prod(nb_grid_pts)

    def test_fourier_subdomain_decomposition_sums_to_total(self, comm):
        """Test that Fourier subdomain sizes sum to total Fourier domain."""
        nb_grid_pts = [16 * max(1, comm.size), 20]
        engine = FFTEngine(nb_grid_pts, comm)

        # Sum Fourier subdomain sizes across all ranks
        local_pts = np.prod(engine.nb_fourier_subdomain_grid_pts)
        total_pts = comm.sum(local_pts)
        assert total_pts == np.prod(engine.nb_fourier_grid_pts)

    def test_normalization_factor(self, comm):
        """Test that normalization factor is correct."""
        nb_grid_pts = [16, 20]
        engine = FFTEngine(nb_grid_pts, comm)
        expected_norm = 1.0 / np.prod(nb_grid_pts)
        assert abs(engine.normalisation - expected_norm) < 1e-15

    def test_backend_name(self, comm):
        """Test that backend name is reported."""
        nb_grid_pts = [16, 20]
        engine = FFTEngine(nb_grid_pts, comm)
        # Should be PocketFFT for serial or MPI version
        assert engine.backend_name == "PocketFFT"


class TestMPIFFTForwardTransform:
    """Test forward FFT with MPI parallelization."""

    @pytest.mark.parametrize(
        "nb_grid_pts",
        [
            [16, 20],
            [32, 24],
        ],
    )
    def test_forward_transform_2d(self, comm, nb_grid_pts):
        """Test 2D forward FFT against numpy reference."""
        # Scale grid with number of processes
        nb_grid_pts = [n * max(1, comm.size) for n in nb_grid_pts]
        engine = FFTEngine(nb_grid_pts, comm)

        real_field = engine.real_space_field("real")
        fourier_field = engine.fourier_space_field("fourier")

        # Create global input array
        np.random.seed(42)
        global_input = np.random.randn(*nb_grid_pts)

        # Compute global reference using numpy
        # numpy uses row-major, we use column-major, so transpose
        global_ref = np.fft.rfftn(global_input.T).T

        # Extract local subdomain
        subdomain_slices = make_subdomain_slices(
            engine.subdomain_locations, engine.nb_subdomain_grid_pts
        )
        local_input = global_input[subdomain_slices]

        # Set local field data (squeeze to remove component dimension)
        real_field.p[0, ...] = local_input

        # Perform FFT
        engine.fft(real_field, fourier_field)

        # Extract expected local Fourier data
        fourier_slices = make_fourier_slices(
            engine.fourier_subdomain_locations, engine.nb_fourier_subdomain_grid_pts
        )
        expected_local = global_ref[fourier_slices]

        # Compare (squeeze to remove component dimension)
        tol = 1e-12 * np.prod(nb_grid_pts)
        assert_allclose(fourier_field.p.squeeze(), expected_local, atol=tol)

    @pytest.mark.parametrize(
        "nb_grid_pts",
        [
            [8, 10, 12],
        ],
    )
    def test_forward_transform_3d(self, comm, nb_grid_pts):
        """Test 3D forward FFT against numpy reference."""
        nb_grid_pts = [n * max(1, comm.size) for n in nb_grid_pts]
        engine = FFTEngine(nb_grid_pts, comm)

        real_field = engine.real_space_field("real")
        fourier_field = engine.fourier_space_field("fourier")

        # Create global input array
        np.random.seed(42)
        global_input = np.random.randn(*nb_grid_pts)

        # Compute global reference using numpy
        global_ref = np.fft.rfftn(global_input.T).T

        # Extract local subdomain
        subdomain_slices = make_subdomain_slices(
            engine.subdomain_locations, engine.nb_subdomain_grid_pts
        )
        local_input = global_input[subdomain_slices]
        real_field.p[0, ...] = local_input

        # Perform FFT
        engine.fft(real_field, fourier_field)

        # Extract expected local Fourier data
        fourier_slices = make_fourier_slices(
            engine.fourier_subdomain_locations, engine.nb_fourier_subdomain_grid_pts
        )
        expected_local = global_ref[fourier_slices]

        tol = 1e-12 * np.prod(nb_grid_pts)
        assert_allclose(fourier_field.p.squeeze(), expected_local, atol=tol)


class TestMPIFFTInverseTransform:
    """Test inverse FFT with MPI parallelization."""

    @pytest.mark.parametrize(
        "nb_grid_pts",
        [
            [16, 20],
            [32, 24],
        ],
    )
    def test_inverse_transform_2d(self, comm, nb_grid_pts):
        """Test 2D inverse FFT against numpy reference."""
        nb_grid_pts = [n * max(1, comm.size) for n in nb_grid_pts]
        engine = FFTEngine(nb_grid_pts, comm)

        real_field = engine.real_space_field("real")
        fourier_field = engine.fourier_space_field("fourier")

        # Create global Fourier input array
        fourier_shape = list(engine.nb_fourier_grid_pts)
        np.random.seed(42)
        global_fourier = np.zeros(fourier_shape, dtype=complex)
        global_fourier.real = np.random.randn(*fourier_shape)
        global_fourier.imag = np.random.randn(*fourier_shape)

        # Compute global reference using numpy
        global_ref = np.fft.irfftn(global_fourier.T).T

        # Extract local Fourier subdomain
        fourier_slices = make_fourier_slices(
            engine.fourier_subdomain_locations, engine.nb_fourier_subdomain_grid_pts
        )
        local_fourier = global_fourier[fourier_slices]
        fourier_field.p[0, ...] = local_fourier

        # Perform inverse FFT
        engine.ifft(fourier_field, real_field)
        real_field.p[0, ...] *= engine.normalisation

        # Extract expected local real data
        subdomain_slices = make_subdomain_slices(
            engine.subdomain_locations, engine.nb_subdomain_grid_pts
        )
        expected_local = global_ref[subdomain_slices]

        tol = 1e-12 * np.prod(nb_grid_pts)
        assert_allclose(real_field.p.squeeze(), expected_local, atol=tol)


class TestMPIFFTRoundtrip:
    """Test FFT roundtrip (forward + inverse)."""

    @pytest.mark.parametrize(
        "nb_grid_pts",
        [
            [16, 20],
            [8, 10, 12],
        ],
    )
    def test_roundtrip(self, comm, nb_grid_pts):
        """Test that ifft(fft(x)) * normalisation == x."""
        nb_grid_pts = [n * max(1, comm.size) for n in nb_grid_pts]
        engine = FFTEngine(nb_grid_pts, comm)

        real_field = engine.real_space_field("real")
        fourier_field = engine.fourier_space_field("fourier")

        # Initialize with random data
        np.random.seed(42 + comm.rank)  # Different seed per rank
        original = np.random.randn(*engine.nb_subdomain_grid_pts)
        real_field.p[0, ...] = original

        # Forward FFT
        engine.fft(real_field, fourier_field)

        # Inverse FFT
        engine.ifft(fourier_field, real_field)

        # Normalize
        real_field.p[0, ...] *= engine.normalisation

        # Check roundtrip
        assert_allclose(real_field.p.squeeze(), original, atol=1e-14)

    def test_roundtrip_sine_wave(self, comm):
        """Test roundtrip with sine wave pattern."""
        nb_grid_pts = [32 * max(1, comm.size), 32]
        engine = FFTEngine(nb_grid_pts, comm)

        real_field = engine.real_space_field("real")
        fourier_field = engine.fourier_space_field("fourier")

        # Create sine wave
        x = np.arange(
            engine.subdomain_locations[0],
            engine.subdomain_locations[0] + engine.nb_subdomain_grid_pts[0],
        )
        y = np.arange(
            engine.subdomain_locations[1],
            engine.subdomain_locations[1] + engine.nb_subdomain_grid_pts[1],
        )
        X, Y = np.meshgrid(x, y, indexing="ij")

        original = np.sin(2 * np.pi * X / nb_grid_pts[0]) * np.cos(
            2 * np.pi * Y / nb_grid_pts[1]
        )
        real_field.p[0, ...] = original

        # Roundtrip
        engine.fft(real_field, fourier_field)
        engine.ifft(fourier_field, real_field)
        real_field.p[0, ...] *= engine.normalisation

        assert_allclose(real_field.p.squeeze(), original, atol=1e-13)


class TestMPIFFTMultipleComponents:
    """Test FFT with multi-component fields."""

    def test_vector_field_2d(self, comm):
        """Test FFT of 2D vector field (2 components)."""
        nb_grid_pts = [16 * max(1, comm.size), 20]
        engine = FFTEngine(nb_grid_pts, comm)

        # Create 2-component fields
        real_field = engine.real_space_field("real_vec", nb_components=2)
        fourier_field = engine.fourier_space_field("fourier_vec", nb_components=2)

        # Initialize with random data
        np.random.seed(42 + comm.rank)
        original = np.random.randn(2, *engine.nb_subdomain_grid_pts)
        real_field.p[:] = original

        # Roundtrip
        engine.fft(real_field, fourier_field)
        engine.ifft(fourier_field, real_field)
        real_field.p[:] *= engine.normalisation

        assert_allclose(real_field.p, original, atol=1e-14)

    def test_tensor_field_3d(self, comm):
        """Test FFT of 3x3 tensor field in 3D."""
        nb_grid_pts = [8, 8, 8]
        engine = FFTEngine(nb_grid_pts, comm)

        # Create 9-component field (3x3 tensor)
        real_field = engine.real_space_field("real_tensor", nb_components=9)
        fourier_field = engine.fourier_space_field("fourier_tensor", nb_components=9)

        # Initialize with random data
        np.random.seed(42)
        original = np.random.randn(9, *engine.nb_subdomain_grid_pts)
        real_field.p[:] = original

        # Roundtrip
        engine.fft(real_field, fourier_field)
        engine.ifft(fourier_field, real_field)
        real_field.p[:] *= engine.normalisation

        assert_allclose(real_field.p, original, atol=1e-14)


class TestMPIFFTDCComponent:
    """Test DC (zero frequency) component handling."""

    def test_constant_field_dc_component(self, comm):
        """Test that constant field has only DC component."""
        nb_grid_pts = [16 * max(1, comm.size), 20]
        engine = FFTEngine(nb_grid_pts, comm)

        real_field = engine.real_space_field("real")
        fourier_field = engine.fourier_space_field("fourier")

        # Constant field
        constant_value = 5.0
        real_field.p[0, ...] = constant_value

        engine.fft(real_field, fourier_field)

        # DC component should be at (0,0) = constant_value * total_points
        expected_dc = constant_value * np.prod(nb_grid_pts)

        # Check if this rank has the DC component
        if (
            engine.fourier_subdomain_locations[0] == 0
            and engine.fourier_subdomain_locations[1] == 0
        ):
            assert abs(fourier_field.p[0, 0, 0].real - expected_dc) < 1e-10
            assert abs(fourier_field.p[0, 0, 0].imag) < 1e-10

        # All other components should be zero (globally)
        local_non_dc_sum = 0.0
        for i in range(engine.nb_fourier_subdomain_grid_pts[0]):
            for j in range(engine.nb_fourier_subdomain_grid_pts[1]):
                gi = engine.fourier_subdomain_locations[0] + i
                gj = engine.fourier_subdomain_locations[1] + j
                if gi != 0 or gj != 0:
                    local_non_dc_sum += abs(fourier_field.p[0, i, j])

        global_non_dc_sum = comm.sum(local_non_dc_sum)
        assert global_non_dc_sum < 1e-10


class TestMPIFFTProcessGrid:
    """Test process grid properties."""

    def test_process_grid_matches_communicator_size(self, comm):
        """Test that process grid size matches communicator size."""
        nb_grid_pts = [16 * max(1, comm.size), 20]
        engine = FFTEngine(nb_grid_pts, comm)

        p1, p2 = engine.process_grid
        assert p1 * p2 == comm.size

    def test_process_coords_unique(self, comm):
        """Test that each rank has unique process coordinates."""
        if comm.size == 1:
            pytest.skip("Need multiple processes for this test")

        nb_grid_pts = [16 * comm.size, 20]
        engine = FFTEngine(nb_grid_pts, comm)

        # Gather all process coordinates
        local_coords = np.array(engine.process_coords, dtype=np.int32).reshape(1, -1)
        all_coords = comm.gather(local_coords)

        if comm.rank == 0:
            # Check all coordinates are unique
            # all_coords is concatenated to [comm.size * 2] array after gather
            # Reshape to [comm.size, 2]
            all_coords = all_coords.reshape(comm.size, 2)
            coord_tuples = [tuple(row) for row in all_coords]
            coord_set = set(coord_tuples)
            assert len(coord_set) == comm.size


class TestMPIFFTEdgeCases:
    """Test edge cases for MPI FFT."""

    def test_small_grid_large_process_count(self, comm):
        """Test behavior with grid smaller than optimal for process count."""
        if comm.size > 4:
            pytest.skip("Test designed for <= 4 processes")

        # Small grid that may not divide evenly
        nb_grid_pts = [7, 5]
        engine = FFTEngine(nb_grid_pts, comm)

        # Should still work correctly
        real_field = engine.real_space_field("real")
        fourier_field = engine.fourier_space_field("fourier")

        np.random.seed(42 + comm.rank)
        subdomain_shape = engine.nb_subdomain_grid_pts
        if np.prod(subdomain_shape) > 0:
            original = np.random.randn(*subdomain_shape)
            real_field.p[0, ...] = original

            engine.fft(real_field, fourier_field)
            engine.ifft(fourier_field, real_field)
            real_field.p[0, ...] *= engine.normalisation

            # Compare without squeeze to handle subdomains with size-1 dimensions
            assert_allclose(real_field.p[0, ...], original, atol=1e-14)

    def test_non_power_of_two_grid(self, comm):
        """Test FFT with non-power-of-two grid sizes."""
        nb_grid_pts = [17 * max(1, comm.size), 23]
        engine = FFTEngine(nb_grid_pts, comm)

        real_field = engine.real_space_field("real")
        fourier_field = engine.fourier_space_field("fourier")

        np.random.seed(42 + comm.rank)
        original = np.random.randn(*engine.nb_subdomain_grid_pts)
        real_field.p[0, ...] = original

        engine.fft(real_field, fourier_field)
        engine.ifft(fourier_field, real_field)
        real_field.p[0, ...] *= engine.normalisation

        assert_allclose(real_field.p.squeeze(), original, atol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

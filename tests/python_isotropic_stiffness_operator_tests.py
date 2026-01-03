#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file    python_isotropic_stiffness_operator_tests.py

@author  Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date    31 Dec 2025

@brief   Tests for IsotropicStiffnessOperator (fused elliptic kernel)

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
from conftest import (
    HAS_CUPY,
    cp,
    create_device,
    get_test_devices,
    skip_if_gpu_unavailable,
)

import muGrid
from muGrid import Timer

# =============================================================================
# Test Configuration
# =============================================================================

GPU_AVAILABLE = muGrid.has_gpu


# =============================================================================
# 2D Operator Tests
# =============================================================================


class TestIsotropicStiffnessOperator2D:
    """Test suite for IsotropicStiffnessOperator2D."""

    def test_construction(self):
        """Test that the operator can be constructed."""
        grid_spacing = [0.1, 0.1]
        op = muGrid.IsotropicStiffnessOperator2D(grid_spacing)

        # Check that G and V matrices have correct shape
        G = op.G
        V = op.V
        assert G.shape == (8, 8)
        assert V.shape == (8, 8)

    def test_symmetry(self):
        """Test that G and V matrices are symmetric."""
        grid_spacing = [0.1, 0.2]
        op = muGrid.IsotropicStiffnessOperator2D(grid_spacing)

        G = op.G
        V = op.V

        np.testing.assert_allclose(G, G.T, rtol=1e-10)
        np.testing.assert_allclose(V, V.T, rtol=1e-10)

    def test_apply_homogeneous_material(self):
        """Test apply with homogeneous material."""
        nx, ny = 8, 8
        grid_spacing = [1.0 / (nx - 1), 1.0 / (ny - 1)]

        op = muGrid.IsotropicStiffnessOperator2D(grid_spacing)

        # Create field collection with ghost cells on both sides
        fc = muGrid.GlobalFieldCollection(
            (nx, ny), nb_ghosts_left=(1, 1), nb_ghosts_right=(1, 1)
        )

        # Create fields
        displacement = fc.real_field("displacement", (2,))
        force = fc.real_field("force", (2,))

        # Create material field collection (same size as node field, with ghosts)
        fc_mat = muGrid.GlobalFieldCollection(
            (nx, ny), nb_ghosts_left=(1, 1), nb_ghosts_right=(1, 1)
        )
        lambda_field = fc_mat.real_field("lambda", (1,))
        mu_field = fc_mat.real_field("mu", (1,))

        # Set homogeneous material properties
        lam = 1.0
        mu = 1.0
        lambda_field.p[...] = lam
        mu_field.p[...] = mu

        # Set a simple displacement field (linear in x)
        x = np.arange(nx) * grid_spacing[0]
        y = np.arange(ny) * grid_spacing[1]
        X, Y = np.meshgrid(x, y, indexing="ij")
        displacement.p[0, ...] = X  # u_x = x
        displacement.p[1, ...] = 0  # u_y = 0

        # Fill ghost cells (periodic BC)
        displacement.pg[0, 1 : nx + 1, 1 : ny + 1] = displacement.p[0]
        displacement.pg[1, 1 : nx + 1, 1 : ny + 1] = displacement.p[1]
        displacement.pg[:, 0, :] = displacement.pg[:, nx, :]  # left ghost
        displacement.pg[:, nx + 1, :] = displacement.pg[:, 1, :]  # right ghost
        displacement.pg[:, :, 0] = displacement.pg[:, :, ny]  # bottom ghost
        displacement.pg[:, :, ny + 1] = displacement.pg[:, :, 1]  # top ghost

        # Fill material ghost cells
        lambda_field.pg[..., 1 : nx + 1, 1 : ny + 1] = lambda_field.p
        lambda_field.pg[..., 0, :] = lambda_field.pg[..., nx, :]
        lambda_field.pg[..., nx + 1, :] = lambda_field.pg[..., 1, :]
        lambda_field.pg[..., :, 0] = lambda_field.pg[..., :, ny]
        lambda_field.pg[..., :, ny + 1] = lambda_field.pg[..., :, 1]
        mu_field.pg[..., 1 : nx + 1, 1 : ny + 1] = mu_field.p
        mu_field.pg[..., 0, :] = mu_field.pg[..., nx, :]
        mu_field.pg[..., nx + 1, :] = mu_field.pg[..., 1, :]
        mu_field.pg[..., :, 0] = mu_field.pg[..., :, ny]
        mu_field.pg[..., :, ny + 1] = mu_field.pg[..., :, 1]

        # Apply operator
        op.apply(displacement, lambda_field, mu_field, force)

        # Force should be non-zero only at boundaries for uniform strain
        # (internal forces cancel out for constant strain)
        # Just check that it runs without error and force is finite
        assert np.all(np.isfinite(force.p))

    def test_compare_with_generic_operator(self):
        """Compare fused kernel with generic B^T C B computation."""
        nn = 6  # Number of nodes
        nel = nn  # Number of elements (periodic BC)
        grid_spacing = [0.25, 0.25]

        # Material properties (homogeneous)
        lam = 2.0
        mu = 1.5

        # Create the fused operator
        fused_op = muGrid.IsotropicStiffnessOperator2D(grid_spacing)

        # Create the generic gradient operator
        grad_op = muGrid.FEMGradientOperator(2, grid_spacing)

        # Create decomposition with ghosts for periodic BC
        comm = muGrid.Communicator()
        decomposition = muGrid.CartesianDecomposition(
            comm,
            (nn, nn),
            nb_subdivisions=(1, 1),
            nb_ghosts_left=(1, 1),
            nb_ghosts_right=(1, 1),
            nb_sub_pts={"quad": grad_op.nb_quad_pts},
        )

        # Create material field collection with ghost cells for periodic BC
        # (ghost cells are filled via communicate_ghosts once before apply)
        fc_mat = muGrid.GlobalFieldCollection(
            (nel, nel), nb_ghosts_left=(1, 1), nb_ghosts_right=(1, 1)
        )
        lambda_field = fc_mat.real_field("lambda", (1,))
        mu_field = fc_mat.real_field("mu", (1,))
        lambda_field.p[...] = lam
        mu_field.p[...] = mu

        # Fill ghost cells with periodic copies
        # (This is done once if material field doesn't change during simulation)
        lambda_field.pg[...] = np.pad(
            lambda_field.p, ((0, 0), (1, 1), (1, 1)), mode="wrap"
        )
        mu_field.pg[...] = np.pad(mu_field.p, ((0, 0), (1, 1), (1, 1)), mode="wrap")

        # Create displacement field
        displacement = decomposition.real_field("displacement", (2,))
        np.random.seed(42)
        displacement.p[...] = np.random.rand(*displacement.p.shape)
        decomposition.communicate_ghosts(displacement)

        # Force fields
        force_fused = decomposition.real_field("force_fused", (2,))
        force_generic = decomposition.real_field("force_generic", (2,))

        # Apply fused operator
        fused_op.apply(displacement, lambda_field, mu_field, force_fused)

        # Apply generic operator (gradient -> stress -> divergence)
        gradient = decomposition.real_field("gradient", (2, 2), "quad")
        grad_op.apply(displacement, gradient)

        # Compute stress using isotropic constitutive law
        stress = decomposition.real_field("stress", (2, 2), "quad")
        grad = gradient.s

        # Symmetric strain
        eps_xx = grad[0, 0, ...]
        eps_yy = grad[1, 1, ...]
        eps_xy = 0.5 * (grad[0, 1, ...] + grad[1, 0, ...])
        trace = eps_xx + eps_yy

        # Stress
        stress.s[0, 0, ...] = lam * trace + 2 * mu * eps_xx
        stress.s[1, 1, ...] = lam * trace + 2 * mu * eps_yy
        stress.s[0, 1, ...] = 2 * mu * eps_xy
        stress.s[1, 0, ...] = 2 * mu * eps_xy

        # Fill ghost cells for periodic contribution
        decomposition.communicate_ghosts(stress)

        # Apply divergence (transpose of gradient)
        force_generic.pg[...] = 0.0
        quad_weights = grad_op.quadrature_weights
        grad_op.transpose(stress, force_generic, quad_weights)

        # Compare results
        np.testing.assert_allclose(
            force_fused.p, force_generic.p, rtol=1e-10, atol=1e-10
        )


# =============================================================================
# 3D Operator Tests
# =============================================================================


class TestIsotropicStiffnessOperator3D:
    """Test suite for IsotropicStiffnessOperator3D."""

    def test_construction(self):
        """Test that the operator can be constructed."""
        grid_spacing = [0.1, 0.1, 0.1]
        op = muGrid.IsotropicStiffnessOperator3D(grid_spacing)

        # Check that G and V matrices have correct shape
        G = op.G
        V = op.V
        assert G.shape == (24, 24)
        assert V.shape == (24, 24)

    def test_symmetry(self):
        """Test that G and V matrices are symmetric."""
        grid_spacing = [0.1, 0.2, 0.15]
        op = muGrid.IsotropicStiffnessOperator3D(grid_spacing)

        G = op.G
        V = op.V

        np.testing.assert_allclose(G, G.T, rtol=1e-10)
        np.testing.assert_allclose(V, V.T, rtol=1e-10)

    def test_apply_homogeneous_material(self):
        """Test apply with homogeneous material."""
        nx, ny, nz = 6, 6, 6
        grid_spacing = [1.0 / (nx - 1), 1.0 / (ny - 1), 1.0 / (nz - 1)]

        op = muGrid.IsotropicStiffnessOperator3D(grid_spacing)

        # Create field collection with ghost cells on both sides
        fc = muGrid.GlobalFieldCollection(
            (nx, ny, nz), nb_ghosts_left=(1, 1, 1), nb_ghosts_right=(1, 1, 1)
        )

        # Create fields
        displacement = fc.real_field("displacement", (3,))
        force = fc.real_field("force", (3,))

        # Create material field collection (same size as node field, with ghosts)
        fc_mat = muGrid.GlobalFieldCollection(
            (nx, ny, nz), nb_ghosts_left=(1, 1, 1), nb_ghosts_right=(1, 1, 1)
        )
        lambda_field = fc_mat.real_field("lambda", (1,))
        mu_field = fc_mat.real_field("mu", (1,))

        # Set homogeneous material properties
        lam = 1.0
        mu = 1.0
        lambda_field.p[...] = lam
        mu_field.p[...] = mu

        # Set a simple displacement field (linear in x)
        x = np.arange(nx) * grid_spacing[0]
        y = np.arange(ny) * grid_spacing[1]
        z = np.arange(nz) * grid_spacing[2]
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        displacement.p[0, ...] = X  # u_x = x
        displacement.p[1, ...] = 0  # u_y = 0
        displacement.p[2, ...] = 0  # u_z = 0

        # Fill ghost cells with periodic copies
        displacement.pg[...] = np.pad(
            displacement.p, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="wrap"
        )
        lambda_field.pg[...] = np.pad(
            lambda_field.p, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="wrap"
        )
        mu_field.pg[...] = np.pad(
            mu_field.p, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="wrap"
        )

        # Apply operator
        op.apply(displacement, lambda_field, mu_field, force)

        # Just check that it runs without error and force is finite
        assert np.all(np.isfinite(force.p))

    def test_compare_with_generic_operator(self):
        """Compare fused kernel with generic B^T C B computation."""
        nn = 5  # Number of nodes
        nel = nn  # Number of elements (periodic BC)
        grid_spacing = [0.25, 0.25, 0.25]

        # Material properties (homogeneous)
        lam = 2.0
        mu = 1.5

        # Create the fused operator
        fused_op = muGrid.IsotropicStiffnessOperator3D(grid_spacing)

        # Create the generic gradient operator
        grad_op = muGrid.FEMGradientOperator(3, grid_spacing)

        # Create decomposition with ghosts for periodic BC
        comm = muGrid.Communicator()
        decomposition = muGrid.CartesianDecomposition(
            comm,
            (nn, nn, nn),
            nb_subdivisions=(1, 1, 1),
            nb_ghosts_left=(1, 1, 1),
            nb_ghosts_right=(1, 1, 1),
            nb_sub_pts={"quad": grad_op.nb_quad_pts},
        )

        # Create material field collection with ghost cells for periodic BC
        fc_mat = muGrid.GlobalFieldCollection(
            (nel, nel, nel), nb_ghosts_left=(1, 1, 1), nb_ghosts_right=(1, 1, 1)
        )
        lambda_field = fc_mat.real_field("lambda", (1,))
        mu_field = fc_mat.real_field("mu", (1,))
        lambda_field.p[...] = lam
        mu_field.p[...] = mu

        # Fill ghost cells with periodic copies
        lambda_field.pg[...] = np.pad(
            lambda_field.p, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="wrap"
        )
        mu_field.pg[...] = np.pad(
            mu_field.p, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="wrap"
        )

        # Create displacement field
        displacement = decomposition.real_field("displacement", (3,))
        np.random.seed(42)
        displacement.p[...] = np.random.rand(*displacement.p.shape)
        decomposition.communicate_ghosts(displacement)

        # Force fields
        force_fused = decomposition.real_field("force_fused", (3,))
        force_generic = decomposition.real_field("force_generic", (3,))

        # Apply fused operator
        fused_op.apply(displacement, lambda_field, mu_field, force_fused)

        # Apply generic operator (gradient -> stress -> divergence)
        gradient = decomposition.real_field("gradient", (3, 3), "quad")
        grad_op.apply(displacement, gradient)

        # Compute stress using isotropic constitutive law
        stress = decomposition.real_field("stress", (3, 3), "quad")
        grad = gradient.s

        # Symmetric strain
        eps = np.zeros_like(stress.s)
        for i in range(3):
            for j in range(3):
                eps[i, j, ...] = 0.5 * (grad[i, j, ...] + grad[j, i, ...])

        trace = eps[0, 0, ...] + eps[1, 1, ...] + eps[2, 2, ...]

        # Stress
        for i in range(3):
            for j in range(3):
                stress.s[i, j, ...] = 2 * mu * eps[i, j, ...]
                if i == j:
                    stress.s[i, j, ...] += lam * trace

        # Fill ghost cells for periodic contribution
        decomposition.communicate_ghosts(stress)

        # Apply divergence (transpose of gradient)
        force_generic.pg[...] = 0.0
        quad_weights = grad_op.quadrature_weights
        grad_op.transpose(stress, force_generic, quad_weights)

        # Compare results
        np.testing.assert_allclose(
            force_fused.p, force_generic.p, rtol=1e-10, atol=1e-10
        )


# =============================================================================
# Apply Increment Tests
# =============================================================================


class TestIsotropicStiffnessApplyIncrement:
    """Test apply_increment functionality."""

    def test_apply_increment_2d(self):
        """Test apply_increment in 2D."""
        nx, ny = 6, 6
        grid_spacing = [0.25, 0.25]

        op = muGrid.IsotropicStiffnessOperator2D(grid_spacing)

        # Create field collections with ghosts on both sides
        fc = muGrid.GlobalFieldCollection(
            (nx, ny), nb_ghosts_left=(1, 1), nb_ghosts_right=(1, 1)
        )
        # Material field same size as node field, with ghost cells
        fc_mat = muGrid.GlobalFieldCollection(
            (nx, ny), nb_ghosts_left=(1, 1), nb_ghosts_right=(1, 1)
        )

        displacement = fc.real_field("displacement", (2,))
        force1 = fc.real_field("force1", (2,))
        force2 = fc.real_field("force2", (2,))

        lambda_field = fc_mat.real_field("lambda", (1,))
        mu_field = fc_mat.real_field("mu", (1,))

        lambda_field.p[...] = 2.0
        mu_field.p[...] = 1.0
        # Fill ghost cells with periodic copies
        lambda_field.pg[...] = np.pad(
            lambda_field.p, ((0, 0), (1, 1), (1, 1)), mode="wrap"
        )
        mu_field.pg[...] = np.pad(mu_field.p, ((0, 0), (1, 1), (1, 1)), mode="wrap")

        np.random.seed(42)
        displacement.pg[...] = np.random.rand(*displacement.pg.shape)

        # Apply once
        op.apply(displacement, lambda_field, mu_field, force1)

        # Apply with increment (alpha=2.0, starting from force1)
        force2.p[...] = force1.p[...]
        op.apply_increment(displacement, lambda_field, mu_field, 1.0, force2)

        # force2 should now be 2 * force1
        np.testing.assert_allclose(force2.p, 2.0 * force1.p, rtol=1e-10)

    def test_apply_increment_3d(self):
        """Test apply_increment in 3D."""
        nx, ny, nz = 5, 5, 5
        grid_spacing = [0.25, 0.25, 0.25]

        op = muGrid.IsotropicStiffnessOperator3D(grid_spacing)

        # Create field collections with ghosts on both sides
        fc = muGrid.GlobalFieldCollection(
            (nx, ny, nz), nb_ghosts_left=(1, 1, 1), nb_ghosts_right=(1, 1, 1)
        )
        # Material field same size as node field, with ghost cells
        fc_mat = muGrid.GlobalFieldCollection(
            (nx, ny, nz), nb_ghosts_left=(1, 1, 1), nb_ghosts_right=(1, 1, 1)
        )

        displacement = fc.real_field("displacement", (3,))
        force1 = fc.real_field("force1", (3,))
        force2 = fc.real_field("force2", (3,))

        lambda_field = fc_mat.real_field("lambda", (1,))
        mu_field = fc_mat.real_field("mu", (1,))

        lambda_field.p[...] = 2.0
        mu_field.p[...] = 1.0
        # Fill ghost cells with periodic copies
        lambda_field.pg[...] = np.pad(
            lambda_field.p, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="wrap"
        )
        mu_field.pg[...] = np.pad(
            mu_field.p, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="wrap"
        )

        np.random.seed(42)
        displacement.pg[...] = np.random.rand(*displacement.pg.shape)

        # Apply once
        op.apply(displacement, lambda_field, mu_field, force1)

        # Apply with increment (alpha=0.5, starting from zero)
        force2.p[...] = 0.0
        op.apply_increment(displacement, lambda_field, mu_field, 0.5, force2)

        # force2 should be 0.5 * force1
        np.testing.assert_allclose(force2.p, 0.5 * force1.p, rtol=1e-10)


# =============================================================================
# Smoke Tests (Parametrized on Device)
# =============================================================================


@pytest.mark.parametrize("device", get_test_devices())
class TestIsotropicStiffnessOperatorApply_smoke:
    """Smoke tests for IsotropicStiffnessOperator apply on different devices."""

    def test_2d_apply_smoke(self, device):
        """Test 2D operator apply smoke test."""
        skip_if_gpu_unavailable(device)

        nx, ny = 8, 8
        grid_spacing = [0.25, 0.25]

        op = muGrid.IsotropicStiffnessOperator2D(grid_spacing)

        # Create device field collections
        device_obj = create_device(device)
        fc_kwargs = {"device": device_obj} if device_obj else {}

        fc = muGrid.GlobalFieldCollection(
            (nx, ny),
            nb_ghosts_left=(1, 1),
            nb_ghosts_right=(1, 1),
            **fc_kwargs,
        )
        fc_mat = muGrid.GlobalFieldCollection(
            (nx, ny),
            nb_ghosts_left=(1, 1),
            nb_ghosts_right=(1, 1),
            **fc_kwargs,
        )

        displacement = fc.real_field("displacement", (2,))
        force = fc.real_field("force", (2,))
        lambda_field = fc_mat.real_field("lambda", (1,))
        mu_field = fc_mat.real_field("mu", (1,))

        # Initialize
        displacement.set_zero()
        lambda_field.pg[...] = 2.0
        mu_field.pg[...] = 1.0

        # Apply should not raise
        op.apply(displacement, lambda_field, mu_field, force)

        # Check that force is finite
        if device == "gpu":
            assert force.is_on_gpu
            force_data = cp.asnumpy(force.p)
        else:
            force_data = force.p
        assert np.all(np.isfinite(force_data))

    def test_3d_apply_smoke(self, device):
        """Test 3D operator apply smoke test."""
        skip_if_gpu_unavailable(device)

        nx, ny, nz = 6, 6, 6
        grid_spacing = [0.25, 0.25, 0.25]

        op = muGrid.IsotropicStiffnessOperator3D(grid_spacing)

        # Create device field collections
        device_obj = create_device(device)
        fc_kwargs = {"device": device_obj} if device_obj else {}

        fc = muGrid.GlobalFieldCollection(
            (nx, ny, nz),
            nb_ghosts_left=(1, 1, 1),
            nb_ghosts_right=(1, 1, 1),
            **fc_kwargs,
        )
        fc_mat = muGrid.GlobalFieldCollection(
            (nx, ny, nz),
            nb_ghosts_left=(1, 1, 1),
            nb_ghosts_right=(1, 1, 1),
            **fc_kwargs,
        )

        displacement = fc.real_field("displacement", (3,))
        force = fc.real_field("force", (3,))
        lambda_field = fc_mat.real_field("lambda", (1,))
        mu_field = fc_mat.real_field("mu", (1,))

        # Initialize
        displacement.set_zero()
        lambda_field.pg[...] = 2.0
        mu_field.pg[...] = 1.0

        # Apply should not raise
        op.apply(displacement, lambda_field, mu_field, force)

        # Check that force is finite
        if device == "gpu":
            assert force.is_on_gpu
            force_data = cp.asnumpy(force.p)
        else:
            force_data = force.p
        assert np.all(np.isfinite(force_data))


# =============================================================================
# GPU vs CPU Consistency Tests (Dual-device comparison)
# =============================================================================


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU backend not available")
class TestIsotropicStiffnessOperatorGPUCorrectness:
    """Test GPU correctness by comparing with CPU."""

    def setup_method(self, method):
        """Skip tests if no GPU is available at runtime."""
        if not muGrid.is_gpu_available():
            pytest.skip("No GPU device available at runtime")
        if not HAS_CUPY:
            pytest.skip("CuPy not available for GPU tests")

    def test_gpu_cpu_consistency_2d(self):
        """Test that GPU and CPU give same results in 2D."""
        nx, ny = 8, 8
        grid_spacing = [0.25, 0.25]

        op = muGrid.IsotropicStiffnessOperator2D(grid_spacing)

        # Both CPU and GPU use same ghost configuration for comparison
        fc_cpu = muGrid.GlobalFieldCollection(
            (nx, ny), nb_ghosts_left=(1, 1), nb_ghosts_right=(1, 1)
        )
        fc_mat_cpu = muGrid.GlobalFieldCollection(
            (nx, ny), nb_ghosts_left=(1, 1), nb_ghosts_right=(1, 1)
        )

        disp_cpu = fc_cpu.real_field("displacement", (2,))
        force_cpu = fc_cpu.real_field("force", (2,))
        lam_cpu = fc_mat_cpu.real_field("lambda", (1,))
        mu_cpu = fc_mat_cpu.real_field("mu", (1,))

        fc_gpu = muGrid.GlobalFieldCollection(
            (nx, ny),
            nb_ghosts_left=(1, 1),
            nb_ghosts_right=(1, 1),
            device=muGrid.Device.gpu(),
        )
        fc_mat_gpu = muGrid.GlobalFieldCollection(
            (nx, ny),
            nb_ghosts_left=(1, 1),
            nb_ghosts_right=(1, 1),
            device=muGrid.Device.gpu(),
        )

        disp_gpu = fc_gpu.real_field("displacement", (2,))
        force_gpu = fc_gpu.real_field("force", (2,))
        lam_gpu = fc_mat_gpu.real_field("lambda", (1,))
        mu_gpu = fc_mat_gpu.real_field("mu", (1,))

        # Initialize with same data
        np.random.seed(42)
        test_disp = np.random.rand(*disp_cpu.pg.shape)
        test_lam = 2.0 + np.random.rand(*lam_cpu.p.shape)
        test_mu = 1.0 + np.random.rand(*mu_cpu.p.shape)

        disp_cpu.pg[...] = test_disp
        lam_cpu.p[...] = test_lam
        mu_cpu.p[...] = test_mu
        # Fill ghost cells with periodic copies
        lam_cpu.pg[...] = np.pad(lam_cpu.p, ((0, 0), (1, 1), (1, 1)), mode="wrap")
        mu_cpu.pg[...] = np.pad(mu_cpu.p, ((0, 0), (1, 1), (1, 1)), mode="wrap")

        disp_gpu.pg[...] = cp.asarray(test_disp)
        lam_gpu.p[...] = cp.asarray(test_lam)
        mu_gpu.p[...] = cp.asarray(test_mu)
        # Fill ghost cells with periodic copies on GPU
        lam_gpu.pg[...] = cp.pad(lam_gpu.p, ((0, 0), (1, 1), (1, 1)), mode="wrap")
        mu_gpu.pg[...] = cp.pad(mu_gpu.p, ((0, 0), (1, 1), (1, 1)), mode="wrap")

        # Apply
        op.apply(disp_cpu, lam_cpu, mu_cpu, force_cpu)
        op.apply(disp_gpu, lam_gpu, mu_gpu, force_gpu)

        # Compare
        force_gpu_np = cp.asnumpy(force_gpu.p)
        np.testing.assert_allclose(force_gpu_np, force_cpu.p, rtol=1e-10)

    def test_gpu_cpu_consistency_3d(self):
        """Test that GPU and CPU give same results in 3D."""
        nx, ny, nz = 6, 6, 6
        grid_spacing = [0.25, 0.25, 0.25]

        op = muGrid.IsotropicStiffnessOperator3D(grid_spacing)

        # Both CPU and GPU use same ghost configuration for comparison
        fc_cpu = muGrid.GlobalFieldCollection(
            (nx, ny, nz), nb_ghosts_left=(1, 1, 1), nb_ghosts_right=(1, 1, 1)
        )
        fc_mat_cpu = muGrid.GlobalFieldCollection(
            (nx, ny, nz), nb_ghosts_left=(1, 1, 1), nb_ghosts_right=(1, 1, 1)
        )

        disp_cpu = fc_cpu.real_field("displacement", (3,))
        force_cpu = fc_cpu.real_field("force", (3,))
        lam_cpu = fc_mat_cpu.real_field("lambda", (1,))
        mu_cpu = fc_mat_cpu.real_field("mu", (1,))

        fc_gpu = muGrid.GlobalFieldCollection(
            (nx, ny, nz),
            nb_ghosts_left=(1, 1, 1),
            nb_ghosts_right=(1, 1, 1),
            device=muGrid.Device.gpu(),
        )
        fc_mat_gpu = muGrid.GlobalFieldCollection(
            (nx, ny, nz),
            nb_ghosts_left=(1, 1, 1),
            nb_ghosts_right=(1, 1, 1),
            device=muGrid.Device.gpu(),
        )

        disp_gpu = fc_gpu.real_field("displacement", (3,))
        force_gpu = fc_gpu.real_field("force", (3,))
        lam_gpu = fc_mat_gpu.real_field("lambda", (1,))
        mu_gpu = fc_mat_gpu.real_field("mu", (1,))

        # Initialize with same data
        np.random.seed(42)
        test_disp = np.random.rand(*disp_cpu.pg.shape)
        test_lam = 2.0 + np.random.rand(*lam_cpu.p.shape)
        test_mu = 1.0 + np.random.rand(*mu_cpu.p.shape)

        disp_cpu.pg[...] = test_disp
        lam_cpu.p[...] = test_lam
        mu_cpu.p[...] = test_mu
        # Fill ghost cells with periodic copies
        lam_cpu.pg[...] = np.pad(
            lam_cpu.p, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="wrap"
        )
        mu_cpu.pg[...] = np.pad(
            mu_cpu.p, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="wrap"
        )

        disp_gpu.pg[...] = cp.asarray(test_disp)
        lam_gpu.p[...] = cp.asarray(test_lam)
        mu_gpu.p[...] = cp.asarray(test_mu)
        # Fill ghost cells with periodic copies on GPU
        lam_gpu.pg[...] = cp.pad(
            lam_gpu.p, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="wrap"
        )
        mu_gpu.pg[...] = cp.pad(
            mu_gpu.p, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="wrap"
        )

        # Apply
        op.apply(disp_cpu, lam_cpu, mu_cpu, force_cpu)
        op.apply(disp_gpu, lam_gpu, mu_gpu, force_gpu)

        # Compare
        force_gpu_np = cp.asnumpy(force_gpu.p)
        np.testing.assert_allclose(force_gpu_np, force_cpu.p, rtol=1e-10)


# =============================================================================
# Unit Impulse Tests (systematic comparison with generic operator)
# =============================================================================


class TestUnitImpulse2D:
    """Test fused operator matches generic operator for unit impulses in 2D."""

    def _generic_apply(self, decomposition, displacement, force, lam, mu, grad_op):
        """Apply generic B^T C B operator using FEMGradientOperator."""
        gradient = decomposition.real_field("gradient", (2, 2), "quad")
        stress = decomposition.real_field("stress", (2, 2), "quad")

        grad_op.apply(displacement, gradient)
        grad = gradient.s

        # Symmetric strain
        eps_xx = grad[0, 0, ...]
        eps_yy = grad[1, 1, ...]
        eps_xy = 0.5 * (grad[0, 1, ...] + grad[1, 0, ...])
        trace = eps_xx + eps_yy

        # Stress (isotropic constitutive law)
        stress.s[0, 0, ...] = lam * trace + 2 * mu * eps_xx
        stress.s[1, 1, ...] = lam * trace + 2 * mu * eps_yy
        stress.s[0, 1, ...] = 2 * mu * eps_xy
        stress.s[1, 0, ...] = 2 * mu * eps_xy

        # Fill stress ghost cells for periodic contributions
        decomposition.communicate_ghosts(stress)

        # Divergence
        force.pg[...] = 0.0
        quad_weights = grad_op.quadrature_weights
        grad_op.transpose(stress, force, quad_weights)

    def _convolution_apply(
        self, decomposition, displacement, force, lam, mu, conv_op, quad_weights
    ):
        """Apply B^T C B operator using ConvolutionOperator.

        This uses a ConvolutionOperator created from FEMGradientOperator.coefficients
        to verify that both gradient operators produce identical results.

        Note: ConvolutionOperator.transpose doesn't apply quadrature weights internally,
        so we pre-multiply the stress by weights before calling transpose.
        """
        gradient = decomposition.real_field("gradient_conv", (2, 2), "quad")
        stress = decomposition.real_field("stress_conv", (2, 2), "quad")

        conv_op.apply(displacement, gradient)
        grad = gradient.s

        # Symmetric strain (same as _generic_apply)
        eps_xx = grad[0, 0, ...]
        eps_yy = grad[1, 1, ...]
        eps_xy = 0.5 * (grad[0, 1, ...] + grad[1, 0, ...])
        trace = eps_xx + eps_yy

        # Stress (isotropic constitutive law)
        stress.s[0, 0, ...] = lam * trace + 2 * mu * eps_xx
        stress.s[1, 1, ...] = lam * trace + 2 * mu * eps_yy
        stress.s[0, 1, ...] = 2 * mu * eps_xy
        stress.s[1, 0, ...] = 2 * mu * eps_xy

        # Pre-multiply stress by quadrature weights (ConvolutionOperator.transpose
        # doesn't apply weights internally like FEMGradientOperator does)
        for q in range(len(quad_weights)):
            stress.s[:, :, q, ...] *= quad_weights[q]

        # Fill stress ghost cells for periodic contributions
        decomposition.communicate_ghosts(stress)

        # Divergence using transpose of ConvolutionOperator (without weights)
        force.pg[...] = 0.0
        conv_op.transpose(stress, force, [])

    def test_unit_impulse_non_periodic(self):
        """Test unit impulse response for non-periodic BC.

        With node-based material field indexing, non-periodic vs periodic is
        distinguished only by how ghost cells are filled (not by kernel logic).
        This test uses zero-filled ghosts to simulate non-periodic BC.

        Compares three approaches:
        1. Fused IsotropicStiffnessOperator2D
        2. Generic FEMGradientOperator
        3. ConvolutionOperator with FEMGradientOperator.coefficients

        Also outputs timing information for the generic vs convolution approaches.
        """
        nx, ny = 8, 8
        grid_spacing = [0.25, 0.25]

        lam = 2.0
        mu = 1.5

        fused_op = muGrid.IsotropicStiffnessOperator2D(grid_spacing)
        grad_op = muGrid.FEMGradientOperator(2, grid_spacing)

        # Create ConvolutionOperator from FEMGradientOperator coefficients
        conv_op = muGrid.GenericLinearOperator([0, 0], grad_op.coefficients)
        quad_weights = grad_op.quadrature_weights

        # Use CartesianDecomposition for ghost handling
        comm = muGrid.Communicator()
        decomposition = muGrid.CartesianDecomposition(
            comm,
            (nx, ny),
            nb_subdivisions=(1, 1),
            nb_ghosts_left=(1, 1),
            nb_ghosts_right=(1, 1),
            nb_sub_pts={"quad": grad_op.nb_quad_pts},
        )

        # Material field same size as node field, with ghost cells
        fc_mat = muGrid.GlobalFieldCollection(
            (nx, ny), nb_ghosts_left=(1, 1), nb_ghosts_right=(1, 1)
        )

        displacement = decomposition.real_field("displacement", (2,))
        force_fused = decomposition.real_field("force_fused", (2,))
        force_generic = decomposition.real_field("force_generic", (2,))
        force_convolution = decomposition.real_field("force_convolution", (2,))

        lambda_field = fc_mat.real_field("lambda", (1,))
        mu_field = fc_mat.real_field("mu", (1,))
        lambda_field.p[...] = lam
        mu_field.p[...] = mu
        # Fill ghost cells with periodic copies (same as periodic BC for material)
        lambda_field.pg[...] = np.pad(
            lambda_field.p, ((0, 0), (1, 1), (1, 1)), mode="wrap"
        )
        mu_field.pg[...] = np.pad(mu_field.p, ((0, 0), (1, 1), (1, 1)), mode="wrap")

        timer = Timer()

        # Test unit impulses at a few interior nodes
        test_positions = [(2, 3), (4, 4), (5, 2)]
        for ix, iy in test_positions:
            for d in range(2):  # Test both displacement directions
                displacement.p[...] = 0.0
                displacement.p[d, ix, iy] = 1.0

                # communicate_ghosts for ghost handling
                decomposition.communicate_ghosts(displacement)

                force_fused.p[...] = 0.0
                force_generic.p[...] = 0.0
                force_convolution.p[...] = 0.0

                with timer("fused"):
                    fused_op.apply(displacement, lambda_field, mu_field, force_fused)

                with timer("FEMGradientOperator"):
                    self._generic_apply(
                        decomposition, displacement, force_generic, lam, mu, grad_op
                    )

                with timer("ConvolutionOperator"):
                    self._convolution_apply(
                        decomposition,
                        displacement,
                        force_convolution,
                        lam,
                        mu,
                        conv_op,
                        quad_weights,
                    )

                # Compare fused vs generic FEMGradientOperator
                # With node-based indexing, boundary forces are computed too
                np.testing.assert_allclose(
                    force_fused.p,
                    force_generic.p,
                    rtol=1e-10,
                    atol=1e-14,
                    err_msg=f"Fused vs generic mismatch at node ({ix}, {iy}), "
                    f"direction {d}",
                )

                # Compare generic FEMGradientOperator vs ConvolutionOperator
                np.testing.assert_allclose(
                    force_generic.p,
                    force_convolution.p,
                    rtol=1e-10,
                    atol=1e-14,
                    err_msg=f"Generic vs convolution mismatch at node ({ix}, {iy}), "
                    f"direction {d}",
                )

        timer.print_summary(title=f"2D Non-periodic ({nx}x{ny} grid)")

    def test_unit_impulse_periodic(self):
        """Test unit impulse response for periodic BC (N elements for N nodes).

        Compares three approaches:
        1. Fused IsotropicStiffnessOperator2D
        2. Generic FEMGradientOperator
        3. ConvolutionOperator with FEMGradientOperator.coefficients

        Also outputs timing information for the generic vs convolution approaches.
        """
        nx, ny = 8, 8
        grid_spacing = [0.25, 0.25]

        lam = 2.0
        mu = 1.5

        fused_op = muGrid.IsotropicStiffnessOperator2D(grid_spacing)
        grad_op = muGrid.FEMGradientOperator(2, grid_spacing)

        # Create ConvolutionOperator from FEMGradientOperator coefficients
        conv_op = muGrid.GenericLinearOperator([0, 0], grad_op.coefficients)
        quad_weights = grad_op.quadrature_weights

        # Use CartesianDecomposition to handle ghost communication for periodic BC
        comm = muGrid.Communicator()
        decomposition = muGrid.CartesianDecomposition(
            comm,
            (nx, ny),
            nb_subdivisions=(1, 1),
            nb_ghosts_left=(1, 1),
            nb_ghosts_right=(1, 1),
            nb_sub_pts={"quad": grad_op.nb_quad_pts},
        )

        # Periodic: material field has N elements (same as nodes) with ghost cells
        fc_mat = muGrid.GlobalFieldCollection(
            (nx, ny), nb_ghosts_left=(1, 1), nb_ghosts_right=(1, 1)
        )

        displacement = decomposition.real_field("displacement", (2,))
        force_fused = decomposition.real_field("force_fused", (2,))
        force_generic = decomposition.real_field("force_generic", (2,))
        force_convolution = decomposition.real_field("force_convolution", (2,))

        lambda_field = fc_mat.real_field("lambda", (1,))
        mu_field = fc_mat.real_field("mu", (1,))
        lambda_field.p[...] = lam
        mu_field.p[...] = mu

        # Fill ghost cells with periodic copies
        lambda_field.pg[...] = np.pad(
            lambda_field.p, ((0, 0), (1, 1), (1, 1)), mode="wrap"
        )
        mu_field.pg[...] = np.pad(mu_field.p, ((0, 0), (1, 1), (1, 1)), mode="wrap")

        timer = Timer()

        # Test unit impulses at various nodes including boundaries
        # For periodic BC, boundary nodes should also match
        test_positions = [(0, 0), (7, 7), (0, 4), (4, 0), (3, 5)]
        for ix, iy in test_positions:
            for d in range(2):
                # Set displacement and use communicate_ghosts for periodic filling
                displacement.p[...] = 0.0
                displacement.p[d, ix, iy] = 1.0

                # communicate_ghosts fills ghost cells with periodic copies
                decomposition.communicate_ghosts(displacement)

                force_fused.p[...] = 0.0
                force_generic.p[...] = 0.0
                force_convolution.p[...] = 0.0

                with timer("fused"):
                    fused_op.apply(displacement, lambda_field, mu_field, force_fused)

                with timer("FEMGradientOperator"):
                    self._generic_apply(
                        decomposition, displacement, force_generic, lam, mu, grad_op
                    )

                with timer("ConvolutionOperator"):
                    self._convolution_apply(
                        decomposition,
                        displacement,
                        force_convolution,
                        lam,
                        mu,
                        conv_op,
                        quad_weights,
                    )

                # Compare fused vs generic FEMGradientOperator
                np.testing.assert_allclose(
                    force_fused.p,
                    force_generic.p,
                    rtol=1e-10,
                    atol=1e-14,
                    err_msg=f"Fused vs generic mismatch at node ({ix}, {iy}), "
                    f"direction {d}",
                )

                # Compare generic FEMGradientOperator vs ConvolutionOperator
                np.testing.assert_allclose(
                    force_generic.p,
                    force_convolution.p,
                    rtol=1e-10,
                    atol=1e-14,
                    err_msg=f"Generic vs convolution mismatch at node ({ix}, {iy}), "
                    f"direction {d}",
                )

        timer.print_summary(title=f"2D Periodic ({nx}x{ny} grid)")


class TestUnitImpulse3D:
    """Test fused operator matches generic operator for unit impulses in 3D."""

    def _generic_apply(self, decomposition, displacement, force, lam, mu, grad_op):
        """Apply generic B^T C B operator using FEMGradientOperator."""
        gradient = decomposition.real_field("gradient", (3, 3), "quad")
        stress = decomposition.real_field("stress", (3, 3), "quad")

        grad_op.apply(displacement, gradient)
        grad = gradient.s

        # Symmetric strain
        eps = np.zeros_like(stress.s)
        for i in range(3):
            for j in range(3):
                eps[i, j, ...] = 0.5 * (grad[i, j, ...] + grad[j, i, ...])

        trace = eps[0, 0, ...] + eps[1, 1, ...] + eps[2, 2, ...]

        # Stress
        for i in range(3):
            for j in range(3):
                stress.s[i, j, ...] = 2 * mu * eps[i, j, ...]
                if i == j:
                    stress.s[i, j, ...] += lam * trace

        # Fill stress ghost cells for periodic contributions
        decomposition.communicate_ghosts(stress)

        # Divergence
        force.pg[...] = 0.0
        quad_weights = grad_op.quadrature_weights
        grad_op.transpose(stress, force, quad_weights)

    def _convolution_apply(
        self, decomposition, displacement, force, lam, mu, conv_op, quad_weights
    ):
        """Apply B^T C B operator using ConvolutionOperator.

        This uses a ConvolutionOperator created from FEMGradientOperator.coefficients
        to verify that both gradient operators produce identical results.

        Note: ConvolutionOperator.transpose doesn't apply quadrature weights internally,
        so we pre-multiply the stress by weights before calling transpose.
        """
        gradient = decomposition.real_field("gradient_conv", (3, 3), "quad")
        stress = decomposition.real_field("stress_conv", (3, 3), "quad")

        conv_op.apply(displacement, gradient)
        grad = gradient.s

        # Symmetric strain (same as _generic_apply)
        eps = np.zeros_like(stress.s)
        for i in range(3):
            for j in range(3):
                eps[i, j, ...] = 0.5 * (grad[i, j, ...] + grad[j, i, ...])

        trace = eps[0, 0, ...] + eps[1, 1, ...] + eps[2, 2, ...]

        # Stress (isotropic constitutive law)
        for i in range(3):
            for j in range(3):
                stress.s[i, j, ...] = 2 * mu * eps[i, j, ...]
                if i == j:
                    stress.s[i, j, ...] += lam * trace

        # Pre-multiply stress by quadrature weights (ConvolutionOperator.transpose
        # doesn't apply weights internally like FEMGradientOperator does)
        for q in range(len(quad_weights)):
            stress.s[:, :, q, ...] *= quad_weights[q]

        # Fill stress ghost cells for periodic contributions
        decomposition.communicate_ghosts(stress)

        # Divergence using transpose of ConvolutionOperator (without weights)
        force.pg[...] = 0.0
        conv_op.transpose(stress, force, [])

    def test_unit_impulse_non_periodic(self):
        """Test unit impulse response for non-periodic BC in 3D.

        With node-based material field indexing, non-periodic vs periodic is
        distinguished only by how ghost cells are filled (not by kernel logic).

        Compares three approaches:
        1. Fused IsotropicStiffnessOperator3D
        2. Generic FEMGradientOperator
        3. ConvolutionOperator with FEMGradientOperator.coefficients

        Also outputs timing information for the generic vs convolution approaches.
        """
        nx, ny, nz = 6, 6, 6
        grid_spacing = [0.25, 0.25, 0.25]

        lam = 2.0
        mu = 1.5

        fused_op = muGrid.IsotropicStiffnessOperator3D(grid_spacing)
        grad_op = muGrid.FEMGradientOperator(3, grid_spacing)

        # Create ConvolutionOperator from FEMGradientOperator coefficients
        conv_op = muGrid.GenericLinearOperator([0, 0, 0], grad_op.coefficients)
        quad_weights = grad_op.quadrature_weights

        # Use CartesianDecomposition for ghost handling
        comm = muGrid.Communicator()
        decomposition = muGrid.CartesianDecomposition(
            comm,
            (nx, ny, nz),
            nb_subdivisions=(1, 1, 1),
            nb_ghosts_left=(1, 1, 1),
            nb_ghosts_right=(1, 1, 1),
            nb_sub_pts={"quad": grad_op.nb_quad_pts},
        )

        # Material field same size as node field, with ghost cells
        fc_mat = muGrid.GlobalFieldCollection(
            (nx, ny, nz), nb_ghosts_left=(1, 1, 1), nb_ghosts_right=(1, 1, 1)
        )

        displacement = decomposition.real_field("displacement", (3,))
        force_fused = decomposition.real_field("force_fused", (3,))
        force_generic = decomposition.real_field("force_generic", (3,))
        force_convolution = decomposition.real_field("force_convolution", (3,))

        lambda_field = fc_mat.real_field("lambda", (1,))
        mu_field = fc_mat.real_field("mu", (1,))
        lambda_field.p[...] = lam
        mu_field.p[...] = mu
        # Fill ghost cells with periodic copies
        lambda_field.pg[...] = np.pad(
            lambda_field.p, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="wrap"
        )
        mu_field.pg[...] = np.pad(
            mu_field.p, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="wrap"
        )

        timer = Timer()

        # Test a few interior nodes
        test_positions = [(2, 2, 2), (3, 2, 4)]
        for ix, iy, iz in test_positions:
            for d in range(3):
                displacement.p[...] = 0.0
                displacement.p[d, ix, iy, iz] = 1.0

                decomposition.communicate_ghosts(displacement)

                force_fused.p[...] = 0.0
                force_generic.p[...] = 0.0
                force_convolution.p[...] = 0.0

                with timer("fused"):
                    fused_op.apply(displacement, lambda_field, mu_field, force_fused)

                with timer("FEMGradientOperator"):
                    self._generic_apply(
                        decomposition, displacement, force_generic, lam, mu, grad_op
                    )

                with timer("ConvolutionOperator"):
                    self._convolution_apply(
                        decomposition,
                        displacement,
                        force_convolution,
                        lam,
                        mu,
                        conv_op,
                        quad_weights,
                    )

                # Compare fused vs generic FEMGradientOperator
                # With node-based indexing, boundary forces are computed too
                np.testing.assert_allclose(
                    force_fused.p,
                    force_generic.p,
                    rtol=1e-10,
                    atol=1e-14,
                    err_msg=f"Fused vs generic mismatch at node ({ix}, {iy}, {iz}), "
                    f"direction {d}",
                )

                # Compare generic FEMGradientOperator vs ConvolutionOperator
                np.testing.assert_allclose(
                    force_generic.p,
                    force_convolution.p,
                    rtol=1e-10,
                    atol=1e-14,
                    err_msg="Generic vs convolution mismatch at node "
                    f"({ix}, {iy}, {iz}), "
                    f"direction {d}",
                )

        timer.print_summary(title=f"3D Non-periodic ({nx}x{ny}x{nz} grid)")

    def test_unit_impulse_periodic(self):
        """Test unit impulse response for periodic BC in 3D.

        Compares three approaches:
        1. Fused IsotropicStiffnessOperator3D
        2. Generic FEMGradientOperator
        3. ConvolutionOperator with FEMGradientOperator.coefficients
        """
        nx, ny, nz = 5, 5, 5
        grid_spacing = [0.25, 0.25, 0.25]

        lam = 2.0
        mu = 1.5

        fused_op = muGrid.IsotropicStiffnessOperator3D(grid_spacing)
        grad_op = muGrid.FEMGradientOperator(3, grid_spacing)

        # Create ConvolutionOperator from FEMGradientOperator coefficients
        conv_op = muGrid.GenericLinearOperator([0, 0, 0], grad_op.coefficients)
        quad_weights = grad_op.quadrature_weights

        # Use CartesianDecomposition to handle ghost communication for periodic BC
        comm = muGrid.Communicator()
        decomposition = muGrid.CartesianDecomposition(
            comm,
            (nx, ny, nz),
            nb_subdivisions=(1, 1, 1),
            nb_ghosts_left=(1, 1, 1),
            nb_ghosts_right=(1, 1, 1),
            nb_sub_pts={"quad": grad_op.nb_quad_pts},
        )

        # Periodic: material field with ghost cells
        fc_mat = muGrid.GlobalFieldCollection(
            (nx, ny, nz), nb_ghosts_left=(1, 1, 1), nb_ghosts_right=(1, 1, 1)
        )

        displacement = decomposition.real_field("displacement", (3,))
        force_fused = decomposition.real_field("force_fused", (3,))
        force_generic = decomposition.real_field("force_generic", (3,))
        force_convolution = decomposition.real_field("force_convolution", (3,))

        lambda_field = fc_mat.real_field("lambda", (1,))
        mu_field = fc_mat.real_field("mu", (1,))
        lambda_field.p[...] = lam
        mu_field.p[...] = mu

        # Fill ghost cells with periodic copies
        lambda_field.pg[...] = np.pad(
            lambda_field.p, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="wrap"
        )
        mu_field.pg[...] = np.pad(
            mu_field.p, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="wrap"
        )

        timer = Timer()

        # Test boundary and interior nodes
        test_positions = [(0, 0, 0), (4, 4, 4), (2, 3, 1)]
        for ix, iy, iz in test_positions:
            for d in range(3):
                # Set displacement and use communicate_ghosts for periodic filling
                displacement.p[...] = 0.0
                displacement.p[d, ix, iy, iz] = 1.0

                # communicate_ghosts fills ghost cells with periodic copies
                decomposition.communicate_ghosts(displacement)

                force_fused.p[...] = 0.0
                force_generic.p[...] = 0.0
                force_convolution.p[...] = 0.0

                with timer("fused"):
                    fused_op.apply(displacement, lambda_field, mu_field, force_fused)

                with timer("FEMGradientOperator"):
                    self._generic_apply(
                        decomposition, displacement, force_generic, lam, mu, grad_op
                    )

                with timer("ConvolutionOperator"):
                    self._convolution_apply(
                        decomposition,
                        displacement,
                        force_convolution,
                        lam,
                        mu,
                        conv_op,
                        quad_weights,
                    )

                # Compare fused vs generic FEMGradientOperator
                np.testing.assert_allclose(
                    force_fused.p,
                    force_generic.p,
                    rtol=1e-10,
                    atol=1e-14,
                    err_msg=f"Fused vs generic mismatch at node ({ix}, {iy}, {iz}), "
                    f"direction {d}",
                )

                # Compare generic FEMGradientOperator vs ConvolutionOperator
                np.testing.assert_allclose(
                    force_generic.p,
                    force_convolution.p,
                    rtol=1e-10,
                    atol=1e-14,
                    err_msg="Generic vs convolution mismatch at node "
                    f"({ix}, {iy}, {iz}), "
                    f"direction {d}",
                )

        timer.print_summary(title=f"3D Periodic ({nx}x{ny}x{nz} grid)")


# =============================================================================
# Validation Guard Tests
# =============================================================================


class TestValidationGuard2D:
    """Test that validation guards raise appropriate exceptions in 2D."""

    def test_wrong_material_field_size(self):
        """Test error when material field has wrong size."""
        nx, ny = 8, 8
        grid_spacing = [0.25, 0.25]

        op = muGrid.IsotropicStiffnessOperator2D(grid_spacing)

        # Displacement field with proper ghosts
        fc = muGrid.GlobalFieldCollection(
            (nx, ny), nb_ghosts_left=(1, 1), nb_ghosts_right=(1, 1)
        )
        displacement = fc.real_field("displacement", (2,))
        force = fc.real_field("force", (2,))

        # Material field with wrong size (not same as node field)
        fc_mat_wrong = muGrid.GlobalFieldCollection(
            (nx - 2, ny - 2), nb_ghosts_left=(1, 1), nb_ghosts_right=(1, 1)
        )
        lambda_field = fc_mat_wrong.real_field("lambda", (1,))
        mu_field = fc_mat_wrong.real_field("mu", (1,))
        lambda_field.pg[...] = 1.0
        mu_field.pg[...] = 1.0

        with pytest.raises(RuntimeError) as exc_info:
            op.apply(displacement, lambda_field, mu_field, force)
        err_msg = str(exc_info.value).lower()
        assert "material field" in err_msg or "dimensions" in err_msg

    def test_missing_left_ghosts_node(self):
        """Test error when left ghost cells are missing for node field."""
        nx, ny = 8, 8
        grid_spacing = [0.25, 0.25]

        op = muGrid.IsotropicStiffnessOperator2D(grid_spacing)

        # Only right ghosts for node field - this is invalid
        fc = muGrid.GlobalFieldCollection(
            (nx, ny), nb_ghosts_left=(0, 0), nb_ghosts_right=(1, 1)
        )
        displacement = fc.real_field("displacement", (2,))
        force = fc.real_field("force", (2,))

        # Proper material field with ghost cells
        fc_mat = muGrid.GlobalFieldCollection(
            (nx, ny), nb_ghosts_left=(1, 1), nb_ghosts_right=(1, 1)
        )
        lambda_field = fc_mat.real_field("lambda", (1,))
        mu_field = fc_mat.real_field("mu", (1,))
        lambda_field.pg[...] = 1.0
        mu_field.pg[...] = 1.0

        with pytest.raises(RuntimeError) as exc_info:
            op.apply(displacement, lambda_field, mu_field, force)
        err_msg = str(exc_info.value).lower()
        assert "ghost" in err_msg or "ghosts" in err_msg

    def test_missing_right_ghosts(self):
        """Test error when right ghost cells are missing."""
        nx, ny = 8, 8
        grid_spacing = [0.25, 0.25]

        op = muGrid.IsotropicStiffnessOperator2D(grid_spacing)

        # No right ghosts for node field
        fc = muGrid.GlobalFieldCollection(
            (nx, ny), nb_ghosts_left=(1, 1), nb_ghosts_right=(0, 0)
        )
        displacement = fc.real_field("displacement", (2,))
        force = fc.real_field("force", (2,))

        # Proper material field with ghost cells
        fc_mat = muGrid.GlobalFieldCollection(
            (nx, ny), nb_ghosts_left=(1, 1), nb_ghosts_right=(1, 1)
        )
        lambda_field = fc_mat.real_field("lambda", (1,))
        mu_field = fc_mat.real_field("mu", (1,))
        lambda_field.pg[...] = 1.0
        mu_field.pg[...] = 1.0

        with pytest.raises(RuntimeError) as exc_info:
            op.apply(displacement, lambda_field, mu_field, force)
        assert "ghost" in str(exc_info.value).lower()

    def test_missing_material_ghosts(self):
        """Test error when ghost cells are missing for material field."""
        nx, ny = 8, 8
        grid_spacing = [0.25, 0.25]

        op = muGrid.IsotropicStiffnessOperator2D(grid_spacing)

        # Proper node field with ghost cells
        fc = muGrid.GlobalFieldCollection(
            (nx, ny), nb_ghosts_left=(1, 1), nb_ghosts_right=(1, 1)
        )
        displacement = fc.real_field("displacement", (2,))
        force = fc.real_field("force", (2,))

        # Material field without ghost cells
        fc_mat = muGrid.GlobalFieldCollection((nx, ny))
        lambda_field = fc_mat.real_field("lambda", (1,))
        mu_field = fc_mat.real_field("mu", (1,))
        lambda_field.s[...] = 1.0
        mu_field.s[...] = 1.0

        with pytest.raises(RuntimeError) as exc_info:
            op.apply(displacement, lambda_field, mu_field, force)
        err_msg = str(exc_info.value).lower()
        assert "ghost" in err_msg or "ghosts" in err_msg

    def test_valid_config(self):
        """Test that valid configuration does not raise."""
        nx, ny = 8, 8
        grid_spacing = [0.25, 0.25]

        op = muGrid.IsotropicStiffnessOperator2D(grid_spacing)

        # Node field with ghost cells
        fc = muGrid.GlobalFieldCollection(
            (nx, ny), nb_ghosts_left=(1, 1), nb_ghosts_right=(1, 1)
        )
        displacement = fc.real_field("displacement", (2,))
        force = fc.real_field("force", (2,))

        # Material field same size as node field, with ghost cells
        fc_mat = muGrid.GlobalFieldCollection(
            (nx, ny), nb_ghosts_left=(1, 1), nb_ghosts_right=(1, 1)
        )
        lambda_field = fc_mat.real_field("lambda", (1,))
        mu_field = fc_mat.real_field("mu", (1,))
        lambda_field.p[...] = 1.0
        mu_field.p[...] = 1.0

        # Fill ghost cells with periodic copies
        lambda_field.pg[...] = np.pad(
            lambda_field.p, ((0, 0), (1, 1), (1, 1)), mode="wrap"
        )
        mu_field.pg[...] = np.pad(mu_field.p, ((0, 0), (1, 1), (1, 1)), mode="wrap")

        # Should not raise
        op.apply(displacement, lambda_field, mu_field, force)


class TestValidationGuard3D:
    """Test that validation guards raise appropriate exceptions in 3D."""

    def test_wrong_material_field_size(self):
        """Test error when material field has wrong size in 3D."""
        nx, ny, nz = 6, 6, 6
        grid_spacing = [0.25, 0.25, 0.25]

        op = muGrid.IsotropicStiffnessOperator3D(grid_spacing)

        fc = muGrid.GlobalFieldCollection(
            (nx, ny, nz), nb_ghosts_left=(1, 1, 1), nb_ghosts_right=(1, 1, 1)
        )
        displacement = fc.real_field("displacement", (3,))
        force = fc.real_field("force", (3,))

        # Wrong size (not same as node field)
        fc_mat_wrong = muGrid.GlobalFieldCollection(
            (nx - 2, ny - 2, nz - 2),
            nb_ghosts_left=(1, 1, 1),
            nb_ghosts_right=(1, 1, 1),
        )
        lambda_field = fc_mat_wrong.real_field("lambda", (1,))
        mu_field = fc_mat_wrong.real_field("mu", (1,))
        lambda_field.pg[...] = 1.0
        mu_field.pg[...] = 1.0

        with pytest.raises(RuntimeError) as exc_info:
            op.apply(displacement, lambda_field, mu_field, force)
        err_msg = str(exc_info.value).lower()
        assert "material field" in err_msg or "dimensions" in err_msg

    def test_missing_ghosts(self):
        """Test error when ghost cells are missing in 3D."""
        nx, ny, nz = 6, 6, 6
        grid_spacing = [0.25, 0.25, 0.25]

        op = muGrid.IsotropicStiffnessOperator3D(grid_spacing)

        # No ghosts at all for node field
        fc = muGrid.GlobalFieldCollection((nx, ny, nz))
        displacement = fc.real_field("displacement", (3,))
        force = fc.real_field("force", (3,))

        # Proper material field with ghost cells
        fc_mat = muGrid.GlobalFieldCollection(
            (nx, ny, nz), nb_ghosts_left=(1, 1, 1), nb_ghosts_right=(1, 1, 1)
        )
        lambda_field = fc_mat.real_field("lambda", (1,))
        mu_field = fc_mat.real_field("mu", (1,))
        lambda_field.pg[...] = 1.0
        mu_field.pg[...] = 1.0

        with pytest.raises(RuntimeError) as exc_info:
            op.apply(displacement, lambda_field, mu_field, force)
        assert "ghost" in str(exc_info.value).lower()


# =============================================================================
# GPU Unit Impulse Tests (Dual-device comparison)
# =============================================================================


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU backend not available")
class TestGPUUnitImpulse:
    """Test GPU unit impulse response matches CPU."""

    def setup_method(self, method):
        """Skip tests if no GPU is available at runtime."""
        if not muGrid.is_gpu_available():
            pytest.skip("No GPU device available at runtime")
        if not HAS_CUPY:
            pytest.skip("CuPy not available for GPU tests")

    def test_gpu_unit_impulse_2d(self):
        """Test GPU unit impulse response in 2D."""
        nx, ny = 8, 8
        grid_spacing = [0.25, 0.25]

        op = muGrid.IsotropicStiffnessOperator2D(grid_spacing)

        # GPU kernels require ghosts on both sides
        fc_cpu = muGrid.GlobalFieldCollection(
            (nx, ny), nb_ghosts_left=(1, 1), nb_ghosts_right=(1, 1)
        )
        # Material field same size as node field, with ghost cells
        fc_mat_cpu = muGrid.GlobalFieldCollection(
            (nx, ny), nb_ghosts_left=(1, 1), nb_ghosts_right=(1, 1)
        )

        disp_cpu = fc_cpu.real_field("displacement", (2,))
        force_cpu = fc_cpu.real_field("force", (2,))
        lam_cpu = fc_mat_cpu.real_field("lambda", (1,))
        mu_cpu = fc_mat_cpu.real_field("mu", (1,))

        lam_cpu.p[...] = 2.0
        mu_cpu.p[...] = 1.5
        # Fill ghost cells with periodic copies
        lam_cpu.pg[...] = np.pad(lam_cpu.p, ((0, 0), (1, 1), (1, 1)), mode="wrap")
        mu_cpu.pg[...] = np.pad(mu_cpu.p, ((0, 0), (1, 1), (1, 1)), mode="wrap")

        # GPU setup
        fc_gpu = muGrid.GlobalFieldCollection(
            (nx, ny),
            nb_ghosts_left=(1, 1),
            nb_ghosts_right=(1, 1),
            device=muGrid.Device.gpu(),
        )
        fc_mat_gpu = muGrid.GlobalFieldCollection(
            (nx, ny),
            nb_ghosts_left=(1, 1),
            nb_ghosts_right=(1, 1),
            device=muGrid.Device.gpu(),
        )

        disp_gpu = fc_gpu.real_field("displacement", (2,))
        force_gpu = fc_gpu.real_field("force", (2,))
        lam_gpu = fc_mat_gpu.real_field("lambda", (1,))
        mu_gpu = fc_mat_gpu.real_field("mu", (1,))

        lam_gpu.p[...] = 2.0
        mu_gpu.p[...] = 1.5
        # Fill ghost cells with periodic copies on GPU
        lam_gpu.pg[...] = cp.pad(lam_gpu.p, ((0, 0), (1, 1), (1, 1)), mode="wrap")
        mu_gpu.pg[...] = cp.pad(mu_gpu.p, ((0, 0), (1, 1), (1, 1)), mode="wrap")

        # Test a few unit impulses
        test_positions = [(2, 2), (3, 4), (5, 5)]
        for ix, iy in test_positions:
            for d in range(2):
                # Set up CPU and GPU data independently (same logical values)
                disp_cpu.pg[...] = 0.0
                disp_cpu.pg[d, ix + 1, iy + 1] = 1.0

                disp_gpu.pg[...] = 0.0
                disp_gpu.pg[d, ix + 1, iy + 1] = 1.0

                force_cpu.p[...] = 0.0
                force_gpu.p[...] = 0.0

                op.apply(disp_cpu, lam_cpu, mu_cpu, force_cpu)
                op.apply(disp_gpu, lam_gpu, mu_gpu, force_gpu)

                force_gpu_np = cp.asnumpy(force_gpu.p)
                np.testing.assert_allclose(
                    force_gpu_np,
                    force_cpu.p,
                    rtol=1e-10,
                    atol=1e-14,
                    err_msg=f"GPU mismatch at node ({ix}, {iy}), direction {d}",
                )

    def test_gpu_unit_impulse_3d(self):
        """Test GPU unit impulse response in 3D."""
        nx, ny, nz = 6, 6, 6
        grid_spacing = [0.25, 0.25, 0.25]

        op = muGrid.IsotropicStiffnessOperator3D(grid_spacing)

        # GPU kernels require ghosts on both sides
        fc_cpu = muGrid.GlobalFieldCollection(
            (nx, ny, nz), nb_ghosts_left=(1, 1, 1), nb_ghosts_right=(1, 1, 1)
        )
        # Material field same size as node field, with ghost cells
        fc_mat_cpu = muGrid.GlobalFieldCollection(
            (nx, ny, nz), nb_ghosts_left=(1, 1, 1), nb_ghosts_right=(1, 1, 1)
        )

        disp_cpu = fc_cpu.real_field("displacement", (3,))
        force_cpu = fc_cpu.real_field("force", (3,))
        lam_cpu = fc_mat_cpu.real_field("lambda", (1,))
        mu_cpu = fc_mat_cpu.real_field("mu", (1,))

        lam_cpu.p[...] = 2.0
        mu_cpu.p[...] = 1.5
        # Fill ghost cells with periodic copies
        lam_cpu.pg[...] = np.pad(
            lam_cpu.p, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="wrap"
        )
        mu_cpu.pg[...] = np.pad(
            mu_cpu.p, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="wrap"
        )

        # GPU setup
        fc_gpu = muGrid.GlobalFieldCollection(
            (nx, ny, nz),
            nb_ghosts_left=(1, 1, 1),
            nb_ghosts_right=(1, 1, 1),
            device=muGrid.Device.gpu(),
        )
        fc_mat_gpu = muGrid.GlobalFieldCollection(
            (nx, ny, nz),
            nb_ghosts_left=(1, 1, 1),
            nb_ghosts_right=(1, 1, 1),
            device=muGrid.Device.gpu(),
        )

        disp_gpu = fc_gpu.real_field("displacement", (3,))
        force_gpu = fc_gpu.real_field("force", (3,))
        lam_gpu = fc_mat_gpu.real_field("lambda", (1,))
        mu_gpu = fc_mat_gpu.real_field("mu", (1,))

        lam_gpu.p[...] = 2.0
        mu_gpu.p[...] = 1.5
        # Fill ghost cells with periodic copies on GPU
        lam_gpu.pg[...] = cp.pad(
            lam_gpu.p, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="wrap"
        )
        mu_gpu.pg[...] = cp.pad(
            mu_gpu.p, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="wrap"
        )

        # Test a few unit impulses
        test_positions = [(2, 2, 2), (3, 2, 3), (4, 3, 2)]
        for ix, iy, iz in test_positions:
            for d in range(3):
                # Set up CPU and GPU data independently (same logical values)
                disp_cpu.pg[...] = 0.0
                disp_cpu.pg[d, ix + 1, iy + 1, iz + 1] = 1.0

                disp_gpu.pg[...] = 0.0
                disp_gpu.pg[d, ix + 1, iy + 1, iz + 1] = 1.0

                force_cpu.p[...] = 0.0
                force_gpu.p[...] = 0.0

                op.apply(disp_cpu, lam_cpu, mu_cpu, force_cpu)
                op.apply(disp_gpu, lam_gpu, mu_gpu, force_gpu)

                force_gpu_np = cp.asnumpy(force_gpu.p)
                np.testing.assert_allclose(
                    force_gpu_np,
                    force_cpu.p,
                    rtol=1e-10,
                    atol=1e-14,
                    err_msg=f"GPU mismatch at node ({ix}, {iy}, {iz}), direction {d}",
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

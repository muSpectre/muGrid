"""
Tests for the trilinear hexahedron with one quadrature point at the voxel
centre (FEMElement.q1_reduced).
"""

import numpy as np
import pytest
from conftest import (
    create_device,
    get_array_module,
    get_test_devices,
    skip_if_gpu_unavailable,
)

import muGrid


def to_host(array):
    return array.get() if hasattr(array, "get") else np.asarray(array)


@pytest.mark.parametrize("device", get_test_devices())
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_reduced_gradient_is_the_mean_of_the_gauss_point_gradients(
        comm, device, dtype):
    """The centre gradient equals the mean of the eight Gauss-point
    gradients of FEMElement.q1, on every rank's part of a random field."""
    skip_if_gpu_unavailable(device)
    nb_grid_pts = (8, 6, 4)
    spacing = [1.0 / n for n in nb_grid_pts]
    xp = get_array_module(device)
    decomposition = muGrid.CartesianDecomposition(
        comm, nb_grid_pts, nb_ghosts_left=(1, 1, 1),
        nb_ghosts_right=(1, 1, 1), device=create_device(device))
    # every rank builds the same global field and takes its subdomain
    u_global = np.random.default_rng(0).standard_normal((3, *nb_grid_pts))
    location = decomposition.subdomain_locations
    size = decomposition.nb_subdomain_grid_pts
    local = (..., *(slice(o, o + n) for o, n in zip(location, size)))
    u = decomposition.real_field("u", (3,), dtype=dtype)
    u.p[...] = xp.asarray(u_global[local], dtype=dtype)

    reduced = muGrid.FEMGradientOperator(
        3, spacing, muGrid.FEMElement.q1_reduced, dtype=dtype)
    full = muGrid.FEMGradientOperator(
        3, spacing, muGrid.FEMElement.q1, dtype=dtype)
    assert reduced.nb_quad_pts == 1
    np.testing.assert_allclose(reduced.quadrature_weights,
                               [np.prod(spacing)])

    collection = decomposition.collection
    collection.set_nb_sub_pts("centre", 1)
    collection.set_nb_sub_pts("gauss", 8)
    at_centre = collection.real_field("centre", (3, 3), "centre", dtype=dtype)
    at_gauss = collection.real_field("gauss", (3, 3), "gauss", dtype=dtype)
    decomposition.communicate_ghosts(u)
    reduced.apply(u, at_centre)
    full.apply(u, at_gauss)
    tolerance = 1e-12 if dtype == np.float64 else 1e-4
    scale = float(abs(to_host(at_gauss.s)).max())
    np.testing.assert_allclose(
        to_host(at_centre.s)[:, :, 0], to_host(at_gauss.s).mean(axis=2),
        atol=tolerance * scale)


def test_reduced_element_is_rejected_where_it_does_not_apply():
    """2D has no reduced element; the fused stiffness would be rank
    deficient."""
    with pytest.raises(ValueError, match="3D only"):
        muGrid.FEMGradientOperator(2, [1, 1], muGrid.FEMElement.q1_reduced)
    with pytest.raises(RuntimeError, match="rank deficient"):
        muGrid.IsotropicStiffnessOperator3D(
            [1, 1, 1], muGrid.FEMElement.q1_reduced)

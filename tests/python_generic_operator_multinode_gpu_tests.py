#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file    python_generic_operator_multinode_gpu_tests.py

@brief   GenericLinearOperator apply/transpose with several nodal sub-points
         per pixel, compared against a plain NumPy reference on CPU and GPU.

A pixel carrying more than one nodal sub-point (e.g. biquadratic finite
elements: 4 nodes per pixel) exercises the nodal sub-point index of the
sparse operator. With a single nodal sub-point that index is always zero,
so the existing tests do not cover it.
"""

import numpy as np
import pytest

from conftest import cp, get_test_devices, skip_if_gpu_unavailable

import muGrid
from muGrid import GenericLinearOperator

NB_GRID = (8, 6)


def _setup(device, nb_nodes, nb_quad, nb_comp):
    """Periodic FFT engine with one ghost layer, nodal and quadrature fields."""
    fft = muGrid.FFTEngine(nb_domain_grid_pts=NB_GRID, nb_ghosts_left=[1, 1],
                           nb_ghosts_right=[1, 1], device=device)
    fc = fft.real_space_collection
    fc.set_nb_sub_pts("quad_points", nb_quad)
    fc.set_nb_sub_pts("nodal_points", nb_nodes)
    nodal = fc.real_field("nodal", (nb_comp,), "nodal_points")
    quad = fc.real_field("quad", (nb_comp, 2), "quad_points")
    return fft, nodal, quad


def _to_host(a, device):
    return cp.asnumpy(a) if device == "gpu" else np.array(a)


def _to_device(a, device):
    return cp.asarray(a) if device == "gpu" else a


@pytest.mark.parametrize("device", get_test_devices())
@pytest.mark.parametrize("nb_comp", [1, 3])
@pytest.mark.parametrize("nb_nodes, nb_quad", [(1, 4), (2, 3), (4, 9)])
def test_apply_multinode(device, nb_nodes, nb_quad, nb_comp):
    """apply: q[c,d,q,x,y] = sum_{n,i,j} S[d,q,n,i,j] u[c,n,x+i,y+j] (periodic)"""
    skip_if_gpu_unavailable(device)
    rng = np.random.default_rng(1)
    stencil = rng.standard_normal((2, nb_quad, nb_nodes, 2, 2))  # [d, q, n, i, j]
    u = rng.standard_normal((nb_comp, nb_nodes, *NB_GRID))

    expected = np.zeros((nb_comp, 2, nb_quad, *NB_GRID))
    for i in range(2):
        for j in range(2):
            expected += np.einsum("dqn,cnxy->cdqxy", stencil[..., i, j],
                                  np.roll(u, (-i, -j), axis=(-2, -1)))

    fft, nodal, quad = _setup(device, nb_nodes, nb_quad, nb_comp)
    nodal.s[...] = _to_device(u, device)
    fft.communicate_ghosts(nodal)
    GenericLinearOperator([0, 0], stencil).apply(nodal_field=nodal,
                                                 quadrature_point_field=quad)

    np.testing.assert_allclose(_to_host(quad.s, device), expected,
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("device", get_test_devices())
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("nb_comp", [1, 3])
@pytest.mark.parametrize("nb_nodes, nb_quad", [(1, 4), (2, 3), (4, 9)])
def test_transpose_multinode(device, nb_nodes, nb_quad, nb_comp, weighted):
    """transpose: u[c,n,x,y] = sum_{d,q,i,j} w_q S[d,q,n,i,j] q[c,d,q,x-i,y-j]"""
    skip_if_gpu_unavailable(device)
    rng = np.random.default_rng(2)
    stencil = rng.standard_normal((2, nb_quad, nb_nodes, 2, 2))
    g = rng.standard_normal((nb_comp, 2, nb_quad, *NB_GRID))
    weights = rng.uniform(0.5, 1.5, nb_quad) if weighted else np.ones(nb_quad)

    expected = np.zeros((nb_comp, nb_nodes, *NB_GRID))
    for i in range(2):
        for j in range(2):
            expected += np.einsum("dqn,q,cdqxy->cnxy", stencil[..., i, j], weights,
                                  np.roll(g, (i, j), axis=(-2, -1)))

    fft, nodal, quad = _setup(device, nb_nodes, nb_quad, nb_comp)
    quad.s[...] = _to_device(g, device)
    fft.communicate_ghosts(quad)
    GenericLinearOperator([0, 0], stencil).transpose(
        quadrature_point_field=quad, nodal_field=nodal,
        weights=list(weights) if weighted else None)

    np.testing.assert_allclose(_to_host(nodal.s, device), expected,
                               rtol=1e-12, atol=1e-12)

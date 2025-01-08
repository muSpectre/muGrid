#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file     python_field_tests.py

@author  Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date    18 Oct 2019

@brief   test field collections and fields

Copyright © 2018 Till Junge

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

import functools

import unittest
import numpy as np

import muGrid


verbose = True


class ConvolutionOperatorCheck(unittest.TestCase):

    def template_test_apply_in_2D_field(self, n_component: int):
        # Create a 2D grid
        n_dim = 2
        nx = 3        # number of pixels in x axis
        ny = 3        # number of pixels in y axis
        n_sub_pt = 2  # number of sub-points (e.g. quadrature) in one pixel
        fc = muGrid.GlobalFieldCollection((nx, ny), sub_pts={'quad': n_sub_pt})

        # Fill the node with some scalar values
        nodal = fc.real_field('nodal-value', n_component)
        nodal.p = np.arange(n_component * nx * ny).reshape(n_component, nx, ny)
        if verbose:
            print(f"\nnodal: shape={nodal.shape}, value=\n{nodal.p}")

        # Create the pixel-wise gradient operator
        d = 10
        # Set values such that the eventual values at different sub-points are different,
        # though they are not necessarily gradients.
        pixel_gradient = np.array([[d, .0, .0, .0],
                                   [.0, .0, d, .0],
                                   [.0, d, .0, .0],
                                   [.0, .0, .0, d]], dtype=float)
        n_nodal_pixel = 1
        d_op = muGrid.ConvolutionOperatorDefault(pixel_gradient, n_dim, n_sub_pt, n_nodal_pixel)

        # Create a quadrature field to hold derivatives
        quad = fc.real_field('quad-grad', n_dim*n_component, 'quad')

        # Apply the graident operator
        d_op.apply(nodal, quad)
        grad_get_s = quad.s.reshape(n_dim, n_component, n_sub_pt, nx, ny)
        grad_get_p = quad.p.reshape(n_sub_pt, n_dim, n_component, nx, ny)

        # Create a pack of nodal values, each with a different offset
        offset_00 = nodal.p
        offset_10 = np.roll(nodal.p, (-1,0), axis=(-2,-1))
        offset_01 = np.roll(nodal.p, (0,-1), axis=(-2,-1))
        offset_11 = np.roll(nodal.p, (-1,-1), axis=(-2,-1))
        # NOTE: The oder of offset must keep the same as impelemented in lib
        offset_nodes = np.stack((offset_00, offset_10, offset_01, offset_11), axis=0)
        n_offset = np.size(offset_nodes, axis=0)

        # Compute the reference value
        pixel_gradient = pixel_gradient.reshape(n_sub_pt, n_dim, n_offset)
        if verbose:
            print(f"\nideal operator: shape={pixel_gradient.shape}, value=\n{pixel_gradient}")

        grad_ref_s = np.einsum("sdo,ocxy->dcsxy", pixel_gradient, offset_nodes)
        grad_ref_p = np.einsum("sdo,ocxy->sdcxy", pixel_gradient, offset_nodes)

        # Check
        np.testing.assert_allclose(grad_get_s, grad_ref_s)
        np.testing.assert_allclose(grad_get_p, grad_ref_p)


    test_apply_2D_field_scalar = functools.partialmethod(template_test_apply_in_2D_field, n_component = 1)
    test_apply_2D_field_3D_vector = functools.partialmethod(template_test_apply_in_2D_field, n_component = 3)


if __name__ == '__main__':
    unittest.main()

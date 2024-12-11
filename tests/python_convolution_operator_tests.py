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

import unittest
import numpy as np

import muGrid


class ConvolutionOperatorCheck(unittest.TestCase):
    _rng = np.random.default_rng()

    def test_gradient(self):
        # Create a 2D grid
        nx = 3  # number of pixels in x axis
        ny = 4  # number of pixels in y axis
        n_quad = 2  # number of quadrature points in one pixel
        fc = muGrid.GlobalFieldCollection((nx, ny), sub_pts={'quad': n_quad})

        # Fill the node with random values (scalar)
        nodal = fc.real_field('nodal-value')
        nodal.p = self._rng.random((nx, ny))

        # Create a quadrature field to hold values
        n_dim = 2
        quad = fc.real_field('quad-grad', n_dim, 'quad')
        n_nodal_pixel = 1

        # Create the pixel-wise gradient operator
        qx = 1 / nx
        qy = 1 / ny
        pixel_gradient = np.array([[-qx, qx, .0, .0],
                                   [-qy, .0, .0, qy],
                                   [ .0, .0, qx,-qx],
                                   [ .0,-qy, qy, .0]], dtype=float)
        d_op = muGrid.ConvolutionOperatorDefault(pixel_gradient, n_dim, n_quad, n_nodal_pixel)

        # Apply the graident operator
        d_op.apply(nodal, quad)

        # Create a pack of nodal values, each with a different offset
        # NOTE:
        # - np.roll create copies. But it is okay for testing.
        # - the shift is the negation of the offset
        offset_00 = nodal.p
        offset_10 = np.roll(nodal.p, (-1,0), axis=(0,1))
        offset_01 = np.roll(nodal.p, (0,-1), axis=(0,1))
        offset_11 = np.roll(nodal.p, (-1,-1), axis=(0,1))
        offset_nodes = np.stack((offset_00, offset_10, offset_01, offset_11), axis=0)
        # Compute the reference value
        grad_ref = np.einsum("rn,nxy->rxy", pixel_gradient, offset_nodes)

        # Check
        np.testing.assert_allclose(quad.p, grad_ref)


if __name__ == '__main__':
    unittest.main()

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


class ConvolutionOperatorCheck(unittest.TestCase):

    def template_test_apply_in_2D_field(self, nb_field_compos: int):
        # Parameters
        nb_x_pts = 5           # number of pixels in x axis
        nb_y_pts = 4           # number of pixels in y axis
        nb_operators = 3       # number of operators
        nb_quad_pts = 2        # number of quadrature points
        nb_nodalpixel_pts = 1  # number of nodal pixel points

        # Create the operator
        conv_pts_shape = (2, 3)
        conv_kern0 = np.array([[1, 0, 0],
                               [0, 0, 0]], dtype=float)
        conv_kern1 = np.array([[0, 1, 0],
                               [0, 0, 0]], dtype=float)
        conv_kern2 = np.array([[0, 0, 1],
                               [0, 0, 0]], dtype=float)
        conv_kern3 = np.array([[0, 0, 0],
                               [1, 0, 0]], dtype=float)
        conv_kern4 = np.array([[0, 0, 0],
                               [0, 1, 0]], dtype=float)
        conv_kern5 = np.array([[0, 0, 0],
                               [0, 0, 1]], dtype=float)
        pixel_map = np.vstack([
            conv_kern0.ravel(order='F'),
            conv_kern1.ravel(order='F'),
            conv_kern2.ravel(order='F'),
            conv_kern3.ravel(order='F'),
            conv_kern4.ravel(order='F'),
            conv_kern5.ravel(order='F')])
        d_op = muGrid.ConvolutionOperator(
            pixel_map, conv_pts_shape, nb_nodalpixel_pts, nb_quad_pts, nb_operators)

        # Create the grid
        fc = muGrid.GlobalFieldCollection((nb_x_pts, nb_y_pts), sub_pts={'quad': nb_quad_pts})

        # A nodal field with some sequence as values
        nodal = fc.real_field('nodal-value', nb_field_compos)
        nodal_vals = 1 + np.arange(nb_field_compos * nb_x_pts * nb_y_pts)
        nodal.p = nodal_vals.reshape(nb_field_compos, nb_x_pts, nb_y_pts)

        # Create a quadrature field to store the result
        quad = fc.real_field('quad-grad', (nb_operators, nb_field_compos), 'quad')

        # Apply the graident operator
        d_op.apply(nodal, quad)
    
        # Compute the reference value
        # Create a pack of nodal values, each with a different offset
        offset_00 = nodal.p
        offset_10 = np.roll(nodal.p, (-1,0), axis=(-2,-1))
        offset_01 = np.roll(nodal.p, (0,-1), axis=(-2,-1))
        offset_11 = np.roll(nodal.p, (-1,-1), axis=(-2,-1))
        offset_02 = np.roll(nodal.p, (0,-2), axis=(-2,-1))
        offset_12 = np.roll(nodal.p, (-1,-2), axis=(-2,-1))
        # NOTE: the offset-ed array must be ordered in column major, as that follows the implementation
        offset_nodes = np.stack((offset_00, offset_10, offset_01, offset_11, offset_02, offset_12), axis=0)

        pixel_map = pixel_map.reshape(nb_operators, nb_quad_pts, -1, order='F')
        grad_ref_s = np.einsum("dqo,ocxy->dcqxy", pixel_map, offset_nodes)

        # Check
        np.testing.assert_allclose(quad.s, grad_ref_s)

    # Test cases of a scalar field & a vector field
    test_apply_2D_field_scalar = functools.partialmethod(template_test_apply_in_2D_field, nb_field_compos = 1)
    test_apply_2D_field_3D_vector = functools.partialmethod(template_test_apply_in_2D_field, nb_field_compos = 3)


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file    python_convolution_operator_tests.py

@author  Yizhen Wang <yizhen.wang@imtek.uni-freiburg.de>

@date    11 Dec 2024

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

    def template_test_apply_in_2D_field(self, nb_field_components: int):
        # Parameters
        nb_x_pts = 5  # number of pixels in x axis
        nb_y_pts = 4  # number of pixels in y axis
        nb_stencil_x = 2  # number of stencil points in x axis
        nb_stencil_y = 3  # number of stencil points in y axis
        nb_operators = 3  # number of operators
        nb_quad_pts = 2  # number of quadrature points
        nb_nodal_pts = 1  # number of nodal pixel points

        # Create the operator
        conv_kern0 = np.array([[1, 0, 0], [0, 0, 0]], dtype=float)
        conv_kern1 = np.array([[0, 1, 0], [0, 0, 0]], dtype=float)
        conv_kern2 = np.array([[0, 0, 1], [0, 0, 0]], dtype=float)
        conv_kern3 = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        conv_kern4 = np.array([[0, 0, 0], [0, 1, 0]], dtype=float)
        conv_kern5 = np.array([[0, 0, 0], [0, 0, 1]], dtype=float)
        stencil_oqij = np.array(
            [
                [conv_kern0, conv_kern3],  # Operator 1 for both quadrature points
                [conv_kern1, conv_kern4],  # Operator 2 for both quadrature points
                [conv_kern2, conv_kern5],  # Operator 3 for both quadrature points
            ]
        )
        assert stencil_oqij.shape == (
            nb_operators,
            nb_quad_pts,
            nb_stencil_x,
            nb_stencil_y,
        )
        d_op = muGrid.ConvolutionOperator(2, stencil_oqij)

        # Check that convolution operator has correct shape
        assert d_op.nb_operators == nb_operators
        assert d_op.nb_quad_pts == nb_quad_pts
        assert d_op.nb_nodal_pts == nb_nodal_pts

        # Create the grid
        fc = muGrid.GlobalFieldCollection(
            (nb_x_pts, nb_y_pts), sub_pts={"quad": nb_quad_pts}
        )

        # A nodal field with some sequence as values
        nodal = fc.real_field("nodal-value", nb_field_components)
        nodal_vals = 1 + np.arange(nb_field_components * nb_x_pts * nb_y_pts)
        nodal.p = nodal_vals.reshape(nb_field_components, nb_x_pts, nb_y_pts)

        # Create a quadrature field to store the result
        quad = fc.real_field("quad-grad", (nb_operators, nb_field_components), "quad")

        # Check that quadrature field has correct shape
        assert quad.s.shape == (
            nb_operators,
            nb_field_components,
            nb_quad_pts,
            nb_x_pts,
            nb_y_pts,
        )

        # Apply the gradient operator to the `nodal` field and write result to the `quad` field
        d_op.apply(nodal, quad)

        # Compute the reference value
        # Create a pack of nodal values, each with a different offset
        offset_00 = nodal.p
        offset_10 = np.roll(nodal.p, (-1, 0), axis=(-2, -1))
        offset_01 = np.roll(nodal.p, (0, -1), axis=(-2, -1))
        offset_11 = np.roll(nodal.p, (-1, -1), axis=(-2, -1))
        offset_02 = np.roll(nodal.p, (0, -2), axis=(-2, -1))
        offset_12 = np.roll(nodal.p, (-1, -2), axis=(-2, -1))
        offset_nodes_ijcxy = np.array(
            [[offset_00, offset_01, offset_02], [offset_10, offset_11, offset_12]]
        )
        assert offset_nodes_ijcxy.shape == (
            nb_stencil_x,
            nb_stencil_y,
            nb_field_components,
            nb_x_pts,
            nb_y_pts,
        )

        grad_ref_ocqxy = np.einsum(
            "oqij,ijcxy->ocqxy", stencil_oqij, offset_nodes_ijcxy
        )
        assert grad_ref_ocqxy.shape == (
            nb_operators,
            nb_field_components,
            nb_quad_pts,
            nb_x_pts,
            nb_y_pts,
        )

        # Check
        np.testing.assert_allclose(quad.s, grad_ref_ocqxy)

    # Test cases of a scalar field & a vector field
    test_apply_2D_field_scalar = functools.partialmethod(
        template_test_apply_in_2D_field, nb_field_components=1
    )
    test_apply_2D_field_3D_vector = functools.partialmethod(
        template_test_apply_in_2D_field, nb_field_components=3
    )


if __name__ == "__main__":
    unittest.main()

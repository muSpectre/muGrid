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
import pytest
from NuMPI.Testing.Subdivision import suggest_subdivisions

import muGrid


class ConvolutionOperatorCheck(unittest.TestCase):
    """Test suite for ConvolutionOperator functionality.

    Tests verify correct behavior of convolution operations applied to
    fields with various configurations, including shape validation and
    component matching.
    """

    def template_test_apply_in_2D_field(self, nb_field_components: int):
        """Test convolution operator application on 2D fields.

        This test verifies that the ConvolutionOperator correctly applies a
        stencil to a nodal field and produces output at quadrature points
        with the expected values. It tests the core functionality of the
        convolution operation.

        The convolution is defined mathematically as:
            f_{o,c,q,p} = sum_n sum_k s_{o,q,n,k} g_{c,n,p-k}
        where:
            - f: output field at quadrature points
            - g: input field at nodal points
            - s: stencil
            - o: operator index
            - c: component index
            - q: quadrature point index
            - n: nodal point index (within stencil)
            - p: pixel coordinate
            - k: stencil offset

        Parameters
        ----------
        nb_field_components : int
            Number of components in the field (1 for scalar, 3 for
            vector, etc.)

        Test Coverage
        -------
        - Correct shape of output field
        - Correct computation of convolution values
        - Proper handling of multiple components
        - Periodic boundary conditions (via np.roll)
        """
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
        d_op = muGrid.ConvolutionOperator([0, 0], stencil_oqij)

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
        quad = fc.real_field("quad-grad", (nb_field_components, nb_operators), "quad")

        # Check that quadrature field has correct shape
        assert quad.s.shape == (
            nb_field_components,
            nb_operators,
            nb_quad_pts,
            nb_x_pts,
            nb_y_pts,
        )

        # Apply the gradient operator to the `nodal` field and write result to the
        # `quad` field
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
            "oqij,ijcxy->coqxy", stencil_oqij, offset_nodes_ijcxy
        )
        assert grad_ref_ocqxy.shape == (
            nb_field_components,
            nb_operators,
            nb_quad_pts,
            nb_x_pts,
            nb_y_pts,
        )

        # print(f"actual\n{quad.s}")
        # print(f"desired\n{grad_ref_ocqxy}")
        # Check
        np.testing.assert_allclose(quad.s, grad_ref_ocqxy)

    # Test cases of a scalar field & a vector field
    test_apply_2D_field_scalar = functools.partialmethod(
        template_test_apply_in_2D_field, nb_field_components=1
    )
    test_apply_2D_field_3D_vector = functools.partialmethod(
        template_test_apply_in_2D_field, nb_field_components=3
    )


@pytest.mark.skip(
    reason="Known limitation: Local fields from CartesianDecomposition "
    "cause segfault. Exception checking for is_global() exists in C++ but "
    "exception propagation needs review."
)
def test_malformed_convolution_input(comm):
    """Test convolution operator with local fields raises error.

    The ConvolutionOperator is designed to work only with global fields.
    Local fields from CartesianDecomposition should be rejected with a
    clear error message about field type.

    Setup
    -----
    - Uses CartesianDecomposition to create local fields (wrong!)
    - Attempts to apply ConvolutionOperator to these local fields

    Expected behavior
    -----------------
    Should raise a RuntimeError with message about field type mismatch.
    """
    nb_domain_grid_pts = [4, 6]

    left_ghosts = [1, 1]
    right_ghosts = [1, 1]

    decomp = muGrid.CartesianDecomposition(
        comm,
        nb_domain_grid_pts,
        suggest_subdivisions(2, comm.size),
        left_ghosts,
        right_ghosts)

    fc = decomp.collection

    # Get nodal field
    nodal_field1 = fc.real_field("nodal-field1", (2,))
    nodal_field2 = fc.real_field("nodal-field2", (2,))

    impulse_locations = (nodal_field1.icoordsg[0] == 0) & (
        nodal_field1.icoordsg[1] == 0
    )
    nodal_field1.pg[0, impulse_locations] = 1
    nodal_field2.pg[1, impulse_locations] = 1

    shift = np.array(
        [
            [
                [
                    [1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 1],
                ],
            ],
        ],
    )
    # Stencil shape: (nb_operators=1, nb_quad_pts=2, stencil_y=3,
    # stencil_x=3) But we're using it on a local field collection
    # (after decomposition)
    shift_op = muGrid.ConvolutionOperator([-1, -1], shift)
    # This should raise an error because fields are local, not global
    with pytest.raises(RuntimeError):
        shift_op.apply(nodal_field1, nodal_field2)


def test_convolution_component_mismatch_global():
    """Test that mismatched component counts are caught.

    The ConvolutionOperator requires that:
        output_components == nb_operators * input_components

    This test verifies that the operator validates this constraint and
    raises an appropriate error rather than causing undefined behavior.

    Setup
    -----
    - Creates a 2D grid with a simple stencil (1 operator)
    - Creates an input nodal field with 2 components
    - Creates an output quadrature field with incorrect component count
      (3 instead of 2)
    - The correct count should be: 1 operator × 2 components = 2
      components

    Expected behavior
    -----------------
    Should raise a RuntimeError with a message explaining the component
    mismatch.
    """
    nb_x_pts = 3
    nb_y_pts = 3

    # Create a simple stencil with 1 operator and 1 quadrature point
    stencil = np.array([
        [
            [1, 0],  # First row of stencil points
            [0, 1],
        ]
    ])

    conv_op = muGrid.ConvolutionOperator([0, 0], stencil)

    # Create field collection
    fc = muGrid.GlobalFieldCollection((nb_x_pts, nb_y_pts), sub_pts={"quad": 1})

    # Input field: 2 components on nodal points (default)
    nodal_field = fc.real_field("nodal", (2,))

    # Output field with WRONG component count
    # Should be: nb_operators (1) × nb_components (2) = 2 components
    quad_field = fc.real_field("quad-wrong", (3,), "quad")

    # This should raise an error about component mismatch
    with pytest.raises(RuntimeError):
        conv_op.apply(nodal_field, quad_field)


def test_convolution_component_mismatch_multiple_operators_global():
    """Test component validation with multiple operators.

    When a ConvolutionOperator has multiple operators (e.g., for gradient
    with multiple directional derivatives), the output field must have
    components = nb_operators * nb_input_components.

    Setup
    -----
    - Creates a 2D grid with a stencil having 3 operators
    - Creates an input nodal field with 1 component
    - Creates an output quadrature field with wrong component count
    - The correct count should be: 3 operators × 1 component = 3
      components

    Expected behavior
    -----------------
    Should raise a RuntimeError with a message explaining the component
    mismatch.
    """
    nb_x_pts = 3
    nb_y_pts = 3

    # Create stencil with 3 operators
    stencil = np.zeros((3, 1, 2, 2))  # 3 ops, 1 quad pt, 2x2 stencil
    stencil[0, 0, 0, 0] = 1.0
    stencil[1, 0, 0, 1] = 1.0
    stencil[2, 0, 1, 0] = 1.0

    conv_op = muGrid.ConvolutionOperator([0, 0], stencil)

    # Create field collection
    fc = muGrid.GlobalFieldCollection((nb_x_pts, nb_y_pts), sub_pts={"quad": 1})

    # Input field: 1 component
    nodal_field = fc.real_field("nodal-scalar", (1,))

    # Output field with WRONG component count (should be 3)
    quad_field = fc.real_field("quad-wrong", (2,), "quad")

    # This should raise an error
    with pytest.raises(RuntimeError):
        conv_op.apply(nodal_field, quad_field)


def test_convolution_wrong_input_subpt_type_global():
    """Test that mismatched sub-point counts are caught.

    The ConvolutionOperator requires that input and output fields have
    the same number of sub-points per pixel. This test verifies that
    fields with mismatched sub-point counts are rejected with a clear
    error message.

    Setup
    -----
    - Creates a 2D grid with a simple stencil
    - Creates an input field with 2 sub-points (quadrature)
    - Creates an output field with 1 sub-point (nodal)
    - Tries to apply the operator

    Expected behavior
    -----------------
    Should raise a RuntimeError with message about sub-point count
    mismatch.
    """
    nb_x_pts = 3
    nb_y_pts = 3

    # Create a simple stencil
    stencil = np.array([[[1, 0], [0, 1]]])
    conv_op = muGrid.ConvolutionOperator([0, 0], stencil)

    # Create field collection with different sub-point counts
    fc_nodal = muGrid.GlobalFieldCollection((nb_x_pts, nb_y_pts))
    fc_quad = muGrid.GlobalFieldCollection(
        (nb_x_pts, nb_y_pts), sub_pts={"quad": 2}
    )

    # Input field with 2 sub-points (from quad collection)
    nodal_field_wrong = fc_quad.real_field("quad-field", (1,), "quad")

    # Output field with 1 sub-point (from nodal collection)
    quad_field = fc_nodal.real_field("quad-output", (1,))

    # This should raise an error about sub-point count mismatch
    with pytest.raises(RuntimeError):
        conv_op.apply(nodal_field_wrong, quad_field)


def test_convolution_wrong_output_subpt_type_global():
    """Test that mismatched output sub-point count is caught.

    The ConvolutionOperator requires that input and output fields have
    the same number of sub-points per pixel. This test verifies that
    an output field with wrong sub-point count is rejected with a clear
    error message.

    Setup
    -----
    - Creates a 2D grid with a simple stencil
    - Creates an input field with 1 sub-point (nodal)
    - Creates an output field with 2 sub-points (quadrature)
    - Tries to apply the operator

    Expected behavior
    -----------------
    Should raise a RuntimeError with message about sub-point count
    mismatch.
    """
    nb_x_pts = 3
    nb_y_pts = 3

    # Create a simple stencil
    stencil = np.array([[[1, 0], [0, 1]]])
    conv_op = muGrid.ConvolutionOperator([0, 0], stencil)

    # Create field collections with different sub-point counts
    fc_nodal = muGrid.GlobalFieldCollection((nb_x_pts, nb_y_pts))
    fc_quad = muGrid.GlobalFieldCollection(
        (nb_x_pts, nb_y_pts), sub_pts={"quad": 2}
    )

    # Input field with 1 sub-point (from nodal collection)
    nodal_field = fc_nodal.real_field("nodal", (1,))

    # Output field with 2 sub-points (from quad collection) - wrong!
    quad_field_wrong = fc_quad.real_field("output-quad", (1,), "quad")

    # This should raise an error about sub-point count mismatch
    with pytest.raises(RuntimeError):
        conv_op.apply(nodal_field, quad_field_wrong)

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file    python_convolution_validation_tests.py

@author  Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date    06 Jun 2026

@brief   Validation/error-branch tests for GenericLinearOperator

Copyright © 2026 Lars Pastewka

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

------------------------------------------------------------------------------

These tests target the field-validation error branches in
``GenericLinearOperator::validate_fields`` (``operators/generic.cc``) that the
existing convolution tests do not reach. The validation runs the checks in a
fixed order — collection match, global-collection, spatial dimension, ghost
sufficiency (left then right), then component count — so each test below is
constructed so that the *intended* branch is the first to fail.

In particular, the pre-existing ``test_convolution_component_mismatch_global``
uses a stencil with offset ``[0, 0]`` on a ghost-free grid, which trips the
*ghost* check before the component check is ever reached. The component-mismatch
branch is therefore exercised here with a centred stencil on a grid that has
sufficient ghosts.
"""

import numpy as np
import pytest

import muGrid

# Centred 5-point Laplacian: offset [-1, -1], 3x3 stencil. Requires exactly one
# ghost layer on every side for both apply and transpose.
LAPLACE_2D = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])


def test_apply_spatial_dimension_mismatch():
    """A 2D operator applied to fields from a 3D collection must raise (the
    spatial-dimension check precedes the ghost and component checks)."""
    op = muGrid.GenericLinearOperator([-1, -1], LAPLACE_2D)  # 2D operator
    fc = muGrid.GlobalFieldCollection(
        [4, 4, 4], nb_ghosts_left=(1, 1, 1), nb_ghosts_right=(1, 1, 1)
    )  # 3D collection, amply ghosted
    nodal = fc.real_field("nodal")
    quad = fc.real_field("quad")
    with pytest.raises(RuntimeError):
        op.apply(nodal, quad)


def test_apply_insufficient_left_ghosts():
    """The centred Laplacian needs one ghost on each side; a ghost-free grid
    must fail the left-ghost check (offset [-1,-1] => min left = 1)."""
    op = muGrid.GenericLinearOperator([-1, -1], LAPLACE_2D)
    fc = muGrid.GlobalFieldCollection([4, 4])  # zero ghosts
    nodal = fc.real_field("nodal")
    quad = fc.real_field("quad")
    with pytest.raises(RuntimeError):
        op.apply(nodal, quad)


def test_apply_component_mismatch_with_sufficient_ghosts():
    """With enough ghosts the validator reaches the component-count check:
    quad components must equal nb_operators (1) * nodal components (2) = 2.
    Supplying 3 must raise."""
    op = muGrid.GenericLinearOperator([-1, -1], LAPLACE_2D)  # 1 output component
    fc = muGrid.GlobalFieldCollection(
        [4, 4], nb_ghosts_left=(1, 1), nb_ghosts_right=(1, 1)
    )
    nodal = fc.real_field("nodal", (2,))
    quad = fc.real_field("quad", (3,))  # should be 2
    with pytest.raises(RuntimeError):
        op.apply(nodal, quad)


def test_transpose_insufficient_ghosts():
    """The transpose swaps the left/right ghost requirements; on a ghost-free
    grid the centred Laplacian transpose must also raise."""
    op = muGrid.GenericLinearOperator([-1, -1], LAPLACE_2D)
    fc = muGrid.GlobalFieldCollection([4, 4])  # zero ghosts
    nodal = fc.real_field("nodal")
    quad = fc.real_field("quad")
    with pytest.raises(RuntimeError):
        op.transpose(quad, nodal)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

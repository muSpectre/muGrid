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
        # Create a pixel-wise operator
        pixel_map = np.array([[1e1, .0, .0, .0],
                              [-1e1, .0, .0, .0],
                              [.0, 1e2, .0, .0],
                              [.0, -1e2, .0, .0],
                              [.0, .0, 1e3, .0],
                              [.0, .0, -1e3, .0],
                              [.0, .0, .0, 1e4],
                              [.0, .0, .0, -1e4]], dtype=float, order='F')
        n_nodal_pixel = 1
        n_operators = 2
        n_quad = 4
        assert n_operators * n_quad == pixel_map.shape[0]
        d_op = muGrid.ConvolutionOperatorDefault(pixel_map, (2, 2), n_component, n_nodal_pixel, n_quad, n_operators)

        # Create a 2D grid
        nx = 3      # number of pixels in x axis
        ny = 3      # number of pixels in y axis
        fc = muGrid.GlobalFieldCollection((nx, ny), sub_pts={'quad': n_quad})

        # A nodal field with some sequence as values
        nodal = fc.real_field('nodal-value', n_component)
        values = 1 + np.arange(n_component * nx * ny)
        nodal.p = values.reshape(n_component, nx, ny, order='F')

        # Create a quadrature field to store the result
        quad = fc.real_field('quad-grad', (n_component, n_operators), 'quad')

        # Apply the graident operator
        d_op.apply(nodal, quad)

        # Compute the reference value
        # Create a pack of nodal values, each with a different offset
        offset_00 = nodal.p
        offset_10 = np.roll(nodal.p, (-1,0), axis=(-2,-1))
        offset_01 = np.roll(nodal.p, (0,-1), axis=(-2,-1))
        offset_11 = np.roll(nodal.p, (-1,-1), axis=(-2,-1))
        # NOTE: The oder of offset must keep the same as impelemented in lib
        offset_nodes = np.stack((offset_00, offset_10, offset_01, offset_11), axis=0)

        grad_ref_p = np.einsum("do,ocxy->cdxy", pixel_map, offset_nodes)
        # explicitly split (n_operators * n_quad) into two axes
        grad_ref_s = grad_ref_p.reshape(n_component, n_operators, n_quad, nx, ny, order='F')

        # Print something
        if verbose:
            print(f"\n"
                  f"NumPy array={values.ravel(order='F')}\n"
                  f"n_component={n_component}, n_operators={n_operators}, n_quad={n_quad}\n"
                  f"Operator=\n{d_op.pixel_operator}\n"
                  f"\n"
                  f"Nodal field, shape={nodal.shape}\n"
                  f"value@(0,0)=\n{nodal.p[...,0,0]}\n"
                  f"value@(1,0)=\n{nodal.p[...,1,0]}\n"
                  f"value@(0,1)=\n{nodal.p[...,0,1]}\n"
                  f"value@(1,1)=\n{nodal.p[...,1,1]}\n"
                  f"\n"
                  f"Quadrature field, shape={quad.s.shape}\n"
                  f"value@(0,0)=\n{quad.p[...,0,0]}\n"
                  f"with the first quadrature=\n{quad.s[...,0,0,0]}\n")

        # Check
        np.testing.assert_allclose(quad.s, grad_ref_s)
        np.testing.assert_allclose(quad.p, grad_ref_p)


    test_apply_2D_field_scalar = functools.partialmethod(template_test_apply_in_2D_field, n_component = 1)
    test_apply_2D_field_3D_vector = functools.partialmethod(template_test_apply_in_2D_field, n_component = 3)


if __name__ == '__main__':
    unittest.main()

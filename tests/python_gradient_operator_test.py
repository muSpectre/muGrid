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


class GradientOperatorCheck(unittest.TestCase):

    def test_constructor(self):
        n_dim = 2
        n_quad_of_elem = 1
        n_elem_of_pixel = 2
        n_node_of_elem = 3
        n_node_of_pixel = 4

        qx = 0.1  # 1 / dx
        qy = 0.2  # 1 / dy
        shape_fn_grad_elem = np.array([[-qx, qx, .0],
                                       [-qy, .0, qy]], dtype=float)
        shape_fn_grad = [[shape_fn_grad_elem], [shape_fn_grad_elem]]

        elem_nodal_coord = [
            (np.array([0,1,2], dtype=int), np.array([[0,0],
                                                     [1,0],
                                                     [0,1]], dtype=int)),
            (np.array([3,2,1], dtype=int), np.array([[1,1],
                                                     [0,1],
                                                     [1,0]], dtype=int)),
        ]
        d_op = muGrid.GradientOperatorDefault(n_dim, n_quad_of_elem, n_elem_of_pixel, 
                                              n_node_of_elem, n_node_of_pixel,
                                              shape_fn_grad, elem_nodal_coord)

        pixel_gradient = np.array([[-qx, qx, .0, .0],
                                   [-qy, .0, .0, qy],
                                   [ .0, .0, qx,-qx],
                                   [ .0,-qy, qy, .0]], dtype=float)
        self.assertEqual(d_op.pixel_gradient, pixel_gradient)


if __name__ == '__main__':
    unittest.main()

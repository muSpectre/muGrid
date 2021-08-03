#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   Stencils2D.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   15 May 2020

@brief  Library of some common stencils for 2D problems

Copyright © 2018 Till Junge

µFFT is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µFFT is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with µFFT; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.

Additional permission under GNU GPL version 3 section 7

If you modify this Program, or any covered work, by linking or combining it
with proprietary FFT implementations or numerical libraries, containing parts
covered by the terms of those libraries' licenses, the licensors of this
Program grant you additional permission to convey the resulting work.
"""

import muFFT

# First order upwind differences
upwind_x = muFFT.DiscreteDerivative([0, 0], [[-1], [1]])
upwind_y = muFFT.DiscreteDerivative([0, 0], [[-1, 1]])
upwind = (upwind_x, upwind_y)

# d-stencil label the corners used for the derivative
d_10_00 = upwind_x
d_01_00 = upwind_y
d_11_01 = muFFT.DiscreteDerivative([0, 0], [[0, -1],
                                            [0, 1]])
d_11_10 = muFFT.DiscreteDerivative([0, 0], [[0, 0],
                                            [-1, 1]])

# Linear finite elements (each pixel is subdivided into two triangles)
# in the manner first triangle (T1), second triangle (T2)
# see: R. J. Leute, M. Ladecký, A. Falsafi, I. Jödicke, I. Pultarová, J. Zeman,
#      T. Junge, L. Pastewka, Elimination of ringing artifacts by
#      finite-element projection in FFT-based homogenization,
#      https://arxiv.org/abs/2105.03297 (2021)
#
#        -------------
#        |\          |
#        |  \        |
# y^     |    \  T2  |
#  |     |      \    |
#  --->  |  T1    \  |
#     x  |          \|
#        -------------
linear_finite_elements = (
    # First triangle
    d_10_00,  # x-derivative
    d_01_00,  # y-derivative
    # Second triangle
    d_11_01,  # x-derivative
    d_11_10   # y-derivative
)

# First order upwind differences, averaged over neighboring pixels
averaged_upwind_x = muFFT.DiscreteDerivative([0, 0], [[-0.5, -0.5],
                                                      [0.5, 0.5]])
averaged_upwind_y = muFFT.DiscreteDerivative([0, 0], [[-0.5, 0.5],
                                                      [-0.5, 0.5]])
averaged_upwind = (averaged_upwind_x, averaged_upwind_y)

# Second order central differences
central_x = muFFT.DiscreteDerivative([-1, 0], [[-0.5], [0], [0.5]])
central_y = muFFT.DiscreteDerivative([0, -1], [[-0.5, 0, 0.5]])
central = (central_x, central_y)

# Fourth order central differences of the second derivative
central_2nd_x = muFFT.DiscreteDerivative([-1, 0], [[1], [-2], [1]])
central_2nd_y = muFFT.DiscreteDerivative([0, -1], [[1, -2, 1]])
central_2nd = (central_2nd_x, central_2nd_y)

# hexagonal grid, two equilateral triangles T1 and T2
# see: I. Jödicke, R. J. Leute, T. Junge, L. Pastewka, Efficient topology
#      optimization using compatibility projection in micromechanical
#      homogenization, https://arxiv.org/abs/2107.04123 (2021)
#             __________
#            /\        /
# y^        /  \  T2  /
#  |       /    \    /
#  --->   /  T1  \  /
#     x  /________\/
#
hexagonal_T1_dx = muFFT.DiscreteDerivative([0, 0], [[-1], [1]])
hexagonal_T1_dy = muFFT.DiscreteDerivative([0, 0], [[-1/2, 1], [-1/2, 0]])
hexagonal_T2_dx = muFFT.DiscreteDerivative([0, 0], [[0, -1], [0, 1]])
hexagonal_T2_dy = muFFT.DiscreteDerivative([0, 0], [[0, 1/2], [-1, 1/2]])
hexagonal_linear_finite_elements = (hexagonal_T1_dx,
                                    hexagonal_T1_dy,
                                    hexagonal_T2_dx,
                                    hexagonal_T2_dy)

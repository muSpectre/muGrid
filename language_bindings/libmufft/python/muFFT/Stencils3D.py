#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   Stencils3D.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   15 May 2020

@brief  Library of some common stencils for 3D problems

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
import numpy as np

upwind_x = muFFT.DiscreteDerivative([0, 0, 0], [[[-1, 1]]]) \
    .rollaxes(-1).rollaxes(-1)
upwind_y = muFFT.DiscreteDerivative([0, 0, 0], [[[-1, 1]]]).rollaxes(-1)
upwind_z = muFFT.DiscreteDerivative([0, 0, 0], [[[-1, 1]]])
upwind = (upwind_x, upwind_y, upwind_z)

averaged_upwind_x = muFFT.DiscreteDerivative([0, 0, 0],
                                             [[[-0.25, -0.25], [-0.25, -0.25]],
                                              [[0.25,  0.25], [0.25,  0.25]]])
averaged_upwind_y = muFFT.DiscreteDerivative([0, 0, 0],
                                             [[[-0.25, -0.25], [0.25, 0.25]],
                                              [[-0.25, -0.25], [0.25, 0.25]]])
averaged_upwind_z = muFFT.DiscreteDerivative([0, 0, 0],
                                             [[[-0.25, 0.25], [-0.25, 0.25]],
                                              [[-0.25, 0.25], [-0.25, 0.25]]])
averaged_upwind = (averaged_upwind_x, averaged_upwind_y, averaged_upwind_z)

central_x = muFFT.DiscreteDerivative([-1, 0, 0], [[[-0.5]], [[0]], [[0.5]]])
central_y = muFFT.DiscreteDerivative([0, -1, 0], [[[-0.5], [0], [0.5]]])
central_z = muFFT.DiscreteDerivative([0, 0, -1], [[[-0.5, 0, 0.5]]])
central = (central_x, central_y, central_z)

# d-stencil label the corners used for the derivative
# x-derivatives
d_100_000 = muFFT.DiscreteDerivative([0, 0, 0], [[[-1]], [[1]]])
d_110_010 = muFFT.DiscreteDerivative([0, 1, 0], [[[-1]], [[1]]])
d_111_011 = muFFT.DiscreteDerivative([0, 1, 1], [[[-1]], [[1]]])
d_101_001 = muFFT.DiscreteDerivative([0, 0, 1], [[[-1]], [[1]]])

# y-derivatives
d_010_000 = muFFT.DiscreteDerivative([0, 0, 0], [[[-1], [1]]])
d_110_100 = muFFT.DiscreteDerivative([1, 0, 0], [[[-1], [1]]])
d_111_101 = muFFT.DiscreteDerivative([1, 0, 1], [[[-1], [1]]])
d_011_001 = muFFT.DiscreteDerivative([0, 0, 1], [[[-1], [1]]])

# z-derivatives
d_001_000 = muFFT.DiscreteDerivative([0, 0, 0], [[[-1, 1]]])
d_101_100 = muFFT.DiscreteDerivative([1, 0, 0], [[[-1, 1]]])
d_111_110 = muFFT.DiscreteDerivative([1, 1, 0], [[[-1, 1]]])
d_011_010 = muFFT.DiscreteDerivative([0, 1, 0], [[[-1, 1]]])

# Linear finite elements in 3D (each voxel is subdivided into six tetrahedra)
linear_finite_elements = (
    # First tetrahedron
    d_100_000,  # x-derivative
    d_110_100,  # y-derivative
    d_111_110,  # z-derivative
    # Second tetrahedron
    d_100_000,  # x-derivative
    d_111_101,  # y-derivative
    d_101_100,  # z-derivative
    # Third tetrahedron
    d_110_010,  # x-derivative
    d_010_000,  # y-derivative
    d_111_110,  # z-derivative
    # Fourth tetrahedron
    d_111_011,  # x-derivative
    d_010_000,  # y-derivative
    d_011_010,  # z-derivative
    # Fifth tetrahedron
    d_101_001,  # x-derivative
    d_111_101,  # y-derivative
    d_001_000,  # z-derivative
    # Sixth tetrahedron
    d_111_011,  # x-derivative
    d_011_001,  # y-derivative
    d_001_000   # z-derivative
)

# Linear finite elements in 3D (each voxel is subdivided into five tetrahedra)
# On the zeros thetrahedron all four corner points are used for the
# derivatives. On the four other thetrahedra the derivatives are given by
# first order forward differences along the tetrahedron edge aligned with the
# respective axis.
# ∂f/∂x = 1/2 * (f₁₁₁ + f₁₀₀ - f₀₁₀ - f₀₀₁)
_dx_helper = np.zeros([2, 2, 2])
_dx_helper[1, 1, 1] = 1/2
_dx_helper[1, 0, 0] = 1/2
_dx_helper[0, 1, 0] = -1/2
_dx_helper[0, 0, 1] = -1/2
# ∂f/∂y = 1/2 * (f₁₁₁ - f₁₀₀ + f₀₁₀ - f₀₀₁)
_dy_helper = np.zeros([2, 2, 2])
_dy_helper[1, 1, 1] = 1/2
_dy_helper[1, 0, 0] = -1/2
_dy_helper[0, 1, 0] = 1/2
_dy_helper[0, 0, 1] = -1/2
# ∂f/∂z = 1/2 * (f₁₁₁ - f₁₀₀ - f₀₁₀ + f₀₀₁)
_dz_helper = np.zeros([2, 2, 2])
_dz_helper[1, 1, 1] = 1/2
_dz_helper[1, 0, 0] = -1/2
_dz_helper[0, 1, 0] = -1/2
_dz_helper[0, 0, 1] = 1/2

linear_finite_elements_5 = (
    # Central tetrahedron
    muFFT.DiscreteDerivative([0, 0, 0], _dx_helper),  # x-derivative
    muFFT.DiscreteDerivative([0, 0, 0], _dy_helper),  # y-derivative
    muFFT.DiscreteDerivative([0, 0, 0], _dz_helper),  # z-derivative
    # First corner tetrahedron
    d_100_000,  # x-derivative
    d_010_000,  # y-derivative
    d_001_000,  # z-derivative
    # Second corner tetrahedron
    d_110_010,  # x-derivative
    d_110_100,  # y-derivative
    d_111_110,  # z-derivative
    # Third corner tetrahedron
    d_101_001,  # x-derivative
    d_111_101,  # y-derivative
    d_101_100,  # z-derivative
    # Fourth corner tetrahedron
    d_111_011,  # x-derivative
    d_011_001,  # y-derivative
    d_011_010,  # z-derivative
)

# Linear finite elements in 3D (each voxel is subdivided into six tetrahedra)
# by tilting the coordinate system twice the tetrahedra are almost regular (two
# of them are indeed regular (T0 and T5) and the other four are more regular
# than before tilting the coordinate system)
# T0
# ∂f/∂x = f₁₀₀ - f₀₀₀ + 1/2 (f₁₀₀ - f₁₁₀)
_dx_helper_T0 = np.zeros([2, 2, 2])
_dx_helper_T0[1, 0, 0] += +1
_dx_helper_T0[0, 0, 0] += -1
_dx_helper_T0[1, 0, 0] += +1/2
_dx_helper_T0[1, 1, 0] += -1/2
# ∂f/∂y = f₁₁₀ - f₁₀₀ + 2/3 (f₁₁₀ - f₁₁₁)
_dy_helper_T0 = np.zeros([2, 2, 2])
_dy_helper_T0[1, 1, 0] += +1
_dy_helper_T0[1, 0, 0] += -1
_dy_helper_T0[1, 1, 0] += +2/3
_dy_helper_T0[1, 1, 1] += -2/3
# ∂f/∂z = f₁₁₁ - f₁₁₀
_dz_helper_T0 = np.zeros([2, 2, 2])
_dz_helper_T0[1, 1, 1] = +1
_dz_helper_T0[1, 1, 0] = -1

# T1
# ∂f/∂x = f₁₀₀ - f₀₀₀ + 1/2 (f₁₀₁ - f₁₁₁)
_dx_helper_T1 = np.zeros([2, 2, 2])
_dx_helper_T1[1, 0, 0] += +1
_dx_helper_T1[0, 0, 0] += -1
_dx_helper_T1[1, 0, 1] += +1/2
_dx_helper_T1[1, 1, 1] += -1/2
# ∂f/∂y = f₁₁₁ - f₁₀₁ + 2/3 (f₁₀₀ - f₁₀₁)
_dy_helper_T1 = np.zeros([2, 2, 2])
_dy_helper_T1[1, 1, 1] += +1
_dy_helper_T1[1, 0, 1] += -1
_dy_helper_T1[1, 0, 0] += +2/3
_dy_helper_T1[1, 0, 1] += -2/3
# ∂f/∂z = f₁₀₁ - f₁₀₀
_dz_helper_T1 = np.zeros([2, 2, 2])
_dz_helper_T1[1, 0, 1] = +1
_dz_helper_T1[1, 0, 0] = -1

# T2
# ∂f/∂x = f₁₁₀ - f₀₁₀ + 1/2 (f₀₀₀ - f₀₁₀)
_dx_helper_T2 = np.zeros([2, 2, 2])
_dx_helper_T2[1, 1, 0] += +1
_dx_helper_T2[0, 1, 0] += -1
_dx_helper_T2[0, 0, 0] += +1/2
_dx_helper_T2[0, 1, 0] += -1/2
# ∂f/∂y = f₀₁₀ - f₀₀₀ + 2/3 (f₁₁₀ - f₁₁₁)
_dy_helper_T2 = np.zeros([2, 2, 2])
_dy_helper_T2[0, 1, 0] += +1
_dy_helper_T2[0, 0, 0] += -1
_dy_helper_T2[1, 1, 0] += +2/3
_dy_helper_T2[1, 1, 1] += -2/3
# ∂f/∂z = f₁₁₁ - f₁₁₀
_dz_helper_T2 = np.zeros([2, 2, 2])
_dz_helper_T2[1, 1, 1] = +1
_dz_helper_T2[1, 1, 0] = -1

# T3
# ∂f/∂x = f₁₁₁ - f₀₁₁ + 1/2 (f₀₀₀ - f₀₁₀)
_dx_helper_T3 = np.zeros([2, 2, 2])
_dx_helper_T3[1, 1, 1] += +1
_dx_helper_T3[0, 1, 1] += -1
_dx_helper_T3[0, 0, 0] += +1/2
_dx_helper_T3[0, 1, 0] += -1/2
# ∂f/∂y = f₀₁₀ - f₀₀₀ + 2/3 (f₀₁₀ - f₀₁₁)
_dy_helper_T3 = np.zeros([2, 2, 2])
_dy_helper_T3[0, 1, 0] += +1
_dy_helper_T3[0, 0, 0] += -1
_dy_helper_T3[0, 1, 0] += +2/3
_dy_helper_T3[0, 1, 1] += -2/3
# ∂f/∂z = f₀₁₁ - f₀₁₀
_dz_helper_T3 = np.zeros([2, 2, 2])
_dz_helper_T3[0, 1, 1] = +1
_dz_helper_T3[0, 1, 0] = -1

# T4
# ∂f/∂x = f₁₀₁ - f₀₀₁ + 1/2 (f₁₀₁ - f₁₁₁)
_dx_helper_T4 = np.zeros([2, 2, 2])
_dx_helper_T4[1, 0, 1] += +1
_dx_helper_T4[0, 0, 1] += -1
_dx_helper_T4[1, 0, 1] += +1/2
_dx_helper_T4[1, 1, 1] += -1/2
# ∂f/∂y = f₁₁₁ - f₁₀₁ + 2/3 (f₀₀₀ - f₀₀₁)
_dy_helper_T4 = np.zeros([2, 2, 2])
_dy_helper_T4[1, 1, 1] += +1
_dy_helper_T4[1, 0, 1] += -1
_dy_helper_T4[0, 0, 0] += +2/3
_dy_helper_T4[0, 0, 1] += -2/3
# ∂f/∂z = f₀₀₁ - f₀₀₀
_dz_helper_T4 = np.zeros([2, 2, 2])
_dz_helper_T4[0, 0, 1] = +1
_dz_helper_T4[0, 0, 0] = -1

# T5
# ∂f/∂x = f₁₁₁ - f₀₁₁ + 1/2 (f₀₀₁ - f₀₁₁)
_dx_helper_T5 = np.zeros([2, 2, 2])
_dx_helper_T5[1, 1, 1] += +1
_dx_helper_T5[0, 1, 1] += -1
_dx_helper_T5[0, 0, 1] += +1/2
_dx_helper_T5[0, 1, 1] += -1/2
# ∂f/∂y = f₀₁₁ - f₀₀₁ + 2/3 (f₀₀₀ - f₀₀₁)
_dy_helper_T5 = np.zeros([2, 2, 2])
_dy_helper_T5[0, 1, 1] += +1
_dy_helper_T5[0, 0, 1] += -1
_dy_helper_T5[0, 0, 0] += +2/3
_dy_helper_T5[0, 0, 1] += -2/3
# ∂f/∂z = f₀₀₁ - f₀₀₀
_dz_helper_T5 = np.zeros([2, 2, 2])
_dz_helper_T5[0, 0, 1] = +1
_dz_helper_T5[0, 0, 0] = -1

linear_finite_elements_6_regular = (
    # T0
    muFFT.DiscreteDerivative([0, 0, 0], _dx_helper_T0),  # x-derivative
    muFFT.DiscreteDerivative([0, 0, 0], _dy_helper_T0),  # y-derivative
    muFFT.DiscreteDerivative([0, 0, 0], _dz_helper_T0),  # z-derivative
    # T1
    muFFT.DiscreteDerivative([0, 0, 0], _dx_helper_T1),  # x-derivative
    muFFT.DiscreteDerivative([0, 0, 0], _dy_helper_T1),  # y-derivative
    muFFT.DiscreteDerivative([0, 0, 0], _dz_helper_T1),  # z-derivative
    # T2
    muFFT.DiscreteDerivative([0, 0, 0], _dx_helper_T2),  # x-derivative
    muFFT.DiscreteDerivative([0, 0, 0], _dy_helper_T2),  # y-derivative
    muFFT.DiscreteDerivative([0, 0, 0], _dz_helper_T2),  # z-derivative
    # T3
    muFFT.DiscreteDerivative([0, 0, 0], _dx_helper_T3),  # x-derivative
    muFFT.DiscreteDerivative([0, 0, 0], _dy_helper_T3),  # y-derivative
    muFFT.DiscreteDerivative([0, 0, 0], _dz_helper_T3),  # z-derivative
    # T4
    muFFT.DiscreteDerivative([0, 0, 0], _dx_helper_T4),  # x-derivative
    muFFT.DiscreteDerivative([0, 0, 0], _dy_helper_T4),  # y-derivative
    muFFT.DiscreteDerivative([0, 0, 0], _dz_helper_T4),  # z-derivative
    # T5
    muFFT.DiscreteDerivative([0, 0, 0], _dx_helper_T5),  # x-derivative
    muFFT.DiscreteDerivative([0, 0, 0], _dy_helper_T5),  # y-derivative
    muFFT.DiscreteDerivative([0, 0, 0], _dz_helper_T5),  # z-derivative
)

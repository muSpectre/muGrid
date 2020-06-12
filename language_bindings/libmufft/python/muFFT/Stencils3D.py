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

upwind_x = muFFT.DiscreteDerivative([0, 0, 0], [[[-1, 1]]]) \
    .rollaxes(-1).rollaxes(-1)
upwind_y = muFFT.DiscreteDerivative([0, 0, 0], [[[-1, 1]]]).rollaxes(-1)
upwind_z = muFFT.DiscreteDerivative([0, 0, 0], [[[-1, 1]]])
upwind = (upwind_x, upwind_y, upwind_z)

averaged_upwind_x = muFFT.DiscreteDerivative([0, 0, 0],
                                             [[[-0.25, -0.25], [-0.25, -0.25]],
                                              [[ 0.25,  0.25], [ 0.25,  0.25]]])
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
d_100_000 = muFFT.DiscreteDerivative([0, 0, 0], [[[-1]], [[ 1]]])
d_110_010 = muFFT.DiscreteDerivative([0, 1, 0], [[[-1]], [[ 1]]])
d_111_011 = muFFT.DiscreteDerivative([0, 1, 1], [[[-1]], [[ 1]]])
d_101_001 = muFFT.DiscreteDerivative([0, 0, 1], [[[-1]], [[ 1]]])

# y-derivatives
d_010_000 = muFFT.DiscreteDerivative([0, 0, 0], [[[-1], [ 1]]])
d_110_100 = muFFT.DiscreteDerivative([1, 0, 0], [[[-1], [ 1]]])
d_111_101 = muFFT.DiscreteDerivative([1, 0, 1], [[[-1], [ 1]]])
d_011_001 = muFFT.DiscreteDerivative([0, 0, 1], [[[-1], [ 1]]])

# z-derivatives
d_001_000 = muFFT.DiscreteDerivative([0, 0, 0], [[[-1, 1]]])
d_101_100 = muFFT.DiscreteDerivative([1, 0, 0], [[[-1, 1]]])
d_111_110 = muFFT.DiscreteDerivative([1, 1, 0], [[[-1, 1]]])
d_011_010 = muFFT.DiscreteDerivative([0, 1, 0], [[[-1, 1]]])

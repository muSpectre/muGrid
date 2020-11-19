#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   Stencils1D.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   15 Nov 2020

@brief  Library of some common stencils for 1D problems

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
upwind = muFFT.DiscreteDerivative([0], [-1, 1])

# Second order central differences
central = muFFT.DiscreteDerivative([-1], [-0.5, 0, 0.5])

# Sixth order central differences
central6 = muFFT.DiscreteDerivative([-3],
                                    [-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])

# Fourth order central differences of the second derivative
central_2nd = muFFT.DiscreteDerivative([-1], [1, -2, 1])

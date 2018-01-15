#!/usr/bin/env python3
"""
file   small_case.py

@author Till Junge <till.junge@epfl.ch>

@date   12 Jan 2018

@brief  small case for debugging

@section LICENCE

Copyright © 2018 Till Junge

µSpectre is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µSpectre is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Emacs; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.getcwd(), "language_bindings/python"))
import pyMuSpectre as µ


resolution = [5, 5]

lengths = [5., 5.]
formulation = µ.Formulation.finite_strain

rve = µ.SystemFactory(resolution,
                      lengths,
                      formulation)
hard = µ.material.MaterialHooke2d.make(
    rve, "hard", 210e9, .33)
soft = µ.material.MaterialHooke2d.make(
    rve, "soft",  70e9, .33)


for i, pixel in enumerate(rve):
    if i < 3:
        hard.add_pixel(pixel)
    else:
        soft.add_pixel(pixel)

    print("{}, {}".format(i, tuple(pixel)))

rve.initialise()

tol = 1e-6

Del0 = np.array([[0, .1],
                 [0,  0]])
maxiter = 31
verbose = 2

r = µ.solvers.de_geus(rve, Del0, tol, tol, maxiter, verbose)
print(r.grad.T)
print(r.stress.T)
print(r)

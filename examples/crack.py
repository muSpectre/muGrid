#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   crack.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   05 Jun 2019

@brief  

Copyright © 2019 Till Junge

µSpectre is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µSpectre is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with µSpectre; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.

Additional permission under GNU GPL version 3 section 7

If you modify this Program, or any covered work, by linking or combining it
with proprietary FFT implementations or numerical libraries, containing parts
covered by the terms of those libraries' licenses, the licensors of this
Program grant you additional permission to convey the resulting work.
"""

import sys
import numpy as np
from python_example_imports import muSpectre as msp
print(msp.__file__)

###

nb_domain_grid_pts = [23, 23]
nx, ny = nb_domain_grid_pts
center = np.array([r//2 for r in nb_domain_grid_pts])
crack_length = 11

## Domain dimensions
#domain_lengths = [7., 5.]
domain_lengths = [float(r) for r in nb_domain_grid_pts]

## define the convergence tolerance for the Newton-Raphson increment
newton_tol = 1e-8
equil_tol = newton_tol
## tolerance for the solver of the linear cell
cg_tol = 1e-14

## Macroscopic strain
applied_strain = np.array([[.1, 0.],
                           [0., .1]])
#applied_strain = .5*(applied_strain + applied_strain.T)


maxiter = 1000 ## for linear cell solver

## Verbosity levels:
## 0: silent,
## 1: info about Newton-Raphson loop,
verbose = 1

###

dim = len(nb_domain_grid_pts)
fourier_gradient = [msp.FourierDerivative(dim, i) for i in range(dim)]
# This is a symmetric stencil
dy = msp.DiscreteDerivative([0, 0], [[-0.5, 0.5], [-0.5, 0.5]])
# This is the upwind differences stencil
#dy = msp.DiscreteDerivative([0, 0], [[-1, 1]])
# For the following stencils see:
# https://en.wikipedia.org/wiki/Finite_difference_coefficient
# This is the second order central differences stencil
#dy = msp.DiscreteDerivative([0, -1], [[-0.5, 0, 0.5]])
# This is the fourth order central differences stencil
#dy = msp.DiscreteDerivative([0, -2], [[1/12, -2/3, 0, 2/3, -1/12]])
# This is the sixth order central differences stencil
#dy = msp.DiscreteDerivative([0, -3], [[-1/60, 3/20, -3/4, 0,
#                                       3/4, -3/20, 1/60]])
# This is the eigth order central differences stencil
#dy = msp.DiscreteDerivative([0, -4], [[1/280, -4/105, 1/5, -4/5, 0,
#                                       4/5, -1/5, 4/105, -1/280]])
# Rotate stencil for the derivative in the other direction
dx = dy.rollaxes()
discrete_gradient = [dx, dy]

###

stress = {}
grad = {}
phase = -np.ones(nb_domain_grid_pts, dtype=int)
for i, derivative in enumerate([fourier_gradient, discrete_gradient]):
    rve = msp.Cell(nb_domain_grid_pts, domain_lengths,
                   msp.Formulation.finite_strain, derivative)
    hard = msp.material.MaterialLinearElastic1_2d.make(rve, "hard", 1., .33)
    vacuum = msp.material.MaterialLinearElastic1_2d.make(rve, "vacuum", 0., 0.)
    for j, pixel in rve.pixels.enumerate():
        if pixel[1] == center[1] and \
            abs(pixel[0] - center[0]) < crack_length//2:
            vacuum.add_pixel(j)
            phase[pixel[0], pixel[1]] = 2
        else:
            hard.add_pixel(j)
            phase[pixel[0], pixel[1]] = 0
    solver = msp.solvers.KrylovSolverCG(rve, cg_tol, maxiter,
                                        verbose=msp.Verbosity.Silent)
    result = msp.solvers.newton_cg(rve, applied_strain, solver,
                                   newton_tol=newton_tol, equil_tol=equil_tol,
                                   verbose=msp.Verbosity.Silent)
    stress[i] = result.stress.T.reshape(*nb_domain_grid_pts, 2, 2)
    grad[i] = result.grad.T.reshape(*nb_domain_grid_pts, 2, 2)


# prevent visual output during ctest
if len(sys.argv[:]) == 2:
    if sys.argv[1] != 1:
        pass
else:
    import matplotlib.pyplot as plt
    fac = 1
    plt.figure()
    plt.subplot(321, aspect=1)
    plt.title('Fourier')
    plt.pcolormesh(grad[0][:, :, 1, 0].T)
    plt.colorbar()
    plt.subplot(322, aspect=1)
    plt.title('discrete')
    plt.pcolormesh(fac*grad[1][:, :, 1, 0].T)
    plt.colorbar()
    plt.subplot(323, aspect=1)
    plt.title('Fourier-discrete')
    plt.pcolormesh(grad[0][:, :, 1, 0].T - fac*grad[1][:, :, 1, 0].T)
    plt.colorbar()
    plt.subplot(324, aspect=1)
    plt.pcolormesh(phase.T)
    plt.colorbar()
    plt.subplot(325)
    plt.title('Fourier')
    plt.plot(grad[0][4, :, 1, 0], 'x-')
    plt.plot(grad[0][5, :, 1, 0], 'x-')
    plt.plot(grad[0][6, :, 1, 0], 'x-')
    plt.subplot(326)
    plt.title('discrete')
    plt.plot(fac*grad[1][4, :, 1, 0], 'x-')
    plt.plot(fac*grad[1][5, :, 1, 0], 'x-')
    plt.plot(fac*grad[1][6, :, 1, 0], 'x-')
    plt.tight_layout()
    plt.show()

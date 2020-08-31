#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   crack.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   05 Jun 2019

@brief  crack example

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

import numpy as np
from mpi4py import MPI
import sys

try:
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    matplotlib_found = True
except ImportError:
    matplotlib_found = False

from python_example_imports import muSpectre as msp
from muFFT import Stencils2D

print(msp.__file__)

###

def make_triangles(displ):
    displ_x, displ_y = displ
    nx, ny = displ_x.shape

    def i(x, y):
        return x + nx*y

    x, y = np.mgrid[:nx-1, :ny-1]
    upper_triangles = np.stack((i(x, y), i(x, y+1), i(x+1, y)), axis=2)
    lower_triangles = np.stack((i(x+1, y), i(x, y+1), i(x+1, y+1)), axis=2)

    triangles = np.array([upper_triangles, lower_triangles]).reshape(-1, 3)

    return Triangulation(
        displ_x.reshape(-1, order='f'),
        displ_y.reshape(-1, order='f'),
        triangles)

###

nb_domain_grid_pts = [27, 23]
nx, ny = nb_domain_grid_pts
center = np.array([r//2 for r in nb_domain_grid_pts])
crack_length = nb_domain_grid_pts[0] // 2

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
verbose = msp.Verbosity.Full

###

dim = len(nb_domain_grid_pts)
fourier_gradient = [msp.FourierDerivative(dim, i) for i in range(dim)]
discrete_gradient = Stencils2D.averaged_upwind
discrete_gradient2 = [Stencils2D.d_10_00, Stencils2D.d_01_00,
                      Stencils2D.d_11_01, Stencils2D.d_11_10]

gradients = [fourier_gradient, discrete_gradient, discrete_gradient2]

###

stress = {}
grad = {}
phase = -np.ones(nb_domain_grid_pts, dtype=int)
for i, gradient in enumerate(gradients):
    rve = msp.Cell(nb_domain_grid_pts, domain_lengths,
                   msp.Formulation.finite_strain, gradient,
                   fft='mpi', communicator=MPI.COMM_WORLD)
    hard = msp.material.MaterialLinearElastic1_2d.make(
        rve, "hard", 1., .33)
    vacuum = msp.material.MaterialLinearElastic1_2d.make(
        rve, "vacuum", 0., 0.33)
    for pixel_index, pixel in enumerate(rve.pixels):
        if pixel[1] == center[1] and \
                abs(pixel[0] - center[0]) < crack_length//2:
            vacuum.add_pixel(pixel_index)
            phase[pixel[0], pixel[1]] = 2
        else:
            hard.add_pixel(pixel_index)
            phase[pixel[0], pixel[1]] = 0
    solver = msp.solvers.KrylovSolverCG(
        rve, cg_tol, maxiter, verbose=msp.Verbosity.Silent)
    result = msp.solvers.newton_cg(
        rve, applied_strain, solver, newton_tol=newton_tol,
        equil_tol=equil_tol, verbose=verbose)
    stress[i] = result.stress.reshape(
        (dim, dim, len(gradient)//dim, *rve.nb_subdomain_grid_pts), order='f')
    grad[i] = result.grad.reshape(
        (dim, dim, len(gradient)//dim, *rve.nb_subdomain_grid_pts), order='f')


if matplotlib_found and MPI.COMM_WORLD.Get_size() == 1:
    plt.figure()

    phase = np.stack((phase, phase), axis=0)

    names = ['Fourier', 'Discrete (1-quad)', 'Discrete (2-quad)']
    for i, gradient in enumerate(gradients):
        displ, r = msp.gradient_integration.compute_placement(
            grad[i], domain_lengths, nb_domain_grid_pts, gradient,
            formulation=msp.Formulation.finite_strain)
        tri = make_triangles(displ)

        g = stress[i]
        if i < 2:
            g = np.stack((g, g), axis=2).reshape(
                dim, dim, 2, *nb_domain_grid_pts)

        plt.subplot(3, len(gradients), 1 + i, aspect=1)
        plt.title(names[i])
        plt.tripcolor(tri, g[0, 1].reshape(-1))
        plt.colorbar()
        plt.subplot(3, len(gradients), 1 + len(gradients) + i, aspect=1)
        plt.tripcolor(tri, phase.reshape(-1), ec='black')
        plt.subplot(3, len(gradients), 1 + 2*len(gradients) + i)
        if i == 2:
            plt.plot(g[0, 1, 0, center[0]-1, :], 'rx-')
            plt.plot(g[0, 1, 0, center[0], :], 'gx-')
            plt.plot(g[0, 1, 0, center[0]+1, :], 'bx-')
            plt.plot(g[0, 1, 1, center[0]-1, :], 'r+--')
            plt.plot(g[0, 1, 1, center[0], :], 'g+--')
            plt.plot(g[0, 1, 1, center[0]+1, :], 'b+--')
        else:
            plt.plot(g[0, 1, 0, center[0]-1, :], 'rx-')
            plt.plot(g[0, 1, 0, center[0], :], 'gx-')
            plt.plot(g[0, 1, 0, center[0]+1, :], 'bx-')

    plt.tight_layout()

    if len(sys.argv[:]) == 2:
        if sys.argv[1] == 1:
            print("I skip the ploting of the results because you gave '1' as "
                  "first argument.")
            pass
    else:
        plt.show()

else:
    if MPI.COMM_WORLD.Get_size() != 1:
        print('Plotting disabled because we are running MPI-parallel.')
    else:
        print('Plotting disabled because matplotlib is not found.')

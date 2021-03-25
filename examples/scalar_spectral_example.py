#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   scalar_spectral_example.py

@author Till Junge <till.junge@altermail.ch>

@date   29 Mar 2021

@brief  Example of a heat flow problem in spectral formulation

Copyright © 2021 Till Junge

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


import optparse
import sys
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    matplotlib_found = True
except ImportError:
    matplotlib_found = False

from python_example_imports import muSpectre as msp


parser = optparse.OptionParser()

parser.add_option('-p', '--plot',
                  action="store_true", default=False, dest="plot",
                  help="show plot")

options, args = parser.parse_args()


resolution = 21
nb_grid_pts = [resolution, resolution]

lengths = [5., 5.]

rve = msp.cell.CellData.make(nb_grid_pts, lengths)
rve.nb_quad_pts = 1

kappa = 100
contrast = 100
conductor = msp.material.MaterialLinearDiffusion_2d.make(
    rve, "conductor", kappa * contrast)
insulator = msp.material.MaterialLinearDiffusion_2d.make(
    rve, "insulator", kappa)

mid_point = np.array([(nb + 1) // 2 for nb in nb_grid_pts])
inclusion_half_size = resolution/4
phases = np.zeros(nb_grid_pts, dtype = bool)
for i, pixel in  rve.pixels.enumerate():
    if np.abs(np.array(pixel) - mid_point).sum() <= inclusion_half_size:
        conductor.add_pixel(i)
        phases[pixel[0], pixel[1]] = True
    else:
        insulator.add_pixel(i)

# define the convergence tolerance for the Newton-Raphson increment
# this is irrelevant it the present linear problem, as it will converge after
# one step
tol = 1e-5
# tolerance for the solver of the linear cell
cg_tol = 1e-8
equi_tol = 0

# Macroscopic strain
Del0 = np.array([[0.],
                 [1.000]])

maxiter = 500  # for linear cell solver

# Choose a solver for the linear cells. Currently avaliable:
# KrylovSolverCG, KrylovSolverCGEigen, KrylovSolverBiCGSTABEigen,
# KrylovSolverGMRESEigen, KrylovSolverDGMRESEigen, KrylovSolverMINRESEigen.
# See Reference for explanations

krylov_solver = msp.solvers.KrylovSolverCG(cg_tol, maxiter,
                                           verbose=msp.Verbosity.Silent)

# Verbosity levels:
# Silent: silent
# Some: some info about Newton-Raphson loop,
# Full: full info about Newton-Raphson loop,
verbose = msp.Verbosity.Full


# Choose a solution strategy.
# construct the solver class using the solution strategy
# Currently available solver classes:
# SolverNewtonCG
# SolverNewtonPCG
# SolverNewtonFEMCG
# SolverNewtonFEMPCG
newton_solver = msp.solvers.SolverNewtonCG(rve, krylov_solver,
                                           verbose,
                                           tol, equi_tol, maxiter)
# obtaining the response of applying a load step
# consecutive load steps can be applied through a loop here
result = newton_solver.solve_load_increment(Del0)

# the resulting heat flux is stored in the stress field, because of µSpectre's
# roots as a solid mechanics code
flux = result.stress.reshape(2, *nb_grid_pts, order="F")
flux_norm = np.sqrt(flux[0,:,:]**2 + flux[1,:,:]**2)

temperature = newton_solver.projection.integrate(newton_solver.grad.field).array()


if matplotlib_found:
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.pcolormesh(flux[1,:,:])
    if (options.plot):
        plt.show()

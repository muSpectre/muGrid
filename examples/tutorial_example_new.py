#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file tutorial_example_new.py

@author Till Junge <till.junge@altermail.ch>

@date   22 Jan 2021

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

# currently, some muSpectre solvers  are restricted to odd numbers of
# grid points in each direction for reasons:
# Explained in T.W.J. de Geus, J. Vondřejc, J. Zeman,
# R.H.J. Peerlings, M.G.D. Geers, Finite strain FFT-based non-linear
# solvers made simple, Computer Methods in Applied Mechanics and
# Engineering, Volume 318, 2017
# https://doi.org/10.1016/j.cma.2016.12.032
dim = 2
nb_grid_pts = [51, 51]
center = np.array([r // 2 for r in nb_grid_pts])
incl = nb_grid_pts[0] // 5

# Domain dimensions
lengths = [7., 5.]
## formulation (small_strain or finite_strain)
formulation = msp.Formulation.small_strain

# build a computational domain
cell = msp.cell.CellData.make(nb_grid_pts, lengths)

# Determining the number of quad_pts in each pixel of solution domain
# might not be necessary for some of the solvers as they use their specific
# discretisation and determined the nb_quad_pts automatically
cell.nb_quad_pts = 1

# define the material properties of the matrix and inclusion
hard = msp.material.MaterialLinearElastic1_2d.make(
    cell, "hard", 90e9, .3)
soft = msp.material.MaterialLinearElastic1_2d.make(
    cell, "soft", 70e7, .3)

# assign each pixel to exactly one material
material_geometry = np.ndarray(nb_grid_pts)
for i, pixel in cell.pixels.enumerate():

    if np.linalg.norm(center - np.array(pixel), dim) < incl:
        material_geometry[np.array(pixel)[0], np.array(pixel)[1]] = 1
        soft.add_pixel(i)

    else:
        material_geometry[np.array(pixel)[0], np.array(pixel)[1]] = 0
        hard.add_pixel(i)


# define the convergence tolerance for the Newton-Raphson increment
tol = 1e-5
# tolerance for the solver of the linear cell
cg_tol = 1e-8
equi_tol = 1e-5

# Macroscopic strain
Del0 = np.array([[0.00, .000],
                 [0.000, .001]])
Del0 = .5 * (Del0 + Del0.T)

maxiter = 500  # for linear cell solver

# Choose a solver for the linear cells. Currently avaliable:
## KrylovSolverCG, KrylovSolverCGEigen, KrylovSolverBiCGSTABEigen, KrylovSolverGMRESEigen,
# KrylovSolverDGMRESEigen, KrylovSolverMINRESEigen.
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
newton_solver = msp.solvers.SolverNewtonCG(cell, krylov_solver,
                                           verbose,
                                           tol, equi_tol, maxiter)
# Determining the formulation for mechanics problems
# Small strain or finite strain
newton_solver.formulation = msp.Formulation.small_strain

# calling the initialiser of the cell associated with the solver
# It checks that the cell is properly prepared to apply a load increment
# and being solved for that
newton_solver.initialise_cell()

# obtaining the response of applying a load step
# consecutive load steps can be applied through a loop here
result = newton_solver.solve_load_increment(Del0)

print(result)

# visualise e.g., stress in y-direction
stress = result.stress
# stress is stored in a flatten stress tensor per pixel, i.e., a
# dim^2 × prod(nb_grid_pts_i) array, so it needs to be reshaped
stress = stress.reshape(dim, dim, *nb_grid_pts)


# prevent visual output during ctest
if len(sys.argv[:]) == 2:
    if sys.argv[1] == '1':
        pass
    else:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.pcolormesh(material_geometry)
        CS_geometry = plt.pcolormesh(material_geometry)

        for i in range(dim):
            for j in range(dim):
                fig = plt.figure()
                CS_stress = plt.pcolormesh(-stress[i,  j,:, :])
        plt.show()

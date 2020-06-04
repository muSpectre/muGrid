#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file tutorial_example.py

@author Till Junge <till.junge@altermail.ch>

@date   14 Feb 2018

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

# currently, muSpectre is restricted to odd numbers of grid points in each
# direction for reasons explained in T.W.J. de Geus, J. Vondřejc, J. Zeman,
# R.H.J. Peerlings, M.G.D. Geers, Finite strain FFT-based non-linear
# solvers made simple, Computer Methods in Applied Mechanics and
# Engineering, Volume 318, 2017
# https://doi.org/10.1016/j.cma.2016.12.032
nb_grid_pts = [51, 51]
center = np.array([r // 2 for r in nb_grid_pts])
incl = nb_grid_pts[0] // 5

# Domain dimensions
lengths = [7., 5.]
## formulation (small_strain or finite_strain)
formulation = msp.Formulation.small_strain

# build a computational domain
rve = msp.Cell(nb_grid_pts, lengths, formulation)

# define the material properties of the matrix and inclusion
hard = msp.material.MaterialLinearElastic1_2d.make(
    rve, "hard", 10e9, .33)
soft = msp.material.MaterialLinearElastic1_2d.make(
    rve, "soft", 70e9, .33)

# assign each pixel to exactly one material
for i, pixel in rve.pixels.enumerate():

    if np.linalg.norm(center - np.array(pixel), 2) < incl:
        hard.add_pixel(i)
    else:
        soft.add_pixel(i)

# define the convergence tolerance for the Newton-Raphson increment
tol = 1e-5
# tolerance for the solver of the linear cell
cg_tol = 1e-8
equi_tol = 1e-5

# Macroscopic strain
Del0 = np.array([[.0, .0],
                 [0, .03]])
Del0 = .5 * (Del0 + Del0.T)

maxiter = 50  # for linear cell solver

# Choose a solver for the linear cells. Currently avaliable:
## KrylovSolverCG, KrylovSolverCGEigen, KrylovSolverBiCGSTABEigen, KrylovSolverGMRESEigen,
# KrylovSolverDGMRESEigen, KrylovSolverMINRESEigen.
# See Reference for explanations

solver = msp.solvers.KrylovSolverCGEigen(rve, cg_tol, maxiter,
                                         verbose=msp.Verbosity.Silent)


# Verbosity levels:
# 0: silent,
# 1: info about Newton-Raphson loop,
verbose = 1

# Choose a solution strategy. Currently available:
# de_geus: is discribed in de Geus et al. see Ref above
# newton_cg: classical Newton-Conjugate Gradient solver. Recommended
result = msp.solvers.newton_cg(rve, Del0, solver, tol, equi_tol,
                               verbose=msp.Verbosity.Silent)

print(result)

# visualise e.g., stress in y-direction
stress = result.stress
# stress is stored in a flatten stress tensor per pixel, i.e., a
# dim^2 × prod(nb_grid_pts_i) array, so it needs to be reshaped
stress = stress.T.reshape(*nb_grid_pts, 2, 2)


# prevent visual output during ctest
if len(sys.argv[:]) == 2:
    if sys.argv[1] != 1:
        pass
else:
    import matplotlib.pyplot as plt
    plt.show()
    plt.pcolormesh(stress[:, :, 1, 1])
    plt.colorbar()

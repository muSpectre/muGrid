#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   small_elasto_plastic_case.py

@author Till Junge <till.junge@epfl.ch>

@date   12 Jan 2018

@brief  small case for debugging elasto-plasticity

Copyright © 2018 Till Junge

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
from python_example_imports import muSpectre as µ

nb_grid_pts = [5, 5]

# Domain dimensions
lengths = [7., 5.]

# formulation (small_strain or finite_strain)
formulation = µ.Formulation.finite_strain

# Material constants
K = .833
mu = .386
H = .04 # Low values of H worsen the condition number of the stiffness-matrix
tauy0 = .006
Young = 9 * K * mu / (3 * K + mu)
Poisson = (3 * K - 2 * mu) / (2 * (3 * K + mu))

# RVE definition
rve = µ.Cell(nb_grid_pts, lengths, formulation)

# define the material properties of the matrix and inclusion
hard = µ.material.MaterialHyperElastoPlastic1_2d.make(
    rve, "hard", Young, Poisson, 2 * tauy0, h=2 * H)
soft = µ.material.MaterialHyperElastoPlastic1_2d.make(
    rve, "soft", Young, Poisson, tauy0, h=H)

# assign each pixel to exactly one material
for pixel_id, pixel_coord in rve.pixels.enumerate():
    if pixel_id < 3:
        hard.add_pixel(pixel_id)
    else:
        soft.add_pixel(pixel_id)

rve.initialise()

# define the convergence tolerance for the Newton-Raphson increment
tol = 1e-5
# tolerance for the solver of the linear cell
cg_tol = 1e-5
equil_tol = 1e-5

# Macroscopic strain
Del0 = np.array([[.0, .0],
                 [0, 3e-2]])
if formulation == µ.Formulation.small_strain:
    Del0 = .5 * (Del0 + Del0.T)

maxiter = 401
verbose = µ.Verbosity.Detailed


solver = µ.solvers.KrylovSolverCG(rve, cg_tol, maxiter,
                                  verbose=verbose)
r = µ.solvers.de_geus(rve, Del0, solver,
                      tol, equil_tol, verbose)

print("nb of {} iterations: {}".format(solver.name, r.nb_fev))

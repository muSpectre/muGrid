#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   hyper-elasticity.py

@author Till Junge <till.junge@epfl.ch>

@date   16 Jan 2018

@brief  Recreation of GooseFFT's hyper-elasticity.py calculation

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
import argparse

from python_example_imports import muSpectre as µ

def compute():
    N = [7, 7, 7]
    lens = [1., 1., 1.]
    incl_size = 3
    inclusion_center = np.array(N)//2

    formulation = µ.Formulation.finite_strain
    cell = µ.Cell(N, lens, formulation)
    hard = µ.material.MaterialLinearElastic1_3d.make(cell, "hard", 210.e9, .33)
    soft = µ.material.MaterialLinearElastic1_3d.make(cell, "soft", 70.e9, .33)
    for pixel_id, pixel_coord in cell.pixels.enumerate():
        # if ((pixel[0] >= N[0]-incl_size) and
        #     (pixel[1] < incl_size) and
        #     (pixel[2] >= N[2]-incl_size)):
        if (((inclusion_center - np.array(pixel_coord))**2).sum() < incl_size**2):
            hard.add_pixel(pixel_id)
        else:
            soft.add_pixel(pixel_id)

    print("{} pixels in the inclusion".format(hard.size()))
    cell.initialise();
    cg_tol, newton_tol, equil_tol = 1e-8, 1e-5, 1e-8
    maxiter = 40
    verbose = 3
    dF_bar = np.array([[0, .02, 0], [0, 0, 0], [0, 0, 0]])

    if formulation == µ.Formulation.small_strain:
        dF_bar = .5*(dF_bar + dF_bar.T)


    test_grad = np.zeros((3,3) + tuple(cell.nb_domain_grid_pts))
    test_grad[:,:] = dF_bar.reshape((3,3,1,1,1))
    solver = µ.solvers.KrylovSolverCG(cell, cg_tol, maxiter,
                                      verbose=µ.Verbosity.Silent)
    optimize_res = µ.solvers.de_geus(cell, dF_bar, solver, newton_tol,
                                     equil_tol, verbose=µ.Verbosity.Full)
    print("nb_cg: {}\nF:\n{}".format(optimize_res.nb_fev,
          µ.gradient_integration.reshape_gradient(optimize_res.grad,
          cell.nb_domain_grid_pts)[:,:,0,0,0]))


def main():
    compute()

if __name__ == "__main__":
    main()

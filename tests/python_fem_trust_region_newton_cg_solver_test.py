#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_fem_trust_region_newton_cg_solver_test.py

@author Ali Falsafi <ali.falsafi@epfl.ch>

@date   04 Sep 2020

@brief  test for the trust region solver class against
        scipy trust region solver

Copyright © 2020 Till Junge

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

import unittest
import numpy as np
from python_test_imports import muSpectre as msp
from scipy.optimize import minimize
from muSpectre import cell
from python_test_1d_scipy_damage_classes.py import (
    mat_lin_dam, mat_lin_undam, func_calculations)


class FEMTRSolverClassCheck(unittest.TestCase):
    def setUp(self):
        self.nb_grid_pts = [3, 3]  # [5,7]
        self.lengths = [1., 1.]  # [5.2, 8.3]
        self.formulation = msp.Formulation.small_strain
        self.cell = msp.cell.CellData.make(self.nb_grid_pts, self.lengths)
        self.cell.nb_quad_pts = 4
        self.young = 2.0e10
        self.poisson = 0.0
        self.kappa = 1.e-1
        self.alpha = 2.e-1
        self.mean_strain = 1.2e-1
        self.trust_radius = 1.e0
        self.eta = 1.0e-4
        self.elastic = msp.material.MaterialLinearElastic1_2d.make(self.cell,
                                                                   "Elastic",
                                                                   self.young,
                                                                   self.poisson)
        self.damage = msp.material.MaterialDunant_2d.make(self.cell,
                                                          "Damage",
                                                          self.young,
                                                          self.poisson,
                                                          self.kappa,
                                                          self.alpha)
        self.fem_stencil = msp.FEMStencil.bilinear_quadrangle(
            self.cell)

        self.discretisation = msp.Discretisation(self.fem_stencil)

    def test_solve_fem(self):
        for pix_id, (pix_x, pix_y) in enumerate(self.cell.pixels):
            if pix_id < 3:
                self.damage.add_pixel(pix_id)
            else:
                self.elastic.add_pixel(pix_id)

        cg_tol = 1e-8
        newton_tol = 1e-8
        equil_tol = 1e-7
        Del0 = np.array([[0, 0.],
                         [0,  self.mean_strain]])
        maxiter = 100
        verbose = msp.Verbosity.Silent

        krylov_solver = msp.solvers.KrylovSolverTrustRegionCG(
            cg_tol, maxiter, self.trust_radius, verbose)

        newton_solver = msp.solvers.SolverFEMTRNewtonCG(
            self.discretisation, krylov_solver,
            verbose, newton_tol,
            equil_tol, maxiter,
            self.trust_radius,
            self.eta)

        newton_solver.formulation = self.formulation
        newton_solver.initialise_cell()
        res = newton_solver.solve_load_increment(Del0)
        grad = res.grad
        print("Trust Region solver result is ".format(grad))

        mats_dam_neg_slope_coeff = [1.0, 0.0, 0.0]
        func_calcs = func_calculations(self.young,
                                       -1.0 * self.alpha * self.young,
                                       self.kappa,
                                       mats_dam_neg_slope_coeff,
                                       e_mac=self.mean_strain)
        my_method = 'trust-ncg'
        my_options = {'disp': True,
                      'gtol': 1e-9,
                      'initial_trust_radius': 0.01,
                      'max_trust_radius': 0.1,
                      'eta': 1e-6}
        x_init = np.array([0.8*self.mean_strain, 1.00*self.mean_strain])
        mats = func_calcs.mats_make()
        res_scipy = minimize(func_calcs.tot_energy, x_init,
                             method=my_method,
                             jac=func_calcs.tot_jac,
                             hess=func_calcs.tot_hess,
                             options=my_options)
        strain_solution_scipy = res_scipy.x
        x = np.array(res_scipy.x)
        print("scipy result is {}".format(x))

        print("solver result shape is: {}".format(grad.shape))
        grad_yy = grad[3, :]

        # check the equality of the corresponding values from the
        # muSepctre result gradient and the output of the scipy solver
        for i in np.arange(0, 12):
            self.assertTrue(((x[0] - grad_yy[i]) / grad_yy[i]) < newton_tol)
        for i in np.arange(12, 36):
            self.assertTrue(((x[1] - grad_yy[i]) / grad_yy[i]) < newton_tol)


if __name__ == '__main__':
    unittest.main()

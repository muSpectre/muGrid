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


class mat_lin_dam:
    def __init__(self, k_pos, k_neg, e0):
        self.e0 = e0
        self.e_init = e0
        self.k_pos = k_pos
        self.k_pos_init = k_pos
        self.k_neg = k_neg

    def reset(self):
        self.e0 = self.e_init
        self.k_pos = self.k_pos_init

    def update_internal(self, e0_new, k_new):
        self.e0 = e0_new
        self.k_pos = k_new

    def stress_tangent_cal(self, e, v=False):
        if abs(e) <= self.e0:
            if v:
                print("NO DAMAGE")
            return self.k_pos * e, self.k_pos
        else:
            ret_stress = 0.0
            if e > 0.0:
                ret_stress = (self.k_pos * self.e0 +
                              self.k_neg * (e - self.e0))
            cor_stress = ret_stress if (ret_stress * e) > 0 else 0.0
            k_update = cor_stress / e
            self.update_internal(e, k_update)
            if v:
                print(
                    "DAMAGE MATERIAL Reduction factor is : " +
                    "{}".format((self.k_pos_init - k_update) / self.k_pos_init))
            return cor_stress, self.k_neg

    def energy_cal(self, e):
        return 0.5 * self.stress_tangent_cal(e)[0] * e


class func_calculations:
    def __init__(self, k_pos, k_neg, e_init,
                 mats_dam_neg_slope_coeff=[1.0, 0.0, 0.0],
                 e_mac=0.0):
        self.k_pos = k_pos
        self.k_neg = k_neg
        self.e_init = e_init
        self.e_mac = e_mac
        self.mats_dam_neg_slope_coeff = mats_dam_neg_slope_coeff

    def mat_dam_make(self, coeff=1):
        return mat_lin_dam(self.k_pos,
                           coeff * self.k_neg,
                           self.e_init)

    def mat_undam_make(self):
        return mat_lin_undam(self.k_pos)

    def mats_make(self):
        ret_tup = list()
        for i, mat_dam_neg_slope_coeff in enumerate(
                self.mats_dam_neg_slope_coeff):
            if mat_dam_neg_slope_coeff == 0:
                ret_tup.append(self.mat_undam_make())
            else:
                ret_tup.append(self.mat_dam_make(mat_dam_neg_slope_coeff))
        return tuple(ret_tup)

    def cal_e_last(self, es):
        return (3.0 * self.e_mac-np.sum(es))

    def tot_energy(self, es):
        es_loc = np.array([es[0], es[1], self.cal_e_last(es)])
        mats = self.mats_make()
        ret_energy = 0.0
        for i, e_loc in enumerate(es_loc):
            ret_energy += mats[i].energy_cal(e_loc)
        return ret_energy

    def tot_jac(self, es):
        es_loc = np.array([es[0], es[1],
                           self.cal_e_last(es)])
        mats = self.mats_make()
        jacs = np.zeros_like(es_loc)
        for i, e_loc in enumerate(es_loc):
            jacs[i] = mats[i].stress_tangent_cal(e_loc)[0]
        return (jacs[:2] - self.k_pos * es_loc[2])

    def tot_stress(self, es):
        es_loc = np.array([es[0], es[1], self.cal_e_last(es)])
        mats = self.mats_make()
        jacs = np.zeros_like(es_loc)
        for i, e_loc in enumerate(es_loc):
            jacs[i] = mats[i].stress_tangent_cal(e_loc)[0]
        return (jacs[:2])

    def tot_hess(self, es):
        es_loc = np.array([es[0], es[1], self.cal_e_last(es)])
        mats = self.mats_make()
        hesses = np.zeros_like(es_loc)
        for i, e_loc in enumerate(es_loc):
            hesses[i] = mats[i].stress_tangent_cal(e_loc)[1]
        return (np.diag(hesses[:2]) +
                self.k_pos)


class mat_lin_undam:
    def __init__(self, k):
        self.k = k

    def stress_tangent_cal(self, e, v=False):
        if v:
            print("NON DAMAGE MATERIAL")
        return self.k * e, self.k

    def energy_cal(self, e):
        return 0.5 * self.stress_tangent_cal(e)[0] * e


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

# !/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_material_hyper_elasto_plastic2_test.py

@author Richard Leute <richard.leute@imtek.uni-freiburg.de>

@date   27 Mar 2018

@brief  description

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

import unittest
import numpy as np

from python_test_imports import µ

class MaterialHyperElastoPlastic2_Check(unittest.TestCase):
    def test_2_vs_1(self):
        ### material geometry
        lens = [10, 10, 10]
        nb_grid_pts  = [5, 5, 5]
        dim = len(nb_grid_pts)

        ### material parameters
        Young   = 210
        Poisson = 0.30
        mu = Young / (2*(1+Poisson))

        np.random.seed(125769235)
        yield_crit = (mu * (0.025 + 0.01 * (np.random.random(nb_grid_pts) > 0.5))).flatten()
        hardening = 100

        ### µSpectre init stuff
        fft = "fftw"
        form = µ.Formulation.finite_strain
        dz = µ.DiscreteDerivative([0, 0, 0],
                                    [[[-0.25, 0.25], [-0.25, 0.25]],
                                     [[-0.25, 0.25], [-0.25, 0.25]]])
        dx = dz.rollaxes(1)
        dy = dx.rollaxes(1)
        discrete_gradient = [dx, dy, dz]

        cell = µ.Cell(nb_grid_pts, lens, form, discrete_gradient, fft)
        cell2 = µ.Cell(nb_grid_pts, lens, form, discrete_gradient, fft)

        mat_vac = µ.material.MaterialLinearElastic1_3d.make(cell.wrapped_cell,
                                                            "3d-vacuum",
                                                            0.5*Young, Poisson)
        mat_vac2 = µ.material.MaterialLinearElastic1_3d.make(cell2.wrapped_cell,
                                                             "3d-vacuum",
                                                             0.5*Young, Poisson)

        mat_min = µ.material.MaterialHyperElastoPlastic1_3d.make(
          cell.wrapped_cell, "3d-small", Young, Poisson, yield_crit.min(), hardening)
        mat_max = µ.material.MaterialHyperElastoPlastic1_3d.make(
          cell.wrapped_cell, "3d-large", Young, Poisson, yield_crit.max(), hardening)

        mat_hpl = µ.material.MaterialHyperElastoPlastic2_3d.make(
          cell2.wrapped_cell, "3d-hpl")

        E        = np.zeros(nb_grid_pts)
        E[:, :, :] = 0.5*Young
        E[:, :, :-1] = Young
        E = E.flatten()

        m = (yield_crit.max() + yield_crit.min())/2

        for i, pixel in enumerate(cell):
            if E[i] < 0.9*Young:
                mat_vac.add_pixel(pixel)
            else:
                if yield_crit[i] < m:
                    mat_min.add_pixel(pixel)
                else:
                    mat_max.add_pixel(pixel)

        for i, pixel in enumerate(cell2):
            if E[i] < 0.9*Young:
                mat_vac2.add_pixel(pixel)
            else:
                mat_hpl.add_pixel(pixel, E[i], Poisson, yield_crit[i], hardening)

        #solver
        newton_tol = 1e-8
        cg_tol     = 1e-8
        equil_tol  = 1e-8
        maxiter = 200
        verbose = 2
        solver = µ.solvers.SolverCG(cell.wrapped_cell, cg_tol, maxiter, verbose)
        cell.initialise()

        solver2 = µ.solvers.SolverCG(cell2.wrapped_cell, cg_tol, maxiter, verbose)
        cell2.initialise()

        #total deformation
        DelF  = np.array([[-0.10 ,  0.00,  0.00],
                          [ 0.00 , -0.10,  0.00],
                          [ 0.00 ,  0.00,  0.00]])

        ### Start muSpectre ###
        #---------------------#
        result = µ.solvers.newton_cg(cell.wrapped_cell, DelF, solver,
                                     newton_tol, equil_tol, verbose)
        result2 = µ.solvers.newton_cg(cell2.wrapped_cell, DelF, solver2,
                                      newton_tol, equil_tol, verbose)

        F = cell.strain
        stress, tangent = cell.evaluate_stress_tangent(F)
        F = cell2.strain
        stress2, tangent2 = cell2.evaluate_stress_tangent(F)

        self.assertTrue(np.allclose(stress, stress2))
        self.assertTrue(np.allclose(tangent, tangent2))

    def test_tangent(self):
        ### Input parameters ###
        #----------------------#
        ### material geometry
        lens = [10, 10, 10]
        nb_grid_pts  = [1, 1, 1]
        dim = len(nb_grid_pts)

        ### material parameters
        Young   = 210
        Poisson = 0.30
        mu = Young / (2*(1+Poisson))

        np.random.seed(125769235)
        yield_crit = (mu * (0.025 + 0.01 * np.random.random(nb_grid_pts))).flatten()
        hardening = 100

        ### µSpectre init stuff
        fft = "fftw"
        form = µ.Formulation.finite_strain
        dz = µ.DiscreteDerivative([0, 0, 0],
                                    [[[-0.25, 0.25], [-0.25, 0.25]],
                                     [[-0.25, 0.25], [-0.25, 0.25]]])
        dx = dz.rollaxes(1)
        dy = dx.rollaxes(1)
        discrete_gradient = [dx, dy, dz]

        cell = µ.Cell(nb_grid_pts, lens, form, discrete_gradient, fft)

        mat_vac = µ.material.MaterialLinearElastic1_3d.make(
            cell.wrapped_cell, "3d-vacuum", 0.5*Young, Poisson)
        mat_hpl = µ.material.MaterialHyperElastoPlastic2_3d.make(
            cell.wrapped_cell, "3d-hpl")

        E        = np.zeros(nb_grid_pts)
        E[:, :, :] = 0.5*Young
        E[:, :, :-1] = Young
        E = E.flatten()

        for i, pixel in enumerate(cell):
            if E[i] < 0.9*Young:
                mat_vac.add_pixel(pixel)
            else:
                mat_hpl.add_pixel(pixel, E[i], Poisson, yield_crit[i],
                                  hardening)

        #solver
        newton_tol = 1e-8
        cg_tol     = 1e-8
        equil_tol  = 1e-8
        maxiter = 200
        verbose = 2
        solver = µ.solvers.SolverCG(cell.wrapped_cell, cg_tol, maxiter,
                                    verbose)
        cell.initialise()

        #total deformation - elastic region
        DelF  = np.array([[-0.01 ,  0.00,  0.00],
                          [ 0.00 , -0.01,  0.00],
                          [ 0.00 ,  0.00,  0.00]])

        result = µ.solvers.newton_cg(cell.wrapped_cell, DelF, solver,
                                     newton_tol, equil_tol, verbose)

        ### Finite differences evaluation of the tangent
        F = cell.strain
        stress, tangent = cell.evaluate_stress_tangent(F)

        numerical_tangent = np.zeros_like(tangent)

        eps = 1e-4
        for i in range(3):
            for j in range(3):
                F[i, j] += eps
                stress_plus = cell.evaluate_stress(F).copy()
                F[i, j] -= 2*eps
                stress_minus = cell.evaluate_stress(F).copy()
                F[i, j] += eps
                numerical_tangent[i, j] = (stress_plus - stress_minus)/(2*eps)

        self.assertTrue(np.allclose(tangent, numerical_tangent))

        #total deformation - plastic region
        DelF  = np.array([[ 0.00 ,  0.20,  0.00],
                          [ 0.00 ,  0.00,  0.15],
                          [ 0.00 ,  0.00,  0.17]])

        result = µ.solvers.newton_cg(cell.wrapped_cell, DelF, solver,
                                     newton_tol, equil_tol, verbose)


        ### Finite differences evaluation of the tangent
        F = cell.strain
        stress, tangent = cell.evaluate_stress_tangent(F)

        numerical_tangent = np.zeros_like(tangent)

        eps = 1e-9
        for i in range(3):
            for j in range(3):
                F[i, j] += eps
                stress_plus = cell.evaluate_stress(F).copy()
                F[i, j] -= eps
                numerical_tangent[i, j] = (stress_plus - stress)/eps

        self.assertTrue(np.allclose(tangent, numerical_tangent))

if __name__ == '__main__':
    unittest.main()

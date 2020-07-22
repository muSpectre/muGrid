#!/usr/bin/env python3
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
import time
import numpy as np

from python_test_imports import µ, muFFT


class MaterialHyperElastoPlastic2_Check(unittest.TestCase):
    def setUp(self):
        # set timing = True for timing information
        self.timing = False
        self.startTime = time.time()

    def tearDown(self):
        if self.timing:
            t = time.time() - self.startTime
            print("{}:\n{:.3f} seconds".format(self.id(), t))

    def test_2_vs_1(self):
        """
        Compares stress and strain computed by material_hyper_elasto_plastic2 vs
        stress and strain computed by material_hyper_elasto_plastic1. The yield
        thresholds and Young moduli are set random.
        """
        # material geometry
        lens = [10, 10, 10]
        nb_grid_pts = [3, 3, 3]
        dim = len(nb_grid_pts)

        # material parameters
        Young = 210
        Poisson = 0.30
        mu = Young / (2*(1+Poisson))

        np.random.seed(102919)  # just the date
        yield_crit = mu * (0.025 + 0.05 * np.random.random(nb_grid_pts))
        E = Young * (0.6 + 0.3 * np.random.random(nb_grid_pts))
        hardening = 1

        # µSpectre init stuff
        fft = "fftw"
        form = µ.Formulation.finite_strain
        # use e.g. average upwind differences
        dz = muFFT.DiscreteDerivative([0, 0, 0],
                                      [[[-0.25, 0.25], [-0.25, 0.25]],
                                       [[-0.25, 0.25], [-0.25, 0.25]]])
        dy = dz.rollaxes(-1)
        dx = dy.rollaxes(-1)
        discrete_gradient = [dx, dy, dz]

        cell = µ.Cell(nb_grid_pts, lens, form, discrete_gradient, fft)
        cell2 = µ.Cell(nb_grid_pts, lens, form, discrete_gradient, fft)

        # stores a hyper elasto plastic 1 material for each pixel
        mat_hpl1_array = np.empty((3, 3, 3), dtype=object)
        for index, mat in np.ndenumerate(mat_hpl1_array):
            mat_hpl1_array[index] = \
                µ.material.MaterialHyperElastoPlastic1_3d.make(
                    cell, "3d-small", E[index], Poisson,
                    yield_crit[index], hardening)

        mat_hpl2 = µ.material.MaterialHyperElastoPlastic2_3d.make(
            cell2, "3d-hpl")

        for i, pixel in cell.pixels.enumerate():
            mat_hpl1_array[tuple(pixel)].add_pixel(i)

        for i, pixel in cell2.pixels.enumerate():
            mat_hpl2.add_pixel(i, E[tuple(pixel)], Poisson,
                               yield_crit[tuple(pixel)], hardening)

        # solver
        newton_tol = 1e-6
        cg_tol = 1e-6
        equil_tol = 1e-6
        maxiter = 2000
        verbose = µ.Verbosity.Silent
        solver = µ.solvers.KrylovSolverCG(cell, cg_tol, maxiter, verbose)
        cell.initialise()

        solver2 = µ.solvers.KrylovSolverCG(cell2, cg_tol, maxiter, verbose)
        cell2.initialise()

        # total deformation. The factor of 0.85 is to achieve convergence. This
        # is the largest strain for which the solver converges.
        DelF = np.array([[-0.05,  0.10,  0.00],
                         [0.00, -0.05,  0.00],
                         [0.00,  0.00,  0.00]])*0.85

        ### Start muSpectre ###
        #---------------------#
        result = µ.solvers.newton_cg(cell, DelF, solver,
                                     newton_tol, equil_tol, verbose)
        result2 = µ.solvers.newton_cg(cell2, DelF, solver2,
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
        # material geometry
        lens = [10, 10, 10]
        nb_grid_pts = [1, 1, 1]
        dim = len(nb_grid_pts)

        # material parameters
        Young = 210
        Poisson = 0.30
        mu = Young / (2*(1+Poisson))

        np.random.seed(125769235)
        yield_crit = (
            mu * (0.025 + 0.01 * np.random.random(nb_grid_pts))).flatten()
        hardening = 100

        # µSpectre init stuff
        fft = "fftw"
        form = µ.Formulation.finite_strain
        dz = muFFT.DiscreteDerivative([0, 0, 0],
                                      [[[-0.25, 0.25], [-0.25, 0.25]],
                                       [[-0.25, 0.25], [-0.25, 0.25]]])
        dy = dz.rollaxes(-1)
        dx = dy.rollaxes(-1)
        discrete_gradient = [dx, dy, dz]

        cell = µ.Cell(nb_grid_pts, lens, form, discrete_gradient, fft)

        mat_vac = µ.material.MaterialLinearElastic1_3d.make(
            cell, "3d-vacuum", 0.5*Young, Poisson)
        mat_hpl = µ.material.MaterialHyperElastoPlastic2_3d.make(
            cell, "3d-hpl")

        E = np.zeros(nb_grid_pts)
        E[:, :, :] = 0.5*Young
        E[:, :, :-1] = Young
        E = E.flatten()

        for i, pixel in cell.pixels.enumerate():
            if E[i] < 0.9*Young:
                mat_vac.add_pixel(i)
            else:
                mat_hpl.add_pixel(i, E[i], Poisson, yield_crit[i],
                                  hardening)

        # solver
        newton_tol = 1e-8
        cg_tol = 1e-8
        equil_tol = 1e-8
        maxiter = 200
        verbose = µ.Verbosity.Silent
        solver = µ.solvers.KrylovSolverCG(cell, cg_tol, maxiter,
                                    verbose)
        cell.initialise()

        # total deformation - elastic region
        DelF = np.array([[-0.01,  0.00,  0.00],
                         [0.00, -0.01,  0.00],
                         [0.00,  0.00,  0.00]])

        result = µ.solvers.newton_cg(cell, DelF, solver,
                                     newton_tol, equil_tol, verbose)

        # Finite differences evaluation of the tangent
        F = cell.strain.array()
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

        # total deformation - plastic region
        DelF = np.array([[0.00,  0.20,  0.00],
                         [0.00,  0.00,  0.15],
                         [0.00,  0.00,  0.17]])

        result = µ.solvers.newton_cg(cell, DelF, solver,
                                     newton_tol, equil_tol, verbose)

        # Finite differences evaluation of the tangent
        F = cell.strain.array()
        stress, tangent = cell.evaluate_stress_tangent(F)

        stress = stress.copy()

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

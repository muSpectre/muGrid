#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_material_phase_field_fracture_test.py

@author W. Beck Andrews <william.beck.andrews@imtek.uni-freiburg.de>

@date   26 Mar 2021

@brief  Tests python functionality of phase field fracture material.  Includes
a simple test of simulation execution and a test that checks the stress
tangent implementation against values computed from the stress by finite
differences.

Copyright © 2018 2021 W. Beck Andrews

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

from python_test_imports import µ, muGrid, muFFT

def test_tangent_worker(testobj, dim):
    ### Input parameters ###
    #----------------------#
    # material geometry
    lens = []
    nb_grid_pts = []
    for i in range(0,dim):
        lens.append(1)
        nb_grid_pts.append(3)

    # material parameters
    Young = 1e3
    Poisson = 0.30
    ksmall = 1e-4

    # µSpectre init stuff
    fft = "fftw"
    form = µ.Formulation.small_strain
    # use e.g. average upwind differences
    fourier_gradient = [µ.FourierDerivative(dim , i) for i in range(dim)]

    cell = µ.Cell(nb_grid_pts, lens, form, fourier_gradient, fft)
    if (dim == 2):
        mat = µ.material.MaterialPhaseFieldFracture_2d.make(
            cell, "material_small", ksmall)
    else:
        mat = µ.material.MaterialPhaseFieldFracture_3d.make(
            cell, "material_small", ksmall)
        
    phivals = np.random.random(np.prod(nb_grid_pts))
    for i, pixel in cell.pixels.enumerate():
        mat.add_pixel(i, Young, Poisson, phivals[i])

    # solver
    newton_tol = 1e-6
    cg_tol = 1e-6
    equil_tol = 1e-6
    maxiter = 2000
    verbose = µ.Verbosity.Silent
    solver = µ.solvers.KrylovSolverCG(cell, cg_tol, maxiter,
                                verbose)
    cell.initialise()
    
    # total deformation - plastic region
    if (dim == 2):
        F = np.array([[0.006,    0.01],
                      [0.01,  0.00]])
    else:
        F = np.array([[0.02,  -0.005,  0.00],
                      [-0.005,  0.00,  0.01],
                      [0.00,  0.01,  0.00]])
    result = µ.solvers.newton_cg(cell, F, solver,
                                 newton_tol, equil_tol, verbose)
                                 
    # Finite differences evaluation of the tangent
    strain = cell.strain.array()
    stress, tangent = cell.evaluate_stress_tangent(strain)
    numerical_tangent = np.zeros_like(tangent)
    eps = 1e-6
    for i in range(dim):
        for j in range(dim):
            if (i == j):
                strain[i, j] += eps
                stress_plus = cell.evaluate_stress(strain).copy()
                strain[i, j] -= 2*eps
                stress_minus = cell.evaluate_stress(strain).copy()
                strain[i, j] += eps
            else:
                strain[i, j] += eps/2
                strain[j, i] += eps/2
                stress_plus = cell.evaluate_stress(strain).copy()
                strain[i, j] -= eps
                strain[j, i] -= eps
                stress_minus = cell.evaluate_stress(strain).copy()
                strain[i, j] += eps/2
                strain[j, i] += eps/2
            numerical_tangent[i, j] = (stress_plus - stress_minus)/(2*eps)
    testobj.assertTrue(((tangent - numerical_tangent)**2).sum()
            /(tangent**2).sum() < 1e-8)


class MaterialPhaseFieldFracture_Check(unittest.TestCase):
    def setUp(self):
        # set timing = True for timing information
        self.timing = False
        self.startTime = time.time()

    def tearDown(self):
        if self.timing:
            t = time.time() - self.startTime
            print("{}:\n{:.3f} seconds".format(self.id(), t))

    def test_sim(self):
        """
        Compares stress and strain computed by material_hyper_elasto_plastic2 vs
        stress and strain computed by material_hyper_elasto_plastic1. The yield
        thresholds and Young moduli are set random.
        """
        # material geometry
        lens = [1, 1, 1]
        nb_grid_pts = [3, 3, 3]
        dim = len(nb_grid_pts)

        # material parameters
        Young = 1e3
        Poisson = 0.30
        ksmall = 1e-4

        # µSpectre init stuff
        fft = "fftw"
        form = µ.Formulation.small_strain
        # use e.g. average upwind differences
        fourier_gradient = [µ.FourierDerivative(dim , i) for i in range(dim)]

        cell = µ.Cell(nb_grid_pts, lens, form, fourier_gradient, fft)
        mat = µ.material.MaterialPhaseFieldFracture_3d.make(
            cell, "material_small", ksmall)

        phi_i = 13
        for i, pixel in cell.pixels.enumerate():
            mat.add_pixel(i, Young, Poisson, 0.0)
            if (i == phi_i):
                phi_pixel = pixel
        
        # solver
        newton_tol = 1e-6
        cg_tol = 1e-6
        equil_tol = 1e-6
        maxiter = 2000
        verbose = µ.Verbosity.Silent
        solver = µ.solvers.KrylovSolverCG(cell, cg_tol, maxiter, verbose)
        cell.initialise()
        mat.set_phase_field(phi_i, 1.0)

        # strain
        F = np.array([[0.00,  0.01,  0.00],
                      [0.01,  0.00,  0.00],
                      [0.00,  0.00,  0.02]])

        ### Start muSpectre ###
        #---------------------#
        result = µ.solvers.newton_cg(cell, F, solver,
                                     newton_tol, equil_tol, verbose)
        strain = cell.strain.array()
        stress, tangent = cell.evaluate_stress_tangent(strain)
        
        # Common sense check on output: pixel with nonzero phase field should
        # have lowest tensile stress and highest tensile strain.
        self.assertTrue(np.argmax(cell.strain.array()[2,2,...]) == phi_i)
        self.assertTrue(np.argmin(stress[2,2,...]) == phi_i)
        
    def test_tangent_2D(self):
        test_tangent_worker(self, 2)

    def test_tangent_3D(self):
        test_tangent_worker(self, 3)    

if __name__ == '__main__':
    unittest.main()

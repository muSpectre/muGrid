#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_richard_test.py

@author Richard Leute <richard.leute@imtek.uni-freiburg.de>

@date   20 Feb 2018

@brief  description

@section LICENSE

Copyright © 2018 Till Junge

µSpectre is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µSpectre is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Emacs; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.
"""

import unittest
import numpy as np

from python_test_imports import µ

class MaterialLinearElastic3_Check(unittest.TestCase):
    """
    Check the implementation of Youngs modulus and Poisson ratio for each cell.
    Assign the same E and nu to each cell, calculate the stress and compare the
    result with stress=2*mu*Del0 (Hooke law for small symmetric strains).
    """
    def setUp(self):
        self.resolution = [5,5]
        self.lengths = [2.5, 3.1]
        self.formulation = µ.Formulation.small_strain
        self.sys = µ.SystemFactory(self.resolution,
                                   self.lengths,
                                   self.formulation)
        self.mat = µ.material.MaterialLinearElastic3_2d.make(
            self.sys, "material")

    def test_solver(self):
        Young   = 10.
        Poisson = 0.3

        for i, pixel in enumerate(self.sys):
            self.mat.add_pixel(pixel, Young, Poisson)

        self.sys.initialise()
        tol = 1e-6
        Del0 = np.array([[0, 0.025],
                         [0.025,  0]])
        maxiter = 100
        verbose = 1

        solver=µ.solvers.SolverCG(self.sys, tol, maxiter, verbose)
        r = µ.solvers.newton_cg(self.sys, Del0,
                                solver, tol, tol, verbose)

        #compare the computed stress with the trivial by hand computed stress
        mu = (Young/(2*(1+Poisson)))
        stress = 2*mu*Del0

        self.assertLess(np.linalg.norm(r.stress-stress.reshape(-1,1)), 1e-8)

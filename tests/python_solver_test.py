#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_solver_test.py

@author Till Junge <till.junge@epfl.ch>

@date   22 Nov 2019

@brief  Unit tests for python bindings

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


class SolverCheck(unittest.TestCase):
    def setUp(self):
        self.nb_grid_pts = [3, 3]  # [5,7]
        self.lengths = [3., 3.]  # [5.2, 8.3]
        self.formulation = µ.Formulation.finite_strain
        self.sys = µ.Cell(self.nb_grid_pts,
                          self.lengths,
                          self.formulation)
        self.hard = µ.material.MaterialLinearElastic1_2d.make(
            self.sys, "hard", 210e9, .33)
        self.soft = µ.material.MaterialLinearElastic1_2d.make(
            self.sys.wrapped_cell, "soft",  70e9, .33)

    def test_solve(self):
        for i, pixel in enumerate(self.sys):
            if i < 3:
                self.hard.add_pixel(pixel)
            else:
                self.soft.add_pixel(pixel)

        self.sys.initialise()
        tol = 1e-6
        Del0 = np.array([[0, .1],
                         [0,  0]])
        maxiter = 100
        verbose = 0

        solver = µ.solvers.SolverCG(
            self.sys.wrapped_cell, tol, maxiter, verbose)
        r = µ.solvers.de_geus(self.sys.wrapped_cell, Del0,
                              solver, tol, verbose)
        # print(r)

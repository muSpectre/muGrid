#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_material_linear_elastic_generic1_test.py

@author Till Junge <till.junge@epfl.ch>

@date   20 Dec 2018

@brief  tests the python bindings of the generic linear elastic material

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

class MaterialLinearElasticGeneric1_Check(unittest.TestCase):
    def setUp(self):
        self.nb_grid_pts = [5,7,5]
        self.dim = len(self.nb_grid_pts)
        self.lengths = [5.2, 8.3, 2.7]
        self.formulation = µ.Formulation.small_strain
        self.cell1 = µ.Cell(self.nb_grid_pts,
                            self.lengths,
                            self.formulation)
        self.Young = 210e9
        self.Poisson = .33
        self.mat1 = µ.material.MaterialLinearElastic1_3d.make(
            self.cell1, "material", self.Young, self.Poisson)
        self.matO1 = µ.material.MaterialLinearElastic1_3d.make(
            self.cell1, "material", 2* self.Young, self.Poisson)

        E, nu = self.Young, self.Poisson
        lam, mu = E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))

        C = np.array([[2 * mu + lam,          lam,          lam,  0,  0,  0],
                      [         lam, 2 * mu + lam,          lam,  0,  0,  0],
                      [         lam,          lam, 2 * mu + lam,  0,  0,  0],
                      [           0,            0,            0, mu,  0,  0],
                      [           0,            0,            0,  0, mu,  0],
                      [           0,            0,            0,  0,  0, mu]])

        self.cell2 = µ.Cell(self.nb_grid_pts,
                            self.lengths,
                            self.formulation)
        self.mat2 = µ.material.MaterialLinearElasticGeneric1_3d.make(
            self.cell2, "material", C)
        self.matO2 = µ.material.MaterialLinearElastic1_3d.make(
            self.cell2, "material", 2* self.Young, self.Poisson)

    def test_equivalence(self):
        sym = lambda x: .5*(x + x.T)
        Del0 = sym((np.random.random((self.dim, self.dim))-.5)/10)
        for pix_id, pixel in self.cell1.pixels.enumerate():
            if pixel[0] == 0:
                self.matO1.add_pixel(pix_id)
                self.matO2.add_pixel(pix_id)
            else:
                self.mat1.add_pixel(pix_id)
                self.mat2.add_pixel(pix_id)

        tol = 1e-6
        equil_tol = tol
        maxiter = 100
        verbose = µ.Verbosity.Silent

        solver1 = µ.solvers.KrylovSolverCG(self.cell1, tol, maxiter, verbose)
        solver2 = µ.solvers.KrylovSolverCG(self.cell2, tol, maxiter, verbose)


        r1 = µ.solvers.de_geus(self.cell1, Del0,
                               solver1, tol, equil_tol, verbose)


        r2 = µ.solvers.de_geus(self.cell2, Del0,
                               solver2, tol, equil_tol, verbose)

        error = (np.linalg.norm(r1.stress - r2.stress) /
                 np.linalg.norm(r1.stress + r2.stress))

        self.assertLess(error, 1e-13)


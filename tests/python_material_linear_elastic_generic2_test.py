#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_material_linear_elastic_generic2_test.py

@author Till Junge <till.junge@epfl.ch>

@date   20 Dec 2018

@brief  tests the bindings for the generic linear law with eigenstrains

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


class MaterialLinearElasticGeneric2_Check(unittest.TestCase):
    def setUp(self):
        self.nb_grid_pts = [3, 3]  # [5,7]
        self.lengths = [3., 3.]  # [5.2, 8.3]
        self.formulation = µ.Formulation.small_strain
        self.cell1 = µ.Cell(self.nb_grid_pts,
                            self.lengths,
                            self.formulation)
        self.cell2 = µ.Cell(self.nb_grid_pts,
                            self.lengths,
                            self.formulation)
        E, nu = 210e9, .33
        lam, mu = E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))

        C = np.array([[2 * mu + lam,          lam,            0],
                      [lam, 2 * mu + lam,            0],
                      [0,            0,           mu]])

        self.mat1 = µ.material.MaterialLinearElasticGeneric1_2d.make(
            self.cell1, "simple", C)
        self.mat2 = µ.material.MaterialLinearElasticGeneric2_2d.make(
            self.cell2, "eigen", C)
        self.mat3 = µ.material.MaterialLinearElastic2_2d.make(
            self.cell2, "eigen2", E, nu)

    def test_solve(self):
        verbose_test = False
        if verbose_test:
            print("start test_solve")
        grad = np.array([[1.1,  .2],
                         [.3, 1.5]])
        gl_strain = -0.5*(grad.T.dot(grad) - np.eye(2))
        gl_strain = -0.5*(grad.T + grad - 2*np.eye(2))
        grad = -gl_strain
        if verbose_test:
            print("grad =\n{}\ngl_strain =\n{}".format(grad, gl_strain))
        for pix_id in self.cell1.pixel_indices:
            self.mat1.add_pixel(pix_id)
            self.mat2.add_pixel(pix_id, gl_strain)
        self.cell1.initialise()
        self.cell2.initialise()
        tol = 1e-6
        Del0_1 = grad
        Del0_2 = np.zeros_like(grad)
        maxiter = 2
        verbose = µ.Verbosity.Silent

        def solve(cell, grad):
            solver =µ.solvers.KrylovSolverCG(cell, tol, maxiter, verbose)
            r = µ.solvers.newton_cg(cell, grad,
                                    solver, tol, tol, verbose)
            return r
        results = [solve(cell, del0) for (cell, del0)
                   in zip((self.cell1, self.cell2),
                          (Del0_1, Del0_2))]
        P1 = results[0].stress
        P2 = results[1].stress
        error = np.linalg.norm(P1-P2)/np.linalg.norm(.5*(P1+P2))

        if verbose_test:
            print("cell 1, no eigenstrain")
            print("P1:\n{}".format(P1[:, 0]))
            print("F1:\n{}".format(results[0].grad[:, 0]))

            print("cell 2, with eigenstrain")
            print("P2:\n{}".format(P2[:, 0]))
            print("F2:\n{}".format(results[1].grad[:, 0]))
            print("end test_solve")
        self.assertLess(error, tol)


if __name__ == '__main__':
    unittest.main()

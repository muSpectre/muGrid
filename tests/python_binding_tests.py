#!/usr/bin/env python3
"""
file   python_binding_tests.py

@author Till Junge <till.junge@epfl.ch>

@date   09 Jan 2018

@brief  Unit tests for python bindings

@section LICENCE

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

from python_fft_tests import FFT_Check
from python_projection_tests import *
from python_material_linear_elastic3_test import MaterialLinearElastic3_Check
from python_material_linear_elastic4_test import MaterialLinearElastic4_Check

class CellCheck(unittest.TestCase):
    def test_Construction(self):
        """
        Simple check for cell constructors
        """
        resolution = [5,7]
        lengths = [5.2, 8.3]
        formulation = µ.Formulation.small_strain
        try:
            sys = µ.Cell(resolution,
                         lengths,
                         formulation)
            mat = µ.material.MaterialLinearElastic1_2d.make(sys, "material", 210e9, .33)
        except Exception as err:
            print(err)
            raise err

class MaterialLinearElastic1_2dCheck(unittest.TestCase):
    def setUp(self):
        self.resolution = [5,7]
        self.lengths = [5.2, 8.3]
        self.formulation = µ.Formulation.small_strain
        self.sys = µ.Cell(self.resolution,
                          self.lengths,
                          self.formulation)
        self.mat = µ.material.MaterialLinearElastic1_2d.make(
            self.sys, "material", 210e9, .33)

    def test_add_material(self):
        self.mat.add_pixel([2,1])


class SolverCheck(unittest.TestCase):
    def setUp(self):
        self.resolution = [3, 3]#[5,7]
        self.lengths = [3., 3.]#[5.2, 8.3]
        self.formulation = µ.Formulation.finite_strain
        self.sys = µ.Cell(self.resolution,
                          self.lengths,
                          self.formulation)
        self.hard = µ.material.MaterialLinearElastic1_2d.make(
            self.sys, "hard", 210e9, .33)
        self.soft = µ.material.MaterialLinearElastic1_2d.make(
            self.sys, "soft",  70e9, .33)

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

        solver=µ.solvers.SolverCG(self.sys, tol, maxiter, verbose)
        r = µ.solvers.de_geus(self.sys, Del0,
                              solver,tol, verbose)
        #print(r)


class EigenStrainCheck(unittest.TestCase):
    def setUp(self):
        self.resolution = [3, 3]#[5,7]
        self.lengths = [3., 3.]#[5.2, 8.3]
        self.formulation = µ.Formulation.small_strain
        self.cell1 = µ.Cell(self.resolution,
                            self.lengths,
                            self.formulation)
        self.cell2 = µ.Cell(self.resolution,
                            self.lengths,
                            self.formulation)
        self.mat1 = µ.material.MaterialLinearElastic1_2d.make(
            self.cell1, "simple", 210e9, .33)
        self.mat2 = µ.material.MaterialLinearElastic2_2d.make(
            self.cell2, "eigen", 210e9, .33)

    def test_solve(self):
        verbose_test = False
        if verbose_test:
            print("start test_solve")
        grad = np.array([[1.1,  .2],
                         [ .3, 1.5]])
        gl_strain = -0.5*(grad.T.dot(grad) - np.eye(2))
        gl_strain = -0.5*(grad.T + grad - 2*np.eye(2))
        grad = -gl_strain
        if verbose_test:
            print("grad =\n{}\ngl_strain =\n{}".format(grad, gl_strain))
        for i, pixel in enumerate(self.cell1):
            self.mat1.add_pixel(pixel)
            self.mat2.add_pixel(pixel, gl_strain)
        self.cell1.initialise()
        self.cell2.initialise()
        tol = 1e-6
        Del0_1 = grad
        Del0_2 = np.zeros_like(grad)
        maxiter = 2
        verbose = 0

        def solve(cell, grad):
            solver=µ.solvers.SolverCG(cell, tol, maxiter, verbose)
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
            print("P1:\n{}".format(P1[:,0]))
            print("F1:\n{}".format(results[0].grad[:,0]))

            print("cell 2, with eigenstrain")
            print("P2:\n{}".format(P2[:,0]))
            print("F2:\n{}".format(results[1].grad[:,0]))
            print("end test_solve")
        self.assertLess(error, tol)





if __name__ == '__main__':
    unittest.main()

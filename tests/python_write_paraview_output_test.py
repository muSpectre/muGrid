#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_write_paraview_output_test.py

@author Ali Falsafi <ali.falsafi@epfl.ch>

@date   22 Nov 2019

@brief  Unit tests for write_2d and wrtie_3d functions

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

import os
import unittest
import numpy as np
from python_test_imports import muSpectre as msp
from muFFT import Stencils2D
from muFFT import Stencils3D


class Write2DCheck(unittest.TestCase):
    def setUp(self):
        self.nb_grid_pts = [3, 3]
        self.lengths = [1., 1.]
        self.formulation = msp.Formulation.small_strain
        self.gradient_operator = Stencils2D.linear_finite_elements
        self.cell = msp.Cell(self.nb_grid_pts,
                             self.lengths,
                             self.formulation,
                             self.gradient_operator)
        self.young = 2.0e10
        self.poisson = 0.3
        self.mean_strain = 1.2e-1
        self.elastic1 = msp.material.MaterialLinearElastic1_2d.make(
            self.cell,
            "Elastic1",
            self.young,
            self.poisson)
        self.elastic2 = msp.material.MaterialLinearElastic1_2d.make(
            self.cell,
            "Elastic2",
            self.young / 10.0,
            self.poisson)

    def test_write(self):
        for pix_id, (pix_x, pix_y) in enumerate(self.cell.pixels):
            if pix_id < 3:
                self.elastic1.add_pixel(pix_id)
            else:
                self.elastic2.add_pixel(pix_id)

        cg_tol = 1e-8
        newton_tol = 1e-8
        equil_tol = 0.
        Del0 = np.array([[0, 0.],
                         [0, self.mean_strain]])
        maxiter = 100
        verbose = msp.Verbosity.Silent

        krylov_solver = msp.solvers.KrylovSolverCG(
            self.cell, cg_tol, maxiter, verbose)

        res = msp.solvers.newton_cg(self.cell, Del0, krylov_solver,
                                    newton_tol, equil_tol, verbose)

        file_name = "test_write_2d"
        msp.linear_finite_elements.write_2d(
            file_name + ".xdmf", self.cell)
        if os.path.exists(file_name + ".xdmf"):
            os.remove(file_name + ".xdmf")
        if os.path.exists(file_name + ".h5"):
            os.remove(file_name + ".h5")


class WriteClassSolver2DCheck(unittest.TestCase):
    def setUp(self):
        self.nb_grid_pts = [3, 3]
        self.lengths = [1., 1.]
        self.formulation = msp.Formulation.small_strain
        self.gradient_operator = Stencils2D.linear_finite_elements
        self.cell = msp.cell.CellData.make(self.nb_grid_pts, self.lengths)
        self.cell.nb_quad_pts = 2
        self.young = 2.0e10
        self.poisson = 0.3
        self.mean_strain = 1.2e-1
        self.elastic1 = msp.material.MaterialLinearElastic1_2d.make(
            self.cell,
            "Elastic1",
            self.young,
            self.poisson)
        self.elastic2 = msp.material.MaterialLinearElastic1_2d.make(
            self.cell,
            "Elastic2",
            self.young / 10.0,
            self.poisson)

    def test_write(self):
        for pix_id, (pix_x, pix_y) in enumerate(self.cell.pixels):
            if pix_id < 3:
                self.elastic1.add_pixel(pix_id)
            else:
                self.elastic2.add_pixel(pix_id)
        cg_tol = 1e-8
        newton_tol = 1e-8
        equil_tol = 0.
        Del0 = np.array([[0, 0.],
                         [0,  self.mean_strain]])
        maxiter = 100
        verbose = msp.Verbosity.Silent

        krylov_solver = msp.solvers.KrylovSolverCG(
            cg_tol, maxiter, verbose)

        newton_solver = msp.solvers.SolverNewtonCG(
            self.cell, krylov_solver,
            verbose, newton_tol,
            equil_tol, maxiter,
            gradient=self.gradient_operator)
        newton_solver.formulation = self.formulation
        newton_solver.initialise_cell()
        res = newton_solver.solve_load_increment(Del0)
        grad = res.grad

        file_name = "test_write_2d_class"
        msp.linear_finite_elements.write_2d_class(
            file_name + ".xdmf", self.cell, newton_solver)
        if os.path.exists(file_name + ".xdmf"):
            os.remove(file_name + ".xdmf")
        if os.path.exists(file_name + ".h5"):
            os.remove(file_name + ".h5")


class Write3DCheck(unittest.TestCase):
    def setUp(self):
        self.nb_grid_pts = [3, 3, 3]
        self.lengths = [1., 1., 1.]
        self.formulation = msp.Formulation.finite_strain
        self.gradient_operator = Stencils3D.linear_finite_elements
        self.cell = msp.Cell(self.nb_grid_pts,
                             self.lengths,
                             self.formulation,
                             self.gradient_operator)
        self.young = 2.0e10
        self.poisson = 0.3
        self.mean_strain = 1.2e-1
        self.elastic1 = msp.material.MaterialLinearElastic1_3d.make(
            self.cell,
            "Elastic1",
            self.young,
            self.poisson)
        self.elastic2 = msp.material.MaterialLinearElastic1_3d.make(
            self.cell,
            "Elastic2",
            self.young / 10.0,
            self.poisson)

    def test_write(self):
        for pix_id, (pix_x, pix_y, pix_z) in enumerate(self.cell.pixels):
            if pix_id < 3:
                self.elastic1.add_pixel(pix_id)
            else:
                self.elastic2.add_pixel(pix_id)

        cg_tol = 1e-8
        newton_tol = 1e-8
        equil_tol = 0.
        Del0 = np.array([[0, 0., 0.],
                         [0, 0., 0.],
                         [0, 0., self.mean_strain]])
        maxiter = 100
        verbose = msp.Verbosity.Silent

        krylov_solver = msp.solvers.KrylovSolverCG(
            self.cell, cg_tol, maxiter, verbose)

        res = msp.solvers.newton_cg(self.cell, Del0, krylov_solver,
                                    newton_tol, equil_tol, verbose)

        file_name = "test_write_3d"
        msp.linear_finite_elements.write_3d(
            file_name + ".xdmf", self.cell)
        if os.path.exists(file_name + ".xdmf"):
            os.remove(file_name + ".xdmf")
        if os.path.exists(file_name + ".h5"):
            os.remove(file_name + ".h5")


class WriteClassSolver3DCheck(unittest.TestCase):
    def setUp(self):
        self.nb_grid_pts = [3, 3, 3]
        self.lengths = [1., 1., 1.]
        self.formulation = msp.Formulation.finite_strain
        self.gradient_operator = Stencils3D.linear_finite_elements
        self.cell = msp.cell.CellData.make(self.nb_grid_pts, self.lengths)
        self.cell.nb_quad_pts = 6
        self.young = 2.0e10
        self.poisson = 0.3
        self.mean_strain = 1.2e-1
        self.elastic1 = msp.material.MaterialLinearElastic1_3d.make(
            self.cell,
            "Elastic1",
            self.young,
            self.poisson)
        self.elastic2 = msp.material.MaterialLinearElastic1_3d.make(
            self.cell,
            "Elastic2",
            self.young / 10.0,
            self.poisson)

    def test_write(self):
        for pix_id, (pix_x, pix_y, pix_z) in enumerate(self.cell.pixels):
            if pix_id < 3:
                self.elastic1.add_pixel(pix_id)
            else:
                self.elastic2.add_pixel(pix_id)
        cg_tol = 1e-8
        newton_tol = 1e-8
        equil_tol = 0.
        Del0 = np.array([[0, 0., 0.],
                         [0, 0., 0.],
                         [0, 0., self.mean_strain]])
        maxiter = 100
        verbose = msp.Verbosity.Silent

        krylov_solver = msp.solvers.KrylovSolverCG(
            cg_tol, maxiter, verbose)

        newton_solver = msp.solvers.SolverNewtonCG(
            self.cell, krylov_solver,
            verbose, newton_tol,
            equil_tol, maxiter,
            gradient=self.gradient_operator)
        newton_solver.formulation = self.formulation
        newton_solver.initialise_cell()
        res = newton_solver.solve_load_increment(Del0)
        grad = res.grad

        file_name = "test_write_3d_class"
        msp.linear_finite_elements.write_3d_class(
            file_name + ".xdmf", self.cell, newton_solver)
        if os.path.exists(file_name + ".xdmf"):
            os.remove(file_name + ".xdmf")
        if os.path.exists(file_name + ".h5"):
            os.remove(file_name + ".h5")

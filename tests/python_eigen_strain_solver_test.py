#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_eigen_strain_solver_test.py

@author Ali Falsafi <ali.falsafi@epfl.ch>

@date   28 May 2020

@brief  Testing handling eigen strain by solver in python bindings

Copyright © 2020 Ali Falsafi

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


def eigen_strain_func(step_nb):
    ndim = 2   # number of dimensions
    N = 3  # number of voxels (assumed equal for all directions)
    Nx = Ny = N
    # identity tensor                  [single tensor]
    i = np.asfortranarray(np.zeros([ndim, ndim]))

    # identity tensors
    identities = np.asfortranarray(
        np.einsum('ij,pxy', i, np.ones([1, N, N])))

    eigen = np.array([[1e-3, 1e-4],
                      [1e-4, 2e-3]])
    eigens = identities
    eigens[:, :, 0, 1, 1] = eigen
    for i in range(3):
        for j in range(2):
            print("\ni: {}, j: {}\n".format(i, j))
            print(eigens[:, :, 0, i, j])

    return eigens


class EigenStrainSolverCheck(unittest.TestCase):
    def setUp(self):
        self.nb_grid_pts = [3, 3]  # [5,7]
        self.lengths = [3., 3.]  # [5.2, 8.3]
        self.formulation = µ.Formulation.small_strain
        self.cell_material = µ.Cell(self.nb_grid_pts,
                                    self.lengths,
                                    self.formulation)
        self.cell_solver = µ.Cell(self.nb_grid_pts,
                                  self.lengths,
                                  self.formulation)

        self.material_1_material = µ.material.MaterialLinearElastic1_2d.make(
            self.cell_material, "material 1 material", 210e9, .33)
        self.material_2_material = µ.material.MaterialLinearElastic2_2d.make(
            self.cell_material, "material 2 material", 210e9, .33)

        self.material_1_solver = µ.material.MaterialLinearElastic1_2d.make(
            self.cell_solver, "material 1 solver", 210e9, .33)
        self.ndim = 2   # number of dimensions
        self.N = 3  # number of voxels (assumed equal for all directions)
        self.Nx = self.Ny = self.N
        # identity tensor                  [single tensor]
        self.i = np.asfortranarray(np.eye(self.ndim))

        # identity tensors
        self.identities = np.asfortranarray(
            np.einsum('ij,pxy', self.i, np.ones([1, self.N, self.N])))

        self.eigen = np.array([[1e-3, 1e-4],
                               [1e-4, 2e-3]])
        self.eigens = self.identities
        self.eigens[:, :, 0, 1, 1] = self.eigen

    def test_eigen_strain_solve(self):
        for pix_id in self.cell_material.pixel_indices:
            if pix_id == 4:
                self.material_2_material.add_pixel(pix_id, self.eigen)
            else:
                self.material_1_material.add_pixel(pix_id)

        for pix_id in self.cell_material.pixel_indices:
            self.material_1_solver.add_pixel(pix_id)

        self.cell_material.initialise()
        self.cell_solver.initialise()

        cg_tol = 1e-8
        newton_tol = 1e-6
        equil_tol = 0.
        Del0 = np.array([[2.5e-3, 1.00e-3],
                         [1.0e-3, 0.00e-3]])
        maxiter = 100
        verbose = µ.Verbosity.Silent

        solver_solver = µ.solvers.KrylovSolverCG(
            self.cell_solver, cg_tol, maxiter, verbose)

        solver_material = µ.solvers.KrylovSolverCG(
            self.cell_material, cg_tol, maxiter, verbose)

        r_solver = µ.solvers.newton_cg(self.cell_solver, Del0,
                                       solver_solver,
                                       newton_tol, equil_tol, verbose,
                                       μ.solvers.IsStrainInitialised.No,
                                       eigen_strain_func)

        r_material = µ.solvers.newton_cg(self.cell_material, Del0,
                                         solver_material,
                                         newton_tol, equil_tol, verbose)
        self.assertTrue((r_material.stress == r_solver.stress).all())
        # self.assertTrue((r_material.grad == r_solver.grad).all())
        # print(r_material.grad-r_solver.grad)


if __name__ == '__main__':
    unittest.main()

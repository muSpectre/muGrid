#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_mpi_gradient_integration_test.py

@author Richard Leute <richard.leute@imtek.uni-freiburg.de>

@date   20 Dez 2021

@brief  tests for parralel gradient integration functions

Copyright © 2021 Till Junge

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

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import unittest
import numpy as np
import itertools
import time

from python_test_imports import µ


def init_rve_mat_solver(dim, communicator, nb_domain_grid_pts, domain_lengths,
                        formulation, fft, Youngs_modulus, Poisson_ratio,
                        cg_tol, maxiter, verbose):
    if dim == 2:
        gradient = µ.linear_finite_elements.gradient_2d
    elif dim == 3:
        gradient = µ.linear_finite_elements.gradient_3d

    rve = µ.Cell(nb_domain_grid_pts[:dim], domain_lengths[:dim],
                 formulation, gradient, fft, communicator)

    if dim == 2:
        material = µ.material.MaterialLinearElastic1_2d.make(
            rve, "material", Youngs_modulus, Poisson_ratio)
    elif dim == 3:
        material = µ.material.MaterialLinearElastic1_3d.make(
            rve, "material", Youngs_modulus, Poisson_ratio)

    for pixel_index, pixel in enumerate(rve.pixels):
        material.add_pixel(pixel_index)

    solver = µ.solvers.KrylovSolverCG(
        rve, cg_tol, maxiter, verbose)

    return rve, material, solver


class GradientIntegrationCheck(unittest.TestCase):
    """
    Check the implementation of parallel muSpectre.gradient_integration
    functions.
    """

    def setUp(self):
        self.comm = MPI.COMM_WORLD
        self.fft = "mpi"

        self.nb_domain_grid_pts = np.array([5, 3, 7])
        self.domain_lengths = np.array([2.4, 3.7, 4.1])
        self.dim = len(self.nb_domain_grid_pts)

        self.Youngs_modulus = 1
        self.Poisson_ratio = 0.33
        self.newton_tol = 1e-6
        self.equil_tol = self.newton_tol
        self.cg_tol = 1e-6

        self.maxiter = 1000
        self.verbose = µ.Verbosity.Silent

        # set timing = True for timing information
        self.timing = False
        self.startTime = time.time()

    def tearDown(self):
        if self.timing:
            t = time.time() - self.startTime
            print("{}:\n{:.3f} seconds".format(self.id(), t))

    def test_get_complemented_positions(self):
        nx, ny, nz = self.nb_domain_grid_pts
        sx, sy, sz = self.domain_lengths

        s = 0.01
        strain_step_all = np.array([[-s, 0, 0], [0, -s, 0], [0, 0, 2*s]])

        for form, dim, pc in itertools.product(
                ["small_strain", "finite_strain"], [2, 3], [True, False]):

            if form == "small_strain":
                formulation = µ.Formulation.small_strain
            elif form == "finite_strain":
                formulation = µ.Formulation.finite_strain

            rve, mat, solver = \
                init_rve_mat_solver(dim, self.comm,
                                    self.nb_domain_grid_pts,
                                    self.domain_lengths, formulation,
                                    self.fft, self.Youngs_modulus,
                                    self.Poisson_ratio, self.cg_tol,
                                    self.maxiter, self.verbose)

            if dim == 2:
                strain_step = np.copy(strain_step_all[:dim, :dim])
            elif dim == 3:
                strain_step = np.copy(strain_step_all)

            µ.solvers.newton_cg(rve, strain_step, solver,
                                self.newton_tol, self.equil_tol,
                                self.verbose,
                                µ.solvers.IsStrainInitialised.No)

            if self.comm.size > 1 and pc:
                # catch RuntimeError and check message
                error_msg = ("Periodically complemented output quantities are "
                             "currently not implemented in the parallel "
                             "version. Either compute serial or use the "
                             "periodically not complemented quantities.")
                self.assertRaisesRegex(
                    RuntimeError, error_msg,
                    µ.gradient_integration.get_complemented_positions,
                    "pdg0n", rve, periodically_complemented=pc)

                # skip the other tests
                continue

            placements, nodal_displ, affine_deformed_grid, \
                grid, non_affine_displ \
                = µ.gradient_integration.get_complemented_positions(
                    "pdg0n", rve, periodically_complemented=pc)

            if dim == 2:
                nxl, nyl = rve.nb_subdomain_grid_pts
                ox, oy = rve.subdomain_locations
                if not pc:
                    pc_l = pc
                elif self.comm.size == 1:
                    pc_l = pc
                else:
                    # find out the last non empty rank
                    pc_l = False
                    if self.comm.rank == self.comm.size - 1:
                        pc_l = pc  # pc_l should be only true if pc is true
                                   # and we are on the last non empty rank.
                                   # For simplicity no empty cores and FFTW are
                                   # assumed. If there are empty cores or you
                                   # use PFFT instead of FFTW you will have to
                                   # adopt the test.

                # run 2D tests
                # placements (node positions)
                self.assertTrue(
                    np.allclose(placements[0],
                                sx/nx * (1-s)*np.arange(nx+(1 if pc else 0))
                                .reshape((-1, 1))))
                self.assertTrue(
                    np.allclose(placements[1], sy/ny * (1-s)
                                * np.arange(oy, oy+nyl+(1 if pc_l else 0))
                                .reshape((1, -1))))

                # grid positions (including applied homogeneous strain)
                self.assertTrue(
                    np.allclose(affine_deformed_grid[0],
                                sx/nx * (1-s)*np.arange(nx+(1 if pc else 0))
                                .reshape((-1, 1))))
                self.assertTrue(
                    np.allclose(affine_deformed_grid[1], sy/ny * (1-s)
                                * np.arange(oy, oy+nyl+(1 if pc_l else 0))
                                .reshape((1, -1))))

                # grid positions (without applied homogeneous strain)
                self.assertTrue(
                    np.allclose(grid[0],
                                sx/nx * np.arange(nx+(1 if pc else 0))
                                .reshape((-1, 1))))
                self.assertTrue(
                    np.allclose(grid[1], sy/ny
                                * np.arange(oy, oy+nyl+(1 if pc_l else 0))
                                .reshape((1, -1))))

                # nodal displacements
                self.assertTrue(
                    np.allclose(placements[0] - grid[0], nodal_displ[0]))
                self.assertTrue(
                    np.allclose(placements[1] - grid[1], nodal_displ[1]))

                # nodal nonaffine displacements
                # (without homogeneous displacement field)
                self.assertTrue(
                    np.allclose(placements[0] - affine_deformed_grid[0],
                                non_affine_displ[0]))
                self.assertTrue(
                    np.allclose(placements[1] - affine_deformed_grid[1],
                                non_affine_displ[1]))

            elif dim == 3:
                nxl, nyl, nzl = rve.nb_subdomain_grid_pts
                ox, oy, oz = rve.subdomain_locations
                if not pc:
                    pc_l = pc
                elif self.comm.size == 1:
                    pc_l = pc
                else:
                    # find out the last non empty rank
                    pc_l = False
                    if self.comm.rank == self.comm.size - 1:
                        pc_l = pc  # pc_l should be only true if pc is true
                                   # and we are on the last non empty rank.
                                   # For simplicity no empty cores and FFTW are
                                   # assumed. If there are empty cores or you
                                   # use PFFT instead of FFTW you will have to
                                   # adopt the test.

                # run 3D tests
                # placements (node positions)
                self.assertTrue(
                    np.allclose(placements[0],
                                sx/nx * (1-s)*np.arange(nx+(1 if pc else 0))
                                .reshape((-1, 1, 1))))
                self.assertTrue(
                    np.allclose(placements[1],
                                sy/ny * (1-s)*np.arange(ny+(1 if pc else 0))
                                .reshape((1, -1, 1))))
                self.assertTrue(
                    np.allclose(placements[2], sz/nz * (1+2*s)
                                * np.arange(oz, oz+nzl+(1 if pc_l else 0))
                                .reshape(1, 1, -1)))

                # grid positions (including applied homogeneous strain)
                self.assertTrue(
                    np.allclose(affine_deformed_grid[0], sx/nx * (1-s)
                                * np.arange(nx+(1 if pc else 0))
                                .reshape((-1, 1, 1))))
                self.assertTrue(
                    np.allclose(affine_deformed_grid[1],
                                sy/ny * (1-s)*np.arange(ny+(1 if pc else 0))
                                .reshape((1, -1, 1))))
                self.assertTrue(
                    np.allclose(affine_deformed_grid[2], sz/nz * (1+2*s)
                                * np.arange(oz, oz+nzl+(1 if pc_l else 0))
                                .reshape(1, 1, -1)))

                # grid positions (without applied homogeneous strain)
                self.assertTrue(
                    np.allclose(grid[0],
                                sx/nx * np.arange(nx+(1 if pc else 0))
                                .reshape((-1, 1, 1))))
                self.assertTrue(
                    np.allclose(grid[1],
                                sy/ny * np.arange(ny+(1 if pc else 0))
                                .reshape((1, -1, 1))))
                self.assertTrue(
                    np.allclose(grid[2], sz/nz
                                * np.arange(oz, oz+nzl+(1 if pc_l else 0))
                                .reshape(1, 1, -1)))

                # nodal displacements
                self.assertTrue(
                    np.allclose(placements[0] - grid[0], nodal_displ[0]))
                self.assertTrue(
                    np.allclose(placements[1] - grid[1], nodal_displ[1]))
                self.assertTrue(
                    np.allclose(placements[2] - grid[2], nodal_displ[2]))

                # nodal nonaffine displacements
                # (without homogeneous displacement field)
                self.assertTrue(
                    np.allclose(placements[0] - affine_deformed_grid[0],
                                non_affine_displ[0]))
                self.assertTrue(
                    np.allclose(placements[1] - affine_deformed_grid[1],
                                non_affine_displ[1]))
                self.assertTrue(
                    np.allclose(placements[2] - affine_deformed_grid[2],
                                non_affine_displ[2]))


if __name__ == '__main__':
    unittest.main()

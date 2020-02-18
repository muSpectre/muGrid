#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_mpi_material_linear_elastic4_test.py

@author Richard Leute <richard.leute@imtek.uni-freiburg.de>

@date   27 Mar 2018

@brief  test MPI-parallel linear elastic material

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

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import unittest
import numpy as np

from python_test_imports import µ, muFFT


def build_test_classes(fft):
    class MaterialLinearElastic4_Check(unittest.TestCase):
        """
        Check the implementation of storing the first and second Lame constant in
        each cell. Assign the same Youngs modulus and Poisson ratio to each cell,
        from which the two Lame constants are internally computed. Then calculate
        the stress and compare the result with stress=2*mu*Del0 (Hooke law for small
        symmetric strains).
        """
        def setUp(self):
            self.nb_grid_pts = [7,7]
            self.lengths = [2.3, 3.9]
            self.formulation = µ.Formulation.small_strain
            self.cell = µ.Cell(self.nb_grid_pts,
                              self.lengths,
                              self.formulation,
                              fft=fft,
                              communicator=MPI.COMM_WORLD)
            self.mat = µ.material.MaterialLinearElastic4_2d.make(
                self.cell, "material")

        def test_decomposition(self):
            self.assertEqual(
                self.cell.communicator.sum(np.prod(self.cell.nb_subdomain_grid_pts)),
                np.prod(self.cell.nb_domain_grid_pts),
                msg='{} engine'.format(fft))

        def test_solver(self):
            Youngs_modulus = 10.
            Poisson_ratio  = 0.3

            for i, pixel in enumerate(self.cell):
                self.mat.add_pixel(pixel, Youngs_modulus, Poisson_ratio)

            self.cell.initialise()
            tol = 1e-6
            Del0 = np.array([[0, 0.025],
                             [0.025,  0]])
            maxiter = 100
            verbose = µ.solvers.Verbosity.Some

            solver = µ.solvers.KrylovSolverCG(self.cell, tol, maxiter, verbose)
            r = µ.solvers.newton_cg(self.cell, Del0,
                                    solver, tol, tol, verbose)

            #compare the computed stress with the trivial by hand computed stress
            mu = (Youngs_modulus/(2*(1+Poisson_ratio)))
            stress = 2*mu*Del0

            self.assertLess(np.linalg.norm(r.stress-stress.reshape(-1,1)), 1e-8)

    return MaterialLinearElastic4_Check

linear_elastic4 = {}
for fft, is_transposed, is_parallel in muFFT.fft_engines:
    if is_parallel:
        linear_elastic4[fft] = build_test_classes(fft)

if __name__ == "__main__":
    unittest.main()

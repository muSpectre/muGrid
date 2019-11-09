#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_mpi_projection_tests.py

@author Till Junge <till.junge@altermail.ch>

@date   18 Jan 2018

@brief  compare µSpectre's MPI-parallel projection operators to GooseFFT

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
import itertools

from mpi4py import MPI

from python_test_imports import µ
from python_goose_ref import (SmallStrainProjectionGooseFFT,
                              FiniteStrainProjectionGooseFFT)
from muFFT import fft_engines, Communicator
from muSpectre import Formulation


def build_test_classes(formulation, RefProjection, fft):
    class ProjectionCheck(unittest.TestCase):
        def __init__(self, methodName='runTest'):
            super().__init__(methodName)

        def setUp(self):
            self.ref = RefProjection
            self.nb_grid_pts = self.ref.nb_grid_pts
            self.ndim = self.ref.ndim
            self.shape = list((self.nb_grid_pts for _ in range(self.ndim)))
            self.projection = Projection(self.shape, self.shape, formulation,
                                         fft, MPI.COMM_WORLD)
            self.projection.initialise()
            self.tol = 1e-12 * np.prod(self.shape)

        def test_constructor(self):
            """Check that engines can be initialized with either bare MPI
            communicator or muFFT communicators"""
            projection = Projection(self.shape, self.shape, formulation,
                                    fft, Communicator(MPI.COMM_WORLD))


        def test_projection_result(self):
            # create a bogus strain field in GooseFFT format
            # dim × dim × N × N (× N)
            strain_shape = (self.ndim, self.ndim, *self.shape)
            strain = np.arange(np.prod(strain_shape)).reshape(strain_shape)
            # if we're testing small strain projections, it needs to be symmetric
            if self.projection.get_formulation() == µ.Formulation.small_strain:
                strain += strain.transpose(1, 0, *range(2, len(strain.shape)))
            strain_g = strain.copy()
            b_g = self.ref.G(strain_g).reshape(strain_g.shape)
            strain_µ = np.zeros((*self.shape, self.ndim, self.ndim))
            for ijk in itertools.product(range(self.nb_grid_pts), repeat=self.ndim):
                index_µ = tuple((*ijk, slice(None), slice(None)))
                index_g = tuple((slice(None), slice(None), *ijk))
                strain_µ[index_µ] = strain_g[index_g].T
            res = self.projection.get_nb_subdomain_grid_pts()
            loc = self.projection.get_subdomain_locations()
            if self.ref.ndim == 2:
                resx, resy = res
                locx, locy = loc
                subdomain_strain_µ = strain_µ[locx:locx+resx, locy:locy+resy]
            else:
                resx, resy, resz = res
                locx, locy, locz = loc
                subdomain_strain_µ = strain_µ[locx:locx+resx, locy:locy+resy,
                                              locz:locz+resz]
            b_µ = self.projection.apply_projection(subdomain_strain_µ.reshape(
                np.prod(res), self.ndim**2).T).T.reshape(subdomain_strain_µ.shape)
            for l in range(np.prod(res)):
                ijk = µ.get_domain_ccoord(res, l)
                index_µ = tuple((*ijk, slice(None), slice(None)))
                ijk = loc + np.array(ijk)
                index_g = tuple((slice(None), slice(None), *ijk))
                b_µ_sl = b_µ[index_µ].T
                b_g_sl = b_g[index_g]
                error = np.linalg.norm(b_µ_sl - b_g_sl)
                condition = error < self.tol
                slice_printer = lambda tup: "({})".format(
                    ", ".join("{}".format(":" if val == slice(None) else val) for val in tup))
                if not condition:
                    print("error = {}, tol = {}".format(error, self.tol))
                    print("b_µ{} =\n{}".format(slice_printer(index_µ), b_µ_sl))
                    print("b_g{} =\n{}".format(slice_printer(index_g), b_g_sl))
                self.assertTrue(condition)

    return ProjectionCheck

get_goose = lambda ndim, proj_type: proj_type(
    ndim, 5, 2, 70e9, .33, 3.)
get_finite_goose = lambda ndim: get_goose(ndim, FiniteStrainProjectionGooseFFT)
get_small_goose = lambda ndim: get_goose(ndim, SmallStrainProjectionGooseFFT)

if ("fftwmpi", True) in fft_engines:
    small_default_fftwmpi_3 = build_test_classes(Formulation.small_strain,
                                                 get_small_goose(3),
                                                 "fftwmpi")
    small_default_fftwmpi_2 = build_test_classes(Formulation.small_strain,
                                                 get_small_goose(2),
                                                 "fftwmpi")

    finite_fast_fftwmpi_3 = build_test_classes(Formulation.finite_strain,
                                               get_finite_goose(3),
                                               "fftwmpi")
    finite_fast_fftwmpi_2 = build_test_classes(Formulation.finite_strain,
                                               get_finite_goose(2),
                                               "fftwmpi")

if ("pfft", True) in fft_engines:
    small_default_pfft_3 = build_test_classes(Formulation.small_strain,
                                              get_small_goose(3),
                                              "pfft")
    small_default_pfft_2 = build_test_classes(Formulation.small_strain,
                                              get_small_goose(2),
                                              "pfft")

    finite_fast_pfft_3 = build_test_classes(Formulation.finite_strain,
                                            get_finite_goose(3),
                                            "pfft")
    finite_fast_pfft_2 = build_test_classes(Formulation.finite_strain,
                                            get_finite_goose(2),
                                            "pfft")

if __name__ == "__main__":
    unittest.main()

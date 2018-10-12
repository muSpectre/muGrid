#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
file   python_projection_tests.py

@author Till Junge <till.junge@altermail.ch>

@date   18 Jan 2018

@brief  compare µSpectre's projection operators to GooseFFT

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
along with µSpectre; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.
"""

import unittest
import numpy as np
import itertools

from python_test_imports import µ
from python_goose_ref import SmallStrainProjectionGooseFFT, FiniteStrainProjectionGooseFFT
import _muSpectre

def build_test_classes(Projection, RefProjection, name):
    class ProjectionCheck(unittest.TestCase):
        def __init__(self, methodName='runTest'):
            super().__init__(methodName)
            self.__class__.__qualname__ = name

        def setUp(self):
            self.ref = RefProjection
            self.resolution = self.ref.resolution
            self.ndim = self.ref.ndim
            self.shape = list((self.resolution for _ in range(self.ndim)))
            self.projection = Projection(self.shape, self.shape)
            self.projection.initialise()
            self.tol = 1e-12*np.prod(self.shape)

        def test_CompareGhat4(self):
            # refG is rowmajor and the dims are i,j,k,l,x,y(,z)
            # reshape refG so they are n² × n² × ¶(resolution)
            refG = self.ref.Ghat4.reshape(
                self.ndim**2, self.ndim**2, np.prod(self.shape))
            # mspG is colmajor (not sure what that's worth, though) with dims
            # ijkl, xy(z)
            # reshape mspG so they are ¶(hermitian) × n² × n²
            ref_sizes = self.shape
            msp_sizes = µ.get_hermitian_sizes(self.shape)
            hermitian_size = np.prod(msp_sizes)
            mspG = self.projection.get_operator()
            #this test only makes sense for fully stored ghats (i.e.,
            #not for the faster alternative implementation
            if mspG.size != hermitian_size*self.ndim**4:
                return

            rando = np.random.random((self.ndim, self.ndim))
            for i in range(hermitian_size):
                coord = µ.get_domain_ccoord(msp_sizes, i)
                ref_id = µ.get_domain_index(ref_sizes, coord)
                msp_id = µ.get_domain_index(msp_sizes, coord)

                # story behind this order vector:
                # There was this issue with the projection operator of
                # de Geus acting on the the transpose of the gradient.
                order = np.arange(self.ndim**2).reshape(
                    self.ndim, self.ndim).T.reshape(-1)
                msp_g = mspG[:, msp_id].reshape(self.ndim**2, self.ndim**2)[order, :]
                error = np.linalg.norm(refG[:, :, ref_id] -
                                       msp_g)
                condition = error < self.tol
                if not condition:
                    print("G_µ{}, at index {} =\n{}".format(coord, msp_id, msp_g))
                    print("G_g{}, at index {} =\n{}".format(coord, ref_id, refG[:, :, ref_id]))
                self.assertTrue(condition)

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
            for ijk in itertools.product(range(self.resolution), repeat=self.ndim):
                index_µ = tuple((*ijk, slice(None), slice(None)))
                index_g = tuple((slice(None), slice(None), *ijk))
                strain_µ[index_µ] = strain_g[index_g].T

            b_µ = self.projection.apply_projection(strain_µ.reshape(
                np.prod(self.shape), self.ndim**2).T).T.reshape(strain_µ.shape)
            for ijk in itertools.product(range(self.resolution), repeat=self.ndim):
                index_µ = tuple((*ijk, slice(None), slice(None)))
                index_g = tuple((slice(None), slice(None), *ijk))
                b_µ_sl = b_µ[index_µ].T
                b_g_sl = b_g[index_g]
                error = np.linalg.norm(b_µ_sl-b_g_sl)
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
get_small_goose  = lambda ndim: get_goose(ndim,  SmallStrainProjectionGooseFFT)


small_default_3 = build_test_classes(_muSpectre.fft.ProjectionSmallStrain_3d,
                                     get_small_goose(3),
                                     "SmallStrainDefaultProjection3d")
small_default_2 = build_test_classes(_muSpectre.fft.ProjectionSmallStrain_2d,
                                     get_small_goose(2),
                                     "SmallStrainDefaultProjection2d")

finite_default_3 = build_test_classes(_muSpectre.fft.ProjectionFiniteStrain_3d,
                                      get_finite_goose(3),
                                      "FiniteStrainDefaultProjection3d")
finite_default_2 = build_test_classes(_muSpectre.fft.ProjectionFiniteStrain_2d,
                                      get_finite_goose(2),
                                      "FiniteStrainDefaultProjection2d")

finite_fast_3 = build_test_classes(_muSpectre.fft.ProjectionFiniteStrainFast_3d,
                                   get_finite_goose(3),
                                   "FiniteStrainFastProjection3d")
finite_fast_2 = build_test_classes(_muSpectre.fft.ProjectionFiniteStrainFast_2d,
                                   get_finite_goose(2),
                                   "FiniteStrainFastProjection2d")
if __name__ == "__main__":
    unittest.main()

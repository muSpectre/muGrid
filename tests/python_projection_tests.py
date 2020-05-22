#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_projection_tests.py

@author Till Junge <till.junge@altermail.ch>

@date   18 Jan 2018

@brief  compare µSpectre's projection operators to GooseFFT

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

from python_test_imports import µ, muFFT
from python_goose_ref import SmallStrainProjectionGooseFFT, FiniteStrainProjectionGooseFFT
import _muSpectre

def build_test_classes(Projection, RefProjection, name):
    class ProjectionCheck(unittest.TestCase):
        def __init__(self, methodName='runTest'):
            super().__init__(methodName)
            self.__class__.__qualname__ = name

        def setUp(self):
            self.ref = RefProjection
            self.nb_grid_pts = self.ref.nb_grid_pts
            self.ndim = self.ref.ndim
            self.shape = list((self.nb_grid_pts for _ in range(self.ndim)))
            self.fft = muFFT.FFT(self.shape)
            self.fft.initialise(self.ndim * self.ndim)
            self.projection = Projection(self.fft, [float(x) for x in self.shape])
            self.projection.initialise()
            self.tol = 1e-12*np.prod(self.shape)

        def test_CompareGhat4(self):
            # refG is rowmajor and the dims are i,j,k,l,x,y(,z)
            # reshape refG so they are n² × n² × ¶(nb_grid_pts)
            order = list(range(self.ndim+4))
            order[-self.ndim:] = reversed(order[-self.ndim:])
            refG = self.ref.Ghat4.transpose(*order).reshape(
                self.ndim**2, self.ndim**2, np.prod(self.shape))
            # mspG is colmajor (not sure what that's worth, though) with dims
            # ijkl, xy(z)
            # reshape mspG so they are ¶(hermitian) × n² × n²
            ref_sizes = self.shape
            msp_sizes = muFFT.get_nb_hermitian_grid_pts(self.shape)
            hermitian_size = np.prod(msp_sizes)
            mspG = self.projection.operator
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
            if self.projection.formulation == µ.Formulation.small_strain:
                strain += strain.transpose(1, 0, *range(2, len(strain.shape)))
            b_g = self.ref.G(strain).reshape(strain.shape)
            b_µ = self.projection.apply_projection(strain)

            assert np.allclose(b_g, b_µ)


    return ProjectionCheck

get_goose = lambda ndim, proj_type: proj_type(
    ndim, 5, 2, 70e9, .33, 3.)
get_finite_goose = lambda ndim: get_goose(ndim, FiniteStrainProjectionGooseFFT)
get_small_goose  = lambda ndim: get_goose(ndim,  SmallStrainProjectionGooseFFT)


small_default_3 = build_test_classes(_muSpectre.ProjectionSmallStrain_3d,
                                     get_small_goose(3),
                                     "SmallStrainDefaultProjection3d")
small_default_2 = build_test_classes(_muSpectre.ProjectionSmallStrain_2d,
                                     get_small_goose(2),
                                     "SmallStrainDefaultProjection2d")

finite_default_3 = build_test_classes(_muSpectre.ProjectionFiniteStrain_3d,
                                      get_finite_goose(3),
                                      "FiniteStrainDefaultProjection3d")
finite_default_2 = build_test_classes(_muSpectre.ProjectionFiniteStrain_2d,
                                      get_finite_goose(2),
                                      "FiniteStrainDefaultProjection2d")

finite_fast_3 = build_test_classes(_muSpectre.ProjectionFiniteStrainFast_3d,
                                   get_finite_goose(3),
                                   "FiniteStrainFastProjection3d")
finite_fast_2 = build_test_classes(_muSpectre.ProjectionFiniteStrainFast_2d,
                                   get_finite_goose(2),
                                   "FiniteStrainFastProjection2d")
if __name__ == "__main__":
    unittest.main()

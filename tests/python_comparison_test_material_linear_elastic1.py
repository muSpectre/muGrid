#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_comparison_test_material_linear_elastic1.py

@author Till Junge <till.junge@epfl.ch>

@date   25 Jan 2019

@brief  compares MaterialLinearElastic1 to the Geus's python implementation

Copyright © 2019 Till Junge

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

import itertools
import numpy as np
np.set_printoptions(linewidth=180)
import unittest

from python_test_imports import µ
LinMat2 = µ.material.MaterialLinearElastic1_2d
LinMat3 = µ.material.MaterialLinearElastic1_3d
LinMats = {2: LinMat2, 3: LinMat3}

#####################
dyad22 = lambda A2,B2: np.einsum('ij  ,kl  ->ijkl',A2,B2)
dyad11 = lambda A1,B1: np.einsum('i   ,j   ->ij  ',A1,B1)
dot22  = lambda A2,B2: np.einsum('ij  ,jk  ->ik  ',A2,B2)
dot24  = lambda A2,B4: np.einsum('ij  ,jkmn->ikmn',A2,B4)
dot42  = lambda A4,B2: np.einsum('ijkl,lm  ->ijkm',A4,B2)
inv2 = np.linalg.inv
trans2 = np.transpose
ddot22 = lambda A2,B2: np.einsum('ij  ,ji  ->    ',A2,B2)
ddot42 = lambda A4,B2: np.einsum('ijkl,lk  ->ij  ',A4,B2)
ddot44 = lambda A4,B4: np.einsum('ijkl,lkmn->ijmn',A4,B4)

class MatTest(unittest.TestCase):
    def constitutive(self, F, dim):
        I  = np.eye(dim)
        II = dyad22(I,I)
        I4 = np.einsum('il,jk',I,I)
        I4rt = np.einsum('ik,jl',I,I)
        I4s  = (I4+I4rt)/2.

        C4 = self.K*II+2.*self.mu*(I4s-1./3.*II)
        S  = ddot42(C4,.5*(dot22(trans2(F),F)-I))
        P  = dot22(F,S)
        K4 = dot24(S,I4)+ddot44(ddot44(I4rt,dot42(dot24(F,C4),trans2(F))),I4rt)
        return P, S, K4, C4

    def setUp(self):
        pass

    def prep(self, dimension):
        self.dim=dimension
        self.Young = 200e9+100*np.random.rand()
        self.Poisson = .3 + .1*(np.random.rand()-.5)
        self.K = self.Young/(3*(1-2*self.Poisson))
        self.mu = self.Young/(2*(1+self.Poisson))
        self.F = (np.eye(self.dim) +
                  (np.random.random((self.dim, self.dim))-.5)/10)
        self.E = .5*(self.F.T.dot(self.F)-np.eye(self.dim))
        self.tol = 1e-13
        self.verbose=True

        self.linmat, self.evaluator = LinMats[self.dim].make_evaluator(
            self.Young, self.Poisson)
        self.linmat.add_pixel(0)

    def test_equivalence_S_C(self):
        for dim in (2, 3):
            self.runner_equivalence_S_C(dim)

    def runner_equivalence_S_C(self, dimension):
        self.prep(dimension)
        S_μ, C_µ_s = self.evaluator.evaluate_stress_tangent(
            self.E, µ.Formulation.small_strain)
        S_μ = S_μ.copy()
        S_μ2 = self.evaluator.evaluate_stress(
            self.E, µ.Formulation.small_strain)
        shape = (self.dim, self.dim, self.dim, self.dim)
        C_µ = C_µ_s.reshape(shape).transpose((0,1,3,2))

        response_p = self.constitutive(self.F, self.dim)
        S_p, C_p = response_p[1], response_p[3]

        S_error = np.linalg.norm(S_µ- S_p)/np.linalg.norm(S_µ)
        if not S_error < self.tol:
            print("Error(S) = {}".format(S_error))
            print("S_µ:\n{}".format(S_µ))
            print("S_µ2:\n{}".format(S_µ2))
            print("S_p:\n{}".format(S_p))
        C_error = np.linalg.norm(C_µ- C_p)/np.linalg.norm(C_µ)
        if not C_error < self.tol:
            print("Error(C) = {}".format(C_error))
            flat_shape = (self.dim**2, self.dim**2)
            print("C_µ:\n{}".format(C_µ.reshape(flat_shape)))
            print("C_p:\n{}".format(C_p.reshape(flat_shape)))

        self.assertLess(S_error,
                        self.tol)

        self.assertLess(C_error,
                        self.tol)

    def test_equivalence_P_K(self):
        for dim in (2, 3):
            self.runner_equivalence_P_K(dim)

    def runner_equivalence_P_K(self, dimension):
        self.prep(dimension)
        P_µ, K_µ_s = self.evaluator.evaluate_stress_tangent(
            self.F, µ.Formulation.finite_strain)
        shape = (self.dim, self.dim, self.dim, self.dim)
        K_µ = K_µ_s.reshape(shape).transpose((0,1,3,2))

        response_p = self.constitutive(self.F, self.dim)
        P_p, K_p = response_p[0], response_p[2]

        P_error = np.linalg.norm(P_µ- P_p)/np.linalg.norm(P_µ)
        if not P_error < self.tol:
            print("Error(P) = {}".format(P_error))
            print("P_µ:\n{}".format(P_µ))
            print("P_p:\n{}".format(P_p))
        K_error = np.linalg.norm(K_µ- K_p)/np.linalg.norm(K_µ)
        if not K_error < self.tol:
            print("Error(K) = {}".format(K_error))
            flat_shape = (self.dim**2, self.dim**2)
            print("K_µ:\n{}".format(K_µ.reshape(flat_shape)))
            print("K_p:\n{}".format(K_p.reshape(flat_shape)))
            print("diff:\n{}".format(K_p.reshape(flat_shape)-
                                     K_µ.reshape(flat_shape)))
        self.assertLess(P_error,
                        self.tol)

        self.assertLess(K_error,
                        self.tol)


if __name__ == "__main__":
    unittest.main()

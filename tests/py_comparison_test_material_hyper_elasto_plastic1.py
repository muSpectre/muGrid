# !/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   py_comparison_test_material_hyper_elasto_plastic1.py

@author Till Junge <till.junge@epfl.ch>

@date   14 Nov 2018

@brief  compares MaterialHyperElastoPlastic1 to de Geus's python
        implementation

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

from  material_hyper_elasto_plastic1 import *
import itertools
import numpy as np
np.set_printoptions(linewidth=180)

import unittest

#####################
dyad22 = lambda A2,B2: np.einsum('ij  ,kl  ->ijkl',A2,B2)
dyad11 = lambda A1,B1: np.einsum('i   ,j   ->ij  ',A1,B1)
dot22  = lambda A2,B2: np.einsum('ij  ,jk  ->ik  ',A2,B2)
dot24  = lambda A2,B4: np.einsum('ij  ,jkmn->ikmn',A2,B4)
dot42  = lambda A4,B2: np.einsum('ijkl,lm  ->ijkm',A4,B2)
inv2 = np.linalg.inv
ddot22 = lambda A2,B2: np.einsum('ij  ,ji  ->    ',A2,B2)
ddot42 = lambda A4,B2: np.einsum('ijkl,lk  ->ij  ',A4,B2)
ddot44 = lambda A4,B4: np.einsum('ijkl,lkmn->ijmn',A4,B4)




class MatTest(unittest.TestCase):
    def constitutive(self, F,F_t,be_t,ep_t, dim):
        I  = np.eye(dim)
        II = dyad22(I,I)
        I4 = np.einsum('il,jk',I,I)
        I4rt = np.einsum('ik,jl',I,I)
        I4s  = (I4+I4rt)/2.

        def ln2(A2):
            vals,vecs = np.linalg.eig(A2)
            return sum(
                [np.log(vals[i])*dyad11(vecs[:,i],vecs[:,i]) for i in range(dim)])

        def exp2(A2):
            vals,vecs = np.linalg.eig(A2)
            return sum(
                [np.exp(vals[i])*dyad11(vecs[:,i],vecs[:,i]) for i in range(dim)])

        # function to compute linearization of the logarithmic Finger tensor
        def dln2_d2(A2):
            vals,vecs = np.linalg.eig(A2)
            K4        = np.zeros([dim, dim, dim, dim])
            for m, n in itertools.product(range(dim),repeat=2):

                if vals[n]==vals[m]:
                    gc = (1.0/vals[m])
                else:
                    gc  = (np.log(vals[n])-np.log(vals[m]))/(vals[n]-vals[m])
                K4 += gc*dyad22(dyad11(vecs[:,m],vecs[:,n]),dyad11(vecs[:,m],vecs[:,n]))
            return K4

        # elastic stiffness tensor
        C4e      = self.K*II+2.*self.mu*(I4s-1./3.*II)

        # trial state
        Fdelta   = dot22(F,inv2(F_t))
        be_s     = dot22(Fdelta,dot22(be_t,Fdelta.T))
        lnbe_s   = ln2(be_s)
        tau_s    = ddot42(C4e,lnbe_s)/2.
        taum_s   = ddot22(tau_s,I)/3.
        taud_s   = tau_s-taum_s*I
        taueq_s  = np.sqrt(3./2.*ddot22(taud_s,taud_s))
        div = np.where(taueq_s < 1e-12, np.ones_like(taueq_s), taueq_s)
        N_s      = 3./2.*taud_s/div
        phi_s    = taueq_s-(self.tauy0+self.H*ep_t)
        phi_s    = 1./2.*(phi_s+np.abs(phi_s))

        # return map
        dgamma   = phi_s/(self.H+3.*self.mu)
        ep       = ep_t  +   dgamma
        tau      = tau_s -2.*dgamma*N_s*self.mu
        lnbe     = lnbe_s-2.*dgamma*N_s
        be       = exp2(lnbe)
        P        = dot22(tau,inv2(F).T)

        # consistent tangent operator
        a0       = dgamma*self.mu/taueq_s
        a1       = self.mu/(self.H+3.*self.mu)
        C4ep     = (((self.K-2./3.*self.mu)/2.+a0*self.mu)*II+(1.-3.*a0)*self.mu*
                    I4s+2.*self.mu*(a0-a1)*dyad22(N_s,N_s))
        dlnbe4_s = dln2_d2(be_s)
        dbe4_s   = 2.*dot42(I4s,be_s)
        #K4a       = ((C4e/2.)*(phi_s<=0.).astype(np.float)+
        #             C4ep*(phi_s>0.).astype(np.float))
        K4a       = np.where(phi_s<=0, C4e/2., C4ep)
        K4b       = ddot44(K4a,ddot44(dlnbe4_s,dbe4_s))
        K4c       = dot42(-I4rt,tau)+K4b
        K4        = dot42(dot24(inv2(F),K4c),inv2(F).T)

        return P,tau,K4,be,ep, dlnbe4_s, dbe4_s, K4a, K4b, K4c

    def setUp(self):
        pass
    def prep(self, dimension):
        self.dim=dimension
        self.K=2.+ np.random.rand()
        self.mu=2.+ np.random.rand()
        self.H=.1 + np.random.rand()/100
        self.tauy0=4. + np.random.rand()/10
        self.F_prev=np.eye(self.dim) + (np.random.random((self.dim, self.dim))-.5)/10
        self.F = self.F_prev +  (np.random.random((self.dim, self.dim))-.5)/10
        noise = np.random.random((self.dim, self.dim))*1e-2
        self.be_prev=.5*(self.F_prev + self.F_prev.T + noise + noise.T)
        self.eps_prev=.5+ np.random.rand()/10
        self.tol = 1e-13
        self.verbose=True

    def test_specific_case(self):
        self.dim = 3
        self.K = 0.833
        self.mu = 0.386
        self.tauy0 = .003
        self.H = 0.004
        self.F_prev = np.eye(self.dim)
        self.F = np.array([[ 1.00357938,  0.0012795,   0.        ],
                           [-0.00126862,  0.99643366,  0.        ],
                           [ 0.,          0.,          0.99999974]])
        self.be_prev = np.eye(self.dim)
        self.eps_prev = 0.0
        self.tol = 1e-13
        self.verbose = True

        τ_µ, C_µ_s = kirchhoff_fun_3d(self.K, self.mu, self.H, self.tauy0,
                                      self.F, self.F_prev, self.be_prev,
                                      self.eps_prev)
        shape = (self.dim, self.dim, self.dim, self.dim)
        C_µ = C_µ_s.reshape(shape).transpose((0,1,3,2))

        P_µ, K_µ_s = PK1_fun_3d(self.K, self.mu, self.H, self.tauy0,
                                self.F, self.F_prev, self.be_prev,
                                self.eps_prev)
        K_µ = K_µ_s.reshape(shape).transpose((0,1,3,2))

        response_p = self.constitutive(self.F, self.F_prev, self.be_prev,
                                       self.eps_prev, self.dim)
        τ_p, C_p = response_p[1], response_p[8]
        P_p, K_p = response_p[0], response_p[2]

        τ_error = np.linalg.norm(τ_µ- τ_p)/np.linalg.norm(τ_µ)
        if not τ_error < self.tol:
            print("Error(τ) = {}".format(τ_error))
            print("τ_µ:\n{}".format(τ_µ))
            print("τ_p:\n{}".format(τ_p))
        self.assertLess(τ_error,
                        self.tol)

        C_error = np.linalg.norm(C_µ- C_p)/np.linalg.norm(C_µ)
        if not C_error < self.tol:
            print("Error(C) = {}".format(C_error))
            flat_shape = (self.dim**2, self.dim**2)
            print("C_µ:\n{}".format(C_µ.reshape(flat_shape)))
            print("C_p:\n{}".format(C_p.reshape(flat_shape)))
        self.assertLess(C_error,
                        self.tol)

        P_error = np.linalg.norm(P_µ- P_p)/np.linalg.norm(P_µ)
        if not P_error < self.tol:
            print("Error(P) = {}".format(P_error))
            print("P_µ:\n{}".format(P_µ))
            print("P_p:\n{}".format(P_p))
        self.assertLess(P_error,
                        self.tol)

        K_error = np.linalg.norm(K_µ- K_p)/np.linalg.norm(K_µ)
        if not K_error < self.tol:
            print("Error(K) = {}".format(K_error))
            flat_shape = (self.dim**2, self.dim**2)
            print("K_µ:\n{}".format(K_µ.reshape(flat_shape)))
            print("K_p:\n{}".format(K_p.reshape(flat_shape)))
            print("diff:\n{}".format(K_p.reshape(flat_shape)-
                                     K_µ.reshape(flat_shape)))
        self.assertLess(K_error,
                        self.tol)

    def test_equivalence_tau_C(self):
        for dim in (2, 3):
            self.runner_equivalence_τ_C(dim)

    def runner_equivalence_τ_C(self, dimension):
        self.prep(dimension)
        fun = kirchhoff_fun_2d if self.dim == 2 else kirchhoff_fun_3d
        τ_µ, C_µ_s = fun(self.K, self.mu, self.H, self.tauy0,
                         self.F, self.F_prev, self.be_prev,
                         self.eps_prev)
        shape = (self.dim, self.dim, self.dim, self.dim)
        C_µ = C_µ_s.reshape(shape).transpose((0,1,3,2))

        response_p = self.constitutive(self.F, self.F_prev, self.be_prev,
                                       self.eps_prev, self.dim)
        τ_p, C_p = response_p[1], response_p[8]

        τ_error = np.linalg.norm(τ_µ- τ_p)/np.linalg.norm(τ_µ)
        if not τ_error < self.tol:
            print("Error(τ) = {}".format(τ_error))
            print("τ_µ:\n{}".format(τ_µ))
            print("τ_p:\n{}".format(τ_p))
        self.assertLess(τ_error,
                        self.tol)

        C_error = np.linalg.norm(C_µ- C_p)/np.linalg.norm(C_µ)
        if not C_error < self.tol:
            print("Error(C) = {}".format(C_error))
            flat_shape = (self.dim**2, self.dim**2)
            print("C_µ:\n{}".format(C_µ.reshape(flat_shape)))
            print("C_p:\n{}".format(C_p.reshape(flat_shape)))
        self.assertLess(C_error,
                        self.tol)

    def test_equivalence_P_K(self):
        for dim in (2, 3):
            self.runner_equivalence_P_K(dim)

    def runner_equivalence_P_K(self, dimension):
        self.prep(dimension)
        fun = PK1_fun_2d if self.dim == 2 else PK1_fun_3d
        P_µ, K_µ_s = fun(self.K, self.mu, self.H, self.tauy0,
                         self.F, self.F_prev, self.be_prev,
                         self.eps_prev)
        shape = (self.dim, self.dim, self.dim, self.dim)
        K_µ = K_µ_s.reshape(shape).transpose((0,1,3,2))

        response_p = self.constitutive(self.F, self.F_prev, self.be_prev,
                                       self.eps_prev, self.dim)
        P_p, K_p = response_p[0], response_p[2]

        P_error = np.linalg.norm(P_µ- P_p)/np.linalg.norm(P_µ)
        if not P_error < self.tol:
            print("Error(P) = {}".format(P_error))
            print("P_µ:\n{}".format(P_µ))
            print("P_p:\n{}".format(P_p))
        self.assertLess(P_error,
                        self.tol)

        K_error = np.linalg.norm(K_µ- K_p)/np.linalg.norm(K_µ)
        if not K_error < self.tol:
            print("Error(K) = {}".format(K_error))
            flat_shape = (self.dim**2, self.dim**2)
            print("K_µ:\n{}".format(K_µ.reshape(flat_shape)))
            print("K_p:\n{}".format(K_p.reshape(flat_shape)))
            print("diff:\n{}".format(K_p.reshape(flat_shape)-
                                     K_µ.reshape(flat_shape)))
        self.assertLess(K_error,
                        self.tol)


if __name__ == "__main__":
    unittest.main()

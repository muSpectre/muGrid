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
along with GNU Emacs; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.
"""

import unittest
import numpy as np
import scipy.sparse.linalg as sp
import itertools

from python_test_imports import µ

def get_bulk_shear(E, nu):
    return E/(3*(1-2*nu)), E/(2*(1+nu))
class FiniteStrainProjectionGooseFFT(object):
    def __init__(self, ndim, resolution, incl_size, E, nu, contrast):
        """
        wraps the GooseFFT hyper-elasticity script into a more user-friendly
        class

        Keyword Arguments:
        ndim       -- number of dimensions of the problem, should be 2 or 3
        resolution -- pixel resolution, integer
        incl_size  -- edge length of cubic hard inclusion in pixels
        E          -- Young's modulus of soft phase
        nu         -- Poisson's ratio
        constrast  -- ratio between hard and soft Young's modulus
        """
        self.ndim = ndim
        self.resolution = resolution
        self.incl_size = incl_size
        self.E = E
        self.nu = nu
        self.contrast = contrast
        self.Kval, self.mu = get_bulk_shear(E, nu)

        self.setup()

    def setup(self):
        ndim=self.ndim
        trans2 = lambda A2   : np.einsum('ij...          ->ji...  ',A2   )
        ddot42 = lambda A4,B2: np.einsum('ijkl...,lk...  ->ij...  ',A4,B2)
        ddot44 = lambda A4,B4: np.einsum('ijkl...,lkmn...->ijmn...',A4,B4)
        dot22  = lambda A2,B2: np.einsum('ij...  ,jk...  ->ik...  ',A2,B2)
        dot24  = lambda A2,B4: np.einsum('ij...  ,jkmn...->ikmn...',A2,B4)
        dot42  = lambda A4,B2: np.einsum('ijkl...,lm...  ->ijkm...',A4,B2)
        dyad22 = lambda A2,B2: np.einsum('ij...  ,kl...  ->ijkl...',A2,B2)
        i      = np.eye(ndim)
        # identity tensors                                   [grid of tensors]
        shape = tuple((self.resolution for _ in range(ndim)))
        oneblock = np.ones(shape)
        def expand(arr):
            new_shape = (np.prod(arr.shape), np.prod(shape))
            ret_arr = np.zeros(new_shape)
            ret_arr[:] = arr.reshape(-1)[:, np.newaxis]
            return ret_arr.reshape((*arr.shape, *shape))
        I     = expand(i)
        self.I = I
        I4    = expand(np.einsum('il,jk',i,i))
        I4rt  = expand(np.einsum('ik,jl',i,i))
        I4s   = (I4+I4rt)/2.
        II    = dyad22(I,I)
        # projection operator                                  [grid of tensors]
        # NB can be vectorized (faster, less readable), see: "elasto-plasticity.py"
        # - support function / look-up list / zero initialize
        delta  = lambda i,j: np.float(i==j)            # Dirac delta function
        N = self.resolution
        freq   = np.fft.fftfreq(N, 1/N)        # coordinate axis -> freq. axis
        Ghat4  = np.zeros([ndim,ndim,ndim,ndim,*shape]) # zero initialize
        # - compute
        for xyz    in itertools.product(range(N),   repeat=self.ndim):
            q = np.array([freq[index] for index in xyz])  # frequency vector
            if not q.dot(q) == 0:                      # zero freq. -> mean
                for i,j,l,m in itertools.product(range(ndim),repeat=4):
                    index = tuple((i,j,l,m,*xyz))
                    Ghat4[index] = delta(i,m)*q[j]*q[l]/(q.dot(q))

        # (inverse) Fourier transform (for each tensor component in each direction)
        fft    = lambda x  : np.fft.fftn (x, shape)
        ifft   = lambda x  : np.fft.ifftn(x, shape)

        # functions for the projection 'G', and the product 'G : K^LT : (delta F)^T'
        G      = lambda A2 : np.real( ifft( ddot42(Ghat4,fft(A2)) ) ).reshape(-1)
        K_dF   = lambda dFm: trans2(ddot42(self.K4,trans2(dFm.reshape(ndim,ndim,*shape))))
        G_K_dF = lambda dFm: G(K_dF(dFm))

        # ------------------- PROBLEM DEFINITION / CONSTITIVE MODEL ----------------

        # phase indicator: cubical inclusion of volume fraction (9**3)/(31**3)
        incl = self.incl_size
        phase  = np.zeros(shape)
        if self.ndim == 2:
            phase[-incl:,:incl] = 1.
        else:
            phase[-incl:,:incl,-incl:] = 1.
        # material parameters + function to convert to grid of scalars
        param  = lambda M0,M1: M0*oneblock*(1.-phase)+M1*oneblock*phase

        K =  param(self.Kval, self.contrast*self.Kval)
        mu = param(self.mu, self.contrast*self.mu)

        # constitutive model: grid of "F" -> grid of "P", "K4"        [grid of tensors]
        def constitutive(F):
            C4 = K*II+2.*mu*(I4s-1./3.*II)
            S  = ddot42(C4,.5*(dot22(trans2(F),F)-I))
            P  = dot22(F,S)
            K4 = dot24(S,I4)+ddot44(ddot44(I4rt,dot42(dot24(F,C4),trans2(F))),I4rt)
            self.K4 = K4
            self.P = P
            return P,K4
        self.constitutive = constitutive
        self.G = G
        self.G_K_dF = G_K_dF
        self.Ghat4 = Ghat4

    def run(self):
        ndim = self.ndim
        shape = tuple((self.resolution for _ in range(ndim)))
        # ----------------------------- NEWTON ITERATIONS -----------------------------

        # initialize deformation gradient, and stress/stiffness       [grid of tensors]
        F     = np.array(self.I,copy=True)
        P,K4  = self.constitutive(F)
        # set macroscopic loading
        zer_shap = (ndim, ndim, *shape)
        DbarF = np.zeros(zer_shap); DbarF[0,1] += 1.0

        # initial residual: distribute "barF" over grid using "K4"
        b     = -self.G_K_dF(DbarF)
        F    +=         DbarF
        Fn    = np.linalg.norm(F)
        iiter = 0

        # iterate as long as the iterative update does not vanish
        class accumul(object):
            def __init__(self):
                self.counter = 0
            def __call__(self, dummy):
                self.counter += 1

        acc = accumul()
        while True:
            dFm,_ = sp.cg(tol=1.e-8,
                          A = sp.LinearOperator(shape=(
                              F.size,F.size),matvec=self.G_K_dF,dtype='float'),
                          b = b,
                          callback=acc
            )                                     # solve linear system using CG
            F    += dFm.reshape(ndim,ndim,*shape)  # update DOFs (array -> tens.grid)
            P,K4  = self.constitutive(F)          # new residual stress and tangent
            b     = -self.G(P)                         # convert res.stress to residual
            print('%10.2e'%(np.linalg.norm(dFm)/Fn)) # print residual to the screen
            if np.linalg.norm(dFm)/Fn<1.e-5 and iiter>0: break # check convergence
            iiter += 1

        print("nb_cg: {0}".format(acc.counter))



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
            rando = np.random.random((self.ndim, self.ndim))
            for i in range(hermitian_size):
                coord = µ.get_ccoord(msp_sizes, i)
                ref_id = µ.get_index(ref_sizes, coord)
                msp_id = µ.get_index(msp_sizes, coord)

                # story behind this order vector:
                # There was this issue with the projection operator of
                # de Geus acting on the the transpose of the gradient. 
                order = np.arange(9).reshape(3, 3).T.reshape(-1)
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
                np.prod(self.shape), self.ndim**2).T).T.reshape(strain_µ.shape, order="f")
            for ijk in itertools.product(range(self.resolution), repeat=self.ndim):
                index_µ = tuple((*ijk, slice(None), slice(None)))
                index_g = tuple((slice(None), slice(None), *ijk))
                b_µ_sl = b_µ[index_µ]
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

get_finite_goose = lambda ndim: FiniteStrainProjectionGooseFFT(
    ndim, 11, 3, 70e9, .33, 3.)

FiniteStrainProjectionCheck = build_test_classes(µ.fft.ProjectionFiniteStrain_3d,
                                                get_finite_goose(3),
                                                "FiniteStrainDefaultProjection")
if __name__ == "__main__":
    unittest.main()

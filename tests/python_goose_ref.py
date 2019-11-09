#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_goose_ref.py

@author Till Junge <till.junge@altermail.ch>

@date   19 Jan 2018

@brief  adapted scripts from GooseFFT, https://github.com/tdegeus/GooseFFT,
        which are MIT licensed

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


import numpy as np
import scipy.sparse.linalg as sp
import itertools


def get_bulk_shear(E, nu):
    return E/(3*(1-2*nu)), E/(2*(1+nu))
class ProjectionGooseFFT(object):
    def __init__(self, ndim, nb_grid_pts, incl_size, E, nu, contrast):
        """
        wraps the GooseFFT hyper-elasticity script into a more user-friendly
        class

        Keyword Arguments:
        ndim        -- number of dimensions of the problem, should be 2 or 3
        nb_grid_pts -- number of grid_points, integer
        incl_size   -- edge length of cubic hard inclusion in pixels
        E           -- Young's modulus of soft phase
        nu          -- Poisson's ratio
        constrast   -- ratio between hard and soft Young's modulus
        """
        self.ndim = ndim
        self.nb_grid_pts = nb_grid_pts
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
        shape = tuple((self.nb_grid_pts for _ in range(ndim)))
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
        N = self.nb_grid_pts
        freq   = np.fft.fftfreq(N, 1/N)        # coordinate axis -> freq. axis
        Ghat4  = np.zeros([ndim,ndim,ndim,ndim,*shape]) # zero initialize
        # - compute
        for xyz    in itertools.product(range(N),   repeat=self.ndim):
            q = np.array([freq[index] for index in xyz])  # frequency vector
            index = tuple((*(slice(None) for _ in range(4)), *xyz))
            Ghat4[index] = self.comp_ghat(q)

        # (inverse) Fourier transform (for each tensor component in each direction)
        fft    = lambda x  : np.fft.fftn (x, shape)
        ifft   = lambda x  : np.fft.ifftn(x, shape)

        # functions for the projection 'G', and the product 'G : K^LT : (delta F)^T'
        G      = lambda A2 : np.real( ifft( ddot42(Ghat4,fft(A2)) ) ).reshape(-1)
        K_dF   = lambda dFm: trans2(ddot42(self.K4,trans2(dFm.reshape(ndim,ndim,*shape))))
        G_K_dF = lambda dFm: G(K_dF(dFm))
        K_deps   = lambda depsm: ddot42(self.C4,depsm.reshape(ndim,ndim,N,N,N))
        G_K_deps = lambda depsm: G(K_deps(depsm))


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
        self.C4 = K*II+2.*mu*(I4s-1./3.*II)
        def constitutive(F):
            C4 = self.C4
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
        self.G_K_deps = G_K_deps

class FiniteStrainProjectionGooseFFT(ProjectionGooseFFT):
    def __init__(self, ndim, nb_grid_pts, incl_size, E, nu, contrast):
        super().__init__(ndim, nb_grid_pts, incl_size, E, nu, contrast)

    def comp_ghat(self, q):
        temp = np.zeros((self.ndim, self.ndim, self.ndim, self.ndim))
        delta  = lambda i,j: np.float(i==j)            # Dirac delta function
        if not q.dot(q) == 0:                      # zero freq. -> mean
            for i,j,l,m in itertools.product(range(self.ndim),repeat=4):
                temp[i, j, l, m] = delta(i,m)*q[j]*q[l]/(q.dot(q))
        return temp

    def run(self):
        ndim = self.ndim
        shape = tuple((self.nb_grid_pts for _ in range(ndim)))
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
            )                                     # solve linear cell using CG
            F    += dFm.reshape(ndim,ndim,*shape)  # update DOFs (array -> tens.grid)
            P,K4  = self.constitutive(F)          # new residual stress and tangent
            b     = -self.G(P)                         # convert res.stress to residual
            print('%10.2e'%(np.linalg.norm(dFm)/Fn)) # print residual to the screen
            if np.linalg.norm(dFm)/Fn<1.e-5 and iiter>0: break # check convergence
            iiter += 1

        print("nb_cg: {0}".format(acc.counter))

class SmallStrainProjectionGooseFFT(ProjectionGooseFFT):
    def __init__(self, ndim, nb_grid_pts, incl_size, E, nu, contrast):
        super().__init__(ndim, nb_grid_pts, incl_size, E, nu, contrast)

    def comp_ghat(self, q):
        temp = np.zeros((self.ndim, self.ndim, self.ndim, self.ndim))
        delta  = lambda i,j: np.float(i==j)            # Dirac delta function
        if not q.dot(q) == 0:                      # zero freq. -> mean
            for i,j,l,m in itertools.product(range(self.ndim),repeat=4):
                temp[i, j, l, m] = -(q[i]*q[j]*q[l]*q[m])/(q.dot(q))**2+\
             (delta(j,l)*q[i]*q[m]+delta(j,m)*q[i]*q[l]+\
              delta(i,l)*q[j]*q[m]+delta(i,m)*q[j]*q[l])/(2.*q.dot(q))
        return temp

    def tangent_stiffness(self, field):
        return self.constitutive(F)[0]

    def run(self):
        ndim = self.ndim
        shape = tuple((self.nb_grid_pts for _ in range(ndim)))
        # ----------------------------- NEWTON ITERATIONS -----------------------------

        # initialize stress and strain tensor              [grid of tensors]
        sig      = np.zeros([ndim,ndim,N,N,N])
        eps      = np.zeros([ndim,ndim,N,N,N])

        # set macroscopic loading
        DE       = np.zeros([ndim,ndim,N,N,N])
        DE[0,1] += 0.01
        DE[1,0] += 0.01

        # initial residual: distribute "barF" over grid using "K4"
        b     = -self.G_K_deps(DE)
        eps     +=           DE
        En       = np.linalg.norm(eps)
        iiter    = 0

        # iterate as long as the iterative update does not vanish
        class accumul(object):
            def __init__(self):
                self.counter = 0
            def __call__(self, dummy):
                self.counter += 1

        acc = accumul()
        while True:
            depsm,_ = sp.cg(tol=1.e-8,
                            A = sp.LinearOperator(shape=(
                                eps.size,eps.size),matvec=self.G_K_deps,dtype='float'),
                            b = b,
                            callback=acc
            )                                     # solve linear cell using CG
            eps  += depsm.reshape(ndim,ndim,*shape)  # update DOFs (array -> tens.grid)
            sig  = ddot42(self.C4, eps)           # new residual stress and tangent
            b     = -self.G(sig)                         # convert res.stress to residual
            print('%10.2e'%(np.linalg.norm(depsm)/En)) # print residual to the screen
            if np.linalg.norm(depsm)/en<1.e-5 and iiter>0: break # check convergence
            iiter += 1

        print("nb_cg: {0}".format(acc.counter))


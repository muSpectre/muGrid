#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_exact_reference_test.py

@author Till Junge <till.junge@epfl.ch>

@date   18 Jun 2018

@brief  Tests exactness of each iterate with respect to python reference
        implementation from GooseFFT for elasticity

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

Additional permission under GNU GPL version 3 section 7

If you modify this Program, or any covered work, by linking or combining it
with proprietary FFT implementations or numerical libraries, containing parts
covered by the terms of those libraries' licenses, the licensors of this
Program grant you additional permission to convey the resulting work.
"""

import unittest
import numpy as np
from numpy.linalg import norm
from python_test_imports import µ

import scipy.sparse.linalg as sp
import itertools

np.set_printoptions(linewidth=180)
comparator_nb_cols=9
# ----------------------------------- GRID ------------------------------------

ndim   = 3   # number of dimensions
N      = 3  # number of voxels (assumed equal for all directions)

Nx = Ny = Nz = N


def deserialise_t4(t4):
    turnaround = np.arange(ndim**2).reshape(ndim,ndim).T.reshape(-1)
    retval = np.zeros([ndim*ndim, ndim*ndim])
    for i,j in itertools.product(range(ndim**2), repeat=2):
        retval[i,j] = t4[:ndim, :ndim, :ndim, :ndim, 0,0].reshape(ndim**2, ndim**2)[turnaround[i], turnaround[j]]
        pass
    return retval

def scalar_to_goose(s_msp):
    s_goose = np.zeros((Nx, Ny, Nz))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                s_goose[i,j,k] = s_msp[Nz*Ny*i + Nz*j + k]
            pass
        pass
    return s_goose

def t2_to_goose(t2_msp):
    t2_goose = np.zeros((ndim, ndim, Nx, Ny, Nz))
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                t2_goose[:,:,i,j,k] = t2_msp[:, Nz*Ny*i + Nz*j + k].reshape(ndim, ndim).T
            pass
        pass
    return t2_goose

def t2_vec_to_goose(t2_msp_vec):
    return t2_to_goose(t2_msp_vec.reshape(ndim*ndim, Nx*Ny*Nz)).reshape(-1)

def scalar_vec_to_goose(s_msp_vec):
    return scalar_to_goose(s_msp_vec.reshape(Nx*Ny*N)).reshape(-1)

def t4_to_goose(t4_msp, right_transposed=True):
    t4_goose = np.zeros((ndim, ndim, ndim, ndim, Nx, Ny, Nz))
    turnaround = np.arange(ndim**2).reshape(ndim,ndim).T.reshape(-1)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                tmp = t4_msp[:, Nz*Ny*i + Nz*j + k].reshape(ndim**2, ndim**2)
                goose_view = t4_goose[:,:,:,:,i,j,k].reshape(ndim**2, ndim**2)
                for a,b in itertools.product(range(ndim**2), repeat=2):
                    a_id = a if right_transposed else turnaround[a]
                    goose_view[a,b] = tmp[a_id, turnaround[b]]
            pass
        pass
    return t4_goose

def t4_vec_to_goose(t4_msp_vec):
    return t4_to_goose(t4_msp_vec.reshape(ndim**4, Nx*Ny*Nz)).reshape(-1)

def t2_from_goose(t2_goose):
    nb_pix = Nx*Ny*Nz
    t2_msp = np.zeros((ndim**2, nb_pix), order='F')
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                view = t2_msp[:, i + Nx*j + Nx*Ny*k].reshape(ndim, ndim).T
                view = t2goose[:,:,i,j,k].T
            pass
        pass
    return t2_msp


# ---------------------- PROJECTION, TENSORS, OPERATIONS ----------------------

# tensor operations/products: np.einsum enables index notation, avoiding loops
# e.g. ddot42 performs $C_ij = A_ijkl B_lk$ for the entire grid
trans2 = lambda A2   : np.einsum('ijxyz          ->jixyz  ',A2   )
ddot42 = lambda A4,B2: np.einsum('ijklxyz,lkxyz  ->ijxyz  ',A4,B2)
ddot44 = lambda A4,B4: np.einsum('ijklxyz,lkmnxyz->ijmnxyz',A4,B4)
dot22  = lambda A2,B2: np.einsum('ijxyz  ,jkxyz  ->ikxyz  ',A2,B2)
dot24  = lambda A2,B4: np.einsum('ijxyz  ,jkmnxyz->ikmnxyz',A2,B4)
dot42  = lambda A4,B2: np.einsum('ijklxyz,lmxyz  ->ijkmxyz',A4,B2)
dyad22 = lambda A2,B2: np.einsum('ijxyz  ,klxyz  ->ijklxyz',A2,B2)

# identity tensor                                               [single tensor]
i      = np.eye(ndim)
# identity tensors                                            [grid of tensors]
I      = np.einsum('ij,xyz'           ,                  i   ,np.ones([N,N,N]))
I4     = np.einsum('ijkl,xyz->ijklxyz',np.einsum('il,jk',i,i),np.ones([N,N,N]))
I4rt   = np.einsum('ijkl,xyz->ijklxyz',np.einsum('ik,jl',i,i),np.ones([N,N,N]))
I4s    = (I4+I4rt)/2.
II     = dyad22(I,I)

# projection operator                                         [grid of tensors]
# NB can be vectorized (faster, less readable), see: "elasto-plasticity.py"
# - support function / look-up list / zero initialize
delta  = lambda i,j: np.float(i==j)            # Dirac delta function
freq   = np.arange(-(N-1)/2.,+(N+1)/2.)        # coordinate axis -> freq. axis
Ghat4  = np.zeros([ndim,ndim,ndim,ndim,N,N,N]) # zero initialize
# - compute
for i,j,l,m in itertools.product(range(ndim),repeat=4):
    for x,y,z    in itertools.product(range(N),   repeat=3):
        q = np.array([freq[x], freq[y], freq[z]])  # frequency vector
        if not q.dot(q) == 0:                      # zero freq. -> mean
            Ghat4[i,j,l,m,x,y,z] = delta(i,m)*q[j]*q[l]/(q.dot(q))

# (inverse) Fourier transform (for each tensor component in each direction)
fft    = lambda x  : np.fft.fftshift(np.fft.fftn (np.fft.ifftshift(x),[N,N,N]))
ifft   = lambda x  : np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x),[N,N,N]))

# functions for the projection 'G', and the product 'G : K^LT : (delta F)^T'
G      = lambda A2 : np.real( ifft( ddot42(Ghat4,fft(A2)) ) ).reshape(-1)
K_dF   = lambda dFm: trans2(ddot42(K4,trans2(dFm.reshape(ndim,ndim,N,N,N))))
G_K_dF = lambda dFm: G(K_dF(dFm))

# ------------------- PROBLEM DEFINITION / CONSTITIVE MODEL -------------------

# phase indicator: cubical inclusion of volume fraction (9**3)/(31**3)
phase  = np.zeros([N,N,N]); phase[:2,:2,:2] = 1.
# material parameters + function to convert to grid of scalars
param  = lambda M0,M1: M0*np.ones([N,N,N])*(1.-phase)+M1*np.ones([N,N,N])*phase
K      = param(0.833,8.33)  # bulk  modulus                   [grid of scalars]
mu     = param(0.386,3.86)  # shear modulus                   [grid of scalars]

# constitutive model: grid of "F" -> grid of "P", "K4"        [grid of tensors]
def constitutive(F):
    C4 = K*II+2.*mu*(I4s-1./3.*II)
    S  = ddot42(C4,.5*(dot22(trans2(F),F)-I))
    P  = dot22(F,S)
    K4 = dot24(S,I4)+ddot44(ddot44(I4rt,dot42(dot24(F,C4),trans2(F))),I4rt)
    return P,K4


F     = np.array(I,copy=True)
P,K4  = constitutive(F)

class Counter(object):
    def __init__(self):
        self.count = self.reset()

    def reset(self):
        self.count = 0
        return self.count

    def get(self):
        return self.count

    def __call__(self, dummy):
        self.count += 1


class LinearElastic_Check(unittest.TestCase):
    def t2_comparator(self, µT2, gT2):
        err_sum = 0.
        err_max = 0.
        for counter, (i, j, k) in enumerate(self.rve):
            print((i,j,k))
            µ_arr = µT2[:, counter].reshape(ndim, ndim).T
            g_arr = gT2[:,:,i,j,k]
            self.assertEqual(Nz*Ny*i+Nz*j + k, counter)
            print("µSpectre:")
            print(µ_arr)
            print("Goose:")
            print(g_arr)
            print(µ_arr-g_arr)
            err = norm(µ_arr-g_arr)
            print("error norm for pixel {} = {}".format((i, j, k), err))
            err_sum += err
            err_max = max(err_max, err)
            pass
        print("∑(err) = {}, max(err) = {}".format (err_sum, err_max))
        return err_sum


    def t4_comparator(self, µT4, gT4, right_transposed=True):
        """ right_transposed: in de Geus's notation, e.g.,
            stiffness tensors have the last two dimensions inverted
        """
        err_sum = 0.
        err_max = 0.
        errs = dict()
        turnaround = np.arange(ndim**2).reshape(ndim,ndim).T.reshape(-1)

        def zero_repr(arr):
            arrcopy = arr.copy()
            arrcopy[abs(arr)<1e-13] = 0.
            return arrcopy
        for counter, (i, j, k) in enumerate(self.rve):
            µ_arr_tmp = µT4[:, counter].reshape(ndim**2, ndim**2).T
            µ_arr = np.empty((ndim**2, ndim**2))
            for a,b in itertools.product(range(ndim**2), repeat=2):
                a = a if right_transposed else turnaround[a]
                µ_arr[a,b] = µ_arr_tmp[a, turnaround[b]]
            g_arr = gT4[:,:,:,:,i,j,k].reshape(ndim**2, ndim**2)
            self.assertEqual(Nz*Ny*i+Nz*j + k, counter)

            print("µSpectre:")
            print(zero_repr(µ_arr[:, :comparator_nb_cols]))
            print("Goose:")
            print(zero_repr(g_arr[:, :comparator_nb_cols]))
            print("Diff")
            print(zero_repr((µ_arr-g_arr)[:, :comparator_nb_cols]))
            err = norm(µ_arr-g_arr)/norm(g_arr)
            print("error norm for pixel {} = {}".format((i, j, k), err))
            err_sum += err
            errs[(i,j,k)] = err
            err_max = max(err_max, err)
            print("count {:>2}: err_norm = {:.5f}, err_sum = {:.5f}".format(
                counter, err, err_sum))
            pass
        print("∑(err) = {}, max(err) = {}".format (err_sum, err_max))
        return err_sum, errs

    def setUp(self):
        #---------------------------- µSpectre init -----------------------------------
        nb_grid_pts = list(phase.shape)
        dim = len(nb_grid_pts)
        self.dim=dim

        center = np.array([r//2 for r in nb_grid_pts])
        incl = nb_grid_pts[0]//5


        ## Domain dimensions
        lengths = [float(r) for r in nb_grid_pts]
        ## formulation (small_strain or finite_strain)
        formulation = µ.Formulation.finite_strain

        ## build a computational domain
        self.rve = µ.Cell(nb_grid_pts, lengths, formulation)
        def get_E_nu(bulk, shear):
            Young = 9*bulk*shear/(3*bulk + shear)
            Poisson = Young/(2*shear) - 1
            return Young, Poisson

        mat = µ.material.MaterialLinearElastic1_3d

        E, nu = get_E_nu(.833, .386)
        hard = mat.make(self.rve, 'hard', 10*E, nu)
        soft = mat.make(self.rve, 'soft',    E, nu)

        for pixel in self.rve:
            if pixel[0] < 2 and pixel[1] < 2 and pixel[2] < 2:
                hard.add_pixel(pixel)
            else:
                soft.add_pixel(pixel)

    def test_solve(self):
        before_cg_tol = 1e-11
        cg_tol = 1e-11
        after_cg_tol = 1e-9
        newton_tol = 1e-4
        # ----------------------------- NEWTON ITERATIONS ---------------------

        # initialize deformation gradient, and stress/stiffness [tensor grid]
        global K4, P, F
        F     = np.array(I,copy=True)
        F2    = np.array(I,copy=True)*1.1
        P2,K42  = constitutive(F2)
        P,K4  = constitutive(F)
        self.rve.set_uniform_strain(np.array(np.eye(ndim)))
        µF = self.rve.get_strain()

        self.assertLess(norm(t2_vec_to_goose(µF) - F.reshape(-1))/norm(F), before_cg_tol)
        # set macroscopic loading
        DbarF = np.zeros([ndim,ndim,N,N,N]); DbarF[0,1] += 1.0

        # initial residual: distribute "barF" over grid using "K4"
        b     = -G_K_dF(DbarF)
        F    +=         DbarF
        Fn    = np.linalg.norm(F)
        iiter = 0

        # µSpectre inits
        µbarF    = np.zeros_like(µF)
        µbarF[ndim, :] += 1.
        µF2 = µF.copy()*1.1
        µP2, µK2 = self.rve.evaluate_stress_tangent(µF2)
        err = norm(t2_vec_to_goose(µP2) - P2.reshape(-1))/norm(P2)


        if not (err < before_cg_tol):
            self.t2_comparator(µP2, µK2)
        self.assertLess(err, before_cg_tol)
        self.rve.set_uniform_strain(np.array(np.eye(ndim)))
        µP, µK = self.rve.evaluate_stress_tangent(µF)
        err = norm(t2_vec_to_goose(µP) - P.reshape(-1))
        if not (err < before_cg_tol):
            print(µF)
            self.t2_comparator(µP, P)
        self.assertLess(err, before_cg_tol)
        err = norm(t4_vec_to_goose(µK) - K4.reshape(-1))/norm(K4)
        if not (err < before_cg_tol):
            print ("err = {}".format(err))

        self.assertLess(err, before_cg_tol)
        µF += µbarF
        µFn = norm(µF)
        self.assertLess(norm(t2_vec_to_goose(µF) - F.reshape(-1))/norm(F), before_cg_tol)
        µG_K_dF = lambda x: self.rve.directional_stiffness(x.reshape(µF.shape)).reshape(-1)
        µG = lambda x: self.rve.project(x).reshape(-1)
        µb = -µG_K_dF(µbarF)

        err = (norm(t2_vec_to_goose(µb.reshape(µF.shape)) - b) /
               norm(b))
        if not (err < before_cg_tol):
            print("|µb| = {}".format(norm(µb)))
            print("|b| = {}".format(norm(b)))
            print("total error = {}".format(err))
            self.t2_comparator(µb.reshape(µF.shape), b.reshape(F.shape))
        self.assertLess(err, before_cg_tol)

        # iterate as long as the iterative update does not vanish
        while True:
            # solve linear system using CG
            g_counter = Counter()
            dFm,_ = sp.cg(tol=cg_tol,
                          A = sp.LinearOperator(shape=(F.size,F.size),
                                                matvec=G_K_dF,dtype='float'),
                          b = b,
                          callback=g_counter
            )

            µ_counter = Counter()
            µdFm,_ = sp.cg(tol=cg_tol,
                           A =  sp.LinearOperator(shape=(F.size,F.size),
                                                  matvec=µG_K_dF,dtype='float'),
                           b = µb,
                           callback=µ_counter)

            err = g_counter.get()-µ_counter.get()
            if err != 0:
                print("n_iter(g) = {}, n_iter(µ) = {}".format(g_counter.get(),
                                                              µ_counter.get()))
                pass


            # in the last iteration, the increment is essentially
            # zero, so we don't care about relative error anymore
            err = norm(t2_vec_to_goose(µdFm) - dFm)/norm(dFm)
            if norm(dFm)/Fn > newton_tol and norm(µdFm)/Fn > newton_tol:
                if not (err < after_cg_tol):
                    self.t2_comparator(µdFm.reshape(µF.shape), dFm.reshape(F.shape))
                    print("µdFm.shape = {}".format(µdFm.shape))
                    print("|µdFm| = {}".format(norm(µdFm)))
                    print("|dFm| = {}".format(norm(dFm)))
                    print("|µdFm - dFm| = {}".format(norm(µdFm-dFm)))
                    print("AssertionWarning: {} is not less than {}".format(err,
                                                                        after_cg_tol))
                self.assertLess(err, after_cg_tol)
            # update DOFs (array -> tens.grid)
            F    += dFm.reshape(ndim,ndim,N,N,N)
            µF   += µdFm.reshape(µF.shape)
            # new residual stress and tangent
            P,K4  = constitutive(F)
            µP, µK = self.rve.evaluate_stress_tangent(µF)

            err = norm(t2_vec_to_goose(µP) - P.reshape(-1))/norm(P)
            self.assertLess(err, before_cg_tol)

            err = norm(t4_vec_to_goose(µK) - K4.reshape(-1))/norm(K4)
            if not (err < before_cg_tol):
                print ("err = {}".format(err))
                self.t4_comparator(µK, K4)
            self.assertLess(err, before_cg_tol)
            # convert res.stress to residual
            b     = -G(P)
            µb = -µG(µP)
            # in the last iteration, the rhs is essentianly zero,
            # leading to large relative errors, which are ok. So we
            # want either the relative error for the rhs to be small,
            # or their absolute error to be small compared to unity
            diff_norm = norm(t2_vec_to_goose(µb) - b.reshape(-1))
            err = diff_norm/norm(b)
            if not ((err < after_cg_tol) or (diff_norm < before_cg_tol)):
                self.t2_comparator(µb.reshape(µF.shape), b.reshape(F.shape))
                print("|µb| = {}".format(norm(µb)))
                print("|b| = {}".format(norm(b)))
                print("err = {}".format(err))
                print("|µb-b| = {}".format(norm(t2_vec_to_goose(µb) - b.reshape(-1))))

                print("AssertionWarning: {} is not less than {}".format(err, before_cg_tol))
            self.assertTrue((err < after_cg_tol) or (diff_norm < after_cg_tol))
            # print residual to the screen
            print('Goose:    %10.15e'%(np.linalg.norm(dFm)/Fn))
            print('µSpectre: %10.15e'%(np.linalg.norm(µdFm)/µFn))
            if np.linalg.norm(dFm)/Fn<newton_tol and iiter>0: break # check convergence
            iiter += 1


if __name__ == '__main__':
    unittest.main()

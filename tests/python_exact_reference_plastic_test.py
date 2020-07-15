#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_exact_reference_plastic_test.py

@author Till Junge <till.junge@epfl.ch>

@date   22 Jun 2018

@brief  Tests exactness of each iterate with respect to python reference
        implementation from GooseFFT for plasticity

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

from python_exact_reference_elastic_test import ndim, N, Nx, Ny, Nz
from material_hyper_elasto_plastic1 import PK1_fun_3d
import python_exact_reference_elastic_test as elastic_ref
from python_exact_reference_elastic_test import Counter
from python_exact_reference_elastic_test import t2_from_goose
from python_exact_reference_elastic_test import t4_vec_to_goose
from python_exact_reference_elastic_test import t4_to_goose
from python_exact_reference_elastic_test import scalar_vec_to_goose
from python_exact_reference_elastic_test import t2_vec_to_goose
from python_exact_reference_elastic_test import deserialise_t4, t2_to_goose
import sys
import itertools
import scipy.sparse.linalg as sp

import unittest
import numpy as np
import numpy.linalg as linalg
from python_test_imports import µ
# turn of warning for zero division
# (which occurs in the linearization of the logarithmic strain)
np.seterr(divide='ignore', invalid='ignore')


# ----------------------------------- GRID ------------------------------------
shape = [Nx, Ny, Nz]


def standalone_dyad22(A2, B2): return np.einsum('ij  ,kl  ->ijkl', A2, B2)


def standalone_dyad11(A1, B1): return np.einsum('i   ,j   ->ij  ', A1, B1)


def standalone_dot22(A2, B2): return np.einsum('ij  ,jk  ->ik  ', A2, B2)


def standalone_dot24(A2, B4): return np.einsum('ij  ,jkmn->ikmn', A2, B4)


def standalone_dot42(A4, B2): return np.einsum('ijkl,lm  ->ijkm', A4, B2)


standalone_inv2 = np.linalg.inv


def standalone_ddot22(A2, B2): return np.einsum('ij  ,ji  ->    ', A2, B2)


def standalone_ddot42(A4, B2): return np.einsum('ijkl,lk  ->ij  ', A4, B2)


def standalone_ddot44(A4, B4): return np.einsum('ijkl,lkmn->ijmn', A4, B4)


def constitutive_standalone(K, mu, H, tauy0, F, F_t, be_t, ep_t, dim):
    I = np.eye(dim)
    II = standalone_dyad22(I, I)
    I4 = np.einsum('il,jk', I, I)
    I4rt = np.einsum('ik,jl', I, I)
    I4s = (I4+I4rt)/2.

    def ln2(A2):
        vals, vecs = np.linalg.eig(A2)
        return sum(
            [np.log(vals[i])*standalone_dyad11(vecs[:, i], vecs[:, i])
             for i in range(dim)])

    def exp2(A2):
        vals, vecs = np.linalg.eig(A2)
        return sum(
            [np.exp(vals[i])*standalone_dyad11(vecs[:, i], vecs[:, i])
             for i in range(dim)])

    # function to compute linearization of the logarithmic Finger tensor
    def dln2_d2(A2):
        vals, vecs = np.linalg.eig(A2)
        K4 = np.zeros([dim, dim, dim, dim])
        for m, n in itertools.product(range(dim), repeat=2):

            if vals[n] == vals[m]:
                gc = (1.0/vals[m])
            else:
                gc = (np.log(vals[n])-np.log(vals[m]))/(vals[n]-vals[m])
            K4 += gc*standalone_dyad22(standalone_dyad11(
                vecs[:, m], vecs[:, n]), standalone_dyad11(vecs[:, m],
                                                           vecs[:, n]))
        return K4

    # elastic stiffness tensor
    C4e = K*II+2.*mu*(I4s-1./3.*II)

    # trial state
    Fdelta = standalone_dot22(F, standalone_inv2(F_t))
    be_s = standalone_dot22(Fdelta, standalone_dot22(be_t, Fdelta.T))
    lnbe_s = ln2(be_s)
    tau_s = standalone_ddot42(C4e, lnbe_s)/2.
    taum_s = standalone_ddot22(tau_s, I)/3.
    taud_s = tau_s-taum_s*I
    taueq_s = np.sqrt(3./2.*standalone_ddot22(taud_s, taud_s))
    div = np.where(taueq_s < 1e-12, np.ones_like(taueq_s), taueq_s)
    N_s = 3./2.*taud_s/div
    phi_s = taueq_s-(tauy0+H*ep_t)
    phi_s = 1./2.*(phi_s+np.abs(phi_s))

    # return map
    dgamma = phi_s/(H+3.*mu)
    ep = ep_t + dgamma
    tau = tau_s - 2.*dgamma*N_s*mu
    lnbe = lnbe_s-2.*dgamma*N_s
    be = exp2(lnbe)
    P = standalone_dot22(tau, standalone_inv2(F).T)

    # consistent tangent operator
    a0 = dgamma*mu/taueq_s
    a1 = mu/(H+3.*mu)
    C4ep = (((K-2./3.*mu)/2.+a0*mu)*II+(1.-3.*a0)*mu *
            I4s+2.*mu*(a0-a1)*standalone_dyad22(N_s, N_s))
    dlnbe4_s = dln2_d2(be_s)
    dbe4_s = 2.*standalone_dot42(I4s, be_s)
    # K4a       = ((C4e/2.)*(phi_s<=0.).astype(np.float)+
    #             C4ep*(phi_s>0.).astype(np.float))
    K4a = np.where(phi_s <= 0, C4e/2., C4ep)
    K4b = standalone_ddot44(K4a, standalone_ddot44(dlnbe4_s, dbe4_s))
    K4c = standalone_dot42(-I4rt, tau)+K4b
    K4 = standalone_dot42(standalone_dot24(
        standalone_inv2(F), K4c), standalone_inv2(F).T)

    return P, tau, K4, be, ep, dlnbe4_s, dbe4_s, K4a, K4b, K4c


# ----------------------------- TENSOR OPERATIONS -----------------------------

# tensor operations / products: np.einsum enables index notation, avoiding loops
# e.g. ddot42 performs $C_ij = A_ijkl B_lk$ for the entire grid
def trans2(A2): return np.einsum('ijxyz          ->jixyz  ', A2)


def ddot22(A2, B2): return np.einsum('ijxyz  ,jixyz  ->xyz    ', A2, B2)


def ddot42(A4, B2): return np.einsum('ijklxyz,lkxyz  ->ijxyz  ', A4, B2)


def ddot44(A4, B4): return np.einsum('ijklxyz,lkmnxyz->ijmnxyz', A4, B4)


def dot11(A1, B1): return np.einsum('ixyz   ,ixyz   ->xyz    ', A1, B1)


def dot22(A2, B2): return np.einsum('ijxyz  ,jkxyz  ->ikxyz  ', A2, B2)


def dot24(A2, B4): return np.einsum('ijxyz  ,jkmnxyz->ikmnxyz', A2, B4)


def dot42(A4, B2): return np.einsum('ijklxyz,lmxyz  ->ijkmxyz', A4, B2)


def dyad22(A2, B2): return np.einsum('ijxyz  ,klxyz  ->ijklxyz', A2, B2)


def dyad11(A1, B1): return np.einsum('ixyz   ,jxyz   ->ijxyz  ', A1, B1)


# eigenvalue decomposition of 2nd-order tensor: return in convention i,j,x,y,z
# NB requires to swap default order of NumPy (in in/output)
def eig2(A2):
    def swap1i(A1): return np.einsum('xyzi ->ixyz ', A1)

    def swap2(A2): return np.einsum('ijxyz->xyzij', A2)

    def swap2i(A2): return np.einsum('xyzij->ijxyz', A2)
    vals, vecs = np.linalg.eig(swap2(A2))
    vals = swap1i(vals)
    vecs = swap2i(vecs)
    return vals, vecs

# logarithm of grid of 2nd-order tensors


def ln2(A2):
    vals, vecs = eig2(A2)
    return sum([np.log(vals[i])*dyad11(vecs[:, i], vecs[:, i])
                for i in range(3)])

# exponent of grid of 2nd-order tensors


def exp2(A2):
    vals, vecs = eig2(A2)
    return sum([np.exp(vals[i])*dyad11(vecs[:, i], vecs[:, i])
                for i in range(3)])

# determinant of grid of 2nd-order tensors


def det2(A2):
    return (A2[0, 0]*A2[1, 1]*A2[2, 2]+A2[0, 1]*A2[1, 2]*A2[2, 0] +
            A2[0, 2]*A2[1, 0]*A2[2, 1]) -\
           (A2[0, 2]*A2[1, 1]*A2[2, 0]+A2[0, 1]*A2[1, 0]
            * A2[2, 2]+A2[0, 0]*A2[1, 2]*A2[2, 1])

# inverse of grid of 2nd-order tensors


def inv2(A2):
    A2det = det2(A2)
    A2inv = np.empty([3, 3, Nx, Ny, Nz])
    A2inv[0, 0] = (A2[1, 1]*A2[2, 2]-A2[1, 2]*A2[2, 1])/A2det
    A2inv[0, 1] = (A2[0, 2]*A2[2, 1]-A2[0, 1]*A2[2, 2])/A2det
    A2inv[0, 2] = (A2[0, 1]*A2[1, 2]-A2[0, 2]*A2[1, 1])/A2det
    A2inv[1, 0] = (A2[1, 2]*A2[2, 0]-A2[1, 0]*A2[2, 2])/A2det
    A2inv[1, 1] = (A2[0, 0]*A2[2, 2]-A2[0, 2]*A2[2, 0])/A2det
    A2inv[1, 2] = (A2[0, 2]*A2[1, 0]-A2[0, 0]*A2[1, 2])/A2det
    A2inv[2, 0] = (A2[1, 0]*A2[2, 1]-A2[1, 1]*A2[2, 0])/A2det
    A2inv[2, 1] = (A2[0, 1]*A2[2, 0]-A2[0, 0]*A2[2, 1])/A2det
    A2inv[2, 2] = (A2[0, 0]*A2[1, 1]-A2[0, 1]*A2[1, 0])/A2det
    return A2inv

# ------------------------ INITIATE (IDENTITY) TENSORS ------------------------


# identity tensor (single tensor)
i = np.eye(3)
# identity tensors (grid)
I = np.einsum('ij,xyz',                  i, np.ones([Nx, Ny, Nz]))
I4 = np.einsum('ijkl,xyz->ijklxyz', np.einsum('il,jk', i, i),
               np.ones([Nx, Ny, Nz]))
I4rt = np.einsum('ijkl,xyz->ijklxyz', np.einsum('ik,jl', i, i),
                 np.ones([Nx, Ny, Nz]))
I4s = (I4+I4rt)/2.
II = dyad22(I, I)

# ------------------------------------ FFT ------------------------------------

# projection operator (only for non-zero frequency, associated with the mean)
# NB: vectorized version of "hyper-elasticity.py"
# - allocate / support function
Ghat4 = np.zeros([3, 3, 3, 3, Nx, Ny, Nz])                # projection operator
x = np.zeros([3, Nx, Ny, Nz], dtype='int64')  # position vectors
q = np.zeros([3, Nx, Ny, Nz], dtype='int64')  # frequency vectors


# Dirac delta function
def delta(i, j): return np.float(i == j)


# - set "x" as position vector of all grid-points   [grid of vector-components]
x[0], x[1], x[2] = np.mgrid[:Nx, :Ny, :Nz]
# - convert positions "x" to frequencies "q"        [grid of vector-components]
for i in range(3):
    freq = np.arange(-(shape[i]-1)/2, +(shape[i]+1)/2, dtype='int64')
    q[i] = freq[x[i]]
# - compute "Q = ||q||", and "norm = 1/Q" being zero for the mean (Q==0)
#   NB: avoid zero division
q = q.astype(np.float)
Q = dot11(q, q)
Z = Q == 0
Q[Z] = 1.
norm = 1./Q
norm[Z] = 0.
# - set projection operator                                   [grid of tensors]
for i, j, l, m in itertools.product(range(3), repeat=4):
    Ghat4[i, j, l, m] = norm*delta(i, m)*q[j]*q[l]

# (inverse) Fourier transform (for each tensor component in each direction)


def fft(x): return np.fft.fftshift(
    np.fft.fftn(np.fft.ifftshift(x), [Nx, Ny, Nz]))


def ifft(x): return np.fft.fftshift(
    np.fft.ifftn(np.fft.ifftshift(x), [Nx, Ny, Nz]))


# functions for the projection 'G', and the product 'G : K^LT : (delta F)^T'
def G(A2): return np.real(ifft(ddot42(Ghat4, fft(A2)))).reshape(-1)


def K_dF(dFm): return trans2(ddot42(K4, trans2(dFm.reshape(3, 3, Nx, Ny, Nz))))


def G_K_dF(dFm): return G(K_dF(dFm))

# --------------------------- CONSTITUTIVE RESPONSE ---------------------------

# constitutive response to a certain loading and history
# NB: completely uncoupled from the FFT-solver, but implemented as a regular
#     grid of quadrature points, to have an efficient code;
#     each point is completely independent, just evaluated at the same time


def constitutive(F, F_t, be_t, ep_t):

    # function to compute linearization of the logarithmic Finger tensor
    def dln2_d2(A2):
        vals, vecs = eig2(A2)
        K4 = np.zeros([3, 3, 3, 3, Nx, Ny, Nz])
        for m, n in itertools.product(range(3), repeat=2):
            gc = (np.log(vals[n])-np.log(vals[m]))/(vals[n]-vals[m])
            gc[vals[n] == vals[m]] = (1.0/vals[m])[vals[n] == vals[m]]
            K4 += gc*dyad22(dyad11(vecs[:, m], vecs[:, n]),
                            dyad11(vecs[:, m], vecs[:, n]))
        return K4

    # elastic stiffness tensor
    C4e = K*II+2.*mu*(I4s-1./3.*II)

    # trial state
    Fdelta = dot22(F, inv2(F_t))
    be_s = dot22(Fdelta, dot22(be_t, trans2(Fdelta)))
    lnbe_s = ln2(be_s)
    tau_s = ddot42(C4e, lnbe_s)/2.
    taum_s = ddot22(tau_s, I)/3.
    taud_s = tau_s-taum_s*I
    taueq_s = np.sqrt(3./2.*ddot22(taud_s, taud_s))
    div = np.where(taueq_s < 1e-12, np.ones_like(taueq_s), taueq_s)
    N_s = 3./2.*taud_s/div
    phi_s = taueq_s-(tauy0+H*ep_t)
    phi_s = 1./2.*(phi_s+np.abs(phi_s))

    # return map
    dgamma = phi_s/(H+3.*mu)
    ep = ep_t + dgamma
    tau = tau_s - 2.*dgamma*N_s*mu
    lnbe = lnbe_s-2.*dgamma*N_s
    be = exp2(lnbe)
    P = dot22(tau, trans2(inv2(F)))

    # consistent tangent operator
    a0 = dgamma*mu/taueq_s
    a1 = mu/(H+3.*mu)
    C4ep = ((K-2./3.*mu)/2.+a0*mu)*II+(1.-3.*a0) * \
        mu*I4s+2.*mu*(a0-a1)*dyad22(N_s, N_s)
    dlnbe4_s = dln2_d2(be_s)
    dbe4_s = 2.*dot42(I4s, be_s)

    K4a = np.where(phi_s <= 0, C4e/2., C4ep)
    K4b = ddot44(K4a, ddot44(dlnbe4_s, dbe4_s))
    K4c = dot42(-I4rt, tau)+K4b
    K4 = dot42(dot24(inv2(F), K4c), trans2(inv2(F)))

    return P, K4, be, ep, dlnbe4_s, dbe4_s, K4a, K4b, K4c


# phase indicator: square inclusion of volume fraction (3*3*15)/(11*13*15)
phase = np.zeros([Nx, Ny, Nz])
phase[0, 0, 0] = 1.
# function to convert material parameters to grid of scalars


def param(M0, M1): return M0*np.ones([Nx, Ny, Nz])*(1.-phase) +\
    M1*np.ones([Nx, Ny, Nz]) * phase


# material parameters
K = param(0.833, 0.833)  # bulk      modulus
Kmat = K
mu = param(0.386, 0.386)  # shear     modulus
H = param(0.004, 0.008)  # hardening modulus
tauy0 = param(0.003, 0.006)  # initial yield stress

# ---------------------------------- LOADING ----------------------------------

# stress, deformation gradient, plastic strain, elastic Finger tensor
# NB "_t" signifies that it concerns the value at the previous increment
ep_t = np.zeros([Nx, Ny, Nz])
P = np.zeros([3, 3, Nx, Ny, Nz])
F = np.array(I, copy=True)
F_t = np.array(I, copy=True)
be_t = np.array(I, copy=True)

# initialize macroscopic incremental loading
ninc = 50
lam = 0.0
barF = np.array(I, copy=True)
barF_t = np.array(I, copy=True)

# initial tangent operator: the elastic tangent
K4 = K*II+2.*mu*(I4s-1./3.*II)


class ElastoPlastic_Check(unittest.TestCase):
    t2_comparator = elastic_ref.LinearElastic_Check.t2_comparator
    t4_comparator = elastic_ref.LinearElastic_Check.t4_comparator

    def scalar_comparator(self, µ, g):
        err_sum = 0.
        err_max = 0.
        for counter, (i, j, k) in enumerate(self.rve):
            print((i, j, k))
            µ_arr = µ[counter]
            g_arr = g[i, j, k]
            self.assertEqual(Nz*Ny*i+Nz*j + k, counter)
            print("µSpectre:")
            print(µ_arr)
            print("Goose:")
            print(g_arr)
            print(µ_arr-g_arr)
            err = linalg.norm(µ_arr-g_arr)
            print("error norm = {}".format(err))
            err_sum += err
            err_max = max(err_max, err)
            pass
        print("∑(err) = {}, max(err) = {}".format(err_sum, err_max))
        return err_sum

    def setUp(self):
        # ---------------------------- µSpectre init --------------------------
        nb_grid_pts = list(phase.shape)
        dim = len(nb_grid_pts)
        self.dim = dim

        center = np.array([r//2 for r in nb_grid_pts])
        incl = nb_grid_pts[0]//5

        # Domain dimensions
        lengths = [float(r) for r in nb_grid_pts]
        ## formulation (small_strain or finite_strain)
        formulation = µ.Formulation.finite_strain

        # build a computational domain
        self.rve = µ.Cell(nb_grid_pts, lengths, formulation)

        def get_E_nu(bulk, shear):
            Young = 9*bulk*shear/(3*bulk + shear)
            Poisson = Young/(2*shear) - 1
            return Young, Poisson

        mat = µ.material.MaterialHyperElastoPlastic1_3d

        E, nu = get_E_nu(.833, .386)
        H = 0.004
        tauy0 = .003
        self.hard = mat.make(self.rve, 'hard', µ.OneQuadPt, E, nu, 2*tauy0, 2*H)
        self.soft = mat.make(self.rve, 'soft', µ.OneQuadPt, E, nu,   tauy0,   H)

        for pixel in self.rve:
            if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
                self.hard.add_pixel(pixel)
            else:
                self.soft.add_pixel(pixel)
                pass
            pass
        return

    def test_solve(self):
        strict_tol = 1e-11
        cg_tol = 1e-11
        after_cg_tol = 1e-10
        newton_tol = 1e-5
        self.rve.set_uniform_strain(np.array(np.eye(ndim)))
        µF = self.rve.get_strain()
        µF_t = µF.copy()
        µbarF_t = µF.copy()
        # incremental deformation
        for inc in range(1, ninc):

            print('=============================')
            print('inc: {0:d}'.format(inc))

            # set macroscopic deformation gradient (pure-shear)
            global lam, F, F_t, barF_t, K4, be_t, ep_t
            lam += 0.2/float(ninc)
            barF = np.array(I, copy=True)
            barF[0, 0] = (1.+lam)
            barF[1, 1] = 1./(1.+lam)

            def rel_error_scalar(µ, g, tol, do_assert=True):
                err = (linalg.norm(scalar_vec_to_goose(µ) - g.reshape(-1)) /
                       linalg.norm(g))
                if not (err < tol):
                    self.scalar_comparator(µ.reshape(-1), g)
                    if do_assert:
                        self.assertLess(err, tol)
                    else:
                        print("AssertionWarning: {} is not less than {}".format(
                            err, tol))
                    pass
                return err

            def rel_error_t2(µ, g, tol, do_assert=True):
                err = linalg.norm(t2_vec_to_goose(
                    µ) - g.reshape(-1)) / linalg.norm(g)
                if not (err < tol):
                    self.t2_comparator(µ.reshape(µF.shape), g.reshape(F.shape))
                    if do_assert:
                        self.assertLess(err, tol)
                    else:
                        print("AssertionWarning: {} is not less than {}".format(
                            err, tol))
                    pass
                return err

            def rel_error_t4(µ, g, tol, right_transposed=True, do_assert=True,
                             pixel_tol=1e-4):
                err = linalg.norm(t4_vec_to_goose(
                    µ) - g.reshape(-1)) / linalg.norm(g)
                errors = None
                if not (err < tol):
                    err_sum, errors = self.t4_comparator(µ.reshape(µK.shape),
                                                         g.reshape(K4.shape),
                                                         right_transposed)
                    if do_assert:
                        self.assertLess(err, tol)
                    else:
                        print("AssertionWarning: {} is not less than {}".format(
                            err, tol))

                    pass
                return err, errors

            def abs_error_t2(µ, g, tol, do_assert=True):
                ref_norm = linalg.norm(g)
                if ref_norm > 1:
                    return rel_error_t2(µ, g, tol, do_assert)
                else:
                    err = linalg.norm(t2_vec_to_goose(µ) - g.reshape(-1))
                    if not (err < tol):
                        self.t2_comparator(
                            µ.reshape(µF.shape), g.reshape(F.shape))
                        if do_assert:
                            self.assertLess(err, tol)
                        else:
                            print(("AssertionWarning: {} is not less than {}" +
                                   "").format(
                                err, tol))
                    return err

            rel_error_t2(µF, F, strict_tol)

            # store normalization
            Fn = np.linalg.norm(F)

            # first iteration residual: distribute "barF" over grid using "K4"
            b = -G_K_dF(barF-barF_t)
            F += barF-barF_t

            # parameters for Newton iterations: normalization and iteration
            # counter
            Fn = np.linalg.norm(F)
            iiter = 0

            # µSpectre inits
            µbarF = np.zeros_like(µF)
            µbarF[0, :] = 1. + lam
            µbarF[ndim + 1, :] = 1./(1. + lam)
            µbarF[-1, :] = 1.
            rel_error_t2(µbarF, barF, strict_tol)
            if inc == 1:
                µP, µK = self.rve.evaluate_stress_tangent(µF)
            rel_error_t4(µK, K4, strict_tol)
            µF += µbarF - µbarF_t
            rel_error_t2(µF, F, strict_tol)

            µFn = linalg.norm(µF)
            self.assertLess((µFn-Fn)/Fn, strict_tol)

            def µG_K_dF(x): return self.rve.directional_stiffness(
                x.reshape(µF.shape)).reshape(-1)

            def µG(x): return self.rve.project(x).reshape(-1)
            µb = -µG_K_dF(µbarF-µbarF_t)
            abs_error_t2(µb, b, strict_tol)
            # because of identical elastic properties, µb has got to
            # be zero before plasticity kicks in
            print("inc = {}".format(inc))
            if inc == 1:
                self.assertLess(linalg.norm(µb), strict_tol)

            global_be_t = self.rggve.get_globalised_current_real_field(
                "Previous left Cauchy-Green deformation bₑᵗ")

            # iterate as long as the iterative update does not vanish
            while True:

                # solve linear system using the Conjugate Gradient iterative
                # solver
                g_counter = Counter()
                dFm, _ = sp.cg(tol=cg_tol,
                               atol=1e-10,
                               A=sp.LinearOperator(
                                   shape=(F.size, F.size), matvec=G_K_dF,
                                   dtype='float'),
                               b=b,
                               callback=g_counter
                               )
                µ_counter = Counter()
                µdFm, _ = sp.cg(tol=cg_tol,
                                atol=1e-10,
                                A=sp.LinearOperator(shape=(F.size, F.size),
                                                    matvec=µG_K_dF,
                                                    dtype='float'),
                                b=µb,
                                callback=µ_counter)

                err = g_counter.get()-µ_counter.get()
                if err != 0:
                    print(
                        "n_iter(g) = {}, n_iter(µ) = {}".format(
                            g_counter.get(), µ_counter.get()))
                    print("AssertionWarning: {} != {}".format(g_counter.get(),
                                                              µ_counter.get()))
                try:
                    err = abs_error_t2(µdFm, dFm, after_cg_tol, do_assert=True)
                except Exception as err:
                    raise err
                # add solution of linear system to DOFs
                F += dFm.reshape(3, 3, Nx, Ny, Nz)
                µF += µdFm.reshape(µF.shape)

                err = rel_error_t2(µF, F, strict_tol, do_assert=True)
                # compute residual stress and tangent, convert to residual
                P, K4, be, ep, dln, dbe4_s, K4a, K4b, K4c = constitutive(
                    F, F_t, be_t, ep_t)
                µP, µK = self.rve.evaluate_stress_tangent(µF)
                err = rel_error_t2(µP, P, strict_tol)
                µbe = self.rve.get_globalised_current_real_field(
                    "Previous left Cauchy-Green deformation bₑᵗ")
                err = rel_error_t2(µbe, be, strict_tol)

                µep = self.rve.get_globalised_current_real_field(
                    "cumulated plastic flow εₚ")
                err = rel_error_scalar(µep, ep, strict_tol)

                err, errors = rel_error_t4(µK, K4, strict_tol, do_assert=False)
                if not err < strict_tol:
                    def t2_disp(name, µ, g, i, j, k, index):
                        g_v = g[:, :, i, j, k].copy()
                        µ_v = µ[:, index].reshape(3, 3).T
                        print("{}_g =\n{}".format(name, g_v))
                        print("{}_µ =\n{}".format(name, µ_v))
                        print("{}_err = {}".format(
                            name, np.linalg.norm(g_v-µ_v)))
                        return g_v, µ_v

                    def t0_disp(name, µ, g, i, j, k, index):
                        g_v = g[i, j, k].copy()
                        µ_v = µ[:, index]
                        print("{}_g =\n{}".format(name, g_v))
                        print("{}_µ =\n{}".format(name, µ_v))
                        print("{}_err = {}".format(name, abs(g_v-µ_v)))
                        return g_v, µ_v

                    for pixel, error in errors.items():
                        for index, (ir, jr, kr) in enumerate(self.rve):
                            i, j, k = pixel
                            if (i, j, k) == (ir, jr, kr):
                                break
                        if error > 1e-4:
                            i, j, k = pixel
                            print("error for pixel {} ({}) = {}".format(
                                pixel, index, error))
                            t2_disp("F", µF, F, i, j, k, index)
                            t2_disp("be", µbe, be, i, j, k, index)
                            F_t_n, dummy = t2_disp(
                                "F_t", µF_t, F_t, i, j, k, index)
                            print("ep.shape = {}".format(µep.shape))
                            t0_disp("ep", µep, ep, i, j, k, index)
                            K_comp_g = K4[:, :, :, :, i, j, k].reshape(9, 9)
                            K_comp_µ = µK[:, index].reshape(
                                3, 3, 3, 3).transpose(
                                    1, 0, 3, 2).reshape(9, 9)
                            F_n = F[:, :, i, j, k]
                            be_n = be_t[:, :, i, j, k]
                            ep_n = ep_t[i, j, k]
                            response = constitutive_standalone(Kmat[pixel],
                                                               mu[pixel],
                                                               H[pixel],
                                                               tauy0[pixel],
                                                               F_n,
                                                               F_t_n, be_n,
                                                               ep_n, 3)
                            P_gn, K_gn = response[0], response[2]
                            P_µn, K_µn = PK1_fun_3d(Kmat[pixel],
                                                    mu[pixel],
                                                    H[pixel],
                                                    tauy0[pixel], F_n,
                                                    F_t_n, be_n, ep_n)
                            K_µn = K_µn.reshape(3, 3, 3, 3).transpose(
                                1, 0, 3, 2).reshape(9, 9)
                            print("P_µn =\n{}".format(P_µn))
                            print("P_gn =\n{}".format(P_gn))
                            P_comp_gn, P_comp_µn = t2_disp(
                                "P_comp", µP, P, i, j, k, index)
                            print("|P_µn - P_comp_µn| = {}".format(
                                np.linalg.norm(P_µn-P_comp_µn)))
                            print("|P_gn - P_comp_gn| = {}".format(
                                np.linalg.norm(P_gn-P_comp_gn)))
                            print("|P_gn - P_µn| = {}".format(
                                np.linalg.norm(P_gn-P_µn)))
                            print("|P_comp_gn - P_comp_µn| = {}".format(
                                np.linalg.norm(P_comp_gn-P_comp_µn)))
                            print()
                            K_gn.shape = 9, 9
                            print("K_µn.shape = {}".format(K_µn.shape))
                            print("K_gn.shape = {}".format(K_gn.shape))
                            print("K_comp_g =\n{}".format(K_comp_g))
                            print("K_comp_µ =\n{}".format(K_comp_µ))
                            print("K_µn=\n{}".format(K_µn))
                            print("K_gn=\n{}".format(K_gn))
                            print("|K_µn - K_comp_µ| = {}".format(
                                np.linalg.norm(K_µn-K_comp_µ)))
                            print("|K_gn - K_comp_g| = {}".format(
                                np.linalg.norm(K_gn-K_comp_g)))
                            print("|K_gn - K_µn| = {}".format(
                                np.linalg.norm(K_gn-K_µn)))
                            print("|K_comp_g - K_comp_µ| = {}".format(
                                np.linalg.norm(K_comp_g-K_comp_µ)))
                            break

                    raise AssertionError(
                        "at iiter = {}, inc = {}, caught this: '{}'".format(
                            iiter, inc, err))

                b = -G(P)
                µb = -µG(µP)

                err = abs_error_t2(µb, b, strict_tol)  # after_cg_tol)

                # check for convergence, print convergence info to screen
                print(
                    'Goose:   rel_residual {:10.15e}, |rhs|: {:10.15e}'.format(
                        np.linalg.norm(dFm)/Fn, linalg.norm(b)))
                print(
                    'µSpectre:rel_residual {:10.15e}, |rhs|: {:10.15e}'.format(
                        np.linalg.norm(µdFm)/µFn, linalg.norm(µb)))
                if np.linalg.norm(dFm)/Fn < 1.e-5 and iiter > 0:
                    break

                # update Newton iteration counter
                print("reached end of iiter = {}".format(iiter))
                iiter += 1

            # end-of-increment: update history
            barF_t = np.array(barF, copy=True)
            µbarF_t[:] = µbarF
            F_t = np.array(F, copy=True)
            be_t = np.array(be, copy=True)
            ep_t = np.array(ep, copy=True)
            µF_t[:] = µF
            self.rve.save_history_variables()


if __name__ == '__main__':
    unittest.main()

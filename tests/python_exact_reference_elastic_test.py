#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_exact_reference_elastic_test.py

@author Till Junge <till.junge@epfl.ch>

@date   16 Dec 2019

@brief  Tests exactness of each iterate with respect to python reference
        implementation from GooseFFT for elasticity

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
import unittest
import numpy as np
from numpy.linalg import norm
from python_test_imports import µ

import scipy.sparse.linalg as sp
import scipy
import itertools

np.set_printoptions(linewidth=180)
comparator_nb_cols = 9

scipy_version = tuple((int(i) for i in scipy.__version__.split('.')))
if scipy_version < (1, 2, 0):
    def cg(*args, **kwargs):
        if "atol" in kwargs.keys():
            del kwargs["atol"]
        return sp.cg(*args, **kwargs)
else:
    cg = sp.cg

f = np.asfortranarray
# ----------------------------------- GRID ------------------------------------

ndim = 2   # number of dimensions
N = 3  # number of voxels (assumed equal for all directions)

Nx = Ny = Nz = N

# ---------------------- PROJECTION, TENSORS, OPERATIONS ----------------------

# tensor operations/products: np.einsum enables index notation, avoiding loops
# e.g. ddot42 performs $C_ij = A_ijkl B_lk$ for the entire grid


def trans2(A2): return np.einsum('ijpxy          ->jipxy  ', A2)


def ddot42(A4, B2): return np.einsum('ijklpxy,lkpxy  ->ijpxy  ', A4, B2)


def ddot44(A4, B4): return np.einsum('ijklpxy,lkmnpxy->ijmnpxy', A4, B4)


def dot22(A2, B2): return np.einsum('ijpxy  ,jkpxy  ->ikpxy  ', A2, B2)


def dot24(A2, B4): return np.einsum('ijpxy  ,jkmnpxy->ikmnpxy', A2, B4)


def dot42(A4, B2): return np.einsum('ijklpxy,lmpxy  ->ijkmpxy', A4, B2)


def dyad22(A2, B2): return np.einsum('ijpxy  ,klpxy  ->ijklpxy', A2, B2)


# identity tensor                                               [single tensor]
i = f(np.eye(ndim))
# identity tensors                                            [grid of tensors]
I = f(np.einsum('ij,pxy',                  i, np.ones([1, N, N])))
I4 = f(np.einsum('ijkl,pxy->ijklpxy',
                 np.einsum('il,jk', i, i), np.ones([1, N, N])))
I4rt = f(np.einsum('ijkl,pxy->ijklpxy',
                   np.einsum('ik,jl', i, i), np.ones([1, N, N])))
I4s = (I4+I4rt)/2.
II = dyad22(I, I)

# projection operator                                         [grid of tensors]
# NB can be vectorized (faster, less readable), see: "elasto-plasticity.py"
# - support function / look-up list / zero initialize


def delta(i, j): return np.float(i == j)            # Dirac delta function


freq = np.arange(-(N-1)/2., +(N+1)/2.)        # coordinate axis -> freq. axis
Ghat4 = np.zeros([ndim, ndim, ndim, ndim, 1, N, N],
                 order="F")  # zero initialize
# - compute
for i, j, l, m in itertools.product(range(ndim), repeat=4):
    for x, y in itertools.product(range(N),   repeat=2):
        q = np.array([freq[x], freq[y]])  # frequency vector
        if not q.dot(q) == 0:                      # zero freq. -> mean
            Ghat4[i, j, l, m, 0, x, y] = delta(i, m)*q[j]*q[l]/(q.dot(q))

# (inverse) Fourier transform (for each tensor component in each direction)


def fft(x): return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x), [N, N]))


def ifft(x): return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x), [N, N]))


# functions for the projection 'G', and the product 'G : K^LT : (delta F)^T'
def G(A2): return np.real(ifft(ddot42(Ghat4, fft(A2)))).reshape(-1)


def Gfull(A2): return np.real(ifft(ddot42(Ghat4, fft(A2))))


def K_dF(dFm): return trans2(
    ddot42(K4, trans2(dFm.reshape(ndim, ndim, 1, N, N))))


def G_K_dF(dFm): return G(K_dF(dFm))


def Gfull_K_dF(dFm): return Gfull(K_dF(dFm))

# ------------------- PROBLEM DEFINITION / CONSTITIVE MODEL -------------------


# phase indicator: cubical inclusion of volume fraction (9**3)/(31**3)
phase = np.zeros([N, N], order="F")
phase[:2, :2] = 1.
# material parameters + function to convert to grid of scalars


def param(M0, M1): return M0 * \
    np.ones([N, N], order="F")*(1.-phase)+M1*np.ones([N, N], order="F")*phase


K = param(0.833, 8.33)  # bulk  modulus                   [grid of scalars]
mu = param(0.386, 3.86)  # shear modulus                   [grid of scalars]

# constitutive model: grid of "F" -> grid of "P", "K4"        [grid of tensors]


def constitutive(F):
    C4 = K*II+2.*mu*(I4s-1./3*II)
    S = ddot42(C4, .5*(dot22(trans2(F), F)-I))
    P = dot22(F, S)
    K4 = dot24(S, I4)+ddot44(ddot44(I4rt, dot42(dot24(F, C4), trans2(F))), I4rt)
    return P, K4


F = np.array(I, copy=True, order="F")
P, K4 = constitutive(F)


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
    def setUp(self):
        self.rel_tol = 1e-13
        # ---------------------------- µSpectre init -----------------------------------
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

        mat = µ.material.MaterialLinearElastic1_2d

        E, nu = get_E_nu(.833, .386)
        hard = mat.make(self.rve, 'hard', 10*E, nu)
        soft = mat.make(self.rve, 'soft',    E, nu)

        for id, pixel in self.rve.pixels.enumerate():
            if phase[pixel[0], pixel[1]]:
                hard.add_pixel(id)
            else:
                soft.add_pixel(id)
                pass
            pass
        return

    def comparator(self, a, b, name, tol=None, ref=None):
        if tol is None:
            tol = self.rel_tol
        if ref is None:
            ref = norm(a)
        error = norm(a-b)/ref

        if not error < tol:
            print("g{0} =\n{1}\nµ{0} =\n{2}".format(name, a, b))
        self.assertLess(error, tol)
        return error

    def test_constitutive_law(self):
        # define some random strain
        F = I + (np.random.random(I.shape)-.5)*1e-3

        gP, gK = constitutive(F)
        µP, µK = self.rve.evaluate_stress_tangent(F)
        P_error = norm(gP-µP)/norm(gP)
        if not P_error < self.rel_tol:
            print("gP =\n{}\nµP =\n{}".format(gP, µP))
        self.assertLess(P_error, self.rel_tol)

        # I can't explain the ordering of de Geus' axes in the tangent moduli,
        # however, they yield the same projected stiffness (see
        # test_directional_stiffness)
        K_error = norm(gK.transpose(1, 0, 2, 3, 4, 5, 6)-µK)/norm(gK)
        self.assertLess(K_error, self.rel_tol)

    def test_directional_stiffness(self):
        # define some random strain
        self.rve.evaluate_stress_tangent(I)
        dF = (np.random.random(I.shape)-.5)*1e-3
        print("\nThe shape of T is: \n {}".format(I.shape))
        gG = Gfull_K_dF(dF)
        µG = self.rve.directional_stiffness(dF)

        error = norm(gG-µG)/norm(gG)
        if not error < self.rel_tol:
            print("gG =\n{}\nµG =\n{}".format(gG, µG))
        self.assertLess(error, self.rel_tol)

    def test_solve(self):
        before_cg_tol = 1e-11
        cg_tol = 1e-11
        after_cg_tol = 1e-9
        newton_tol = 1e-4

        global K4, P, F
        F[:] = I
        self.rve.evaluate_stress_tangent(F)
        # set macroscopic loading
        DbarF = np.zeros([ndim, ndim, 1, N, N], order='F')
        DbarF[0, 1] = 1.0

        # initial residual: distribute "barF" over grid using "K4"
        b = -G_K_dF(DbarF)
        def µG_K_dF(x): return self.rve.directional_stiffness(
            x.reshape(F.shape)).reshape(-1)
        µb = -µG_K_dF(DbarF)
        def µG(x): return self.rve.project(x.reshape(F.shape)).reshape(-1)

        self.comparator(b, µb, 'b')
        F += DbarF
        µF = F.copy()
        Fn = np.linalg.norm(F)
        iiter = 0

        P, K4 = constitutive(F)
        µP, µK = self.rve.evaluate_stress_tangent(µF)

        self.comparator(P, µP, "P")
        self.comparator(K4.transpose(1, 0, 2, 3, 4, 5, 6), µK, "K")

        # iterate as long as the iterative update does not vanish
        while True:
            # solve linear system using CG
            g_counter = Counter()
            dFm, _ = cg(tol=cg_tol,
                        A=sp.LinearOperator(shape=(F.size, F.size),
                                            matvec=G_K_dF, dtype='float'),
                        b=b,
                        callback=g_counter, atol=0
                        )
            µ_counter = Counter()
            µdFm, _ = cg(tol=cg_tol,
                         A=sp.LinearOperator(shape=(F.size, F.size),
                                             matvec=µG_K_dF, dtype='float'),
                         b=µb,
                         callback=µ_counter, atol=0)

            err = g_counter.get()-µ_counter.get()

            if err != 0:
                print("n_iter(g) = {}, n_iter(µ) = {}".format(g_counter.get(),
                                                              µ_counter.get()))
                pass

            # in the last iteration, the increment is essentially
            # zero, so we don't care about relative error anymore
            err = self.comparator(dFm, µdFm, "dFm", tol=cg_tol, ref=Fn)
            if norm(dFm)/Fn > newton_tol:
                if not (err < after_cg_tol):
                    print("µdFm.shape = {}".format(µdFm.shape))
                    print("|µdFm| = {}".format(norm(µdFm)))
                    print("|dFm| = {}".format(norm(dFm)))
                    print("|µdFm - dFm| = {}".format(norm(µdFm-dFm)))
                    print("AssertionWarning: {} is not less than {}".format(err,
                                                                            after_cg_tol))
                self.assertLess(err, after_cg_tol)

            # update DOFs (array -> tens.grid)
            F += dFm.reshape(F.shape)
            µF += µdFm.reshape(F.shape)
            # new residual stress and tangent
            P, K4 = constitutive(F)
            µP, µK = self.rve.evaluate_stress_tangent(µF)

            self.comparator(P, µP, "P", tol=after_cg_tol)
            self.comparator(K4.transpose(1, 0, 2, 3, 4, 5, 6),
                            µK, "K", tol=after_cg_tol)

            # convert res.stress to residual
            b = -G(P)
            µb = -µG(µP)
            self.comparator(b, µb, 'b', tol=after_cg_tol, ref=Fn)

            print('Goose:    %10.15e' % (np.linalg.norm(dFm)/Fn))
            print('µSpectre: %10.15e' % (np.linalg.norm(µdFm)/Fn))
            if np.linalg.norm(dFm)/Fn < newton_tol and iiter > 0:
                break  # check convergence
            iiter += 1

        print("done")


if __name__ == '__main__':
    unittest.main()

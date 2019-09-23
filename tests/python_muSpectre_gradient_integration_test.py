# !/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file  python_muSpectre_gradient_integration_test.py

@author Richard Leute <richard.leute@imtek.uni-freiburg.de>

@date   23 Nov 2018

@brief  test the functionality of gradient_integration.py

Copyright © 2018 Till Junge, Richard Leute

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
import scipy.misc as sm
import itertools
from python_test_imports import µ

### Helper functions
def init_X_F_Chi(lens, res, rank=2):
    """
    Setup all the needed parameters for initialization of the deformation
    gradient F and the corresponding deformation map/field Chi_X.

    Keyword Arguments:
    lens -- list [Lx, Ly, ...] of box lengths in each direction (dtype=float)
    res  -- list [Nx, Ny, ...] of grid resoultions (dtype = int)
    rank -- int (default=2), rank of the deformation gradient tensor F.
            (dtype = int)

    Returns:
    d    : np.array of grid spacing for each spatial direction (dtype = float)
    dim  : int dimension of the structure, derived from len(res).
    x_n  : np.ndarray shape=(res.shape+1, dim) initial nodal/corner positions
           as created by gradient_integration.compute_grid (dtype = float)
    x_c  : np.ndarray shape=(res.shape+1, dim) initial cell center positions
           as created by gradient_integration.compute_grid (dtype = float)
    F    : np.zeros shape=(res.shape, dim*rank) initialise deformation gradient
           (dtype = float)
    Chi_n: np.zeros shape=((res+1).shape, dim) initialise deformation field
           (dtype = float)
    Chi_c: np.zeros shape=(res.shape, dim) initialise deformation field
           (dtype = float)
    freqs: np.ndarray as returned by compute_wave_vectors(). (dtype = float)
    """
    lens = np.array(lens)
    res  = np.array(res)
    d    = lens / res
    dim  = len(res)
    x_n, x_c = µ.gradient_integration.compute_grid(lens, res)
    F     = np.zeros(x_c.shape + (dim,)*(rank-1))
    Chi_n = np.zeros(x_n.shape)
    Chi_c = np.zeros(x_c.shape)
    freqs = µ.gradient_integration.compute_wave_vectors(lens, res)

    return d, dim, x_n, x_c, F, Chi_n, Chi_c, freqs

def correct_to_zero_mean(Chi, d, nodal=False):
    """
    corrects the displacement field such that it's integral is zero. By this one
    can get rid of a constant factor in the deformation gradient. This function
    is specialized for this file and should not be used somewhere else.

    Keywords:
    Chi   : np.ndarray of the uncorrected analytic placements (dtype = float)
    d     : np.array of the gridspacing in each spatial direction (dtype = float)
    nodal : bool (default False) specifies if the input are nodal or cell/center
            values. Default interpretation are cell/center values.

    Returns:
    Chi   : np.ndarray of zero mean corrected analytic placements (dtype = float)
    """
    Chi_zm = np.copy(Chi)
    res = np.array(Chi_zm.shape[:-1])
    dim = res.size
    Chi_zm -= (Chi_zm.sum(axis=tuple(range(dim)))/np.prod(res))\
              .reshape((1,)*dim + (dim,))
    return Chi_zm

def test_integrate(order, F, Chi_n, Chi_c, tol):
    """
    make the convergence tests for the integration

    Keywords:
    order : list of integration orders which are tested (dtype = int)
    F     : np.ndarray applied deformation gradient (dtype = float)
    Chi_n : np.ndarray expected node positions (dtype = float)
    Chi_c : np.ndarray expected cell positions (dtype = float)
    tol   : list of tolerances for each order. If it is a single value the same
            tolerance is used for each order. (dtype = float)
    """
    print('Maybe implement a function like this...')

def central_diff_derivative(data, d, order, rank=1):
    """
    Compute the first derivative of a function with values 'data' with the
    central difference approximation to the order 'order'. The function values
    are on a rectangualar grid with constant grid spacing 'd' in each direction.

    CAUTION:
    The function is assumed to be periodic (pbc)!
    Thus, if there is a discontinuity at the boundaries you have to expect
    errors in the derivative at the vicinity close to the discontinuity.

    Keyword Arguments:
    data  -- np.ndarray of shape=(nb_grid_pts_per_dim, dim*rank) function values
             on an equally spaced grid, with grid spacing 'd' (dtype = float)
    d     -- scalar or np.array of grid spacing in each direction. Scalar is
             interpreted as equal spacing in each direction (dtype = float)
    order -- int >= 1, gives the accuracy order of the central difference
             approximation (dtype = int)
    rank  -- int, rank of the data tensor

    Returns:
    deriv: np.ndarray of shape=(nb_grid_pts_per_dim, dim, dim) central
           difference derivative of given order (dtype = float)
    """
    dim = len(data.shape)-rank
    weights = sm.central_diff_weights(2*order + 1)
    deriv = np.zeros(data.shape + (dim,))
    for ax in range(dim):
        for i in range(2*order + 1):
            deriv[...,ax] += weights[i]*np.roll(data, order-i, axis=ax) / d[ax]

    return deriv


class MuSpectre_gradient_integration_Check(unittest.TestCase):
    """
    Check the implementation of all muSpectre.gradient_integration functions.
    """

    def setUp(self):
        self.lengths    = np.array([2.4, 3.7, 4.1])
        self.nb_grid_pts = np.array([5, 3, 5])

        self.norm_tol = 1e-8

    def test_central_diff_derivative(self):
        """
        Test of the central difference approximation by central_diff_derivative
        of the first derivative of a function on a grid.
        """
        res  = self.nb_grid_pts * 15
        lens = self.lengths
        for j in range(1, len(res)):
            d, dim, x_n, x_c, deriv, f_n, f_c, freqs = init_X_F_Chi(lens[:j],
                                                                    res[:j])
            f_c = np.sin(2*np.pi/lens[:j] * x_c)
            for i in range(j):
                deriv[...,i,i] =2*np.pi/lens[i]*np.cos(2*np.pi*
                                                       x_c[...,i]/lens[i])
            approx_deriv = central_diff_derivative(f_c, d[:j+1], order=5)
            self.assertLess(np.linalg.norm(deriv-approx_deriv),
                            self.norm_tol)

    def test_compute_wave_vectors(self):
        """
        Test the construction of a wave vector grid by compute_wave_vectors
        for an arbitrary dimension.
        """
        lens  = [4, 6, 7]
        res   = [3, 4, 5]
        Nx, Ny, Nz = res
        q_1d = lambda i: 2*np.pi/lens[i] * \
               np.append(np.arange(res[i]-res[i]//2),
                         -np.arange(1, res[i]//2 + 1)[::-1])
        qx = q_1d(0)
        qy = q_1d(1)
        qz = q_1d(2)
        q  = np.zeros(tuple(res) + (3,))
        for i,j,k in itertools.product(range(Nx), range(Ny), range(Nz)):
            q[i,j,k,:] = np.array([qx[i], qy[j], qz[k]])
        for n in range(1,4):
            comp_q = µ.gradient_integration.compute_wave_vectors(lens[:n],
                                                                 res[:n])
            s = (np.s_[:],)*n + (0,)*(3-n) + (np.s_[:n],)
            self.assertLess(np.linalg.norm(comp_q - q[s]), self.norm_tol)

    def test_compute_grid(self):
        """
        Test the function compute_grid which creates an orthogonal
        equally spaced grid of the given number of grid points in each dimension
        and the corresponding  lengths.
        """
        lens = self.lengths
        res  = self.nb_grid_pts
        d    = np.array(lens)/np.array(res)
        grid_n = np.zeros(tuple(res+1) + (len(res),))
        Nx, Ny, Nz = res+1
        for i,j,k in itertools.product(range(Nx), range(Ny), range(Nz)):
            grid_n[i,j,k,:] = np.array([i*d[0], j*d[1], k*d[2]])
        grid_c = (grid_n - d/2)[1:,1:,1:,:]
        for n in range(1,4):
            x_n, x_c  = µ.gradient_integration.compute_grid(lens[:n], res[:n])
            s = (np.s_[:],)*n + (0,)*(3-n) + (np.s_[:n],)
            self.assertLess(np.linalg.norm(x_c - grid_c[s]), self.norm_tol)
            self.assertLess(np.linalg.norm(x_n - grid_n[s]), self.norm_tol)

    def test_reshape_gradient(self):
        """
        Test if reshape gradient transforms a flattend second order tensor in
        the right way to a shape nb_grid_pts + [dim, dim].
        """
        lens = list(self.lengths)
        res  = list(self.nb_grid_pts)
        tol  = 1e-5
        formulation = µ.Formulation.finite_strain
        DelF = np.array([[0   , 0.01, 0.02],
                         [0.03, 0   , 0.04],
                         [0.05, 0.06, 0   ]])
        one  = np.eye(3,3)
        for n in range(2,4):
            sys = µ.Cell(res[:n], lens[:n], formulation)
            if n == 2:
                mat = µ.material.MaterialLinearElastic1_2d.make(sys.wrapped_cell, "material",
                                                                10, 0.3)
            if n == 3:
                mat = µ.material.MaterialLinearElastic1_3d.make(sys.wrapped_cell, "material",
                                                                10, 0.3)
            for pixel in sys:
                mat.add_pixel(pixel)
            solver = µ.solvers.SolverCG(sys.wrapped_cell, tol, maxiter=100, verbose=0)
            r = µ.solvers.newton_cg(sys.wrapped_cell, DelF[:n, :n],
                                    solver, tol, tol , verbose=0)
            grad = µ.gradient_integration.reshape_gradient(r.grad,list(res[:n]))
            grad_theo = (DelF[:n, :n] + one[:n, :n]).reshape((1,)*n+(n,n,))
            self.assertEqual(grad.shape, tuple(res[:n])+(n,n,))
            self.assertLess(np.linalg.norm(grad - grad_theo), self.norm_tol)

    def test_complement_periodically(self):
        """
        Test the periodic reconstruction of an array. Lower left entries are
        added into the upper right part of the array.
        """
        #1D grid scalars
        x_test   = np.array([0,1,2,3])
        x_test_p = np.array([0,1,2,3, 0])
        x_p      = µ.gradient_integration.complement_periodically(x_test, 1)
        self.assertLess(np.linalg.norm(x_p-x_test_p), self.norm_tol)

        #2D grid scalars
        x_test   = np.array([[1,2,3,4],
                             [5,6,7,8]])
        x_test_p = np.array([[1,2,3,4,1],
                             [5,6,7,8,5],
                             [1,2,3,4,1]])
        x_p      = µ.gradient_integration.complement_periodically(x_test, 2)
        self.assertLess(np.linalg.norm(x_p-x_test_p), self.norm_tol)

        #2D grid vectors
        x_test   = np.array([[[1,2,3]   , [3,4,5]]  ,
                             [[6,7,8]   , [9,10,11]],
                             [[12,13,14], [15,6,17]] ])
        x_test_p = np.array([[[1,2,3]   , [3,4,5]  , [1,2,3]]   ,
                             [[6,7,8]   , [9,10,11], [6,7,8]]   ,
                             [[12,13,14], [15,6,17], [12,13,14]],
                             [[1,2,3]   , [3,4,5]  , [1,2,3]]    ])
        x_p      = µ.gradient_integration.complement_periodically(x_test, 2)
        self.assertLess(np.linalg.norm(x_p-x_test_p), self.norm_tol)

    def test_get_integrator(self):
        """
        Test if the right integrator is computed.
        """
        #even grid
        lens_e = np.array([1,1,1])
        res_e  = np.array([2,2,2])
        x_n_e, x_c_e = µ.gradient_integration.compute_grid(lens_e, res_e)
        freqs_e = µ.gradient_integration.compute_wave_vectors(lens_e, res_e)
        #odd grid
        lens_o = np.array([1,1])
        res_o  = np.array([3,3])
        x_n_o, x_c_o = µ.gradient_integration.compute_grid(lens_o, res_o)
        delta_x = 1/3
        freqs_o = µ.gradient_integration.compute_wave_vectors(lens_o, res_o)

        ### order=0
        int_ana = 1j/(2*np.pi)*np.array([[[[ 0  , 0  , 0], [ 0  , 0  ,-1  ]] ,
                                          [[ 0  ,-1  , 0], [ 0  ,-1/2,-1/2]]],
                                         [[[-1  , 0  , 0], [-1/2, 0  ,-1/2]] ,
                                          [[-1/2,-1/2, 0], [-1/3,-1/3,-1/3]]] ])
        dim,shape,integrator = µ.gradient_integration.\
                               get_integrator(x_c_e, freqs_e, order=0)
        self.assertEqual(dim, len(res_e))
        self.assertEqual(shape, tuple(res_e))
        self.assertLess(np.linalg.norm(integrator-int_ana), self.norm_tol)

        ### order=1
        #even grid
        int_ana = np.zeros(res_e.shape)
        dim,shape,integrator = µ.gradient_integration.\
                               get_integrator(x_c_e, freqs_e, order=1)
        self.assertEqual(dim, len(res_e))
        self.assertEqual(shape, tuple(res_e))
        self.assertLess(np.linalg.norm(integrator-int_ana), self.norm_tol)

        #odd grid
        s  = lambda q: np.sin(2*np.pi*q*delta_x)
        sq = lambda q: (np.sin(2*np.pi*np.array(q)*delta_x)**2).sum()
        int_ana = 1j * delta_x *\
                  np.array([[[ 0                 , 0                 ],
                             [ 0                 , s(1)/sq([0,1])    ],
                             [ 0                 , s(-1)/sq([0,-1])  ]],
                            [[ s(1)/sq([1,0])    , 0                 ],
                             [ s(1)/sq([1,1])    , s(1)/sq([1,1])    ],
                             [ s(1)/sq([1,-1])   , s(-1)/sq([1,-1])  ]],
                            [[ s(-1)/sq([-1,0])  , 0                 ],
                             [ s(-1)/sq([-1,1])  , s(1)/sq([-1,1])   ],
                             [ s(-1)/sq([-1,-1]) , s(-1)/sq([-1,-1]) ]]])

        dim,shape,integrator = µ.gradient_integration.\
                               get_integrator(x_c_o, freqs_o, order=1)
        self.assertEqual(dim, len(res_o))
        self.assertEqual(shape, tuple(res_o))
        self.assertLess(np.linalg.norm(integrator-int_ana), self.norm_tol)

        ### order=2
        #even grid
        int_ana = np.zeros(res_e.shape)
        dim,shape,integrator = µ.gradient_integration.\
                               get_integrator(x_c_e, freqs_e, order=2)
        self.assertEqual(dim, len(res_e))
        self.assertEqual(shape, tuple(res_e))
        self.assertLess(np.linalg.norm(integrator-int_ana), self.norm_tol)

        #odd grid
        lens_o = np.array([1,1])
        res_o  = np.array([3,3])
        x_n, x_c = µ.gradient_integration.compute_grid(lens_o, res_o)
        delta_x = 1/3
        freqs    = µ.gradient_integration.compute_wave_vectors(lens_o, res_o)
        s  = lambda q: 8*np.sin(2*np.pi*q*delta_x) + np.sin(2*2*np.pi*q*delta_x)
        sq = lambda q: ( (64*np.sin(2*np.pi*np.array(q)*delta_x)**2).sum() -
                         (np.sin(2*2*np.pi*np.array(q)*delta_x)**2).sum() )
        int_ana = 6 * 1j * delta_x *\
                  np.array([[[ 0                 , 0                 ],
                             [ 0                 , s(1)/sq([0,1])    ],
                             [ 0                 , s(-1)/sq([0,-1])  ]],
                            [[ s(1)/sq([1,0])    , 0                 ],
                             [ s(1)/sq([1,1])    , s(1)/sq([1,1])    ],
                             [ s(1)/sq([1,-1])   , s(-1)/sq([1,-1])  ]],
                            [[ s(-1)/sq([-1,0])  , 0                 ],
                             [ s(-1)/sq([-1,1])  , s(1)/sq([-1,1])   ],
                             [ s(-1)/sq([-1,-1]) , s(-1)/sq([-1,-1]) ]]])

        dim,shape,integrator = µ.gradient_integration.\
                               get_integrator(x_c_o, freqs_o, order=2)
        self.assertEqual(dim, len(res_o))
        self.assertEqual(shape, tuple(res_o))
        self.assertLess(np.linalg.norm(integrator-int_ana), self.norm_tol)

    def test_integrate_tensor_2(self):
        """
        Test the correct integration of a second-rank tensor gradient field,
        like the deformation gradient.
        """
        order = [1, 2] #list of higher order finite difference integration which
                       #will be checked.

        ### cosinus, diagonal deformation gradient
        res  = [15, 15, 14]
        lens = [7, 1.4, 3]
        d, dim, x_n, x_c, F, Chi_n, Chi_c, freqs = init_X_F_Chi(lens, res)
        for i in range(dim):
            F[:,:,:,i,i] = 0.8*np.pi/lens[i]*np.cos(4*np.pi*
                                                    x_c[:,:,:,i]/lens[i])
        Chi_n = 0.2 * np.sin(4*np.pi*x_n/lens)
        Chi_c = 0.2 * np.sin(4*np.pi*x_c/lens)
        # zeroth order correction
        placement_n = µ.gradient_integration.integrate_tensor_2(F, x_n, freqs,
                        staggered_grid=True, order=0)
        placement_c = µ.gradient_integration.integrate_tensor_2(F, x_c, freqs,
                        staggered_grid=False, order=0)
        self.assertLess(np.linalg.norm(Chi_n - placement_n), self.norm_tol)
        self.assertLess(np.linalg.norm(Chi_c - placement_c), self.norm_tol)
        # higher order correction
        for n in order:
            tol_n = [1.334, 0.2299] #adjusted tolerances for node points
            F_c = central_diff_derivative(Chi_c, d, order=n)
            placement_n = µ.gradient_integration.integrate_tensor_2(F_c, x_n,
                            freqs, staggered_grid=True, order=n)
            placement_c = µ.gradient_integration.integrate_tensor_2(F_c, x_c,
                            freqs, staggered_grid=False,order=n)
            self.assertLess(np.linalg.norm(Chi_n - placement_n), tol_n[n-1])
            self.assertLess(np.linalg.norm(Chi_c - placement_c), self.norm_tol)

        ### cosinus, non-diagonal deformation gradient
        res  = [15, 12, 11]
        lens = [8, 8, 8]
        d, dim, x_n, x_c, F, Chi_n, Chi_c, freqs = init_X_F_Chi(lens, res)
        F[:,:,:,0,0] = 4*np.pi/lens[0]*np.cos(2*np.pi/lens[0]*x_c[:,:,:,0])
        F[:,:,:,1,1] = 2*np.pi/lens[1]*np.cos(2*np.pi/lens[1]*x_c[:,:,:,1])
        F[:,:,:,2,2] = 2*np.pi/lens[2]*np.cos(2*np.pi/lens[2]*x_c[:,:,:,2])
        F[:,:,:,1,0] = 2*np.pi/lens[0]*np.cos(2*np.pi/lens[0]*x_c[:,:,:,0])
        F[:,:,:,2,0] = 2*np.pi/lens[0]*np.cos(2*np.pi/lens[0]*x_c[:,:,:,0])
        for i in range(dim):
            Chi_c[:,:,:,i]= np.sin(2*np.pi*x_c[:,:,:,i]/lens[i])  \
                            + np.sin(2*np.pi*x_c[:,:,:,0]/lens[0])
            Chi_n[:,:,:,i]= np.sin(2*np.pi*x_n[:,:,:,i]/lens[i])  \
                            + np.sin(2*np.pi*x_n[:,:,:,0]/lens[0])
        # zeroth order correction
        placement_n = µ.gradient_integration.integrate_tensor_2(F, x_n, freqs,
                        staggered_grid=True, order=0)
        placement_c = µ.gradient_integration.integrate_tensor_2(F, x_c, freqs,
                        staggered_grid=False, order=0)
        self.assertLess(np.linalg.norm(Chi_n - placement_n), self.norm_tol)
        self.assertLess(np.linalg.norm(Chi_c - placement_c), self.norm_tol)
        # higher order correction
        for n in order:
            tol_n = [2.563, 0.1544] #adjusted tolerances for node points
            F_c = central_diff_derivative(Chi_c, d, order=n)
            placement_n = µ.gradient_integration.integrate_tensor_2(F_c, x_n,
                            freqs, staggered_grid=True, order=n)
            placement_c = µ.gradient_integration.integrate_tensor_2(F_c, x_c,
                            freqs, staggered_grid=False,order=n)
            self.assertLess(np.linalg.norm(Chi_n - placement_n), tol_n[n-1])
            self.assertLess(np.linalg.norm(Chi_c - placement_c), self.norm_tol)

        ### polynomial, diagonal deformation gradient
        # Choose the prefactors of the polynomial such that at least Chi_X and F
        # have respectively the same values at the boundaries (here X_i=0 and
        # X_i=4).
        res  = [13, 14, 11]
        lens = [4, 4, 4]
        d, dim, x_n, x_c, F, Chi_n, Chi_c, freqs = init_X_F_Chi(lens, res)
        for i in range(dim):
            F[:,:,:,i,i] = 32*x_c[:,:,:,i] -24*x_c[:,:,:,i]**2+4*x_c[:,:,:,i]**3
        Chi_n = -128/15 + 16*x_n**2 -8*x_n**3 +x_n**4
        Chi_c = -128/15 + 16*x_c**2 -8*x_c**3 +x_c**4
        #subtract the mean of Chi_c, because the numeric integration is done to
        #give a zero mean fluctuating deformation field.
        mean_c = Chi_c.sum(axis=tuple(range(dim)))/ \
                 np.array(Chi_c.shape[:-1]).prod()
        Chi_n -= mean_c.reshape((1,)*dim + (dim,))
        Chi_c -= mean_c.reshape((1,)*dim + (dim,))
        # zeroth order correction
        placement_n = µ.gradient_integration.integrate_tensor_2(F, x_n, freqs,
                        staggered_grid=True, order=0)
        placement_c = µ.gradient_integration.integrate_tensor_2(F, x_c, freqs,
                        staggered_grid=False, order=0)
        self.assertLess(np.linalg.norm(Chi_n - placement_n), 0.19477)
        self.assertLess(np.linalg.norm(Chi_c - placement_c), 0.67355)
        # higher order correction
        for n in order:
            tol_n = [18.266, 2.9073] #adjusted tolerances for node points
            F_c = central_diff_derivative(Chi_c, d, order=n)
            placement_n = µ.gradient_integration.integrate_tensor_2(F_c, x_n,
                            freqs, staggered_grid=True, order=n)
            placement_c = µ.gradient_integration.integrate_tensor_2(F_c, x_c,
                            freqs, staggered_grid=False,order=n)
            self.assertLess(np.linalg.norm(Chi_n - placement_n), tol_n[n-1])
            self.assertLess(np.linalg.norm(Chi_c - placement_c), self.norm_tol)

        ### Realistic test:
        #   shear of a two dimensional material with two different Young moduli.
        order_all = [0]+order #orders for which the test is run
        #initialize material structure
        res  = [ 9, 21] #nb_grid_pts
        lens = [ 9, 21] #lengths
        d, dim, x_n, x_c, _, _, _, freqs = init_X_F_Chi(lens, res)
        formulation = µ.Formulation.finite_strain
        Young   = [10, 20]   #Youngs modulus for each phase (soft, hard)
        Poisson = [0.3, 0.3] #Poissons ratio for each phase

        #geometry (two slabs stacked in y-direction with,
        #hight h (soft material) and hight res[1]-h (hard material))
        h            = res[1]//2
        phase        = np.zeros(tuple(res), dtype=int)
        phase[:, h:] = 1
        phase        = phase.flatten(order='F')
        cell = µ.Cell(res, lens, formulation)
        mat  = µ.material.MaterialLinearElastic4_2d.make(cell.wrapped_cell, "material")
        for i, pixel in enumerate(cell):
            mat.add_pixel(pixel, Young[phase[i]], Poisson[phase[i]])
        cell.initialise()
        DelF = np.array([[0 , 0.01],
                         [0 , 0   ]])

        # µSpectre solution
        solver = µ.solvers.SolverCG(cell.wrapped_cell, tol=1e-6, maxiter=100, verbose=0)
        result = µ.solvers.newton_cg(cell.wrapped_cell, DelF, solver, newton_tol=1e-6,
                                     equil_tol=1e-6, verbose=0)
        F = µ.gradient_integration.reshape_gradient(result.grad, res)
        fin_pos = {} #µSpectre computed center and node positions for all orders
        for n in order_all:
            placement_n = µ.gradient_integration.integrate_tensor_2(F, x_n,
                            freqs, staggered_grid=True, order=n)
            placement_c = µ.gradient_integration.integrate_tensor_2(F, x_c,
                            freqs, staggered_grid=False, order=n)
            fin_pos[str(n)+'_n'] = placement_n
            fin_pos[str(n)+'_c'] = placement_c

        # analytic solution, "placement_ana" (node and center)
        l_soft = d[1] * h           #height soft material
        l_hard = d[1] * (res[1]-h)  #height hard material
        Shear_modulus = np.array(Young) / (2 * (1+np.array(Poisson)))
        mean_shear_strain = 2*DelF[0,1]
        shear_strain_soft = (lens[1]*mean_shear_strain) / (l_soft
                                + l_hard * Shear_modulus[0]/Shear_modulus[1])
        shear_strain_hard = (lens[1]*mean_shear_strain) / (l_soft
                                * Shear_modulus[1]/Shear_modulus[0] + l_hard)
        placement_ana_n = np.zeros(x_n.shape)
        placement_ana_c = np.zeros(x_c.shape)

        #x-coordinate
        #soft material
        placement_ana_n[:,:h+1,0] = shear_strain_soft/2 * x_n[:, :h+1, 1]
        placement_ana_c[:,:h  ,0] = shear_strain_soft/2 * x_c[:, :h  , 1]
        #hard material
        placement_ana_n[:,h+1:,0] =shear_strain_hard/2 * (x_n[:,h+1:,1]-l_soft)\
                                   + shear_strain_soft/2 * l_soft
        placement_ana_c[:,h:  ,0] =shear_strain_hard/2 * (x_c[:,h:  ,1]-l_soft)\
                                    + shear_strain_soft/2 * l_soft
        #y-coordinate
        placement_ana_n[:, :, 1] = 0
        placement_ana_c[:, :, 1] = 0

        #shift the analytic solution such that the average nonaffine deformation
        #is zero (integral of the nonaffine deformation gradient + N*const != 0)
        F_homo    = (1./(np.prod(res)) * F.sum(axis=tuple(np.arange(dim))))\
                    .reshape((1,)*dim+(dim,)*2)
        #integration constant = integral of the nonaffine deformation gradient/N
        int_const = - ((placement_ana_c[:,:,0] - F_homo[:,:,0,1] * x_c[:,:,1])
                       .sum(axis=1))[0] / res[1]
        ana_sol_n = placement_ana_n + x_n + \
                    np.array([int_const, 0]).reshape((1,)*dim+(dim,))
        ana_sol_c = placement_ana_c + x_c + \
                    np.array([int_const, 0]).reshape((1,)*dim+(dim,))

        # check the numeric vs the analytic solution
        tol_n = [2.2112e-3, 1.3488e-3, 1.8124e-3]
        tol_c = [3.1095e-3, 3.2132e-2, 1.8989e-2]
        for n in order_all:
            norm_n = np.linalg.norm(fin_pos[str(n)+'_n'] - ana_sol_n)
            norm_c = np.linalg.norm(fin_pos[str(n)+'_c'] - ana_sol_c)
            self.assertLess(norm_n, tol_n[n])
            self.assertLess(norm_c, tol_c[n])


    def test_integrate_vector(self):
        """Test the integration of a first-rank tensor gradient field."""
        order = [1,2]

        ### cosinus deformation gradient vector field
        res  = [13, 14, 13]
        lens = [ 7,  4, 5]
        d, dim, x_n, x_c, df, Chi_n, Chi_c, freqs = init_X_F_Chi(lens, res, 1)
        for i in range(dim):
            df[:,:,:,i] = 0.8*np.pi/lens[i]*np.cos(4*np.pi*x_c[:,:,:,i]/lens[i])
        Chi_n = 0.2 * np.sin(4*np.pi*x_n/lens).sum(axis=-1)
        Chi_c = 0.2 * np.sin(4*np.pi*x_c/lens).sum(axis=-1)
        # zeroth order correction
        placement_n = µ.gradient_integration.integrate_vector(
            df, x_n, freqs, staggered_grid=True, order=0)
        placement_c = µ.gradient_integration.integrate_vector(
            df, x_c, freqs, staggered_grid=False, order=0)
        self.assertLess(np.linalg.norm(Chi_n - placement_n), self.norm_tol)
        self.assertLess(np.linalg.norm(Chi_c - placement_c), self.norm_tol)
        # higher order correction
        for n in order:
            tol_n = [1.404, 0.2882] #adjusted tolerances for node points
            df_c = central_diff_derivative(Chi_c, d, order=n, rank=0)
            placement_n = µ.gradient_integration.integrate_vector(
                df_c, x_n, freqs, staggered_grid=True, order=n)
            placement_c = µ.gradient_integration.integrate_vector(
                df_c, x_c, freqs, staggered_grid=False, order=n)
            self.assertLess(np.linalg.norm(Chi_n - placement_n), tol_n[n-1])
            self.assertLess(np.linalg.norm(Chi_c - placement_c), self.norm_tol)

        ### polynomial deformation gradient vector field
        # Choose the prefactors of the polynomial such that at least Chi_X and F
        # have respectively the same values at the boundaries (here X_i=0 and
        # X_i=4).
        res  = [12, 11, 13]
        lens = [4, 4, 4]
        d, dim, x_n, x_c, df, Chi_n, Chi_c, freqs = init_X_F_Chi(lens, res, 1)
        for i in range(dim):
            df[:,:,:,i] = 32*x_c[:,:,:,i]-24*x_c[:,:,:,i]**2+4*x_c[:,:,:,i]**3
        Chi_n = (-128/15 + 16*x_n**2 -8*x_n**3 +x_n**4).sum(axis=-1)
        Chi_c = (-128/15 + 16*x_c**2 -8*x_c**3 +x_c**4).sum(axis=-1)
        #subtract the mean of Chi_c, because the numeric integration is done to
        #give a zero mean fluctuating deformation field.
        mean_c = Chi_c.sum() / np.array(Chi_c.shape).prod()
        Chi_n -= mean_c
        Chi_c -= mean_c
        # zeroth order correction
        placement_n = µ.gradient_integration.integrate_vector(
            df, x_n, freqs, staggered_grid=True, order=0)
        placement_c = µ.gradient_integration.integrate_vector(
            df, x_c, freqs, staggered_grid=False, order=0)
        self.assertLess(np.linalg.norm(Chi_n - placement_n), 0.20539)
        self.assertLess(np.linalg.norm(Chi_c - placement_c), 0.67380)
        # higher order correction
        for n in order:
            tol_n = [18.815, 3.14153] #adjusted tolerances for node points
            df_c = central_diff_derivative(Chi_c, d, order=n, rank=0)
            placement_n = µ.gradient_integration.integrate_vector(
                df_c, x_n, freqs, staggered_grid=True, order=n)
            placement_c = µ.gradient_integration.integrate_vector(
                df_c, x_c, freqs, staggered_grid=False, order=n)
            self.assertLess(np.linalg.norm(Chi_n - placement_n), tol_n[n-1])
            self.assertLess(np.linalg.norm(Chi_c - placement_c), self.norm_tol)


    def test_compute_placement(self):
        """Test the computation of placements and the original positions."""
        ### shear of a homogeneous material ###
        res   = [ 3, 11] #nb_grid_pts
        lens  = [10, 10] #lengths
        dim   = len(res) #dimension
        x_n=µ.gradient_integration.compute_grid(np.array(lens),np.array(res))[0]

        ### finite strain
        formulation = µ.Formulation.finite_strain
        cell = µ.Cell(res, lens, formulation)
        mat  = µ.material.MaterialLinearElastic1_2d.make(cell.wrapped_cell, "material",
                                                         Young=10, Poisson=0.3)
        for pixel in cell:
            mat.add_pixel(pixel)
        cell.initialise()
        DelF = np.array([[0 , 0.05],
                         [0 , 0   ]])
        # analytic
        placement_ana = np.copy(x_n)
        placement_ana[:,:,0] += DelF[0,1]*x_n[:,:,1]
        # µSpectre solution
        solver = µ.solvers.SolverCG(cell.wrapped_cell, tol=1e-6, maxiter=100, verbose=0)
        result = µ.solvers.newton_cg(cell.wrapped_cell, DelF, solver, newton_tol=1e-6,
                                     equil_tol=1e-6, verbose=0)
        for r in [result, result.grad]:
            #check input of result=OptimiseResult and result=np.ndarray
            placement, x = µ.gradient_integration.compute_placement(
                r, lens, res, order=0, formulation=µ.Formulation.finite_strain)
            self.assertLess(np.linalg.norm(placement_ana - placement), 1e-12)
            self.assertTrue((x_n == x).all())

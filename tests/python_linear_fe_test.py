#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_linear_fe_test.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   04 Mar 2021

@brief  Test muSpectre solution against traditional finite-elements

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
from scipy.sparse import coo_matrix
from python_test_imports import µ


def node_index(i, j, nb_grid_pts):
    """
    Turn node coordinates (i, j) into their global node index.

    Parameters
    ----------
    i : int
        x-coordinate (integer) of the node
    j : int
        y-coordinate (integer) of the node
    nb_grid_pts : tuple of ints
        Number of nodes in the Cartesian directions

    Returns
    -------
    g : int
        Global node index
    """
    Nx, Ny = nb_grid_pts
    return Ny*np.mod(i, Nx) + np.mod(j, Ny)


def derivative_matrix(dx, dy):
    """
    Compute matrix of shape function derivatives

    Parameters
    ----------
    dx : float
        Grid spacing in x-direction
    dy : float
        Grid spacing in y-direction
    """
    return np.array([[dy/dx, 1,      -dy/dx, 0, 0, -1],
                     [1,     dx/dy,  -1,     0, 0, -dx/dy],
                     [-dy/dx, -1,    dy/dx,  0, 0, 1],
                     [0,     0,      0,      0, 0, 0],
                     [0,     0,      0,      0, 0, 0],
                     [-1,    -dx/dy, 1,      0, 0, dx/dy]])/2


def load_vector(sigma_cc, dx, dy):
    xx = sigma_cc[0, 0]
    yy = sigma_cc[1, 1]
    xy = sigma_cc[0, 1]
    return np.array([-xx * dy - xy * dx,
                     -xy * dy - yy * dx,
                     xx * dy,
                     xy * dy,
                     xy * dx,
                     yy * dx])/2


def make_grid(nb_grid_pts, periodic=False):
    """
    Make an array that contains all elements of the grid. The
    elements are described by the global node indices of
    their corners. The order of the corners is in order of
    the local node index.

    They are sorted in geometric positive order and the first
    is the node with the right angle corner at the bottom
    left. Elements within the same box are consecutive.

    This is the first element per box:

        2
        | \
        |  \
    dy  |   \
        |    \
        0 --- 1

          dx

    This is the second element per box:

           dx
         1 ---0
          \   |
           \  |  dy
            \ |
             \|
              2

    Parameters
    ----------
    nb_grid_pts : tuple of ints
        Number of nodes in the Cartesian directions
    periodic : bool
        Periodic boundary conditions

    Returns
    -------
    triangles_el : numpy.ndarray
        Array containing the global node indices of the
        element corners. The first index (suffix _e)
        identifies the element number and the second index
        (suffix _l) the local node index of that element.
    """
    Nx, Ny = nb_grid_pts
    if periodic:
        nb_nodes = nb_grid_pts
    else:
        nb_nodes = Nx + 1, Ny + 1
    # These are the node position on a subsection of the grid
    # that excludes the rightmost and topmost nodes. The
    # suffix _G indicates this subgrid.
    y_G, x_G = np.mgrid[:Ny, :Nx]
    x_G.shape = (-1,)
    y_G.shape = (-1,)

    # List of triangles
    lower_triangles = np.vstack((node_index(x_G, y_G, nb_nodes),
                                 node_index(x_G + 1, y_G, nb_nodes),
                                 node_index(x_G, y_G + 1, nb_nodes)))
    upper_triangles = np.vstack((node_index(x_G + 1, y_G + 1, nb_nodes),
                                 node_index(x_G, y_G + 1, nb_nodes),
                                 node_index(x_G + 1, y_G, nb_nodes)))
    # Suffix _e indicates global element index
    return np.vstack(
        (lower_triangles, upper_triangles)).T.reshape(-1, 3)


def assemble_system_matrix(lame1_e, lame2_e, nb_grid_pts, lengths,
                           sigma0_ecc=None, periodic=False):
    """
    Assemble system matrix from the matrix of shape function derivatives

    Parameters
    ----------
    lame1_e : array_like
        First Lame constant (per element)
    lame2_e : array_like
        Second Lame constant (per element)
    nb_grid_pts : tuple of ints
        Number of nodes in the Cartesian directions
    lengths : tuple of float
        Physical lengths of the simulation cell
    periodic : bool
        Periodic boundary conditions
    sigma0_ecc : array_like
        Eigenstress (per element)

    Returns
    -------
    system_matrix_gg : numpy.ndarray
        System matrix
    """
    Nx, Ny = nb_grid_pts
    Lx, Ly = lengths
    dx = Lx/Nx
    dy = Ly/Ny

    # NOTATION
    # _l is a local node index (0, 1 or 2)
    # _c is a Cartesian index (0 or 1)
    # _e is the element index
    # _g is the global combined node and Cartesian direction index

    # Compute derivative matrix
    derivative_matrix_lclc = derivative_matrix(dx, dy).reshape((3, 2, 3, 2))

    # Compute load vector
    load_vector_elc = None if sigma0_ecc is None else \
        load_vector(sigma0_ecc.T, dx, dy).T.reshape((-1, 3, 2))
    # The upper triangle picks up a factor of -1
    load_vector_elc[::2] *= -1

    # Compute grid
    grid_el = make_grid(nb_grid_pts, periodic=periodic)

    # We have two degrees of freedom per node
    # (components of displacement vector)
    grid_elc = np.stack([2*grid_el, 2*grid_el+1], axis=2)

    # Get number of elements
    nb_elements, _ = grid_el.shape

    assert len(lame1_e) == nb_elements
    assert len(lame2_e) == nb_elements

    # Spread out grid and element matrix such that they can be used as global
    # node coordinates for the sparse matrix
    grid1_elclc = np.stack(
        [grid_elc, grid_elc, grid_elc, grid_elc, grid_elc, grid_elc], axis=1
    ).reshape((-1, 3, 2, 3, 2))
    grid2_elclc = np.stack(
        [grid_elc, grid_elc, grid_elc, grid_elc, grid_elc, grid_elc], axis=3
    ).reshape((-1, 3, 2, 3, 2))
    derivative_matrix_elclc = np.stack(
        [derivative_matrix_lclc]*nb_elements, axis=0)

    # Compute per element contribution of the system matrix, i.e.
    # this is Hooke's law (in 2D, this is a plane strain condition)
    tr_ell = np.trace(derivative_matrix_elclc, axis1=2, axis2=4)
    tr_elclc = np.stack([tr_ell, np.zeros_like(tr_ell),
                         np.zeros_like(tr_ell), tr_ell], axis=3) \
        .reshape((-1, 3, 3, 2, 2)).swapaxes(2, 3)
    element_matrix_elclc = (
            lame1_e * derivative_matrix_elclc.T +
            lame2_e * (tr_elclc.T +
                       derivative_matrix_elclc.swapaxes(2, 4).T)
    ).T

    # Construct sparse system matrix
    # `coo_matrix` will automatically sum duplicate entries
    if periodic:
        nb_nodes = 2*Nx*Ny
    else:
        nb_nodes = 2*(Nx+1)*(Ny+1)

    system_matrix_gg = coo_matrix(
        (element_matrix_elclc.reshape(-1),
         (grid1_elclc.reshape(-1), grid2_elclc.reshape(-1))),
        shape=(nb_nodes, nb_nodes))

    if load_vector_elc is None:
        return system_matrix_gg
    else:
        load_vector_g = np.bincount(
            grid_elc.reshape(-1), load_vector_elc.reshape(-1), minlength=nb_nodes)

        return system_matrix_gg, load_vector_g


def plot_tri(nb_grid_pts, x_g=None, y_g=None, values_g=None, values_e=None,
             mesh_style=None, ax=None, periodic=False):
    """
    Plot results of a finite-element calculation on a
    two-dimensional structured grid using matplotlib.

    Parameters
    ----------
    nb_grid_pts : tuple of ints
        Number of nodes in the Cartesian directions
    x_g : array_like
        x-positions of the nodes
    y_g : array_like
        y-positions of the nodes
    values_g : array_like
        Expansion coefficients (values of the field) on the
        global nodes
    values_e : array_like
        Values on elements
    mesh_style : str, optional
        Will show the underlying finite-element mesh with
        the given style if set, e.g. 'ko-' to see edges
        and mark nodes by points
        (Default: None)
    ax : matplotlib.Axes, optional
        Axes object for plotting
        (Default: None)

    Returns
    -------
    trim : matplotlib.collections.Trimesh
        Result of tripcolor
    """
    import matplotlib
    import matplotlib.pyplot as plt

    Nx, Ny = nb_grid_pts

    # These are the node positions on the full global grid.
    if x_g is None and y_g is None:
        if periodic:
            y_g, x_g = np.mgrid[:Ny, :Nx]
        else:
            y_g, x_g = np.mgrid[:Ny+1, :Nx+1]
        x_g.shape = (-1,)
        y_g.shape = (-1,)
    elif not (x_g is not None and y_g is not None):
        raise ValueError('You need to specify both, x_g and y_g.')

    # Gouraud shading linearly interpolates the color between
    # the nodes
    if ax is None:
        ax = plt
    triangulation = matplotlib.tri.Triangulation(
        x_g, y_g, make_grid(nb_grid_pts, periodic=periodic))
    if values_e is not None:
        c = ax.tripcolor(triangulation, facecolors=values_e)
    elif values_g is not None:
        c = ax.tripcolor(triangulation, values_g,
                         shading='gouraud')
    else:
        c = ax.tripcolor(triangulation, np.zeros_like(x_g),
                         shading='gouraud')
    if mesh_style is not None:
        ax.triplot(triangulation, mesh_style)
    return c


def unit_vector(n, i):
    """
    Returns a vector of zeros with a single entry that is 1

    Parameters
    ----------
    n : int
        Length of the vector
    i : int
        Position where the 1 should be placed
    """
    a = np.zeros((n))
    a[i] = 1
    return a


class SimpleCompositeTest(unittest.TestCase):
    def setUp(self):
        self.nb_grid_pts = [3, 1]
        self.lengths = [4, 2]
        self.formulation = µ.Formulation.small_strain
        self.gradient = µ.linear_finite_elements.gradient_2d
        self.cell = µ.Cell(self.nb_grid_pts,
                           self.lengths,
                           self.formulation,
                           self.gradient)

        self.young1 = 12.3
        self.poisson1 = 0.33
        #self.young2 = self.young1
        #self.poisson2 = self.poisson1
        self.young2 = 3.2
        self.poisson2 = 0.45

        self.young = self.young1*np.ones(self.nb_grid_pts)
        self.poisson = self.poisson1*np.ones(self.nb_grid_pts)
        self.young[0, :] = self.young2
        self.poisson[0, :] = self.poisson2
        if True:
            self.material = µ.material.MaterialLinearElastic4_2d.make(
                self.cell, "material")
            for pix_id, young, poisson in zip(
                    self.cell.pixel_indices, self.young.T.ravel(),
                    self.poisson.T.ravel()):
                self.material.add_pixel(pix_id, young, poisson)
        elif True:
            self.material1 = µ.material.MaterialLinearElastic1_2d.make(
                self.cell, "material", self.young1, self.poisson1)
            self.material2 = µ.material.MaterialLinearElastic1_2d.make(
                self.cell, "material", self.young2, self.poisson2)
            self.material2.add_pixel(0)
            self.material1.add_pixel(1)
            self.material1.add_pixel(2)
        else:
            self.material = µ.material.MaterialLinearElastic1_2d.make(
                self.cell, "material", self.young1, self.poisson1)
            for pix_id in self.cell.pixel_indices:
                self.material.add_pixel(pix_id)

        self.cell.initialise()

        self.lame1_1 = self.young1*self.poisson1/((1+self.poisson1)*(1-2*self.poisson1))
        self.lame2_1 = self.young1/(2*(1+self.poisson1))

        self.lame1_2 = self.young2*self.poisson2/((1+self.poisson2)*(1-2*self.poisson2))
        self.lame2_2 = self.young2/(2*(1+self.poisson2))

        self.stiffmat1 = np.array([[self.lame1_1 + 2*self.lame2_1, self.lame1_1, 0],
                                   [self.lame1_1, self.lame1_1 + 2*self.lame2_1, 0],
                                   [0, 0, 2*self.lame2_1]])
        self.stiffmat2 = np.array([[self.lame1_2 + 2*self.lame2_2, self.lame1_2, 0],
                                   [self.lame1_2, self.lame1_2 + 2*self.lame2_2, 0],
                                   [0, 0, 2*self.lame2_2]])


    def test_composite(self):
        #delxx = -0.005
        #delyy = 0.01
        #delxy = 0
        delxx = -0.005
        delyy = 0.01
        delxy = 0.1
        applied_strain = np.array([[delxx, delxy],
                                   [delxy, delyy]])

        #
        # Solve via muSpectre
        #
        newton_tol = 1e-6
        equil_tol = 0.
        cg_tol = 1e-6
        maxiter = 100
        verbose = µ.Verbosity.Silent

        solver = µ.solvers.KrylovSolverCG(self.cell, cg_tol, maxiter, verbose)
        µ.solvers.newton_cg(self.cell, applied_strain, solver,
                            newton_tol, equil_tol, verbose)
        numerical_stress = self.cell.stress.array()
        (x, y), (x0, y0) = \
            µ.gradient_integration.get_complemented_positions(
                'p0', self.cell, periodically_complemented=True)

        # Analytic result
        Nx, Ny = self.nb_grid_pts
        A1 = self.stiffmat1[0, 0]
        A2 = self.stiffmat2[0, 0]
        B1 = self.stiffmat1[0, 1]
        B2 = self.stiffmat2[0, 1]
        C1 = self.stiffmat1[2, 2]
        C2 = self.stiffmat2[2, 2]
        analytic_strain_xx2 = (Nx*A1*delxx +
                               (Nx-1)*(B1-B2)*delyy)/(A1 + A2*(Nx-1))
        analytic_strain_xx1 = (Nx * delxx - analytic_strain_xx2)/(Nx-1)
        analytic_strain_xy2 = C1*Nx*delxy/(C1 + C2*(Nx-1))
        analytic_strain_xy1 = (Nx * delxy - analytic_strain_xy2)/(Nx-1)

        analytic_stress1 = self.stiffmat1.dot([analytic_strain_xx1, delyy, analytic_strain_xy1])
        analytic_stress2 = self.stiffmat2.dot([analytic_strain_xx2, delyy, analytic_strain_xy2])

        numerical_strain = self.cell.strain.array()

        # Compare strain
        self.assertAlmostEqual(numerical_strain[0, 0, 0, 1, 0], analytic_strain_xx1)
        self.assertAlmostEqual(numerical_strain[1, 1, 0, 1, 0], delyy)
        self.assertAlmostEqual(numerical_strain[0, 1, 0, 1, 0], analytic_strain_xy1)

        self.assertAlmostEqual(numerical_strain[0, 0, 0, 0, 0], analytic_strain_xx2)
        self.assertAlmostEqual(numerical_strain[1, 1, 0, 0, 0], delyy)
        self.assertAlmostEqual(numerical_strain[0, 1, 0, 0, 0], analytic_strain_xy2)

        # Compare stresses
        self.assertAlmostEqual(numerical_stress[0, 0].mean(),
                               analytic_stress1[0])
        self.assertAlmostEqual(numerical_stress[0, 0].mean(),
                               analytic_stress2[0])
        self.assertAlmostEqual(
            numerical_stress[1, 1].mean(),
            (analytic_stress1[1]*(Nx-1) + analytic_stress2[1])/Nx)
        self.assertAlmostEqual(numerical_stress[0, 1].mean(),
                               analytic_stress1[2])
        self.assertAlmostEqual(numerical_stress[0, 1].mean(),
                               analytic_stress2[2])

        # Compare displacements
        Lx, Ly = self.lengths
        self.assertAlmostEqual(Lx*analytic_strain_xx2/Nx, x[1, 0]-x0[1, 0] - (x[0, 0]-x0[0, 0]))

        #
        # "Traditional" FE
        #
        E = np.stack([self.young.T.ravel(), self.young.T.ravel()],
                     axis=1).ravel()
        nu = np.stack([self.poisson.T.ravel(), self.poisson.T.ravel()],
                      axis=1).ravel()
        lame1_e = E*nu/((1+nu)*(1-2*nu))
        lame2_e = E/(2*(1+nu))

        # Eigenstress for eigenstrain Del0
        sigma0_ecc = (lame1_e * np.trace(applied_strain) *
                      np.identity(2).reshape(2, 2, 1) +
                      2 * lame2_e * applied_strain.reshape(2, 2, 1)).T

        # System matrix with natural boundary conditions
        # (stress-free surface)
        system_matrix_gg, rhs_g = assemble_system_matrix(
            lame1_e, lame2_e, self.nb_grid_pts, self.lengths,
            sigma0_ecc=sigma0_ecc, periodic=True)
        system_matrix_gg = system_matrix_gg.todense()

        # Rank of matrix must be dimension - 3 (translation + rotation)
        nb_nodes = np.prod(self.nb_grid_pts)
        self.assertEqual(np.linalg.matrix_rank(system_matrix_gg),
                         2*nb_nodes-2)

        # Fix average x-displacement to zero
        system_matrix_gg[0] = np.array([1, 0]*nb_nodes)
        rhs_g[0] = 0

        # Fix average y-displacement to zero
        system_matrix_gg[1] = np.array([0, 1]*nb_nodes)
        rhs_g[1] = 0

        # Rank of matrix is now regular
        self.assertEqual(np.linalg.matrix_rank(system_matrix_gg),
                         2*nb_nodes)

        # Solve the linear system
        u_g = np.linalg.solve(system_matrix_gg, rhs_g).reshape((Nx, Ny, 2))
        ux_ij, uy_ij = µ.gradient_integration.complement_periodically(u_g.T, 2)
        ux_ij = ux_ij.T
        uy_ij = uy_ij.T

        # Compute strain...
        Nx, Ny = self.nb_grid_pts
        Lx, Ly = self.lengths
        dx = Lx/Nx
        dy = Ly/Ny

        # ...on the bottom-left element
        strain_fe1_xx_ij = delxx + (ux_ij[1:, :-1] - ux_ij[:-1, :-1])/dx
        strain_fe1_yy_ij = delyy + (uy_ij[:-1, 1:] - uy_ij[:-1, :-1])/dy
        strain_fe1_xy_ij = delxy + ((ux_ij[:-1, 1:] - ux_ij[:-1, :-1])/dy + \
                                    (uy_ij[1:, :-1] - uy_ij[:-1, :-1])/dx)/2

        # ...on the top-right element
        strain_fe2_xx_ij = delxx + (ux_ij[1:, 1:] - ux_ij[:-1, 1:])/dx
        strain_fe2_yy_ij = delyy + (uy_ij[1:, 1:] - uy_ij[1:, :-1])/dy
        strain_fe2_xy_ij = delxy + ((ux_ij[1:, 1:] - ux_ij[1:, :-1])/dy + \
                                    (uy_ij[1:, 1:] - uy_ij[:-1, 1:])/dx)/2

        self.assertAlmostEqual(strain_fe1_xx_ij[1, 0], analytic_strain_xx1)
        self.assertAlmostEqual(strain_fe1_yy_ij[1, 0], delyy)
        self.assertAlmostEqual(strain_fe1_xy_ij[1, 0], analytic_strain_xy1)
        self.assertAlmostEqual(strain_fe1_xx_ij[0, 0], analytic_strain_xx2)
        self.assertAlmostEqual(strain_fe1_yy_ij[0, 0], delyy)
        self.assertAlmostEqual(strain_fe1_xy_ij[0, 0], analytic_strain_xy2)
        self.assertAlmostEqual(strain_fe2_xx_ij[1, 0], analytic_strain_xx1)
        self.assertAlmostEqual(strain_fe2_yy_ij[1, 0], delyy)
        self.assertAlmostEqual(strain_fe2_xy_ij[1, 0], analytic_strain_xy1)
        self.assertAlmostEqual(strain_fe2_xx_ij[0, 0], analytic_strain_xx2)
        self.assertAlmostEqual(strain_fe2_yy_ij[0, 0], delyy)
        self.assertAlmostEqual(strain_fe2_xy_ij[0, 0], analytic_strain_xy2)

        # Now feed this strain through the projection operator
        strain_fe_ccqij = np.array(
            [[[strain_fe1_xx_ij - delxx, strain_fe2_xx_ij - delxx],
              [strain_fe1_xy_ij - delxy, strain_fe2_xy_ij - delxy]],
             [[strain_fe1_xy_ij - delxy, strain_fe2_xy_ij - delxy],
              [strain_fe1_yy_ij - delyy, strain_fe2_yy_ij - delyy]]])
        proj_strain_fe_ccqij = \
            self.cell.projection.apply_projection(strain_fe_ccqij)
        self.assertTrue(np.allclose(proj_strain_fe_ccqij, strain_fe_ccqij))

        # Check stress
        #young = self.young2*np.ones(self.nb_grid_pts)
        #poisson = self.poisson2*np.ones(self.nb_grid_pts)
        #E = np.stack([young.T.ravel(), young.T.ravel()],
        #             axis=1).ravel()
        #nu = np.stack([poisson.T.ravel(), poisson.T.ravel()],
        #              axis=1).ravel()
        #lame1_e = E*nu/((1+nu)*(1-2*nu))
        #lame2_e = E/(2*(1+nu))

        lame1_qij = lame1_e.reshape(1, 3, 2).T
        lame2_qij = lame2_e.reshape(1, 3, 2).T

        stress_fe_ccqij = lame1_qij * np.trace(strain_fe_ccqij) * \
                          np.identity(2).reshape(2, 2, 1, 1, 1) + \
                          2 * lame2_qij * strain_fe_ccqij
        stress_msp_ccqij = self.cell.evaluate_stress(strain_fe_ccqij)
        self.assertTrue(np.allclose(stress_fe_ccqij, stress_msp_ccqij))


def build_fe_test(nb_grid_pts, lengths, Young=1.0, Poisson=0.33):
    class FETest(unittest.TestCase):
        def setUp(self):
            self.nb_grid_pts = nb_grid_pts
            self.lengths = lengths
            self.formulation = µ.Formulation.small_strain
            self.gradient = µ.linear_finite_elements.gradient_2d
            self.cell = µ.Cell(self.nb_grid_pts,
                               self.lengths,
                               self.formulation,
                               self.gradient)
            self.young = Young*np.ones(self.nb_grid_pts)
            self.poisson = Poisson*np.ones(self.nb_grid_pts)
            self.young = 1.0 + 0.2*np.random.random(self.nb_grid_pts)
            self.poisson = 0.33 + 0.1*np.random.random(self.nb_grid_pts)
            self.material = µ.material.MaterialLinearElastic4_2d.make(
                self.cell, "material")
            for pix_id, young, poisson in zip(
                    self.cell.pixel_indices, self.young.T.ravel(),
                    self.poisson.T.ravel()):
                self.material.add_pixel(pix_id, young, poisson)
            self.cell.initialise()

        def test_solve(self):
            Nx, Ny = self.nb_grid_pts
            nb_nodes = Nx*Ny

            # Solve via muSpectre
            newton_tol = 1e-6
            equil_tol = 0.
            applied_strain = np.array([[0.1, 0.1],
                                       [0.1, 0]])
            cg_tol = 1e-6
            maxiter = 100
            verbose = µ.Verbosity.Silent

            solver = µ.solvers.KrylovSolverCG(self.cell, cg_tol, maxiter,
                                              verbose)
            µ.solvers.newton_cg(self.cell, applied_strain, solver,
                                newton_tol, equil_tol, verbose)

            (x, y), (x0, y0) = \
                µ.gradient_integration.get_complemented_positions(
                    'pg', self.cell, periodically_complemented=True)
            x = x.reshape((-1,))
            y = y.reshape((-1,))
            x0 = x0.reshape((-1,))
            y0 = y0.reshape((-1,))

            # "Traditional" FE
            E = np.stack([self.young.T.ravel(), self.young.T.ravel()],
                         axis=1).ravel()
            nu = np.stack([self.poisson.T.ravel(), self.poisson.T.ravel()],
                          axis=1).ravel()
            lame1_e = E*nu/((1+nu)*(1-2*nu))
            lame2_e = E/(2*(1+nu))

            # Eigenstress for eigenstrain Del0
            sigma0_ecc = (lame1_e * np.trace(applied_strain) *
                          np.identity(2).reshape(2, 2, 1) +
                          2 * lame2_e * applied_strain.reshape(2, 2, 1)).T

            # System matrix with natural boundary conditions
            # (stress-free surface)
            system_matrix_gg, rhs_g = assemble_system_matrix(
                lame1_e, lame2_e, self.nb_grid_pts, self.lengths,
                sigma0_ecc=sigma0_ecc, periodic=True)
            system_matrix_gg = system_matrix_gg.todense()

            # Rank of matrix must be dimension - 3 (translation + rotation)
            self.assertEqual(np.linalg.matrix_rank(system_matrix_gg),
                             2*nb_nodes-2)

            # Fix average x-displacement to zero
            system_matrix_gg[0] = np.array([1, 0]*nb_nodes)
            rhs_g[0] = 0

            # Fix average y-displacement to zero
            system_matrix_gg[1] = np.array([0, 1]*nb_nodes)
            rhs_g[1] = 0

            # Rank of matrix is now regular
            self.assertEqual(np.linalg.matrix_rank(system_matrix_gg),
                             2*nb_nodes)

            # Solve the linear system
            u_g = np.linalg.solve(system_matrix_gg, rhs_g).reshape((Nx, Ny, 2))
            ux, uy = µ.gradient_integration.complement_periodically(u_g.T, 2)
            ux = ux.T.ravel()
            uy = uy.T.ravel()

            if False:
                import matplotlib.pyplot as plt
                plt.subplot(121, aspect=1)
                plot_tri(self.nb_grid_pts, x, y,
                         #values_e=lame1_e,
                         values_g=x-x0 - ux,
                         mesh_style='k-', periodic=False)
                plt.colorbar()
                plt.subplot(122, aspect=1)
                plot_tri(self.nb_grid_pts, x0 + ux, y0 + uy,
                         #values_e=lame1_e,
                         values_g=y-y0 - uy,
                         mesh_style='k-', periodic=False)
                plt.colorbar()
                plt.show()

            self.assertTrue(np.allclose(x-x0, ux, atol=1e-6))
            self.assertTrue(np.allclose(y-y0, uy, atol=1e-6))

    return FETest

test_odd = build_fe_test([7, 9], [3.5, 2.3])
test_even = build_fe_test([8, 8], [7.2, 3.3])
test_oddeven = build_fe_test([7, 5], [1, 1])

if __name__ == '__main__':
    unittest.main()

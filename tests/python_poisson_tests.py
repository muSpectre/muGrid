"""
@file    python_poisson_tests.py

@author  Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date    10 April 2025

@brief   solving the Poisson equation as a compound test

Copyright © 2018 Till Junge

µGrid is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µGrid is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with µGrid; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.

Additional permission under GNU GPL version 3 section 7

If you modify this Program, or any covered work, by linking or combining it
with proprietary FFT implementations or numerical libraries, containing parts
covered by the terms of those libraries' licenses, the licensors of this
Program grant you additional permission to convey the resulting work.
"""

import numpy as np
from NuMPI.Testing.Subdivision import suggest_subdivisions

import muGrid
from muGrid.Solvers import conjugate_gradients


def test_fd_stencil():
    stencil = np.array(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    )  # FD-stencil for the Laplacian
    laplace = muGrid.ConvolutionOperator([-1, -1], stencil)
    assert laplace.nb_operators == 1
    assert laplace.nb_quad_pts == 1

    fc = muGrid.GlobalFieldCollection([3, 3])
    ifield = fc.real_field("input-field")
    ofield = fc.real_field("output-field")
    ifield.p[...] = 1
    ifield.p[1, 1] = 2
    laplace.apply(ifield, ofield)
    np.testing.assert_allclose(ofield.p, stencil)
    np.testing.assert_allclose(np.sum(ifield.p * ofield.p), -4)


def test_fd_poisson_solver(comm, nb_grid_pts=(256, 256)):
    """Finite-differences Poisson solver"""
    s = suggest_subdivisions(len(nb_grid_pts), comm.size)

    decomposition = muGrid.CartesianDecomposition(comm, nb_grid_pts, s, (1, 1), (1, 1))
    fc = decomposition.collection
    # fc = muGrid.GlobalFieldCollection(nb_grid_pts)
    grid_spacing = 1 / np.array(nb_grid_pts)  # Grid spacing

    stencil = np.array(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    )  # FD-stencil for the Laplacian
    laplace = muGrid.ConvolutionOperator([-1, -1], stencil)
    assert laplace.nb_operators == 1
    assert laplace.nb_quad_pts == 1

    # np.testing.assert_array_equal(decomposition.nb_subdivisions, s)

    x, y = decomposition.coords  # Domain-local coords for each pixel
    # x, y = np.meshgrid(
    #     np.arange(nb_grid_pts[0]) / nb_grid_pts[0],
    #     np.arange(nb_grid_pts[1]) / nb_grid_pts[1],
    #     indexing="ij",
    # )

    rhs = fc.real_field("rhs")
    solution = fc.real_field("solution")

    rhs.p = np.sin(2 * np.pi * x)

    def callback(it, x, r, p):
        """
        Callback function to print the current solution, residual, and search direction.
        """
        print(it, np.dot(r.ravel(), r.ravel()))

    def hessp(x, Ax):
        """
        Function to compute the product of the Hessian matrix with a vector.
        The Hessian is represented by the convolution operator.
        """
        laplace.apply(x, Ax)
        # We need the minus sign because the Laplace operator is negative
        # definite, but the conjugate-gradients solver assumes a
        # positive-definite operator.
        Ax.s /= -np.mean(grid_spacing) ** 2  # Scale by grid spacing
        return Ax

    conjugate_gradients(
        comm,
        fc,
        hessp,  # linear operator
        rhs,
        solution,
        tol=1e-6,
        callback=callback,
    )

    np.testing.assert_allclose(
        solution.p, np.sin(2 * np.pi * x) / (2 * np.pi) ** 2, atol=1e-5
    )

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


import numpy as np


def conjugate_gradients(
    comm: muGrid.Communicator,
    fc: muGrid.FieldCollection,
    hessp: callable,
    b: muGrid.Field,
    x: muGrid.Field,
    tol: float = 1e-6,
    max_iter: int = 1000,
    callback: callable = None,
):
    """
    Conjugate gradient method for matrix-free solution of the linear problem
    Ax = b, where A is represented by the function hessp (which computes the
    product of A with a vector). The method iteratively refines the solution x
    until the residual ||Ax - b|| is less than tol or until max_iter iterations
    are reached.

    Parameters
    ----------
    comm : muGrid.Communicator
        Communicator for parallel processing.
    fc : muGrid.FieldCollection
        Collection holding temporary fields of the CG algorithm.
    hessp : callable
        Function that computes the product of the Hessian matrix A with a vector.
    b : muGrid.Field
        Right-hand side vector.
    x : muGrid.Field
        Initial guess for the solution.
    tol : float, optional
        Tolerance for convergence. The default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. The default is 1000.
    callback : callable, optional
        Function to call after each iteration with the current solution, residual,
        and search direction.

    Returns
    -------
    x : array_like
        Approximate solution to the system Ax = b. (Same as input field x.)
    """
    tol_sq = tol * tol
    p = fc.real_field("cg-search-direction")
    Ap = fc.real_field("cg-hessian-product")
    hessp(x, p)
    p.s = b.s - p.s
    r = np.copy(p.s)  # residual

    if callback:
        callback(x.s, r, p.s)

    rr = comm.sum(np.dot(r, r))  # initial residual dot product
    if rr < tol_sq:
        return x

    for _ in range(max_iter):
        # Compute Hessian product
        hessp(p, Ap)

        # Update x (and residual)
        pAp = comm.sum(np.dot(p.s, Ap.s))
        if pAp <= 0:
            raise RuntimeError("Hessian is not positive definite")

        alpha = rr / pAp
        x.s += alpha * p.s
        r -= alpha * Ap.s

        if callback:
            callback(x.s, r, p.s)

        # Check convergence
        next_rr = comm.sum(np.dot(r, r))
        if next_rr < tol_sq:
            return x

        # Update search direction
        beta = next_rr / rr
        rr = next_rr
        p.s *= beta
        p.s += r

    raise RuntimeError("Conjugate gradient algorithm did not converge")


def test_fd_poisson_solver(comm, nb_grid_pts=(32, 32)):
    """Finite-differences Poisson solver"""
    s = suggest_subdivisions(len(nb_grid_pts), comm.size)

    decomposition = muGrid.CartesianDecomposition(comm, nb_grid_pts, s, (1, 1), (1, 1))

    stencil = np.array(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    )  # FD-stencil for the Laplacian
    laplace = muGrid.ConvolutionOperator(len(nb_grid_pts), stencil)
    assert laplace.nb_operators == 1
    assert laplace.nb_quad_pts == 1

    np.testing.assert_array_equal(decomposition.nb_subdivisions, s)

    x, y = decomposition.coords  # Domain-local coords for each pixel

    rhs = decomposition.collection.real_field("rhs")
    rhs.p = np.sin(2 * np.pi * x)

    solution = decomposition.collection.real_field("solution")

    def callback(x, r, p):
        """Callback function to print the current solution, residual, and search direction."""
        print(x)

    conjugate_gradients(
        comm,
        decomposition.collection,
        lambda x, Ax: laplace.apply(x, Ax),  # linear operator
        solution,
        rhs,
        tol=1e-6,
        callback=callback,
    )

    print(solution.s)

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
import pytest
from NuMPI.Testing.Subdivision import suggest_subdivisions

import muGrid
from muGrid import wrap_field
from muGrid.Solvers import conjugate_gradients


def test_fd_stencil():
    stencil = np.array(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    )  # FD-stencil for the Laplacian
    laplace = muGrid.ConvolutionOperator([-1, -1], stencil)
    assert laplace.nb_operators == 1
    assert laplace.nb_quad_pts == 1

    nb_ghosts = (1, 1)
    fc = muGrid.GlobalFieldCollection(
        [3, 3], nb_ghosts_left=nb_ghosts, nb_ghosts_right=nb_ghosts
    )
    ifield = fc.real_field("input-field")
    ofield = fc.real_field("output-field")
    ifield.p[...] = 1
    ifield.p[1, 1] = 2
    # Manually correct a periodic boundary
    ifield.pg[...] = np.pad(ifield.p, tuple(zip(nb_ghosts, nb_ghosts)), mode="wrap")
    laplace.apply(ifield, ofield)
    np.testing.assert_allclose(ofield.p, stencil)
    np.testing.assert_allclose(np.sum(ifield.p * ofield.p), -4)


def test_fd_poisson_solver(comm, nb_grid_pts=(128, 128)):
    """Finite-differences Poisson solver"""
    s = suggest_subdivisions(len(nb_grid_pts), comm.size)

    decomposition = muGrid.CartesianDecomposition(comm, nb_grid_pts, s, (1, 1), (1, 1))
    grid_spacing = 1 / np.array(nb_grid_pts)  # Grid spacing

    stencil = np.array(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    )  # FD-stencil for the Laplacian
    laplace = muGrid.ConvolutionOperator([-1, -1], stencil)
    assert laplace.nb_operators == 1
    assert laplace.nb_quad_pts == 1

    # np.testing.assert_array_equal(decomposition.nb_subdivisions, s)

    x, y = decomposition.coords  # Domain-local coords for each pixel

    rhs = decomposition.real_field("rhs")
    solution = decomposition.real_field("solution")

    rhs.p[...] = np.sin(2 * np.pi * x)

    def callback(it, x, r, p):
        """
        Callback function to print the current solution, residual, and search direction.
        """
        print(it, np.dot(r.ravel(), r.ravel()))

    def hessp(x_field, Ax_field):
        """
        Function to compute the product of the Hessian matrix with a vector.
        The Hessian is represented by the convolution operator.
        """
        decomposition.communicate_ghosts(x_field)
        laplace.apply(x_field, Ax_field)
        # We need the minus sign because the Laplace operator is negative
        # definite, but the conjugate-gradients solver assumes a
        # positive-definite operator.
        Ax_field.s[...] /= -np.mean(grid_spacing) ** 2  # Scale by grid spacing
        return Ax_field

    conjugate_gradients(
        comm,
        decomposition,
        hessp,  # linear operator
        rhs,
        solution,
        tol=1e-6,
        callback=callback,
        maxiter=10,
    )

    np.testing.assert_allclose(
        solution.p, np.sin(2 * np.pi * x) / (2 * np.pi) ** 2, atol=1e-5
    )


@pytest.mark.skip("Currently fails; reenable after migration")
def test_unit_impulse(
    comm,
):
    nx, ny = nb_grid_pts = (4, 6)  # grid size should be arbitrary
    s = suggest_subdivisions(len(nb_grid_pts), comm.size)
    decomposition = muGrid.CartesianDecomposition(comm, nb_grid_pts, s, (1, 1), (1, 1))
    fc = decomposition._cpp.collection
    fc.set_nb_sub_pts("quad_points", 2)
    fc.set_nb_sub_pts("nodal_points", 1)

    # set up gradient operator
    gradient = np.array(
        [
            [  # Derivative in x-direction
                [[[-1, 0], [1, 0]]],  # Bottom-left triangle (first quadrature point)
                [[[0, -1], [0, 1]]],  # Top-right triangle (second quadrature point)
            ],
            [  # Derivative in y-direction
                [[[-1, 1], [0, 0]]],  # Bottom-left triangle (first quadrature point)
                [[[0, 0], [-1, 1]]],  # Top-right triangle (second quadrature point)
            ],
        ],
    )
    gradient_op = muGrid.ConvolutionOperator([0, 0], gradient)

    # Get nodal field
    nodal_field_cpp = fc.real_field("nodal-field", (1,), "nodal_points")
    impuls_response_field_cpp = fc.real_field(
        "impuls_response_field", (1,), "nodal_points"
    )
    nodal_field = wrap_field(nodal_field_cpp)
    impuls_response_field = wrap_field(impuls_response_field_cpp)

    # Get quadrature field of shape (2, quad, nx, ny)
    quad_field_cpp = fc.real_field("quad-field", (2,), "quad_points")

    # set up impulse
    impuls_locations = (impuls_response_field.icoordsg[0] == 0) & (
        impuls_response_field.icoordsg[1] == 0
    )

    left_location = (impuls_response_field.icoordsg[0] == nx - 1) & (
        impuls_response_field.icoordsg[1] == 0
    )
    right_location = (impuls_response_field.icoordsg[0] == 1) & (
        impuls_response_field.icoordsg[1] == 0
    )
    top_location = (impuls_response_field.icoordsg[0] == 0) & (
        impuls_response_field.icoordsg[1] == 1
    )
    bottom_location = (impuls_response_field.icoordsg[0] == 0) & (
        impuls_response_field.icoordsg[1] == ny - 1
    )

    nodal_field.sg[0, 0, impuls_locations] = 1
    impuls_response_field.sg[0, 0, impuls_locations] = 4
    impuls_response_field.sg[0, 0, left_location] = -1
    impuls_response_field.sg[0, 0, right_location] = -1
    impuls_response_field.sg[0, 0, top_location] = -1
    impuls_response_field.sg[0, 0, bottom_location] = -1

    decomposition.communicate_ghosts(nodal_field)
    print(
        f"unit impuls: nodal field with buffers in rank {comm.rank} \n "
        + f"{nodal_field.sg}"
    )
    print(
        f"impuls_response_field: nodal field with buffers in rank {comm.rank} \n "
        + f"{impuls_response_field.sg}"
    )

    # Apply the gradient operator to the nodal field and write result to the quad field
    gradient_op.apply(nodal_field, wrap_field(quad_field_cpp))

    decomposition.communicate_ghosts(wrap_field(quad_field_cpp))

    # Apply the gradient transposed operator to the quad field and write result to
    # the nodal field
    gradient_op.transpose(
        quadrature_point_field=wrap_field(quad_field_cpp),
        nodal_field=nodal_field,
        weights=[
            1 / 2,
            1 / 2,
        ],  # size of the element is half of the pixel. Pixel size is 1
    )

    decomposition.communicate_ghosts(nodal_field)
    print(
        "computed unit impuls response: nodal field with buffers in rank "
        f"{comm.rank} \n {nodal_field.sg}"
    )

    print(
        f"local sum on core (nodal_field.s) = {np.sum(nodal_field.s)}"
    )  # does not have to be zero
    local_sum = np.sum(nodal_field.s)
    total_sum = comm.sum(local_sum)
    print(f"total_sum= {total_sum}")  # have to be zero

    # Check that the nodal_field has zero mean
    np.testing.assert_allclose(
        total_sum,
        0,
        atol=1e-10,
    )
    # Check that the impulse response is correct
    np.testing.assert_allclose(
        nodal_field.s,
        impuls_response_field.s,
        atol=1e-5,
    )

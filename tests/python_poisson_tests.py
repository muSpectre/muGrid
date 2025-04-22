import numpy as np
from NuMPI.Testing.Subdivision import suggest_subdivisions

import muGrid


def test_poisson_solver(comm, nb_grid_pts=(32, 32)):
    s = suggest_subdivisions(len(nb_grid_pts), comm.size)

    decomposition = muGrid.CartesianDecomposition(comm, nb_grid_pts, s, (1, 1), (1, 1))

    stencil = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    laplace = muGrid.ConvolutionOperator(len(nb_grid_pts), stencil)
    assert laplace.nb_operators == 1
    assert laplace.nb_quad_pts == 1

    np.testing.assert_array_equal(decomposition.nb_subdivisions, s)

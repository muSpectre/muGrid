import numpy as np

from muGrid import ConvolutionOperator, GlobalFieldCollection, CartesianDecomposition, Communicator
try:
    from mpi4py import MPI
    comm = Communicator(MPI.COMM_WORLD)
except ImportError:
    comm = Communicator()

# Two dimensional grid
nx, ny = nb_grid_pts = (1024, 2)
decomposition = CartesianDecomposition(comm, nb_grid_pts, (1, 1), (1, 1), (1, 1))
fc = decomposition.collection
fc.set_nb_sub_pts("quad", 2)

# Get nodal field
nodal_field = fc.real_field("nodal-field")

# Get quadrature field of shape (2, quad, nx, ny)
quad_field = fc.real_field("quad-field", (2,), "quad")

# Fill nodal field with a sine-wave
x, y = nodal_field.icoords
nodal_field.p = np.sin(2 * np.pi * x / nx)
decomposition.communicate_ghosts(nodal_field)

# Derivative stencil of shape (2, quad, 2, 2)
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
op = ConvolutionOperator([0, 0], gradient)

# Apply the gradient operator to the nodal field and write result to the quad field
op.apply(nodal_field, quad_field)

# Check that the quadrature field has the correct derivative
np.testing.assert_allclose(
    # because the gradient stencil is linearly interpolating, it should compare against
    # the analytic value evaluated at the middle of two grid points, hence (x + 0.5)
    quad_field.s[0, 0], 2 * np.pi * np.cos(2 * np.pi * (x + 0.5) / nx) / nx, atol=1e-8
)
np.testing.assert_allclose(
    quad_field.s[1, 0], 0
)

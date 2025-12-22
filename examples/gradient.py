import numpy as np

import muGrid

try:
    from mpi4py import MPI
    comm = muGrid.Communicator(MPI.COMM_WORLD)
except ImportError:
    comm = muGrid.Communicator()

# Two dimensional grid
nx, ny = nb_grid_pts = (1024, 2)
# Add some ghosts for computing
nb_ghosts_left = (0, 0)
nb_ghosts_right = (1, 1)
fc = muGrid.GlobalFieldCollection(
    nb_grid_pts,
    nb_sub_pts={"quad": 2},
    nb_ghosts_left=nb_ghosts_left,
    nb_ghosts_right=nb_ghosts_right,
)

# Get nodal field
nodal_field = fc.real_field("nodal-field")

# Get quadrature field of shape (2, quad, nx, ny)
quad_field = fc.real_field("quad-field", (2,), "quad")

# Fill nodal field with a sine-wave
# Generate grid coordinates
x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
nodal_field.p[...] = np.sin(2 * np.pi * x / nx)
# Padding to mimic periodic boundary (sine wave is also periodic)
nodal_field.pg[...] = np.pad(nodal_field.p, tuple(zip(nb_ghosts_left, nb_ghosts_right)), mode="wrap")

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
op = muGrid.ConvolutionOperator([0, 0], gradient)

# Apply the gradient operator to the nodal field and write result to the quad field
op.apply(nodal_field, quad_field)

# Check that the quadrature field has the correct derivative
np.testing.assert_allclose(
    # Even though the quadrature point can have different locations, because the gradient
    # stencil is linearly interpolating, it should compare against the analytic value
    # evaluated at the middle of two grid points, hence (x + 0.5)
    quad_field.s[0, 0], 2 * np.pi * np.cos(2 * np.pi * (x + 0.5) / nx) / nx, atol=1e-8
)
np.testing.assert_allclose(
    quad_field.s[1, 0], 0
)

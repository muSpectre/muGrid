import numpy as np

from muGrid import ConvolutionOperator, GlobalFieldCollection

# Two dimensional grid
nb_grid_pts = (11, 12)  # nx, ny
fc = GlobalFieldCollection(nb_grid_pts, sub_pts={"quad": 2})

# Get nodal field
nodal_field = fc.real_field("nodal-field")

# Get quadrature field of shape (2, quad, nx, ny)
quad_field = fc.real_field("quad-field", (2,), "quad")

# Fill nodal field with a sine-wave
x, y = nodal_field.coords
nodal_field.p = np.sin(2 * np.pi * x)

# Derivative stencil of shape (2, quad, 2, 2)
gradient = np.array(
    [
        [  # Derivative in x-direction
            [[-1, 1], [0, 0]],  # Bottom-left triangle (first quadrature point)
            [[0, 0], [-1, 1]],  # Top-right triangle (second quadrature point)
        ],
        [  # Derivative in y-direction
            [[-1, 0], [1, 0]],  # Bottom-left triangle (first quadrature point)
            [[0, -1], [0, 1]],  # Top-right triangle (second quadrature point)
        ],
    ],
)
op = ConvolutionOperator(gradient)

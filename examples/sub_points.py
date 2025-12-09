import numpy as np
from muGrid import GlobalFieldCollection, wrap_field

# Two dimensional grid
nb_grid_pts = (11, 12, 13)
fc = GlobalFieldCollection(nb_grid_pts, sub_pts={"element": 5})

# Get a tensor-field (for example to represent the strain)
strain_cpp = fc.real_field(
    "strain",  # name of the field
    (3, 3),  # shape of components
    "element",  # sub-point type
)
strain = wrap_field(strain_cpp)

# Fill the field with random numbers
strain.s[...] = np.random.rand(*((3, 3, 5) + nb_grid_pts))

# Note that the convenience accessor `p` yields the same field but a different shape
strain.p[...] = np.random.rand(*((3, 3 * 5) + nb_grid_pts))

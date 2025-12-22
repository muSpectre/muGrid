import numpy as np
import muGrid

# Two dimensional grid
nb_grid_pts = (11, 12, 13)
fc = muGrid.GlobalFieldCollection(nb_grid_pts, nb_sub_pts={"element": 5})

# Get a tensor-field (for example to represent the strain)
strain = fc.real_field("strain", (3, 3), "element")

# Fill the field with random numbers
strain.s[...] = np.random.rand(*((3, 3, 5) + nb_grid_pts))

# Note that the convenience accessor `p` yields the same field but a different shape
strain.p[...] = np.random.rand(*((3, 3 * 5) + nb_grid_pts))

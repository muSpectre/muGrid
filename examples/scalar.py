import numpy as np
import muGrid

# Two dimensional grid
nb_grid_pts = (11, 12)
fc = muGrid.GlobalFieldCollection(nb_grid_pts)

# Get a scalar field
phase = fc.real_field("phase")
# The 1 is the number of sub points
assert phase.shape == [1, *nb_grid_pts]

# Get a single component field
single = fc.real_field("single", (1,))
# Now there is an explicit component dimension of 1
assert single.shape == [1, 1, *nb_grid_pts]

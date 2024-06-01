import numpy as np
from muGrid import GlobalFieldCollection

# Two dimensional grid
nb_grid_pts = (11, 12)
fc = GlobalFieldCollection(nb_grid_pts)

# Get a tensor-field (for example to represent the strain)
strain = fc.real_field('strain', (2, 2))

# Fill the field with random numbers
strain.p = np.random.rand(*((2, 2) + nb_grid_pts))

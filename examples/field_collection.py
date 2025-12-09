import numpy as np
from muGrid import GlobalFieldCollection, wrap_field

# Two dimensional grid
nb_grid_pts = (11, 12)
fc = GlobalFieldCollection(nb_grid_pts)

# Get a scalar (real-valued) field
# This call creates a new field if it does not exist.
field_cpp = fc.real_field("my-real-valued-field")
field = wrap_field(field_cpp)

# Check the registered fields
assert fc.field_names == ["my-real-valued-field"]

# Fill the field with random numbers
field.p[...] = np.random.rand(*nb_grid_pts)

# Set a specific value
field.p[5, 6] = 42

# Get the field, but now into a different variable
field2_cpp = fc.real_field("my-real-valued-field")
field2 = wrap_field(field2_cpp)

# Check that the value is the same
assert field2.p[5, 6] == 42

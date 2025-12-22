import numpy as np
import muGrid

if not muGrid.has_netcdf:
    print("IO example is skipped: muGrid was built without NetCDF support")
    exit(0)

# Two dimensional grid
nb_grid_pts = (11, 12, 13)
fc = muGrid.GlobalFieldCollection(nb_grid_pts, nb_sub_pts={"element": 5})

# Get a tensor-field (for example to represent the strain)
strain = fc.real_field("strain", (3, 3), "element")

# Fill the field with random numbers
strain.s[...] = np.random.rand(*((3, 3, 5) + nb_grid_pts))

# Initialize a file I/O object (using string for open mode)
file = muGrid.FileIONetCDF("example.nc", open_mode="overwrite")
file.register_field_collection(fc)  # Register the field collection with the file
file.append_frame().write()  # Write all fields of the field collection to the file

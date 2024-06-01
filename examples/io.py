import numpy as np
from muGrid import GlobalFieldCollection, FileIONetCDF, OpenMode

# Two dimensional grid
nb_grid_pts = (11, 12, 13)
fc = GlobalFieldCollection(nb_grid_pts, sub_pts={'element': 5})

# Get a tensor-field (for example to represent the strain)
strain = fc.real_field(
    'strain',  # name of the field
    (3, 3),  # shape of components
    'element'  # sub-point type
)

# Fill the field with random numbers
strain.s = np.random.rand(*((3, 3, 5) + nb_grid_pts))

# Initialize a file I/O object
file = FileIONetCDF('example.nc', OpenMode.Overwrite)
file.register_field_collection(fc)  # Register the field collection with the file
file.append_frame().write()  # Write all fields of the field collection to the file

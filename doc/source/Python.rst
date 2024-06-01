Python Bindings
###############

Fields
******

The µGrid library handles field quantities, i.e. scalar, vectors or tensors,
that vary in space. It supports only a uniform discretization of space. In the
language of µGrid, we call the discrete coordinates **pixels**. Each pixel
is associated with a physical position in space. µGrid supports fields in
two and three dimensional Cartesian grids. Note that a common name for a pixel
in three dimensions is a **voxel**, but we refer to it as a pixel throughout
this documentation.

Each pixel
can be subdivided into logical elements, for example into a number of
support points for numerical quadrature. These subdivisions are called
**sub-points**. Note that while µGrid understands the physical location of
each *pixel*, the *sub-points* are logical subdivisions and not associated
with any position within the pixel. Each *sub-point* carries the field quantity,
which can be a scalar, vector or any type of tensor. The field quantity is
called the **component**.

Multidimensional arrays
***********************

Each µGrid field has a representation in a multidimensional array. This
representation is used when accessing a field with Python via `numpy <https://numpy.org/>`_
or when writing the field to a file. The default representation has the shape

.. code-block:: Python

    (components, sub-points, pixels)

As a concrete example, a second-rank tensor (for example the deformation
gradient) living on two quadrature points in three dimensions with a spatial
discretization of 11 x 12 grid points would have the following shape:

.. code-block:: Python

    (3, 3, 2, 11, 12)

Note that the components are omitted if there is only a single component (i.e.
a scalar value), but the sub-points are always included even if there is only
a single sub-point.

Throughout the code, we call the field quantities **entries**. For example,
`nb_pixels` refers to the number of pixels while `nb_sub_pts` refers to the
number of sub-points per pixel. `nb_entries` is then the product of the two.

The above field has another multidimensional array representation that folds
the sub-point into the last dimension of the components. For the above example,
this means a multidimensional array of shape:

.. code-block:: Python

    (3, 6, 11, 12)

Depending on the numerical use case, it can be useful to work with either
representation.

Field collections
*****************

In µGrid, fields are grouped into field collections. A field collection knows
about the spatial discretization of the fields, but each field can differ in number of
sub-points and components.

There are two kinds of field collections:

* A **global** collection that groups fields that have values at all pixels.
* A **local** collection that groups fields that have values only at a subset of pixels.
  An example would be the elastic constants of a material that only exists at
  certain locations in the domain.

The following example shows how to initialize a field collection and create scalar fields:

.. literalinclude:: ../../examples/field_collection.py
    :language: python

The first argument to the constructor of `FieldCollection` is the spatial dimension.
Fields are then registered with the field accessor methods of the field collection,
e.g. `real_field` in the example above. Fields are *named*, and the name needs to be
unique. Accessing a field of the same name yield the same field object.

Note the `FieldCollection` additionally has register methods, e.g. `register_real_field`.
The different to `real_field` method is that the explicit registration of the field fails if
it already exists, while the accessor method `real_field` registers it if it does not exist
but returns the existing field if it does. You can also query a field with `get_field`,
which will raise an exception if the field does not exist.

Components
**********

In the above example, we registered a scalar field, which has one component. Vector or
tensor-valued field can be defined by specifying either simply a number of components or
the shape of the components explicitly. The following example shows how to create a
tensor-valued field that contains 2 x 2 matrices:

.. literalinclude:: ../../examples/components.py
    :language: python

Sub-points
**********

A pixel can be subdivided into multiple sub-points, each of which holds a value (scalar,
vector or tensor) of the field. Examples for sub-points are elements or quadrature points
in a the finite element method.

Sub-points are named, e.g. common names would be `element` for subdivision into elements or
`quad` for quadrature points. The name of the subdivision must be specified when the field
is created.

The following example initializes a three-dimension grid with
a sub-division of each pixel into five elements:

.. literalinclude:: ../../examples/sub_points.py
    :language: python

The example demonstrates two ways of accessing the field. The convenience accessor `p` (that
we also used in the above examples) and the new accessor `s`. Both yield a numpy array that is
a view on the underlying data, but with different shapes. The `s` accessor has the shape
`(components, sub-points, pixels)` and exposes the sub-points explicitly. The `p` accessor folds
the sub-points into the last dimension of the components.

numpy views
***********

The multidimensional array representation of a field is accessible via the
`array` method.

.. code-block:: Python

    a = displacement_field.array(muGrid.IterUnit.SubPt)

yields the multidimensional array with the explicit sub-point dimension.
The pixel-representation can be obtained by

.. code-block:: Python

    a = displacement_field.array(muGrid.IterUnit.Pixel)

Because those operations are used to frequently, there are shortcuts already
introduced in the examples above:

.. code-block:: Python

    displacement_field.s  # sub-point representation
    displacement_field.p  # pixel representation

The entries of the field occur as the first indices in the multidimensional
because a numerical code is typically vectorized of the spatial domain, i.e.
we carry out the same operation on each pixel but not on each component.
This means for the above displacement field, we can simply get the components
of the field from

.. code-block:: Python

    ux, uy, uz = displacement_field.s

Each of the variable `ux`, `uy`, and `uz` is a three-dimensional array with shape
`(2, 11, 12)`. *numpy*'s
`broadcasting rules <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_
make it simple to vectorize over pixels,
for example normalizing the displacement field could look like:

.. code-block:: Python

    displacement_field.s /= np.sqrt(ux**2 + uy**2 + uz**2)

Note that the default storage order of the field is column-major, which means the
field components are stored next to each other in memory.

I/O
***

Fields can be written to disk in the `NetCDF <https://en.wikipedia.org/wiki/NetCDF>`_ format.
µGrid uses `Unidata NetCDF <https://www.unidata.ucar.edu/software/netcdf/>`_ when build
with just serial capabilities and `PnetCDF <https://parallel-netcdf.github.io/>`_ when build
with MPI enabled.

I/O is handled by the `FileIONetCDF` class. The following example shows how to write all fields
from a field collection to disk:

.. literalinclude:: ../../examples/io.py
    :language: python

The file has the following structure (output of `ncdump -h`):

.. code-block::

    netcdf example {
    dimensions:
            frame = UNLIMITED ; // (1 currently)
            tensor_dim__strain-0 = 3 ;
            tensor_dim__strain-1 = 3 ;
            subpt__element-5 = 5 ;
            nx = 11 ;
            ny = 12 ;
            nz = 13 ;
    variables:
            double strain(frame, tensor_dim__strain-0, tensor_dim__strain-1, subpt__element-5, nx, ny, nz) ;
                    strain:unit = "no unit provided" ;

    // global attributes:
                    :creation_date = "01-06-2024 (d-m-Y)" ;
                    :creation_time = "23:02:06 (H:M:S)" ;
                    :last_modified_date = "01-06-2024 (d-m-Y)" ;
                    :last_modified_time = "23:02:06 (H:M:S)" ;
                    :muGrid_version_info = "µGrid version: 0.90.1+40-g5bc73b30\n",
                            "" ;
                    :muGrid_git_hash = "5bc73b305881ef837ce6598568d41eb3d0307c41" ;
                    :muGrid_description = "0.90.1+40-g5bc73b30" ;
                    :muGrid_git_branch_is_dirty = "false" ;
    }

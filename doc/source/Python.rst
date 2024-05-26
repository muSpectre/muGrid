Python Bindings
###############

Fields
******

The *µ*\Grid library handles field quantities, i.e. scalar, vectors or tensors,
that vary in space. It supports only a uniform discretization of space. In the
language of *µ*\Grid, we call the discrete coordinates **pixels**. Each pixel
is associated with a physical position in space. *µ*\Grid supports fields in
two and three dimensional Cartesian grids. Note that a common name for a pixel
in three dimensions is a **voxel**, but we refer to it as a pixel throughout
this documentation.

Each pixel
can be subdivided into logical elements, for example into a number of
support points for numerical quadrature. These subdivisions are called
**sub-points**. Note that while *µ*\Grid understands the physical location of
each *pixel*, the *sub-points* are logical subdivisions and not associated
with any position within the pixel. Each *sub-point* carries the field quantity,
which can be a scalar, vector or any type of tensor. The field quantity is
called the **component**.

Multidimensional arrays
***********************

Each *µ*\Grid field has a representation in a multidimensional array. This
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

In *µ*\Grid, fields are grouped into field collections. A field collection knows
about the spatial discretization of the fields, but each field can differ in number of
sub-points and components.

There are two kinds of field collections:

* A **global** collection that groups fields that have values at all pixels.
* A **local** collection that groups fields that have values only at a subset of pixels.
  An example would be the elastic constants of a material that only exists at
  certain locations in the domain.

A global field collection can be initialized as follows:

.. code-block:: Python

    from muGrid import GlobalFieldCollection
    fc = GlobalFieldCollection(3, {'element': 2})
    fc.initialise([11, 12, 13], [11, 12, 13], [0, 0, 0])

The first argument is the spatial dimension, the second argument defines sub-points.
A field that is defined on `element` has two sub-points that in the example that we
will follow through now are two (finite) elements that are the triangles that constitute
half of a pixel each. The `initialise` method defines
the spatial discretization of the domain, taking the number of pixels as the first
argument.

We can now register a new field in the collection:

.. code-block:: Python

    displacement_field = fc.register_real_field('displacement', 3, 'element')

Note that a field can only be registered once in a collection and needs to have a unique
name. We can query the field by name:

.. code-block:: Python

    displacement_field = fc.get_real_field('displacement')

Working with fields
*******************

The multidimensional array representation of a field is accessible via the
`array` method.

.. code-block:: Python

    a = displacement_field.array(muGrid.IterUnit.SubPt)

yields the multidimensional array with the explicit sub-point dimension.
The pixel-representation can be obtained by

.. code-block:: Python

    a = displacement_field.array(muGrid.IterUnit.Pixel)

Because those operations are used to frequently, there are shortcuts:

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

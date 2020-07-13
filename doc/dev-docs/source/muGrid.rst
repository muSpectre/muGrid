
.. toctree::
   :maxdepth: 2
   :caption: Contents:


.. _mugrid:


*µ*\Grid
########

The *µ*\Grid library handles field quantities, i.e. scalar, vectors or tensors,
that vary in space. It supports only a uniform discretization of space. In the
language of *µ*\Grid, we call the discrete coordinates **pixels**. Each pixel
can be subdivided into logical elements, for example into a number of
support points for numerical quadrature. These subdivisions are called
**sub-points**. Note that while *µ*\Grid can understand the physical location of
each *pixel*, the *sub-points* are logical subdivisions and not associated
with any position within the pixel. Each *sub-point* carries the field quantity,
which can be a scalar, vector or any type of tensor. The field quantity is
called the **component**.

Each *µ*\Grid field has a representation in a multidimensional array. This
representation is used when accessing a field with Python via *numpy* or when
writing the field to a file. The default representation has the shape

.. code-block:: Python

    (components, sub-points, pixels)

As a concrete example, a second-rank tensor (for example the deformation
gradient) living on two quadrature points in three dimensions with a spatial
discretization of 11 x 12 x 13 grid points would have the following shape:

.. code-block:: Python

    (3, 3, 2, 11, 12, 13)

Note that the components are omitted if there is only a single component (i.e.
a scalar value), but the sub-points are always included even if there is only
a single sub-point. The default storage order of that representation is
column-major.

Throughout the code, we call the field quantities **entries**. For example,
`nb_pixels` refers to the number of pixels while `nb_sub_pts` refers to the
number of sub-points per pixel. `nb_entries` is then the product of the two.

Common Type Aliases
%%%%%%%%%%%%%%%%%%%
As the lowest-level component of *µ*\Spectre, *µ*\Grid defines all the commonly used type aliases and data structures used throuhgout the project. The most common aliases are described below, but it is worth having a look at the file `grid_common.hh <https://gitlab.com/muspectre/muspectre/blob/master/src/libmugrid/grid_common.hh>`_ for the details.

All mathematical calculations should use these types.

.. doxygengroup:: Scalars
   :members:

While it is possible to use other types in principle, these are the ones for which all data structures are tested and which are known to work. Also, other *µ*\Spectre developers will expect and understand these types.

Dimensions are counted using the signed integer type :cpp:type:`muGrid::Dim_t`. This is necessary because `Eigen <eigen.tuxfamily.org>`_ uses -1 to signify a dynamic number of dimensions.

The types :cpp:type:`muGrid::Rcoord_t` and :cpp:type:`muGrid::Ccoord_t`  are used to represent real-valued coordinates and integer-valued coordinates (i.e., pixel- or cell-coordinates).

.. doxygengroup:: Coordinates
   :members:

These types are also used to define nb_grid_pts or spatial lengths for computational domains.


Field Data Types
%%%%%%%%%%%%%%%%

The most important part of *µ*\Grid to understand is how it handles field data
and access to it. By field we mean the discretisation of a mathematical field on
the grid points, i.e., numerical data associated with all pixels/voxels of an
FFT grid or a subset thereof. The numerical data can be **scalar**,
**vectorial**, **matricial**, **tensorial** or a generic **array** of
**integer**, **real** or **complex** values per pixel.

Fields that are defined on every pixel/voxel of a grid are called **global
fields** while fields defined on a subset of pixels/voxels **local fields**. As
an example, the strain field is a global field for any calculation (it exists in
the entire domain), while for instance the field of an internal (state) variable
of a material in a composite is only defined for the pixels that belong to that
material.

There are several ways in which we interact with fields, and the same field
might be interacted with in different ways by different parts of a
problem. Let's take the (global) strain field in a three-dimensional finite
strain problem with 255 × 255 × 255 voxels as an example: the solver treats it
as a long vector (of length 3² · 255³), the FFT sees it as a four-dimensional
array of shape 255 × 255 × 255 × 3², and from the constitutive laws'
perspective, it is just a sequence of second-rank tensors (i.e., shape 255³ × 3
× 3).



Basic *µ*\Grid Field Concepts
+++++++++++++++++++++++++++++

In order to reconcile these different interpretations without copying data
around, *µ*\Grid splits the concept of a field into three components:

* storage

  This refers managing the actual memory in which field data is held. For this,
  the storage abstraction needs to know the scalar type of data (``Int``,
  ``Real``, ``Complex``, e.t.c.), the number of pixels/voxels for which the
  field is defined, and the number of scalar components per pixel/voxel (e.g., 9
  for a second-rank asymmetric tensor in a three-dimensional problem).

  *µ*\Grid's abstraction for field data storage is the **field** represented by a
  child class of
  :cpp:class:`FieldBase\<FieldCollection_t><muGrid::internal::FieldBase>`, see
  :ref:`fields`.

* representation

  Meaning how to interpret the data at a given pixel/voxel (i.e., is it a
  vector, a matrix, ...). This will also determine which mathematical operations
  can be performed on per-pixel/voxel data. The representation allows also to
  iterate over a field pixel/voxel by pixel/voxel.

  *µ*\Grid's abstraction for field representations is the **field map**
  represented by a child class of :cpp:class:`FieldMap\<FieldCollection_t,
  Scalar_t, NbComponents[, IsConst]><muGrid::internal::FieldMap>`, see
  :ref:`field_map`.

* per-pixel/voxel access/iteration

  Given a pixel/voxel coordinate or index, the position of the associated
  pixel/voxel data is a function of the type of field (global or local). Since
  the determination procedure is identical for every field defined on the same
  domain, this ability (and the associated overhead) can be centralised into a
  manager of field collections.

  *µ*\Grid's abstraction for field access and management is the **field
  collection** represented by the two classes
  :cpp:class:`LocalFieldCollection\<Dim><muGrid::LocalFieldCollection>` and
  :cpp:class:`GlobalFieldCollection\<Dim><muGrid::GlobalFieldCollection>`, see
  :ref:`field_collection`.

.. _fields:

Fields
``````

Fields are where the data is stored, so they are mainly distinguished by the
scalar type they store (``Int``, ``Real`` or ``Complex``), and the number of
components (statically fixed size, or dynamic size).

The most commonly used fields are the statically sized ones,
:cpp:class:`TensorField<muGrid::TensorField>`,
:cpp:class:`MatrixField<muGrid::MatrixField>`, and the
:cpp:type:`ScalarField<muGrid::ScalarField>` (which is really just a 1×1 matrix
field).

Less commonly, we use the dynamically sized
:cpp:class:`TypedField<muGrid::TypedField>`, but more on this later.

Fields have a protected constructor, which means that you cannot directly build
a field object, instead you have to go through the factory function
:cpp:func:`make_field\<Field_t>(name, collection)<muGrid::make_field>` (or
:cpp:func:`make_statefield\<Field_t>(name, collection)<muGrid::make_statefield>`
if you're building a statefield, see :ref:`state_field`) to create them and you only
receive a reference to the built field. The field itself is stored in a
:cpp:class:`std::unique_ptr` which is registered in and managed by a :ref:`field
collection<field_collection>`. This mechanism is meant to ensure that fields are
not copied around or free'd so that :ref:`field maps<field_map>` always remain
valid and unambiguously linked to a field.

Fields give access to their bulk memory in form of an
:cpp:class:`Eigen::Map`. This is useful for instance for accessing the global
strain, stress, and tangent moduli fields in the solver.

Example: Using fields as global arrays:
.......................................
The following is a code example from the standard :ref:`Cell<muSpectre::CellBase>`:

.. code-block:: c++
   :linenos:

   template <Dim_t DimS, Dim_t DimM>
   auto CellBase<DimS, DimM>::get_strain_vector() -> Vector_ref {
     return this->get_strain().eigenvec();
   }

The return value of :cpp:function:`Cell::get_strain_vector()<muSpectre::CellBase::get_strain_vector>` is an :cpp:class:`Eigen::Map` onto a matrix of shape 1×N.

If you wish to handle field data on a per-pixel/voxel basis, the mechanism for
that is the field map and is described in :ref:`field_map`.

.. _field_map:

Field Maps
``````````

Field maps are light-weight resource handles (meaning they can be created and
destroyed cheaply) that are iterable and provide direct per-pixel/voxel access
to the data stored in the mapped field.

The choice of field map defines the type of reference you obtain when
dereferencing an iterator or using the direct random acccess operator ``[]``.

Typically used field maps include:

  - :cpp:class:`ScalarFieldMap<muGrid::ScalarFieldMap>`,
  - :cpp:class:`ArrayFieldMap<muGrid::ArrayFieldMap>`,
  - :cpp:class:`MatrixFieldMap<muGrid::MatrixFieldMap>`, and the
  - :cpp:type:`T4MatrixFieldMap<muGrid::T4MatrixFieldMap>`.

All of these are fixed size (meaning their size is set at compile time) and
therefore support fast linear algebra on the iterates. There is also a
dynamically sized field map type, the
:cpp:class:`TypedFieldMap<muGrid::TypedFieldMap>` which is useful for debugging
and python bindings. It supports all the features of the fixed-size maps, but
linear algebra on the iterates will be slow because it cannot be effectively
vectorised.

Example 1: Iterating over fields and do math on the iterates:
..............................................................

The following is a code example from the tests of the finite-strain projection
operator defined by `T.W.J. de Geus, J. Vondřejc, J. Zeman, R.H.J. Peerlings,
M.G.D. Geers <https://doi.org/10.1016/j.cma.2016.12.032>`_.

.. literalinclude:: ../../../tests/test_projection_finite.cc
   :language: c++
   :linenos:
   :start-after: start_field_iteration_snippet
   :end-before: end_field_iteration_snippet
   :dedent: 4

.. _field_collection:

Field Collections
`````````````````

Field collections come in two flavours;
:cpp:class:`LocalFieldCollection\<Dim><muGrid::LocalFieldCollection>` and
:cpp:class:`GlobalFieldCollection\<Dim><muGrid::GlobalFieldCollection>` and are
templated by the spatial dimension of the problem. They adhere to the interface
defined by their common base class,
:cpp:class:`FieldCollectionBase<muGrid::FieldCollectionBase>`. Both types are
iterable (the iterates are the coordinates of the pixels/voxels for which the
fields of the collection are defiened.

Global field collections need to be given the problem nb_grid_pts (i.e. the size
of the grid) at initialisation, while local collections need to be filled with
pixels/voxels through repeated calls to
:cpp:func:`add_pixel(pixel)<muGrid::LocalFieldCollection::add_pixel>`. At
initialisation, they derive their size from the number of pixels that have been
added.

Fields (State Fields) are identified by their unique name (prefix) and can be retrieved in multiple ways:

.. doxygenfunction:: muGrid::FieldCollectionBase::operator[]
.. doxygenfunction:: muGrid::FieldCollectionBase::at
.. doxygenfunction:: muGrid::FieldCollectionBase::get_typed_field(std::string)
.. doxygenfunction:: muGrid::FieldCollectionBase::get_statefield(std::string)
.. doxygenfunction:: muGrid::FieldCollectionBase::get_typed_statefield(std::string)

.. _state_field:

* per-pixel/voxel access/iteration

  Given a pixel/voxel coordinate or index, the position of the associated
  pixel/voxel data is a function of the type of field (global or local). Since
  the determination procedure is identical for every field defined on the same
  domain, this ability (and the associated overhead) can be centralised into a
  manager of field collections.

  *µ*\Grid's abstraction for field access and management is the **field
  collection** represented by the two classes
  :cpp:class:`LocalFieldCollection\<Dim><muGrid::LocalFieldCollection>` and
  :cpp:class:`LocalFieldCollection\<Dim><muGrid::LocalFieldCollection>`, see
  :ref:`field_collection`.

.. _fields:

Fields
``````

Fields are where the data is stored, so they are mainly distinguished by the
scalar type they store (``Int``, ``Real`` or ``Complex``), and the number of
components (statically fixed size, or dynamic size).

The most commonly used fields are the statically sized ones,
:cpp:class:`TensorField<muGrid::TensorField>`,
:cpp:class:`MatrixField<muGrid::MatrixField>`, and the
:cpp:type:`ScalarField<muGrid::ScalarField>` (which is really just a 1×1 matrix
field).

Less commonly, we use the dynamically sized
:cpp:class:`TypedField<muGrid::TypedField>`, but more on this later.

Fields have a protected constructor, which means that you cannot directly build
a field object, instead you have to go through the factory function
:cpp:func:`make_field\<Field_t>(name, collection)<muGrid::make_field>` (or
:cpp:func:`make_statefield\<Field_t>(name, collection)<muGrid::make_statefield>`
if you're building a statefield, see :ref:`state_field`) to create them and you only
receive a reference to the built field. The field itself is stored in a
:cpp:class:`std::unique_ptr` which is registered in and managed by a :ref:`field
collection<field_collection>`. This mechanism is meant to ensure that fields are
not copied around or free'd so that :ref:`field maps<field_map>` always remain
valid and unambiguously linked to a field.

Fields give access to their bulk memory in form of an
:cpp:class:`Eigen:Map`. This is useful for instance for accessing the global
strain, stress, and tangent moduli fields in the solver.

If you wish to handle field data on a per-pixel/voxel basis, the mechanism for
that is the field map and is described in :ref:`field_map`.

Example: Using fields as global arrays:
.......................................
The following is a code example from the standard :ref:`Cell<muSpectre::CellBase>`:

.. code-block:: c++
   :linenos:

   template <Dim_t DimS, Dim_t DimM>
   auto CellBase<DimS, DimM>::get_strain_vector() -> Vector_ref {
     return this->get_strain().eigenvec();
   }

The return value of :cpp:function:`Cell::get_strain_vector()<muSpectre::CellBase::get_strain_vector>` is an :cpp:class:`Eigen::Map` onto a matrix of shape 1×N.

If you wish to handle field data on a per-pixel/voxel basis, the mechanism for
that is the field map and is described in :ref:`field_map`.

.. _field_map:

Field Maps
``````````

Field maps are light-weight resource handles (meaning they can be created and
destroyed cheaply) that are iterable and provide direct per-pixel/voxel access
to the data stored in the mapped field.

The choice of field map defines the type of reference you obtain when
dereferencing an iterator or using the direct random acccess operator ``[]``.

Typically used field maps include:

  - :cpp:class:`ScalarFieldMap<muGrid::ScalarFieldMap>`,
  - :cpp:class:`ArrayFieldMap<muGrid::ArrayFieldMap>`,
  - :cpp:class:`MatrixFieldMap<muGrid::MatrixFieldMap>`, and the
  - :cpp:type:`T4MatrixFieldMap<muGrid::T4MatrixFieldMap>`.

All of these are fixed size (meaning their size is set at compile time) and
therefore support fast linear algebra on the iterates. There is also a
dynamically sized field map type, the
:cpp:class:`TypedFieldMap<muGrid::TypedFieldMap>` which is useful for debugging
and python bindings. It supports all the features of the fixed-size maps, but
linear algebra on the iterates will be slow because it cannot be effectively
vectorised.

Example 1: Iterating over fields and do math on the iterates:
..............................................................

The following is a code example from the tests of the finite-strain projection
operator defined by `T.W.J. de Geus, J. Vondřejc, J. Zeman, R.H.J. Peerlings,
M.G.D. Geers <https://doi.org/10.1016/j.cma.2016.12.032>`_.

.. literalinclude:: ../../../tests/test_projection_finite.cc
   :language: c++
   :linenos:
   :start-after: start_field_iteration_snippet
   :end-before: end_field_iteration_snippet
   :dedent: 4

.. _field_collection:

Field Collections
`````````````````

Field collections come in two flavours;
:cpp:class:`LocalFieldCollection\<Dim><muGrid::LocalFieldCollection>` and
:cpp:class:`GlobalFieldCollection\<Dim><muGrid::GlobalFieldCollection>` and are
templated by the spatial dimension of the problem. They adhere to the interface
defined by their common base class,
:cpp:class:`FieldCollectionBase<muGrid::FieldCollectionBase>`. Both types are
iterable (the iterates are the coordinates of the pixels/voxels for which the
fields of the collection are defiened.

Global field collections need to be given the problem nb_grid_pts (i.e. the size
of the grid) at initialisation, while local collections need to be filled with
pixels/voxels through repeated calls to
:cpp:func:`add_pixel(pixel)<muGrid::LocalFieldCollection::add_pixel>`. At
initialisation, they derive their size from the number of pixels that have been
added.

Fields (State Fields) are identified by their unique name (prefix) and can be retrieved in multiple ways:

.. doxygenfunction:: muGrid::FieldCollectionBase::operator[]
.. doxygenfunction:: muGrid::FieldCollectionBase::at
.. doxygenfunction:: muGrid::FieldCollectionBase::get_typed_field(std::string)
.. doxygenfunction:: muGrid::FieldCollectionBase::get_statefield(std::string)
.. doxygenfunction:: muGrid::FieldCollectionBase::get_typed_statefield(std::string)

.. _state_field:

State or History Variables
++++++++++++++++++++++++++

Some fields hold state or history variables, i.e., such fields have a current
value and one or more old values. This is particularly common for internal
variables of inelastic materials (e.g., damage variables, plastic flow,
e.t.c.). The straight-forward way of handling this situation is to define a
current field, and one or more fields of the same type to hold old values. This
approach has the disadvantages that it leads to a multitude of variables to keep
track of, and that the values need to by cycled between the fields using a copy;
this approach is both inefficient and error-prone.

*µ*\Grid addresses this situation with the state field abstraction. A state
 field is an encapsulated container of fields in a single variable. It allows to
 access the current field values globally, and gives read-only access to old
 field values globally. Iterative per-pixel access is handled through state
 field maps which, similarly to the field map, allow to iterate though all
 pixels/voxels on which the field is defined, and the iterates give access to
 the current value at the pixel/voxel or read-only access to the old values.

Mapped Fields
+++++++++++++

Some fields are only ever going to be used by one entity (e.g., internal
variables of a material). For these fields, the flexibility of the field/field
collection/field map paradigm can be a burden. Mapped fields are an
encapsulation of a field and a corresponding map into a single object,
drastically reducing boilerplate code.

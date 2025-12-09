#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   Field.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   09 Dec 2024

@brief  Python wrapper for muGrid fields with numpy array views

Copyright © 2024 Lars Pastewka

µGrid is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µGrid is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with µGrid; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.

Additional permission under GNU GPL version 3 section 7

If you modify this Program, or any covered work, by linking or combining it
with proprietary FFT implementations or numerical libraries, containing parts
covered by the terms of those libraries' licenses, the licensors of this
Program grant you additional permission to convey the resulting work.
"""

import numpy as np


class Field:
    """
    Python wrapper for muGrid fields providing numpy array views.

    This class wraps a C++ muGrid field and provides the following properties
    for accessing the underlying data as numpy arrays:

    - `s`: SubPt layout, excluding ghost regions
    - `sg`: SubPt layout, including ghost regions
    - `p`: Pixel layout, excluding ghost regions
    - `pg`: Pixel layout, including ghost regions

    The arrays are views into the underlying C++ data, so modifications
    to the arrays will modify the field data directly (zero-copy).
    """

    def __init__(self, cpp_field):
        """
        Initialize the Field wrapper.

        Parameters
        ----------
        cpp_field : _muGrid.RealFieldBase or similar
            The underlying C++ field object
        """
        self._cpp = cpp_field
        self._buffer = None

    def _get_buffer(self):
        """Lazy-load the full buffer via DLPack."""
        if self._buffer is None:
            self._buffer = np.from_dlpack(self._cpp)
        return self._buffer

    def _make_slice(self, offsets, shape):
        """Create slice tuple from offsets and shape."""
        return tuple(slice(o, o + s) for o, s in zip(offsets, shape))

    @property
    def sg(self):
        """
        SubPt layout array including ghost regions.

        Shape: (*components_shape, nb_sub_pts, *spatial_dims_with_ghosts)
        """
        return self._get_buffer()

    @property
    def s(self):
        """
        SubPt layout array excluding ghost regions.

        Shape: (*components_shape, nb_sub_pts, *spatial_dims_without_ghosts)
        """
        buf = self._get_buffer()
        offsets = self._cpp.offsets_s
        shape = self._cpp.shape_s
        slices = self._make_slice(offsets, shape)
        return buf[slices]

    @property
    def pg(self):
        """
        Pixel layout array including ghost regions.

        Shape: (nb_components * nb_sub_pts, *spatial_dims_with_ghosts)
        """
        buf = self._get_buffer()
        shape_pg = self._cpp.shape_pg
        return buf.reshape(shape_pg)

    @property
    def p(self):
        """
        Pixel layout array excluding ghost regions.

        Shape: (nb_components * nb_sub_pts, *spatial_dims_without_ghosts)
        """
        buf = self._get_buffer()
        shape_pg = self._cpp.shape_pg
        pixel_buf = buf.reshape(shape_pg)
        offsets = self._cpp.offsets_p
        shape = self._cpp.shape_p
        slices = self._make_slice(offsets, shape)
        return pixel_buf[slices]

    # Delegate attribute access to the underlying C++ object
    def __getattr__(self, name):
        """Delegate attribute access to the underlying C++ field."""
        return getattr(self._cpp, name)

    def __repr__(self):
        return f"Field({self._cpp.name!r}, shape={self._cpp.shape})"


def wrap_field(cpp_field):
    """
    Wrap a C++ field in a Python Field object.

    Parameters
    ----------
    cpp_field : _muGrid field object
        The underlying C++ field

    Returns
    -------
    Field
        Wrapped field with numpy array access
    """
    return Field(cpp_field)


def real_field(collection, name, components=(), sub_pt="pixel"):
    """
    Create a real-valued field and return it wrapped with numpy accessors.

    Parameters
    ----------
    collection : FieldCollection or CartesianDecomposition
        The field collection or decomposition to create the field in.
        If a CartesianDecomposition is passed, uses its internal collection.
    name : str
        Name of the field
    components : tuple, optional
        Shape of the field components (default: scalar field)
    sub_pt : str, optional
        Sub-point type (default: "pixel")

    Returns
    -------
    Field
        Wrapped field with .s, .sg, .p, .pg accessors
    """
    # Handle CartesianDecomposition by getting its collection
    fc = getattr(collection, 'collection', collection)
    cpp_field = fc.real_field(name, components, sub_pt)
    return Field(cpp_field)


def int_field(collection, name, components=(), sub_pt="pixel"):
    """
    Create an integer field and return it wrapped with numpy accessors.

    Parameters
    ----------
    collection : FieldCollection or CartesianDecomposition
        The field collection or decomposition to create the field in.
        If a CartesianDecomposition is passed, uses its internal collection.
    name : str
        Name of the field
    components : tuple, optional
        Shape of the field components (default: scalar field)
    sub_pt : str, optional
        Sub-point type (default: "pixel")

    Returns
    -------
    Field
        Wrapped field with .s, .sg, .p, .pg accessors
    """
    # Handle CartesianDecomposition by getting its collection
    fc = getattr(collection, 'collection', collection)
    cpp_field = fc.int_field(name, components, sub_pt)
    return Field(cpp_field)


def uint_field(collection, name, components=(), sub_pt="pixel"):
    """
    Create an unsigned integer field and return it wrapped with numpy accessors.

    Parameters
    ----------
    collection : FieldCollection or CartesianDecomposition
        The field collection or decomposition to create the field in.
        If a CartesianDecomposition is passed, uses its internal collection.
    name : str
        Name of the field
    components : tuple, optional
        Shape of the field components (default: scalar field)
    sub_pt : str, optional
        Sub-point type (default: "pixel")

    Returns
    -------
    Field
        Wrapped field with .s, .sg, .p, .pg accessors
    """
    # Handle CartesianDecomposition by getting its collection
    fc = getattr(collection, 'collection', collection)
    cpp_field = fc.uint_field(name, components, sub_pt)
    return Field(cpp_field)


def complex_field(collection, name, components=(), sub_pt="pixel"):
    """
    Create a complex-valued field and return it wrapped with numpy accessors.

    Parameters
    ----------
    collection : FieldCollection or CartesianDecomposition
        The field collection or decomposition to create the field in.
        If a CartesianDecomposition is passed, uses its internal collection.
    name : str
        Name of the field
    components : tuple, optional
        Shape of the field components (default: scalar field)
    sub_pt : str, optional
        Sub-point type (default: "pixel")

    Returns
    -------
    Field
        Wrapped field with .s, .sg, .p, .pg accessors
    """
    # Handle CartesianDecomposition by getting its collection
    fc = getattr(collection, 'collection', collection)
    cpp_field = fc.complex_field(name, components, sub_pt)
    return Field(cpp_field)

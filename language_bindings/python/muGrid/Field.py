#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   Field.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   09 Dec 2024

@brief  Python wrapper for muGrid fields with numpy/cupy array views

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

try:
    from . import _muGrid
except ImportError:
    import _muGrid

# CuPy is optional - only imported when GPU fields are used
_cupy = None


def _get_cupy():
    """Lazy import of CuPy for GPU array support."""
    global _cupy
    if _cupy is None:
        try:
            import cupy
            _cupy = cupy
        except ImportError:
            raise ImportError(
                "CuPy is required for GPU field access. "
                "Install it with: pip install cupy-cuda12x "
                "(or appropriate CUDA or ROCm version)"
            )
    return _cupy


class Field:
    """
    Python wrapper for muGrid fields providing numpy/cupy array views.

    This class wraps a C++ muGrid field and provides the following properties
    for accessing the underlying data as arrays:

    - `s`: SubPt layout, excluding ghost regions
    - `sg`: SubPt layout, including ghost regions
    - `p`: Pixel layout, excluding ghost regions
    - `pg`: Pixel layout, including ghost regions

    For CPU fields, the arrays are numpy arrays. For GPU fields (CUDA),
    the arrays are CuPy arrays. Both are views into the underlying C++ data,
    so modifications to the arrays will modify the field data directly (zero-copy).
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

    @property
    def is_on_gpu(self):
        """Check if this field resides on GPU memory."""
        return self._cpp.is_on_gpu

    @property
    def device(self):
        """Get the device where this field resides ('cpu' or 'cuda:N')."""
        return self._cpp.device

    def _get_buffer(self):
        """Lazy-load the full buffer via DLPack (requires numpy >= 2.1)."""
        if self._buffer is None:
            if self.is_on_gpu:
                cp = _get_cupy()
                self._buffer = cp.from_dlpack(self._cpp)
            else:
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

    def _check_pixel_layout_is_portable(self):
        """Refuse the pixel layout where its order depends on storage.

        The pixel layout folds the sub-points into the last component axis.
        In array-of-structures storage (the host default) that axis runs
        component-fastest, ``j = c + C * s``; in structure-of-arrays storage
        (the device default) sub-point-fastest, ``j = c * S + s``. Each is a
        zero-copy view only in its own storage order, so ``.p[j]`` named a
        different (component, sub-point) pair on host and device for the same
        data -- the same code silently read and wrote different entries.
        A copy in one canonical order is not an option either: writes to
        ``.p`` would go into the copy and be lost. The orders agree when the
        last component axis or the sub-point count is 1, so only the other
        case, in structure-of-arrays storage, is refused; ``.s``/``.sg``
        index components and sub-points separately and are identical in both
        storage orders.
        """
        components = tuple(self._cpp.components_shape)
        if (self._cpp.storage_order == _muGrid.StorageOrder.StructureOfArrays
                and self._cpp.nb_sub_pts > 1 and components
                and components[-1] > 1):
            raise ValueError(
                f"The pixel layout (.p/.pg) of field '{self._cpp.name}' "
                f"(components {components}, {self._cpp.nb_sub_pts} "
                "sub-points) orders components and sub-points differently "
                "in structure-of-arrays storage (the device default) than "
                "on the host. Use .s/.sg, which index them separately and "
                "are the same on host and device.")

    def _pixel_view(self):
        """
        Pixel-layout view of the full buffer (components and sub-points
        merged into one axis).

        Built from the C++ pixel shape and strides so it is always a
        zero-copy view with the C++ Pixel element order. A plain
        ``reshape`` cannot do this: merging the component and sub-point
        axes of the (Fortran-ordered) buffer in C order would silently
        copy and permute elements whenever both counts exceed one.
        """
        self._check_pixel_layout_is_portable()
        buf = self._get_buffer()
        shape = tuple(self._cpp.shape_pg)
        strides = tuple(s * buf.itemsize for s in self._cpp.strides_p)
        if self.is_on_gpu:
            cp = _get_cupy()
            return cp.lib.stride_tricks.as_strided(
                buf, shape=shape, strides=strides
            )
        return np.lib.stride_tricks.as_strided(
            buf, shape=shape, strides=strides
        )

    @property
    def pg(self):
        """
        Pixel layout array including ghost regions.

        Shape: (nb_components * nb_sub_pts, *spatial_dims_with_ghosts)
        """
        return self._pixel_view()

    @property
    def p(self):
        """
        Pixel layout array excluding ghost regions.

        Shape: (nb_components * nb_sub_pts, *spatial_dims_without_ghosts)
        """
        pixel_buf = self._pixel_view()
        offsets = self._cpp.offsets_p
        shape = self._cpp.shape_p
        slices = self._make_slice(offsets, shape)
        return pixel_buf[slices]

    @property
    def dtype(self):
        """
        NumPy dtype of the field's scalar entries (e.g. ``float64``,
        ``float32``, ``complex128``, ``complex64``). Read from the underlying
        buffer, so it works identically on host (NumPy) and device (CuPy).
        """
        return self._get_buffer().dtype

    # Delegate attribute access to the underlying C++ object
    def __getattr__(self, name):
        """Delegate attribute access to the underlying C++ field."""
        return getattr(self._cpp, name)

    def __repr__(self):
        return f"Field({self._cpp.name!r}, shape={self._cpp.shape})"


def wrap_field(field):
    """
    Wrap a C++ field in a Python Field object.

    Parameters
    ----------
    field : _muGrid field object
        The underlying C++ field

    Returns
    -------
    Field
        Wrapped field with numpy array access
    """
    if isinstance(field, Field):
        return field
    return Field(field)

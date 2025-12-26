#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file     python_field_tests.py

@author  Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date    18 Oct 2019

@brief   test field collections and fields

Copyright © 2018 Till Junge

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

import unittest

import numpy as np

import muGrid
from muGrid import wrap_field


class FieldCheck(unittest.TestCase):
    def setUp(self):
        self.nb_grid_pts = (10, 11, 21)

    def test_buffer_size_one_quad_pt(self):
        values = np.random.rand(*self.nb_grid_pts)

        nb_grid_pts2 = (4,) + self.nb_grid_pts
        values2 = np.random.rand(*nb_grid_pts2)

        fc = muGrid.GlobalFieldCollection(self.nb_grid_pts)

        # Scalar - wrap field to access p/s/pg/sg properties
        fs_cpp = fc.register_real_field("test-field-scalar")
        fs = wrap_field(fs_cpp)
        self.assertEqual(fs.p.shape, self.nb_grid_pts)
        self.assertEqual(fs.s.shape, (1,) + self.nb_grid_pts)

        # Single component
        f_cpp = fc.register_real_field("test-field", 1)
        f = wrap_field(f_cpp)
        self.assertEqual(f.p.shape, (1,) + self.nb_grid_pts)
        self.assertEqual(f.s.shape, (1, 1) + self.nb_grid_pts)

        # Four components
        f2_cpp = fc.register_real_field("test-field2", 4)
        f2 = wrap_field(f2_cpp)
        self.assertEqual(f2.p.shape, nb_grid_pts2)
        self.assertEqual(f2.s.shape, (4, 1) + self.nb_grid_pts)

        # Check that we get those fields back
        self.assertEqual(fc.get_field("test-field-scalar"), fs_cpp)
        self.assertEqual(fc.get_field("test-field"), f_cpp)
        self.assertEqual(fc.get_field("test-field2"), f2_cpp)

        # Check strides
        self.assertEqual(fs.stride(muGrid.Pixel), 1)
        self.assertEqual(f.stride(muGrid.Pixel), 1)
        self.assertEqual(f2.stride(muGrid.Pixel), 4)
        self.assertEqual(fs.stride(muGrid.SubPt), 1)
        self.assertEqual(f.stride(muGrid.SubPt), 1)
        self.assertEqual(f2.stride(muGrid.SubPt), 4)

        # Check subpoint-shaped convenience access (read) and data assignment
        self.assertEqual(fs.s.shape, (1,) + self.nb_grid_pts)
        fs.s[...] = values.reshape((1,) + self.nb_grid_pts)
        np.testing.assert_allclose(fs.s, values.reshape((1,) + self.nb_grid_pts))

        self.assertEqual(f.s.shape, (1, 1) + self.nb_grid_pts)
        f.s[...] = values.reshape((1, 1) + self.nb_grid_pts)
        np.testing.assert_allclose(f.s, values.reshape((1, 1) + self.nb_grid_pts))

        self.assertEqual(f2.s.shape, (4, 1) + self.nb_grid_pts)

        # Check pixel-shaped convenience access (read) and data assignment
        self.assertEqual(fs.p.shape, self.nb_grid_pts)
        fs.p[...] = values
        np.testing.assert_allclose(fs.p, values)

        self.assertEqual(f.p.shape, (1,) + self.nb_grid_pts)
        f.p[...] = values.reshape((1,) + self.nb_grid_pts)
        np.testing.assert_allclose(f.p, values.reshape((1,) + self.nb_grid_pts))

        self.assertEqual(f2.p.shape, nb_grid_pts2)
        f2.p[...] = values2
        np.testing.assert_allclose(f2.p, values2)

    def test_buffer_size_four_quad_pt(self):
        fc = muGrid.GlobalFieldCollection(self.nb_grid_pts, sub_pts={"quad": 4})
        # Single component
        f_cpp = fc.register_real_field("test-field", 1, "quad")
        f = wrap_field(f_cpp)
        self.assertEqual(f.p.shape, (4,) + self.nb_grid_pts)
        self.assertEqual(f.s.shape, (1, 4) + self.nb_grid_pts)
        # Four components
        f2_cpp = fc.register_real_field("test-field2", 3, "quad")
        f2 = wrap_field(f2_cpp)
        self.assertEqual(f2.p.shape, (4 * 3,) + self.nb_grid_pts)
        self.assertEqual(f2.s.shape, (3, 4) + self.nb_grid_pts)
        # Check that we get those fields back
        self.assertEqual(fc.get_field("test-field"), f_cpp)
        self.assertEqual(fc.get_field("test-field2"), f2_cpp)
        # Check strides
        self.assertEqual(f.stride(muGrid.Pixel), 4)
        self.assertEqual(f2.stride(muGrid.Pixel), 3 * 4)
        self.assertEqual(f.stride(muGrid.SubPt), 1)
        self.assertEqual(f2.stride(muGrid.SubPt), 3)

    def test_buffer_access(self):
        fc = muGrid.GlobalFieldCollection(self.nb_grid_pts, sub_pts={"quad": 4})
        # Single component
        f_cpp = fc.register_real_field("test-field", 1, "quad")
        f = wrap_field(f_cpp)

        # Access via DLPack - arrays from sg (full buffer)
        arr1 = f.sg  # SubPt layout with ghosts (full buffer)
        self.assertFalse(arr1.flags.owndata)
        self.assertTrue(arr1.flags.f_contiguous)

        arr2 = f.p  # Pixel layout
        self.assertFalse(arr2.flags.owndata)
        self.assertTrue(arr2.flags.f_contiguous)

        arr1[0, 0, 2, 1, 0] = 10  # f-contiguous
        self.assertEqual(arr2[0, 2, 1, 0], 10)  # f-contiguous

        f2_cpp = fc.register_real_field("test-field-2", 3, "quad")
        # Different fields should have different data buffers
        self.assertTrue(
            np.from_dlpack(f_cpp).ctypes.data != np.from_dlpack(f2_cpp).ctypes.data
        )

    def test_col_major(self):
        dims = (3, 4)
        fc = muGrid.GlobalFieldCollection(
            self.nb_grid_pts, self.nb_grid_pts, (0,) * len(self.nb_grid_pts), {}
        )
        f_cpp = fc.register_real_field("test-field", dims)
        a = np.from_dlpack(f_cpp)
        self.assertTrue(a.flags.f_contiguous)
        self.assertEqual(
            a.shape,
            tuple(
                list(dims)
                + [
                    1,
                ]
                + list(self.nb_grid_pts)
            ),
        )
        strides = np.append([1], np.cumprod(dims))
        strides = np.append(strides, strides[-1])
        strides = np.append(strides, strides[-1] * np.cumprod(self.nb_grid_pts))
        strides = 8 * strides[:-1]
        self.assertEqual(a.strides, tuple(strides))

    def test_row_major(self):
        # We need to specify the storage order twice and they mean different
        # things.
        dims = (3, 4)
        fc = muGrid.GlobalFieldCollection(
            self.nb_grid_pts,
            self.nb_grid_pts,
            [0] * len(self.nb_grid_pts),
            muGrid.StorageOrder.RowMajor,
            {},
            muGrid.StorageOrder.RowMajor,
        )
        f_cpp = fc.register_real_field("test-field", dims)
        a = np.from_dlpack(f_cpp)
        self.assertTrue(a.flags.c_contiguous)
        self.assertEqual(
            a.shape,
            tuple(
                list(dims)
                + [
                    1,
                ]
                + list(self.nb_grid_pts)
            ),
        )
        strides = np.append([1], np.cumprod(self.nb_grid_pts[::-1]))
        strides = np.append(strides, strides[-1])
        strides = np.append(strides, strides[-1] * np.cumprod(dims[::-1]))
        strides = 8 * strides[-2::-1]
        self.assertEqual(a.strides, tuple(strides))

    def test_local_field_collection(self):
        lfc_name = "ArbitraryLocalFieldCollectionName"
        lfc_field_name = "MyLocalField"
        fc = muGrid.LocalFieldCollection(3, lfc_name)
        fc.add_pixel(1)  # add pixel with global_index = 1
        fc.initialise()
        field_cpp = fc.register_real_field(lfc_field_name, 1, "pixel")
        field_array = np.from_dlpack(field_cpp)
        field_array[:] = 5  # assigne value 5 to pixel 1

        self.assertEqual(fc.get_name(), lfc_name)

    def test_accessors(self):
        fc = muGrid.GlobalFieldCollection(self.nb_grid_pts)
        field_cpp = fc.register_real_field("test-field", 1, "pixel")
        field = wrap_field(field_cpp)
        # Without ghosts, p and pg should have the same shape
        # Similarly for s and sg
        assert field.p.shape == field.pg.shape
        assert field.s.shape == field.sg.shape

    def test_dlpack_writability(self):
        """Test that arrays obtained via DLPack are writable."""
        fc = muGrid.GlobalFieldCollection(self.nb_grid_pts)
        field_cpp = fc.register_real_field("test-field", 1, "pixel")
        arr = np.from_dlpack(field_cpp)

        # Should be writable
        self.assertTrue(arr.flags.writeable)

        # Test actual write
        arr[0, 0, 0, 0, 0] = 42.0
        self.assertEqual(arr[0, 0, 0, 0, 0], 42.0)

    def test_field_wrapper_properties(self):
        """Test the Python Field wrapper properties."""
        fc = muGrid.GlobalFieldCollection(self.nb_grid_pts)
        field_cpp = fc.register_real_field("test-field", 3, "pixel")
        field = wrap_field(field_cpp)

        # Check that wrapper provides all expected properties
        self.assertEqual(field.name, "test-field")
        self.assertEqual(field.nb_components, 3)

        # Check array access
        self.assertEqual(field.sg.shape, (3, 1) + self.nb_grid_pts)
        self.assertEqual(field.s.shape, (3, 1) + self.nb_grid_pts)
        self.assertEqual(field.pg.shape, (3,) + self.nb_grid_pts)
        self.assertEqual(field.p.shape, (3,) + self.nb_grid_pts)


if __name__ == "__main__":
    unittest.main()

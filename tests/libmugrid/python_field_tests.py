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

from python_test_imports import muGrid


class FieldCheck(unittest.TestCase):
    def setUp(self):
        self.nb_grid_pts = (10, 11, 21)

    def test_buffer_size_one_quad_pt(self):
        fc = muGrid.GlobalFieldCollection(len(self.nb_grid_pts))
        fc.initialise(self.nb_grid_pts)
        # Single component
        f = fc.register_real_field("test-field", 1)
        self.assertEqual(f.array(muGrid.Pixel).shape,
                         (1,) + self.nb_grid_pts)
        # Four components
        f2 = fc.register_real_field("test-field2", 4)
        self.assertEqual(f2.array(muGrid.Pixel).shape,
                         (4,) + self.nb_grid_pts)
        # Check that we get those fields back
        self.assertEqual(fc.get_field('test-field'), f)
        self.assertEqual(fc.get_field('test-field2'), f2)
        # Check strides
        self.assertEqual(f.stride(muGrid.Pixel), 1)
        self.assertEqual(f2.stride(muGrid.Pixel), 4)
        self.assertEqual(f.stride(muGrid.SubPt), 1)
        self.assertEqual(f2.stride(muGrid.SubPt), 4)

    def test_buffer_size_four_quad_pt(self):
        fc = muGrid.GlobalFieldCollection(len(self.nb_grid_pts), {'quad': 4})
        fc.initialise(self.nb_grid_pts)
        # Single component
        f = fc.register_real_field("test-field", 1, 'quad')
        self.assertEqual(f.array(muGrid.Pixel).shape,
                         (4,) + self.nb_grid_pts)
        self.assertEqual(f.array(muGrid.SubPt).shape,
                         (1, 4) + self.nb_grid_pts)
        # Four components
        f2 = fc.register_real_field("test-field2", 3, 'quad')
        self.assertEqual(f2.array(muGrid.Pixel).shape,
                         (4 * 3,) + self.nb_grid_pts)
        self.assertEqual(f2.array(muGrid.SubPt).shape,
                         (3, 4) + self.nb_grid_pts)
        # Check that we get those fields back
        self.assertEqual(fc.get_field('test-field'), f)
        self.assertEqual(fc.get_field('test-field2'), f2)
        # Check strides
        self.assertEqual(f.stride(muGrid.Pixel), 4)
        self.assertEqual(f2.stride(muGrid.Pixel), 3 * 4)
        self.assertEqual(f.stride(muGrid.SubPt), 1)
        self.assertEqual(f2.stride(muGrid.SubPt), 3)

    def test_buffer_access(self):
        fc = muGrid.GlobalFieldCollection(len(self.nb_grid_pts), {'quad': 4})
        fc.initialise(self.nb_grid_pts)
        # Single component
        f = fc.register_real_field("test-field", 1, 'quad')
        arr1 = np.array(f, copy=False)
        self.assertFalse(arr1.flags.owndata)
        self.assertTrue(arr1.flags.f_contiguous)
        arr2 = f.array(muGrid.Pixel)
        self.assertFalse(arr2.flags.owndata)
        self.assertTrue(arr2.flags.f_contiguous)

        arr1[0, 0, 2, 1, 0] = 10  # f-contiguous
        self.assertEqual(arr2[0, 2, 1, 0], 10)  # f-contiguous

    def test_col_major(self):
        dims = (3, 4)
        fc = muGrid.GlobalFieldCollection(len(self.nb_grid_pts),
                                          self.nb_grid_pts,
                                          (0,) * len(self.nb_grid_pts),
                                          {})
        f = fc.register_real_field("test-field", dims)
        a = np.array(f)
        self.assertTrue(a.flags.f_contiguous)
        self.assertEqual(a.shape, tuple(list(dims) + list(self.nb_grid_pts)))
        strides = np.append([1], np.cumprod(dims))
        strides = np.append(strides,
                            strides[-1] * np.cumprod(self.nb_grid_pts))
        strides = 8 * strides[:-1]
        self.assertEqual(a.strides, tuple(strides))

    def test_row_major(self):
        # We need to specify the storage order twice and they mean different
        # things.
        dims = (3, 4)
        fc = muGrid.GlobalFieldCollection(
            len(self.nb_grid_pts), self.nb_grid_pts,
            [0] * len(self.nb_grid_pts),
            muGrid.StorageOrder.RowMajor, {}, muGrid.StorageOrder.RowMajor)
        f = fc.register_real_field("test-field", dims,
                                   storage_order=muGrid.StorageOrder.RowMajor)
        a = np.array(f)
        self.assertTrue(a.flags.c_contiguous)
        self.assertEqual(a.shape, tuple(list(dims) + list(self.nb_grid_pts)))
        strides = np.append([1], np.cumprod(self.nb_grid_pts[::-1]))
        strides = np.append(strides, strides[-1] * np.cumprod(dims[::-1]))
        strides = 8 * strides[-2::-1]
        self.assertEqual(a.strides, tuple(strides))

    def test_mixed_storage(self):
        # We need to specify the storage order twice and they mean different
        # things.
        dims = (3, 4)
        fc = muGrid.GlobalFieldCollection(
            len(self.nb_grid_pts), self.nb_grid_pts,
            [0] * len(self.nb_grid_pts),
            muGrid.StorageOrder.RowMajor, {}, muGrid.StorageOrder.ColMajor)
        f = fc.register_real_field("test-field", dims,
                                   storage_order=muGrid.StorageOrder.RowMajor)
        a = np.array(f)
        self.assertFalse(a.flags.c_contiguous)
        self.assertFalse(a.flags.f_contiguous)
        self.assertEqual(a.shape, tuple(list(dims) + list(self.nb_grid_pts)))
        strides = np.append([1], np.cumprod(dims[::-1]))[::-1]
        strides = np.append(strides, strides[0] *
                            np.cumprod(self.nb_grid_pts[::-1])[-2::-1])
        strides = 8 * np.roll(strides, -1)
        self.assertEqual(a.strides, tuple(strides))


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file     python_common_tests.py

@author  Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date    18 Oct 2019

@brief   test common muGrid functionality

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


class DynCcoordCheck(unittest.TestCase):
    def test_dyn_ccoord_3d(self):
        c = muGrid.DynCcoord([1, 20, 3])
        self.assertEqual(c[0], 1)
        self.assertEqual(c[1], 20)
        self.assertEqual(c[2], 3)
        with self.assertRaises(IndexError):
            c[3]


class OptionDictionaryCheck(unittest.TestCase):
    def test_simple_int(self):
        option_dict = muGrid.Dictionary("int", 1)
        self.assertEqual(option_dict["int"], 1)
        self.assertTrue(isinstance(option_dict["int"], int))

    def test_simple_real(self):
        option_dict = muGrid.Dictionary("real", 1.2)
        self.assertEqual(option_dict["real"], 1.2)
        self.assertTrue(isinstance(option_dict["real"], float))

    def test_simple_matrix(self):
        test_mat = np.random.random((2, 3))
        option_dict = muGrid.Dictionary("matrix", test_mat)
        self.assertEqual(np.linalg.norm(option_dict["matrix"] - test_mat), 0.)
        self.assertTrue(isinstance(option_dict["matrix"], np.ndarray))

    def test_simple_dict(self):
        d = dict(int=1, float=2.2, matrix=np.eye(2))
        option_dict = muGrid.Dictionary(d)
        self.assertEqual(option_dict["int"], 1)
        self.assertTrue(isinstance(option_dict["int"], int))
        self.assertEqual(option_dict["float"], 2.2)
        self.assertTrue(isinstance(option_dict["float"], float))
        self.assertEqual(np.linalg.norm(option_dict["matrix"] - np.eye(2)), 0.)
        self.assertTrue(isinstance(option_dict["matrix"], np.ndarray))

    def test_nested_dict(self):
        d = dict(int1=1, int2=24, myfloat=2.2, yourfloat=-24.8,
                 matrix=np.eye(2), dict=dict(int2=2, float2=4.4))
        option_dict = muGrid.Dictionary(d)
        self.assertEqual(option_dict["int1"], 1)
        self.assertTrue(isinstance(option_dict["int1"], int))
        self.assertEqual(option_dict["int2"], 24)
        self.assertTrue(isinstance(option_dict["int2"], int))
        self.assertEqual(option_dict["myfloat"], 2.2)
        self.assertTrue(isinstance(option_dict["myfloat"], float))
        self.assertEqual(option_dict["yourfloat"], -24.8)
        self.assertTrue(isinstance(option_dict["yourfloat"], float))
        self.assertEqual(np.linalg.norm(option_dict["matrix"] - np.eye(2)), 0.)
        self.assertTrue(isinstance(option_dict["matrix"], np.ndarray))
        with self.assertRaises(RuntimeError):
            isinstance(option_dict["dict"], muGrid.Dictionary)

    def test_simple_reject(self):
        with self.assertRaises(TypeError):
            option_dict = muGrid.Dictionary("string", "clown")

    def test_add(self):
        option_dict = muGrid.Dictionary()
        option_dict.add("int", 2)
        self.assertEqual(option_dict["int"], 2)
        self.assertTrue(isinstance(option_dict["int"], int))
        option_dict.add("real", 3.5)
        self.assertEqual(option_dict["real"], 3.5)
        self.assertTrue(isinstance(option_dict["real"], float))
        option_dict.add("matrix", np.eye(2))
        self.assertEqual(np.linalg.norm(option_dict["matrix"] - np.eye(2)), 0.)
        self.assertTrue(isinstance(option_dict["matrix"], np.ndarray))

        with self.assertRaises(RuntimeError):
            option_dict.add("int", 3)

    def test_assign(self):
        option_dict = muGrid.Dictionary("int", 1)
        self.assertEqual(option_dict["int"], 1)
        self.assertTrue(isinstance(option_dict["int"], int))

        option_dict["int"] = 2
        self.assertEqual(option_dict["int"], 2)
        self.assertTrue(isinstance(option_dict["int"], int))

        option_dict["int"] = 3.5
        self.assertEqual(option_dict["int"], 3.5)
        self.assertTrue(isinstance(option_dict["int"], float))

        option_dict["int"] = np.eye(2)
        self.assertEqual(np.linalg.norm(option_dict["int"] - np.eye(2)), 0.)
        self.assertTrue(isinstance(option_dict["int"], np.ndarray))


if __name__ == "__main__":
    unittest.main()

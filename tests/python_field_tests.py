#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_field_tests.py

@author Till Junge <till.junge@epfl.ch>

@date   06 Jul 2018

@brief  tests the python bindings for fieldcollections, fields, and statefields

Copyright © 2018 Till Junge

µSpectre is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µSpectre is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with µSpectre; see the file COPYING. If not, write to the
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

from python_test_imports import µ

class FieldCollection_Check(unittest.TestCase):
    """Because field collections do not have python-accessible
    constructors, this test creates a problem with a material with
    statefields

    """
    def setUp(self):
        self.nb_grid_pts = [3, 3]
        self.lengths = [1.58, 5.87]
        self.formulation = µ.Formulation.finite_strain
        self.cell = µ.Cell(self.nb_grid_pts,
                           self.lengths,
                           self.formulation)
        self.dim = len(self.lengths)
        self.mat = µ.material.MaterialLinearElastic2_2d.make(
            self.cell, "material", 210e9, .33)

    def test_fields(self):
        eigen_strain = np.array([[.01,  .02],
                                 [.03, -.01]])
        for i, pixel in enumerate(self.cell.pixels):
            self.mat.add_pixel(i, i/len(self.cell.pixels)*eigen_strain)

        self.cell.initialise()
        dir(µ.material.MaterialBase)
        self.assertTrue(isinstance(self.mat, µ.material.MaterialBase))
        collection = self.mat.collection
        field_name = collection.field_names[0]
        self.assertRaises(Exception, collection.get_complex_field, field_name)
        self.assertRaises(Exception, collection.get_int_field, field_name)
        self.assertRaises(Exception, collection.get_uint_field, field_name)
        eigen_strain_field = collection.get_real_field(field_name)

        for i, row in enumerate(eigen_strain_field.array().T):
            error = np.linalg.norm(i/len(self.cell.pixels)*eigen_strain -
                                   row.reshape(eigen_strain.shape).T)
            self.assertEqual(0, error)


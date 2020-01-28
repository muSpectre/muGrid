#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_cell_tests.py

@author Till Junge <till.junge@epfl.ch>

@date   22 Nov 2019

@brief  Unit tests for python bindings

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


class CellCheck(unittest.TestCase):
    def test_Construction(self):
        """
        Simple check for cell constructors
        """
        nb_grid_pts = [5, 7]
        lengths = [5.2, 8.3]
        formulation = µ.Formulation.small_strain
        try:
            sys = µ.Cell(nb_grid_pts,
                         lengths,
                         formulation)
            mat = µ.material.MaterialLinearElastic1_2d.make(sys, "material",
                                                            210e9, .33)

        except Exception as err:
            print(err)
            raise err

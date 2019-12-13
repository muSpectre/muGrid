#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_material_evaluator_test.py

@author Till Junge <till.junge@epfl.ch>

@date   17 Jan 2019

@brief  tests the python bindings of the material evaluator

Copyright © 2019 Till Junge

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
LinMat = µ.material.MaterialLinearElastic1_2d

def errfun(a, b):
    return np.linalg.norm(a-b)/np.linalg.norm(a+b)

class MaterialEvaluator_Check(unittest.TestCase):
    def test_linear_elasticity(self):
        young, poisson = 210e9, .33
        material, evaluator = LinMat.make_evaluator(young, poisson)
        material.add_pixel(0)
        stress = evaluator.evaluate_stress(np.array([[1., 0],[0, 1]]),
                                           µ.Formulation.finite_strain)
        stress=stress.copy()
        self.assertEqual(np.linalg.norm(stress), 0)

        stress2, tangent = evaluator.evaluate_stress_tangent(
            np.array([[1., .0],[0, 1.0]]),
            µ.Formulation.finite_strain)

        tangent = tangent.copy()

        self.assertEqual(np.linalg.norm(stress-stress2), 0)

        num_tangent = evaluator.estimate_tangent(
            np.array([[1., .0],[0, 1.0]]),
            µ.Formulation.finite_strain,
            1e-6, µ.FiniteDiff.centred)
        tol = 1e-8
        err = errfun(num_tangent, tangent)
        if not err < tol:
            print("tangent:\n{}".format(tangent))
            print("num_tangent:\n{}".format(num_tangent))
        self.assertLess(err, tol)
        num_tangent = evaluator.estimate_tangent(
            np.array([[1., .0],[0, 1.0]]),
            µ.Formulation.finite_strain,
            1e-6)

        numlin_tangent= evaluator.estimate_tangent(
            np.array([[1, .0],[0, 1.0]]),
            µ.Formulation.small_strain,
            1e-6, µ.FiniteDiff.centred)

        lin_stress, lin_tangent = evaluator.evaluate_stress_tangent(
            np.array([[0, .0],[0, .0]]),
            µ.Formulation.small_strain)


        err = errfun(numlin_tangent, lin_tangent)
        self.assertLess(err, tol)

if __name__ == '__main__':
    unittest.main()

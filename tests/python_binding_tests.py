#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_binding_tests.py

@author Till Junge <till.junge@epfl.ch>

@date   09 Jan 2018

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
from python_material_linear_elastic1_test import MaterialLinearElastic1_2dCheck
from python_material_linear_elastic3_test import MaterialLinearElastic3_Check
from python_material_linear_elastic4_test import MaterialLinearElastic4_Check
from python_material_linear_elastic_generic1_test import \
    MaterialLinearElasticGeneric1_Check
from python_material_linear_elastic_generic2_test import \
    MaterialLinearElasticGeneric2_Check
from python_comparison_test_material_linear_elastic1 import MatTest as MatTest1
from python_material_hyper_elasto_plastic2_test import \
    MaterialHyperElastoPlastic2_Check
from python_field_tests import FieldCollection_Check

from python_exact_reference_elastic_test import LinearElastic_Check

from python_field_tests import FieldCollection_Check
from python_gradient_integration_test import GradientIntegration_Check
from python_vtk_export_test import VtkExport_Check
from python_eshelby_slow_test import MuSpectre_Eshelby_Slow_Check

from python_material_evaluator_test import MaterialEvaluator_Check

# FIXME: @Richard L. - please fix parallel stochastic plasticity model
# from python_stochastic_plasticity_search_test import \
#    StochasticPlasticitySearch_Check


from python_projection_tests import *

from python_cell_tests import CellCheck
from python_eigen_strain_test import EigenStrainCheck
from python_eigen_strain_solver_test import EigenStrainSolverCheck
from python_solver_test import SolverCheck


if __name__ == '__main__':
    unittest.main()

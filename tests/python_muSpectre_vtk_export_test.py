#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_muSpectre_vtk_export_test.py

@author Richard Leute <richard.leute@imtek.uni-freiburg.de>

@date   10 Jan 2019

@brief  test the functionality of vtk_export.py

Copyright © 2019 Till Junge, Richard Leute

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
import tempfile
import os
from python_test_imports import µ

class MuSpectre_vtk_export_Check(unittest.TestCase):
    def setUp(self):
        self.lengths    = np.array([1.1, 2.2, 3])
        self.nb_grid_pts = np.array([3, 5, 7])
    def test_vtk_export(self):
        """
        Check the possibility to write scalar-, vector- and second rank tensor-
        fields on the cell and node grid. The throw of exceptions is not checked
        """
        grad = np.array([[0, 0.01, 0],
                         [0,    0, 0],
                         [0,    0, 0]])
        for dim in [2,3]:
            #test if files are written for 2D and 3D
            lens = self.lengths[:dim]
            res  = self.nb_grid_pts[:dim]
            F = grad[:dim, :dim].reshape((1,)*dim + (dim,)*2)
            x_n, x_c = µ.gradient_integration.compute_grid(lens, res)
            freqs = µ.gradient_integration.compute_wave_vectors(lens, res)
            placement_n = µ.gradient_integration.integrate_tensor_2(F, x_n,
                            freqs, staggered_grid=True, order=0)
            p_d = {'scalar'  : np.random.random(x_n.shape[:-1]),
                   'vector'  : np.random.random(x_n.shape[:-1] + (dim,)),
                   '2-tensor': np.random.random(x_n.shape[:-1] + (dim,)*2)}
            c_d = {'scalar'  : np.random.random(x_c.shape[:-1]),
                   'vector'  : np.random.random(x_c.shape[:-1] + (dim,)),
                   '2-tensor': np.random.random(x_c.shape[:-1] + (dim,)*2)}
            #The temporary directory is atomatically cleand up after one is
            #exiting the block.
            with tempfile.TemporaryDirectory(dir=os.getcwd()) as dir_name:
                os.chdir(dir_name)
                file_name = 'vtk_export_'+str(dim)+'D_test'
                µ.vtk_export.vtk_export(file_name, x_n, placement_n,
                                        point_data = p_d, cell_data = c_d)
                assert os.path.exists(file_name+'.vtr') == 1,\
                "vtk_export() was not able to write the {}D output file "\
                "'{}.vtr'.".format(dim, file_name)
                os.chdir('../')

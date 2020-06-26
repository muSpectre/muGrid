#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_netcdf_tests.py

@author Till Junge <till.junge@altermail.ch>

@date   17 Jan 2018

@brief  Compare µSpectre's fft implementations to numpy reference

Copyright © 2018 Till Junge

µFFT is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µFFT is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with µFFT; see the file COPYING. If not, write to the
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

try:
    from netCDF4 import Dataset

    from python_test_imports import muFFT
    from muFFT.NetCDF import NCStructuredGrid

    def build_test_classes(nb_grid_pts):
        class NetCDF_Check(unittest.TestCase):
            def setUp(self):
                self.nb_grid_pts = nb_grid_pts
                self.tensor_shape = tuple(list(self.nb_grid_pts) + [3, 3])
                self.scalar_grid = np.arange(np.prod(self.nb_grid_pts)).reshape(self.nb_grid_pts)
                self.tensor_grid = np.arange(np.prod(self.tensor_shape)) \
                    .reshape(self.tensor_shape)

                if muFFT.has_mpi:
                    from mpi4py import MPI
                    self.communicator = MPI.COMM_WORLD
                else:
                    self.communicator = None
                self.fft = muFFT.FFT(self.nb_grid_pts, fft='mpi',
                                     communicator=self.communicator)

            def test_write_read_domain(self):
                if self.communicator is None:
                    nc = NCStructuredGrid('test_{}d.nc'.format(len(self.nb_grid_pts)), mode='w',
                                          nb_domain_grid_pts=self.nb_grid_pts)
                else:
                    nc = NCStructuredGrid('test_{}d.nc'.format(len(self.nb_grid_pts)), mode='w',
                                          nb_domain_grid_pts=self.nb_grid_pts,
                                          decomposition='domain',
                                          subdomain_locations=self.fft.subdomain_locations,
                                          nb_subdomain_grid_pts=self.fft.nb_subdomain_grid_pts,
                                          communicator=self.communicator)
                nc.scalar = self.scalar_grid
                nc.tensor = self.tensor_grid
                nc[3].per_frame_tensor = self.tensor_grid
                nc.close()

                # Check that the file structure is correct
                nc = Dataset('test_{}d.nc'.format(len(self.nb_grid_pts)), 'r')
                dimensions = ['frame', 'grid_x', 'grid_y', 'tensor_3']
                if len(self.nb_grid_pts) == 3:
                    dimensions += ['grid_z']
                self.assertEqual(set(nc.dimensions), set(dimensions))
                self.assertEqual(len(nc.dimensions['frame']), 4)
                self.assertEqual(len(nc.dimensions['grid_x']), self.nb_grid_pts[0])
                self.assertEqual(len(nc.dimensions['grid_y']), self.nb_grid_pts[1])
                if len(self.nb_grid_pts) == 3:
                    self.assertEqual(len(nc.dimensions['grid_z']), self.nb_grid_pts[2])
                self.assertEqual(len(nc.dimensions['tensor_3']), 3)
                nc.close()

                # Read file and check data
                nc = NCStructuredGrid('test_{}d.nc'.format(len(self.nb_grid_pts)), mode='r')
                self.assertEqual(tuple(nc.nb_domain_grid_pts), tuple(self.nb_grid_pts))
                self.assertTrue(np.equal(nc.scalar, self.scalar_grid).all())
                self.assertTrue(np.equal(nc.tensor, self.tensor_grid).all())
                self.assertTrue(np.equal(nc[3].per_frame_tensor, self.tensor_grid).all())
                nc.close()

            def test_write_read_subdomain(self):
                scalar_grid = self.scalar_grid[self.fft.subdomain_slices]
                tensor_grid = self.tensor_grid[self.fft.subdomain_slices]

                if self.communicator is None:
                    nc = NCStructuredGrid('test_{}d.nc'.format(len(self.nb_grid_pts)), mode='w',
                                          nb_domain_grid_pts=self.nb_grid_pts)
                else:
                    nc = NCStructuredGrid('test_{}d.nc'.format(len(self.nb_grid_pts)), mode='w',
                                          nb_domain_grid_pts=self.nb_grid_pts,
                                          decomposition='subdomain',
                                          subdomain_locations=self.fft.subdomain_locations,
                                          nb_subdomain_grid_pts=self.fft.nb_subdomain_grid_pts,
                                          communicator=self.communicator)
                nc.scalar = scalar_grid
                nc.tensor = tensor_grid
                nc[3].per_frame_tensor = tensor_grid
                nc.close()

                # Check that the file structure is correct
                nc = Dataset('test_{}d.nc'.format(len(self.nb_grid_pts)), 'r')
                dimensions = ['frame', 'grid_x', 'grid_y', 'tensor_3']
                if len(self.nb_grid_pts) == 3:
                    dimensions += ['grid_z']
                self.assertEqual(set(nc.dimensions), set(dimensions))
                self.assertEqual(len(nc.dimensions['frame']), 4)
                self.assertEqual(len(nc.dimensions['grid_x']), self.nb_grid_pts[0])
                self.assertEqual(len(nc.dimensions['grid_y']), self.nb_grid_pts[1])
                if len(self.nb_grid_pts) == 3:
                    self.assertEqual(len(nc.dimensions['grid_z']), self.nb_grid_pts[2])
                self.assertEqual(len(nc.dimensions['tensor_3']), 3)
                nc.close()

                # Read file and check data
                nc = NCStructuredGrid('test_{}d.nc'.format(len(self.nb_grid_pts)), mode='r')
                self.assertEqual(tuple(nc.nb_domain_grid_pts), tuple(self.nb_grid_pts))
                self.assertTrue(np.equal(nc.scalar, self.scalar_grid).all())
                self.assertTrue(np.equal(nc.tensor, self.tensor_grid).all())
                self.assertTrue(np.equal(nc[3].per_frame_tensor, self.tensor_grid).all())
                nc.close()

        return NetCDF_Check

    # Check that it works with tuples or lists
    NetCDF_Check_2d = build_test_classes((11, 23))
    NetCDF_Check_3d = build_test_classes([11, 23, 7])

except Exception:
    pass

if __name__ == '__main__':
    unittest.main()

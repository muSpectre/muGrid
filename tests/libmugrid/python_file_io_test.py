#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file     python_file_io_test.py

@author  Richard Leute <richard.leute@imtek.uni-freiburg.de>

@date    11 Sep 2020

@brief   test the python bindings for the file I/O interface

Copyright © 2020 Till Junge

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

from python_test_imports import muGrid

from mpi4py import MPI
import numpy as np
import os


class FileIOTest(unittest.TestCase):
    def setUp(self):
        self.nb_domain_grid_pts = (3, 3, 3)
        self.nb_subdomain_grid_pts = (3, 3, 3)
        self.file_name = "python_binding_tests.nc"
        self.comm = muGrid.Communicator(MPI.COMM_WORLD)

        # global field collection
        self.fc_glob = \
            muGrid.GlobalFieldCollection(len(self.nb_domain_grid_pts))
        self.fc_glob.initialise(
            self.nb_domain_grid_pts, self.nb_subdomain_grid_pts)

        # local field collection
        self.local_pixels = [2, 15, 9, 7, 3]
        self.fc_loc = muGrid.LocalFieldCollection(len(self.nb_domain_grid_pts),
                                                  "MyLocalFieldCollection")
        for global_index in self.local_pixels:
            self.fc_loc.add_pixel(global_index)
        self.fc_loc.initialise()

    def test_FileIONetCDF(self):
        if os.path.exists(self.file_name):
            os.remove(self.file_name)

        file_io_object = muGrid.FileIONetCDF(self.file_name,
                                             muGrid.FileIONetCDF.OpenMode.Write,
                                             self.comm)

        glob_field_name = "global-test-field"
        loc_field_name = "local-test-field"
        f_glob = self.fc_glob.register_real_field(glob_field_name, 1, 'pixel')
        a_glob = np.array(f_glob, copy=False)
        a_glob[:] = np.random.random(self.nb_domain_grid_pts)
        file_io_object.register_field_collection(self.fc_glob)

        f_loc = self.fc_loc.register_real_field(loc_field_name, 1, 'pixel')
        a_loc = np.array(f_loc, copy=False)
        a_loc[:] = np.random.random(len(self.local_pixels))
        file_io_object.register_field_collection(self.fc_loc)

        # you can not write to a frame before you have appended it to the
        # file_io_object
        with self.assertRaises(RuntimeError):
            file_io_object.write(0, [glob_field_name])

        ### different versions to access a single frame of the file_io_object
        # append the first frame to the file io object
        file_frame = file_io_object.append_frame()
        # write both fields
        file_frame.write()
        # overwrite the global field
        file_io_object.write(0, [glob_field_name])
        # overwrite the local field
        file_io_object[0].write([loc_field_name])
        # write both fields to the second frame
        file_io_object.append_frame().write()

        # loop over frames and read both fields
        for frame in file_io_object:
            frame.read()

        # read field by name
        file_io_object.read(0, [glob_field_name])
        file_io_object.read(0, [loc_field_name])

        file_io_object.close()


if __name__ == "__main__":
    unittest.main()

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

try:
    from mpi4py import MPI
except ImportError:
    MPI = None
import numpy as np
import os

import random
import string
import time


class FileIOTest(unittest.TestCase):
    def setUp(self):
        self.nb_domain_grid_pts = (3, 3, 3)
        self.nb_subdomain_grid_pts = (3, 3, 3)
        self.file_f_name = "python_binding_io_field-tests.nc"
        self.file_sf_name = "python_binding_io_state-field-tests.nc"
        self.file_ga_name = "python_binding_io_global-attributes-tests.nc"
        if MPI:
            self.comm = muGrid.Communicator(MPI.COMM_WORLD)
        else:
            self.comm = muGrid.Communicator()

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

    def test_FileIONetCDF_Fields(self):
        if os.path.exists(self.file_f_name):
            os.remove(self.file_f_name)

        file_io_object = muGrid.FileIONetCDF(
            self.file_f_name, muGrid.FileIONetCDF.OpenMode.Write, self.comm)

        glob_field_name = "global-test-field"
        loc_field_name = "local-test-field"
        f_glob = self.fc_glob.register_real_field(glob_field_name, 1, 'pixel')
        a_glob = np.array(f_glob, copy=False)
        # set seed to reproduce random numbers for read test
        np.random.seed(12345)
        a_glob[:] = np.random.random(self.nb_domain_grid_pts)
        a_glob_frame0_ref = np.copy(a_glob)
        file_io_object.register_field_collection(self.fc_glob)

        f_loc = self.fc_loc.register_real_field(loc_field_name, 1, 'pixel')
        a_loc = np.array(f_loc, copy=False)
        # set seed to reproduce random numbers for read test
        np.random.seed(98765)
        a_loc[:] = np.random.random(len(self.local_pixels))
        a_loc_frame0_ref = np.copy(a_loc)
        file_io_object.register_field_collection(self.fc_loc)

        # you can not write to a frame before you have appended it to the
        # file_io_object
        with self.assertRaises(RuntimeError):
            file_io_object.write(0, [glob_field_name])

        # ## different versions to access a single frame of the file_io_object
        # append the first frame to the file io object
        file_frame = file_io_object.append_frame()
        # write both fields
        file_frame.write()
        # overwrite the global field
        file_io_object.write(0, [glob_field_name])
        # overwrite the local field
        file_io_object[0].write([loc_field_name])

        # write new values into the fields
        np.random.seed(6789)
        a_glob[:] = np.random.random(self.nb_domain_grid_pts)
        a_glob_frame1_ref = np.copy(a_glob)
        np.random.seed(4321)
        a_loc[:] = np.random.random(len(self.local_pixels))
        a_loc_frame1_ref = np.copy(a_loc)
        # write both fields to the second frame
        file_io_object.append_frame().write()

        # loop over frames and read both fields
        for frame in file_io_object:
            frame.read()

        # read fields by name in frame 0
        file_io_object.read(0, [glob_field_name])
        self.assertTrue((a_glob == a_glob_frame0_ref).all())
        file_io_object.read(0, [loc_field_name])
        self.assertTrue((a_loc == a_loc_frame0_ref).all())

        # read fields in frame 1
        file_io_object.read(1)
        self.assertTrue((a_glob == a_glob_frame1_ref).all())
        self.assertTrue((a_loc == a_loc_frame1_ref).all())

        file_io_object.close()

        # ## append frame 2 to the already written netcdf file
        file_io_object = muGrid.FileIONetCDF(
            self.file_f_name, muGrid.FileIONetCDF.OpenMode.Append, self.comm)
        file_io_object.register_field_collection(self.fc_glob)
        file_io_object.register_field_collection(self.fc_loc)
        a_glob[:] = 1234
        a_loc[:] = 5678
        file_io_object.append_frame().write()

        # read fields in frame 2
        file_io_object.read(2)
        self.assertTrue((a_glob == 1234).all())
        self.assertTrue((a_loc == 5678).all())

    def test_FileIONetCDF_StateFields(self):
        if os.path.exists(self.file_sf_name):
            os.remove(self.file_sf_name)

        file_io_object = muGrid.FileIONetCDF(
            self.file_sf_name, muGrid.FileIONetCDF.OpenMode.Write, self.comm)

        glob_prefix = "global-state-field"
        loc_prefix = "local-state-field"
        sf_glob = \
            self.fc_glob.register_real_state_field(glob_prefix, 3, 1, 'pixel')
        # iterate over state field where only the current field has write access
        for i in range(sf_glob.get_nb_memory() + 1):
            f_glob = sf_glob.current()
            a_glob = np.array(f_glob, copy=False)
            a_glob[:] = i + 1
            sf_glob.cycle()
        # bring field in the correct order
        for i in range(sf_glob.get_nb_memory()):
            sf_glob.cycle()
        file_io_object.register_field_collection(self.fc_glob)


        sf_loc = self.fc_loc.register_int_state_field(loc_prefix, 3, 1, 'pixel')
        # iterate over state field where only the current field has write access
        for i in range(sf_loc.get_nb_memory() + 1):
            f_loc = sf_loc.current()
            a_loc = np.array(f_loc, copy=False)
            a_loc[:] = 2*i + 1
            sf_loc.cycle()
        # bring field in the correct order
        for i in range(sf_loc.get_nb_memory()):
            sf_loc.cycle()
        file_io_object.register_field_collection(self.fc_loc)

        # write only global state field in frame 0
        file_frame = file_io_object.append_frame()
        file_frame.write([glob_prefix])

        # write only local state field in frame 1
        file_io_object.append_frame().write([loc_prefix])

        # read fields by name in frame 0
        file_io_object.read(0, [glob_prefix, loc_prefix])
        for i in range(sf_glob.get_nb_memory() + 1):
            f_glob = sf_glob.old(i)
            a_glob = np.array(f_glob, copy=False)
            ref = sf_glob.get_nb_memory() + 1 - i
            self.assertTrue((a_glob == ref).all(),
                            str(a_glob) + " != " + str(ref))
        for i in range(sf_loc.get_nb_memory() + 1):
            f_loc = sf_loc.old(i)
            a_loc = np.array(f_loc, copy=False)
            ref = 0
            self.assertTrue((a_loc == ref).all(),
                            str(a_loc) + " != " + str(ref))

        # read fields in frame 1
        file_io_object.read(1)
        for i in range(sf_glob.get_nb_memory() + 1):
            f_glob = sf_glob.old(i)
            a_glob = np.array(f_glob, copy=False)
            ref = 0
            self.assertTrue((a_glob == ref).all(),
                            str(a_glob) + " != " + str(ref))
        for i in range(sf_loc.get_nb_memory() + 1):
            f_loc = sf_loc.old(i)
            a_loc = np.array(f_loc, copy=False)
            ref = 2 * (sf_loc.get_nb_memory() - i) + 1
            self.assertTrue((a_loc == ref).all(),
                            str(a_loc) + " != " + str(ref))

        file_io_object.close()

        # ## append frame 2 to the already written netcdf file
        file_io_object = muGrid.FileIONetCDF(
            self.file_sf_name, muGrid.FileIONetCDF.OpenMode.Append, self.comm)
        file_io_object.register_field_collection(self.fc_glob)
        file_io_object.register_field_collection(self.fc_loc)
        a_glob[:] = 1234
        a_loc[:] = 5678
        file_io_object.append_frame().write()

        # read fields in frame 2
        file_io_object.read(2)
        self.assertTrue((a_glob == 1234).all())
        self.assertTrue((a_loc == 5678).all())

    def test_FileIONetCDF_global_attributes(self):
        # --- write global attributes --- #
        if os.path.exists(self.file_ga_name):
            os.remove(self.file_ga_name)

        file_io_object_w = muGrid.FileIONetCDF(
            self.file_ga_name, muGrid.FileIONetCDF.OpenMode.Write, self.comm)

        global_att_1_name = "global_att_str"
        global_att_1_value = "123abc"
        global_att_2_name = "global_att_Int"
        global_att_2_value = np.array([1234, 5678],
                                      dtype=int).tolist()
        global_att_3_name = "global_att_Uint"
        global_att_3_value = np.array([1234, 5678],
                                      dtype=np.uintc).tolist()
        global_att_4_name = "global_att_Index_t"
        global_att_4_value = np.array([1234, 5678],
                                      dtype=np.int64).tolist()
        global_att_5_name = "global_att_Real"
        global_att_5_value = np.array([1234, 5678],
                                      dtype=np.double).tolist()
        file_io_object_w.write_global_attribute(global_att_1_name,
                                                global_att_1_value)
        file_io_object_w.write_global_attribute(global_att_2_name,
                                                global_att_2_value)
        file_io_object_w.write_global_attribute(global_att_3_name,
                                                global_att_3_value)
        file_io_object_w.write_global_attribute(global_att_4_name,
                                                global_att_4_value)
        file_io_object_w.write_global_attribute(global_att_5_name,
                                                global_att_5_value)
        file_io_object_w.register_field_collection(self.fc_glob)
        file_io_object_w.register_field_collection(self.fc_loc)
        file_io_object_w.close()

        # by default written global attributes
        global_att_names_default = ['creation_date', 'creation_time',
                                    'last_modified_date', 'last_modified_time',
                                    'muGrid_version_info',
                                    'muGrid_git_hash',
                                    'muGrid_description',
                                    'muGrid_git_branch_is_dirty']
        global_att_values_default = [None]*len(global_att_names_default)

        # actually added global attributes
        global_att_names_ref = global_att_names_default.copy()
        global_att_names_ref.extend([global_att_1_name, global_att_2_name,
                                     global_att_3_name, global_att_4_name,
                                     global_att_5_name])
        global_att_values_ref = global_att_values_default.copy()
        global_att_values_ref.extend([global_att_1_value, global_att_2_value,
                                      global_att_3_value, global_att_4_value,
                                      global_att_5_value])

        # --- read global attribute --- #
        file_io_object_r = muGrid.FileIONetCDF(
            self.file_ga_name, muGrid.FileIONetCDF.OpenMode.Read, self.comm)

        with self.assertRaises(RuntimeError) as cm:
            file_io_object_r.write_global_attribute(global_att_1_name,
                                                    global_att_1_value)
            error_message = "It is only possible to write global attributes "\
                + "when the FileIONetCDF object was open with "\
                + "'FileIOBase::OpenMode::Write'."
            self.assertEqual(cm.exception.error_code, error_message)

        global_att_names = file_io_object_r.read_global_attribute_names()
        self.assertTrue(global_att_names == global_att_names_ref)

        for i, g_att_name in enumerate(global_att_names):
            g_att_value = file_io_object_r.read_global_attribute(g_att_name)
            if g_att_name not in global_att_names_default:
                if type(g_att_value) == str or isinstance(g_att_value, list):
                    self.assertTrue(g_att_value == global_att_values_ref[i])
                elif isinstance(g_att_value, np.ndarray):
                    self.assertTrue(
                        (g_att_value == global_att_values_ref[i]).all())
                else:
                    raise RuntimeError("Unknow type '" + str(type(g_att_value)) +
                                       "'of attribute value in 'FileIOTest."
                                       "test_FileIONetCDF_global_attributes()'.")
            else:
                # add date and time to reference
                global_att_values_ref[i] = g_att_value

        # construct reference dictionary
        global_att_ref_dic = dict(
            zip(global_att_names_ref, global_att_values_ref))
        global_att_dic = file_io_object_r.read_global_attributes()
        for k in global_att_dic.keys():
            self.assertTrue(global_att_dic[k] == global_att_ref_dic[k])

        # --- change global attribute in append mode --- #
        file_io_object_a = muGrid.FileIONetCDF(
            self.file_ga_name, muGrid.FileIONetCDF.OpenMode.Append, self.comm)

        old_name = global_att_1_name  # this is a str attribute
        old_value = file_io_object_a.read_global_attribute(old_name)
        too_long_value = old_value + "."  # exceeds old att value by one symbol
        too_long_name = old_name + "."  # exceeds old att name by one symbol

        # check for possible errors
        with self.assertRaises(RuntimeError) as cm:
            file_io_object_a.update_global_attribute(old_name,
                                                     old_name,
                                                     too_long_value)
        error_code = ("The new global attribute value data exceeds the"
                      + " old value data size which is not allowed!")
        self.assertTrue(str(cm.exception)[:len(error_code)] == error_code)

        with self.assertRaises(RuntimeError) as cm:
            file_io_object_a.update_global_attribute(old_name,
                                                     too_long_name,
                                                     old_value)
        error_code = ("The new global attribute name exceeds the old name"
                      + " size which is not allowed!")
        self.assertTrue(str(cm.exception)[:len(error_code)] == error_code)

        def random_name_generator(size=6, chars=string.ascii_uppercase
                                  + string.ascii_lowercase + string.digits):
            return ''.join(random.choice(chars) for _ in range(size))

        valid_new_value = random_name_generator(len(old_value))
        valid_new_name = random_name_generator(len(old_name))

        file_io_object_a.update_global_attribute(old_name,
                                                 valid_new_name,
                                                 valid_new_value)

        time.sleep(1)  # delay to get a new time in last modified time
        file_io_object_a.close()  # now the last modified time should be updated

        # check if the global att and the last_modified_time were updated
        file_io_object_r2 = muGrid.FileIONetCDF(
            self.file_ga_name, muGrid.FileIONetCDF.OpenMode.Read, self.comm)

        with self.assertRaises(RuntimeError) as cm:
            file_io_object_r2.read_global_attribute(old_name)
        error_code = ("The global attribute with name 'global_att_str' was"
                      " not found. Maybe you forgot to register the"
                      " corresponding NetCDFGlobalAtt?")
        self.assertTrue(str(cm.exception)[:len(error_code)] == error_code)

        updated_value = file_io_object_r2.read_global_attribute(valid_new_name)
        self.assertEqual(updated_value, valid_new_value)

        old_lmt = global_att_ref_dic["last_modified_time"]
        new_lmt = file_io_object_r2.read_global_attribute("last_modified_time")
        self.assertTrue(old_lmt != new_lmt)


if __name__ == "__main__":
    unittest.main()

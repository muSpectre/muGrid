#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_vtk_export_test.py

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
import muFFT

import xml.etree.ElementTree as ET
import logging

# helper class to compare vtr files
#   partly copied from/inspired by:
#   https://stackoverflow.com/questions/24492895/comparing-two-xml-files-in-python


class XmlTree():
    def __init__(self):

        self.logger = logging.getLogger('xml_compare')
        self.logger.setLevel(logging.DEBUG)
        self.hdlr = logging.FileHandler('xml-comparison.log')
        self.formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s- %(message)s')
        self.hdlr.setLevel(logging.DEBUG)
        self.hdlr.setFormatter(self.formatter)
        self.logger.addHandler(self.hdlr)

        # save the offsets of AppendedData, here x_, y_, z_coordinates
        self.ref_offsets = {}  # store x-off, y-off, z-off in a dictionary
        self.comp_offsets = {}  # store x-off, y-off, z-off in a dictionary

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.hdlr.close()
        return

    @staticmethod
    def binary_file_to_tree(xmlBinaryFileName):
        return ET.parse(xmlBinaryFileName).getroot()

    def xml_compare(self, x_ref, x_comp, excludes=[]):
        """
        Compares two xml etrees
        :param x_ref: the first tree (reference tree)
        :param x_comp: the second tree (tree to compare with reference)
        :param excludes: list of string of attributes to exclude from
                         comparison
        :return:
            True if both files match
        """
        r1 = self.tag_compare(x_ref, x_comp)
        r2 = self.attribute_name_compare(x_ref, x_comp, excludes)
        r3 = self.attribute_value_compare(x_ref, x_comp, excludes)

        if not (r1 and r2 and r3):
            self.tag_compare(x_ref, x_comp, write_log=True)
            self.attribute_name_compare(
                x_ref, x_comp, excludes, write_log=True)
            self.attribute_value_compare(
                x_ref, x_comp, excludes, write_log=True)
            print('Probably there is a mistake in the upper most layer of your'
                  ' xml tree!')
            return False
        if not self.text_compare(x_ref.text, x_comp.text):
            # exception for the appended Data because it can be writen in
            # arbitrary order
            if not x_ref.tag == "AppendedData":
                self.logger.debug('\nreference text:  %r \n!=\ncomparison '
                                  'text: %r' % (x_ref.text, x_comp.text))
                return False

            # special treatement
            status, t_ref_ordered, t_comp_ordered = \
                self.reordered_text_compare(x_ref,  x_comp)
            if not status:
                self.logger.debug('\nreordered reference text:  %r \n!='
                                  '\n reorderd comparison text: %r'
                                  % (t_ref_ordered, t_comp_ordered))
                return False

        if not self.text_compare(x_ref.tail, x_comp.tail):
            self.logger.debug('tail: %r != %r' % (x_ref.tail, x_comp.tail))
            return False
        cl_ref = list(x_ref)
        cl_comp = list(x_comp)
        if len(cl_ref) != len(cl_comp):
            self.logger.debug('children length differs, %i != %i'
                              % (len(cl_ref), len(cl_comp)))
            return False
        i = 0
        for c1 in cl_ref:
            i += 1
            if c1.tag not in excludes:
                # compare the right children with each other,
                # they should have the "names" and belonging "values"
                j = 0
                for c2 in cl_comp:
                    j += 1
                    if self.tag_compare(c1, c2) and \
                       self.attribute_name_compare(c1, c2, excludes) and \
                       self.attribute_value_compare(c1, c2, excludes):
                        if not self.xml_compare(c1, c2, excludes):
                            self.logger.debug('children %i do not match: %s'
                                              % (i, c1.tag))
                            return False
                        j -= 1
                        break  # end "for c2 in cl_comp:" at the first match
                if j == len(cl_comp):
                    self.tag_compare(c1, c2, write_log=True)
                    self.attribute_name_compare(c1, c2, excludes,
                                                write_log=True)
                    self.attribute_value_compare(c1, c2, excludes,
                                                 write_log=True)
                    self.logger.debug('children %i, can not find a matching %s'
                                      ' with same names and belonging values '
                                      'in the file for comparison.'
                                      % (i, c1.tag))
                    return False

        return True

    def tag_compare(self, x_ref, x_comp, write_log=False):
        """
        Compare the tags of two xml etrees
        :param x_ref:  xml etree one (reference etree)
        :param t_comp: xml etree two (etree for comparison)
        :param write_log: bool, if True log file is written (default = False)
        :return:
            True if all names match
        """
        if x_ref.tag != x_comp.tag:
            if write_log:
                self.logger.debug('Tags do not match: %s and %s'
                                  % (x_ref.tag, x_comp.tag))
            else:
                return False
        if not write_log:
            return True

    def attribute_name_compare(self, x_ref, x_comp, excludes, write_log=False):
        """
        Compare the attribute names of two xml etrees
        :param x_ref:  xml etree one (reference etree)
        :param t_comp: xml etree two (etree for comparison)
        :param excludes: list of string of attributes to exclude from
                         comparison
        :param write_log: bool, if True log file is written (default = False)
        :return:
            True if all names match
        """
        for name in x_comp.attrib.keys():
            if name not in excludes:
                if name not in x_ref.attrib:
                    if write_log:
                        self.logger.debug('x_comp has an attribute x_ref is '
                                          'missing: %s' % name)
                    else:
                        return False
        if not write_log:
            return True

    def attribute_value_compare(
            self, x_ref, x_comp, excludes, write_log=False):
        """
        Compare the attribute values of two xml etrees
        :param x_ref:  xml etree one (reference etree)
        :param x_comp: xml etree two (etree for comparison)
        :param excludes: list of string of attributes to exclude from
                         comparison
        :param write_log: bool, if True log file is written (default = False)
        :return:
            True if all values match
        """
        for name, value in x_ref.attrib.items():
            if name == "offset":
                # get store the offset to each coordinate direction in a
                # dicitonary, e.g. {'x': '0', 'y': '52', 'z': '124'}
                self.ref_offsets[(x_ref.attrib['Name'])[0:1]] = \
                    int(x_ref.attrib.get("offset"))
                self.comp_offsets[(x_comp.attrib['Name'])[0:1]] = \
                    int(x_comp.attrib.get("offset"))

            if name not in excludes:
                if x_comp.attrib.get(name) != value:
                    if write_log:
                        self.logger.debug(
                            'Attributes do not match: %s=%r, %s=%r'
                            % (name, value, name, x_comp.attrib.get(name)))
                    else:
                        return False
        if not write_log:
            return True

    def text_compare(self, t_ref, t_comp):
        """
        Compare two text strings
        :param t_ref:  text one (reference text)
        :param t_comp: text two (text for comparison)
        :return:
            True if a match
        """
        if not t_ref and not t_comp:
            return True
        if t_ref == '*' or t_comp == '*':
            return True
        return (t_ref or '').strip() == (t_comp or '').strip()

    def reordered_text_compare(self, x_ref,  x_comp):
        """
        Compare two text strings after reordering them coresponding to the
        offsets
        :param x_ref:  xml etree one (reference etree)
        :param x_comp: xml etree two (etree for comparison)
        :return:
            status         -- True if the reordered texts match otherwise False
            t_ref_ordered  -- the reordered text of x_ref
            t_comp_ordered -- the reordered text of x_comp
        """
        # reorder text
        t_ref_ordered = self.reorder(x_ref.text, self.ref_offsets)
        t_comp_ordered = self.reorder(x_comp.text, self.comp_offsets)

        # compare
        status = self.text_compare(t_ref_ordered, t_comp_ordered)

        return status, t_ref_ordered, t_comp_ordered

    def reorder(self, unordered_text, order_dic):
        """
        reorder the unordered_text in such a way that first the x values come
        followed by the y values and last by the z values.
        :param unordered_text: unordered input text which has the offsets given
                               in the 'order_dic'
        :param order_dic:      dictionary which gives the offsets for each
                               coordinate direction
        """
        t = unordered_text[1:]  # cutoff leading '_'
        t_len = len(t)
        direc = \
            [d for _, d in sorted(zip(order_dic.values(), order_dic.keys()))]
        starts = list(order_dic.values())
        starts.sort()
        starts.append(t_len)
        text_lengths = [int(e)-int(s) for s, e in zip(starts[:-1], starts[1:])]

        ordered_direc = direc.copy()
        ordered_direc.sort()
        t_ordered = "_"  # text always starts with '_'
        for d in ordered_direc:
            pos = direc.index(d)
            t_ordered += t[starts[pos]:starts[pos]+text_lengths[pos]]

        return t_ordered


class VtkExport_Check(unittest.TestCase):
    def setUp(self):
        self.lengths = np.array([1.1, 2.2, 3])
        self.nb_grid_pts = np.array([3, 5, 7])
        self.grid_spacing = self.lengths / self.nb_grid_pts

        self.temporary = False  # decides whether the compared files are written
        # in a temporary folder and deleted after
        # comaprison (True), or if they are written into
        # the afterwards existing folder
        # 'vtr-test-folder' (False)

    def test_vtk_export(self):
        """
        Check the possibility to write scalar-, vector- and second rank tensor-
        fields on the cell and node grid. The writen file is compared to the
        reference files "vtk_export_2D_test.ref.vtr" and
        "vtk_export_3D_test.ref.vtr". A throw of exceptions is not checked.
        """
        # print info about temporary file/folder settings
        if self.temporary:
            print("\nWriting *.vtr test files into a temporary not accessible "
                  "folder. Set self.temporary=False if you want to inspect the"
                  " files.\n")
        else:
            print("\nWriting log and *.vtr test files into '/vtr-test-folder/'"
                  ". To write them only in a temporary folder set "
                  "self.temporary=True.\n")
        # fix random seed to make comparison to reference possible
        np.random.seed(14102019)  # 14.10.2019
        f = np.eye(3)
        f[0, 1] = 0.2

        for dim in [2, 3]:
            # test if correct files are written for 2D and 3D
            lens = self.lengths[:dim]
            res = self.nb_grid_pts[:dim]
            f.shape = (3, 3) + (1, )*dim

            F = np.zeros((dim, dim) + tuple(res))
            F[:, :, ...] = f[:dim, :dim, ...]

            x_n, x_c = µ.gradient_integration.make_grid(lens[:dim], res[:dim])
            gradient_op = [µ.FourierDerivative(dim, i) for i in range(dim)]
            fft_engine = muFFT.FFT(list(self.nb_grid_pts[:dim]))
            fft_engine.create_plan(dim)
            fft_engine.create_plan(dim * dim)
            placement_n = µ.gradient_integration.integrate_tensor_2(
                F, fft_engine, gradient_op,
                list(self.grid_spacing[:dim]))

            p_d = {'scalar': np.random.random(x_n.shape[1:]),
                   'vector': np.random.random((dim,) + x_n.shape[1:]),
                   '2-tensor': np.random.random((dim,)*2 + x_n.shape[1:])}
            c_d = {'scalar': np.random.random(x_c.shape[1:]),
                   'vector': np.random.random((dim,) + x_c.shape[1:]),
                   '2-tensor': np.random.random((dim,)*2 + x_c.shape[1:])}

            def compare_files():
                """
                This function actually compares the two *.vtr files
                """
                file_name = 'vtk_export_'+str(dim)+'D_test'
                uvw_obj = µ.vtk_export.vtk_export(file_name, x_n, placement_n,
                                                  point_data=p_d, cell_data=c_d)
                assert os.path.exists(file_name + '.vtr') == 1,\
                    "vtk_export() was not able to write the {}D output file "\
                    "'{}.vtr'.".format(dim, file_name)
                cmp_data = file_name + '.vtr'
                ref_data = '../reference_computations/' \
                           + file_name + '.ref.vtr'
                ref_tree = XmlTree.binary_file_to_tree(ref_data)
                cmp_tree = XmlTree.binary_file_to_tree(cmp_data)
                with XmlTree() as comparator:
                    are_same_files = comparator.xml_compare(ref_tree, cmp_tree,
                                                            excludes=['offset'])
                    if not are_same_files:
                        with open('xml-comparison.log', 'r') as log:
                            print("Written file '{}' does not coincide with "
                                  "reference file '{}'!".format(cmp_data,
                                                                ref_data))
                            print("They differ in:")
                            print(log.read())
                    self.assertTrue(are_same_files)
                os.chdir('../')

            if self.temporary:
                # This temporary directory is atomatically cleand up after one
                # is exiting the block.
                with tempfile.TemporaryDirectory(dir=os.getcwd()) as dir_name:
                    os.chdir(dir_name)

                    compare_files()

            elif not self.temporary:
                # you can look up the computed *.vtr files and a log file in
                # the folder '/vtr-test-folder/'
                try:
                    os.makedirs(os.getcwd() + '/vtr-test-folder/')
                except FileExistsError:
                    pass
                os.chdir(os.getcwd() + '/vtr-test-folder/')

                compare_files()

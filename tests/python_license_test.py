#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file python_license_test.py

@author Ali Falsafi<ali.falsafi @epfl.ch>

@date 17 Sep 2019

@brief description

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
import os
import re
import argparse


def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


class arg_parser:

    def parse_args(args):
        muSpectre_sources = []
        parser = argparse.ArgumentParser(description='address')
        parser.add_argument('muSpectre_dirs', type=str, nargs='+',
                            help='Path to directories containing source files')
        sources = parser.parse_args()
        muSpectre_sources = re.split(" +", sources.muSpectre_dirs[0])
        return muSpectre_sources


test_case = unittest.TestCase('__init__')


def header_license_test(source_dirs, lic_paras):
    for source_dir in source_dirs:
        header_files = []
        for r, d, f in walklevel(source_dir, 0):
            for file in f:
                if ('.hh' in file and '.hh~' not in file and '#'
                   not in file and 'iterators.hh' not in file):
                    header_files.append(os.path.join(r, file))

        for header_file in header_files:
            file_obj = open(header_file, "r")
            file_text = file_obj.read()
            license_text = re.split("#ifndef", file_text, 2)[0]
            lines_text = re.split("\*\n", license_text)
            file_name_line = lines_text[1]
            file_name_line = re.sub(' +', '', file_name_line)
            file_name_line = re.sub('\n', '', file_name_line)
            file_name = re.sub('\*\@file', '', file_name_line)
            header_file_name = re.split('\/', header_file)[-1]
            header_file_name = re.sub(' +', '', header_file_name)
            test_case.assertEqual(file_name,
                                  header_file_name,
                                  msg="The name supplied in the first line"
                                  "of the license should match the file name")
            mu_license_text = re.split('Copyright', license_text, 2)[1]
            mu_license_text_lines = re.split('\*\n', mu_license_text)
            mu_license_text_lines_new = []
            for line in mu_license_text_lines:
                new_line = re.sub('\n', '', line)
                new_line = re.sub('\*', '', new_line)
                new_line = re.sub(' +', ' ', new_line)
                mu_license_text_lines_new.append(new_line)
            del mu_license_text_lines_new[0]
            msg = ""
            for num, lic_para in enumerate(lic_paras):
                if mu_license_text_lines_new[num] != lic_para:
                    msg = test_case._formatMessage(msg,
                                                   "the paragraph number {}"
                                                   " of the license in file "
                                                   "\"{}\" is not correct."
                                                   "It should be replaced by"
                                                   "\n\'{}\'"
                                                   .format(num+1,
                                                           header_file,
                                                           lic_para))
                    raise test_case.failureException(msg)
            print("\"{}\" License text approved".format(header_file))
            file_obj.close()


def source_license_test(source_dirs, lic_paras):
    for source_dir in source_dirs:
        source_files = []
        for r, d, f in walklevel(source_dir, 0):
            for file in f:
                if '.cc' in file and '.cc~' not in file and '#' not in file:
                    source_files.append(os.path.join(r, file))

        for source_file in source_files:
            file_obj = open(source_file, "r")
            file_text = file_obj.read()
            license_text = re.split("#include", file_text, 2)[0]
            lines_text = re.split("\*\n", license_text)
            file_name_line = lines_text[1]
            file_name_line = re.sub(' +', '', file_name_line)
            file_name_line = re.sub('\n', '', file_name_line)
            file_name = re.sub('\*\@file', '', file_name_line)
            source_file_name = re.split('\/', source_file)[-1]
            source_file_name = re.sub(' +', '', source_file_name)
            test_case.assertEqual(file_name,
                                  source_file_name,
                                  msg="The name supplied in the first line"
                                  "of the license should match the file name")
            mu_license_text = re.split('Copyright', license_text, 2)[1]
            mu_license_text_lines = re.split('\*\n', mu_license_text)
            mu_license_text_lines_new = []
            for line in mu_license_text_lines:
                new_line = re.sub('\n', '', line)
                new_line = re.sub('\*', '', new_line)
                new_line = re.sub(' +', ' ', new_line)
                mu_license_text_lines_new.append(new_line)
            del mu_license_text_lines_new[0]
            msg = ""
            for num, lic_para in enumerate(lic_paras):
                if mu_license_text_lines_new[num] != lic_para:
                    msg = test_case._formatMessage(msg,
                                                   "the paragraph number {} of"
                                                   "the license in file \"{}\""
                                                   "is not correct. It should "
                                                   "be replaced by \n\'{}\'"
                                                   .format(num+1,
                                                           source_file,
                                                           lic_para))
                    raise test_case.failureException(msg)
            print("\"{}\" License text approved".format(source_file))
            file_obj.close()


def python_license_test(source_dirs, py_lic_paras):
    for source_dir in source_dirs:
        python_files = []
        for r, d, f in walklevel(source_dir, 0):
            for file in f:
                if ('.py' in file and '.py~' not in file and '#'
                   not in file and 'pyc' not in file):
                    python_files.append(os.path.join(r, file))

        for python_file in python_files:
            file_obj = open(python_file, "r")
            file_text = file_obj.read()
            hash_bang_text = re.split("\"\"\"", file_text, 3)[0]
            test_case.assertEqual(hash_bang_text,
                                  "#!/usr/bin/env python3\n"
                                  "# -*- coding:utf-8 -*-\n",
                                  msg="the hashbang of file {} is incorrect"
                                  .format(python_file))
            license_text = re.split("\"\"\"", file_text, 3)[1]
            lines_text = re.split("\n\n", license_text)
            file_name_line = lines_text[0]
            file_name_line = re.sub(' +', '', file_name_line)
            file_name_line = re.sub('\n', '', file_name_line)
            file_name = re.sub('\@file', '', file_name_line)
            python_file_name = re.split('\/', python_file)[-1]
            python_file_name = re.sub(' +', '', python_file_name)
            test_case.assertEqual(
                file_name,
                python_file_name,
                msg=("The name supplied in the first line of"
                     "the license should match the file name for :{}").format(
                         python_file))
            del lines_text[0:5]
            msg = ""
            for num, lic_para in enumerate(py_lic_paras):
                re.sub("\ \n", "\n", lines_text[num])
                if lines_text[num] != lic_para:
                    msg = test_case._formatMessage(msg,
                                                   "the paragraph number "
                                                   "{} of the license in file"
                                                   "\"{}\" is not correct. It"
                                                   " should be replaced by \n"
                                                   "\"'{}\' "
                                                   .format(num+1,
                                                           python_file,
                                                           lic_para,))
                    raise test_case.failureException(msg)
            print("\"{}\" License text approved".format(python_file))
            file_obj.close()

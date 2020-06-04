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
    msg_list = ""
    for source_dir in source_dirs:
        header_files = []
        for r, d, f in walklevel(source_dir, 0):
            print("source_dir = {}, r,d,f = {}".format(source_dir, (r, d, f)))
            for file in f:
                if (file.endswith('.hh')
                    and 'iterators.hh' not in file
                        and '#' not in file):
                    # The files containing # are temp files created by emacs.
                    # Therefore, they need to be excluded.
                    header_files.append(os.path.join(r, file))

        for header_file in header_files:
            file_obj = open(header_file, "r")
            file_text = file_obj.read()
            license_text = re.split("#ifndef", file_text, 2)[0]
            lines_text = re.split("\*\n", license_text)
            msg = ""
            if (len(lines_text) == 1):
                msg = test_case._formatMessage(
                    msg,
                    "\nThe file "
                    "\n{}:1:1:\n"
                    " does not have license."
                    " please add the licnese"
                    " (most conveniently form the µSpectre snippets)"
                    .format(os.path.join(r, header_file)))
                msg_list = msg_list + msg
                continue
            file_name_line = lines_text[1]
            file_name_line = re.sub(' +', '', file_name_line)
            file_name_line = re.sub('\n', '', file_name_line)
            file_name = re.sub('\*\@file', '', file_name_line)
            header_file_name = re.split('\/', header_file)[-1]
            header_file_name = re.sub(' +', '', header_file_name)
            msg = ""
            msg = test_case._formatMessage(
                msg,
                "\nThe name supplied in the first "
                "line of the license should match "
                "file name\n {}:1:1:\n"
                .format(os.path.join(r, header_file)))
            if (not (file_name == header_file_name)):
                msg_list = msg_list + msg

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
                if (mu_license_text_lines_new[num] != lic_para):
                    msg = test_case._formatMessage(
                        msg,
                        "\n The paragraph number {} of"
                        "the license in file"
                        "\n{}:1:1:\n"
                        "is not correct. It should "
                        "be replaced by \n\'{}\'\n"
                        .format(num+1,
                                os.path.join(r, header_file_name),
                                lic_para))
                    msg_list = msg_list + msg

            print("\"{}\" License text approved".format(header_file))
            file_obj.close()
    return msg_list


def source_license_test(source_dirs, lic_paras):
    msg_list = ""
    for source_dir in source_dirs:
        source_files = []
        for r, d, f in walklevel(source_dir, 0):
            for file in f:
                if (file.endswith('.cc')
                        and '#' not in file):
                    # The files containing # are temp files created by emacs.
                    # Therefore, they need to be excluded.
                    source_files.append(os.path.join(r, file))

        for source_file in source_files:
            file_obj = open(source_file, "r")
            file_text = file_obj.read()
            license_text = re.split("#include", file_text, 2)[0]
            lines_text = re.split("\*\n", license_text)
            msg = ""
            if (len(lines_text) == 1):
                msg = test_case._formatMessage(
                    msg,
                    "\nThe file "
                    "\n{}:1:1:\n"
                    " does not have license."
                    " please add the licnese"
                    " (most conveniently form the µSpectre snippets)"
                    .format(os.path.join(r, source_file)))
                msg_list = msg_list + msg
                continue
            file_name_line = lines_text[1]
            file_name_line = re.sub(' +', '', file_name_line)
            file_name_line = re.sub('\n', '', file_name_line)
            file_name = re.sub('\*\@file', '', file_name_line)
            source_file_name = re.split('\/', source_file)[-1]
            source_file_name = re.sub(' +', '', source_file_name)
            msg = ""
            msg = test_case._formatMessage(
                msg,
                "\n The name supplied in the first "
                "line of the license should match "
                "file name\n {}:1:1:"
                .format(os.path.join(r,
                                     source_file)))
            if (not (file_name == source_file_name)):
                msg_list = msg_list + msg

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
                    msg = test_case._formatMessage(
                        msg,
                        "\n The paragraph number {} of"
                        "the license in file"
                        "\n{}:1:1:\n"
                        "is not correct. It should "
                        "be replaced by \n\'{}\'\n"
                        .format(num+1,
                                os.path.join(r, source_file_name),
                                lic_para))
                    msg_list = msg_list + msg

            print("\"{}\" License text approved".format(source_file))
            file_obj.close()
    return msg_list


def python_license_test(source_dirs, py_lic_paras):
    msg_list = ""
    for source_dir in source_dirs:
        python_files = []
        for r, d, f in walklevel(source_dir, 0):
            for file in f:
                if (file.endswith('.py')
                    and not file.startswith('#')
                    and 'pyc' not in file
                        and 'comparison_small_strain.py' not in file):
                    # The files containing # are temp files created by emacs.
                    # Therefore, they need to be excluded.
                    python_files.append(os.path.join(r, file))

        for python_file in python_files:
            file_obj = open(python_file, "r")
            file_text = file_obj.read()
            msg = ""
            if ("\"\"\"" not in file_text):
                msg = test_case._formatMessage(
                    msg,
                    "\n The file "
                    "\n{}:1:1:\n"
                    "does not have license/hash bang."
                    " please add it(them)"
                    .format(os.path.join(r, python_file)))
                msg_list = msg_list + msg
                continue

            hash_bang_text = re.split("\"\"\"", file_text, 3)[0]
            msg = ""
            print(len(hash_bang_text))
            if (len(hash_bang_text) == 0):
                msg = test_case._formatMessage(
                    msg,
                    "the file "
                    "\n{}:1:1:\n"
                    "does not have hash bang."
                    " please add it\n"
                    "#!/usr/bin/env python3\n"
                    "# -*- coding:utf-8 -*-"
                    .format(os.path.join(r, python_file)))
                msg_list = msg_list + msg
                continue
            license_text = re.split("\"\"\"", file_text, 3)[1]
            lines_text = re.split("\n\n", license_text)
            msg = ""
            if (len(lines_text) == 1):
                msg = test_case._formatMessage(
                    msg,
                    "\n The file "
                    "\n{}:1:1:\n"
                    "does not have license."
                    "please add the licnese."
                    .format(os.path.join(r, python_file)))
                msg_list = msg_list + msg
                continue
            file_name_line = lines_text[0]
            file_name_line = re.sub(' +', '', file_name_line)
            file_name_line = re.sub('\n', '', file_name_line)
            file_name = re.sub('\@file', '', file_name_line)
            python_file_name = re.split('\/', python_file)[-1]
            python_file_name = re.sub(' +', '', python_file_name)
            print("the length of lines_text ={}".format(len(lines_text)))
            msg = ""
            msg = test_case._formatMessage(
                msg, "\n The has_bang of file:"
                "\n{}:1:1\n is incorrect"
                " it should be:\n"
                "#!/usr/bin/env python3\n"
                "# -*- coding:utf-8 -*-\n"
                .format(os.path.join(r, python_file_name)))
            if (not (hash_bang_text ==
                     "#!/usr/bin/env python3\n"
                     "# -*- coding:utf-8 -*-\n")):
                msg_list = msg_list + msg

            msg = ""
            msg = test_case._formatMessage(
                msg,
                "\n The name supplied in the first "
                "line of the license should match "
                "file name\n {}:1:1:"
                .format(os.path.join(r,
                                     python_file_name)))
            if (not (file_name == python_file_name)):
                msg_list = msg_list + msg

            del lines_text[0:5]
            msg = ""
            for num, lic_para in enumerate(py_lic_paras):
                re.sub("\ \n", "\n", lines_text[num])
                if lines_text[num] != lic_para:
                    msg = test_case._formatMessage(
                        msg,
                        "\n The paragraph number {} of"
                        "the license in file"
                        "\n{}:1:1:\n"
                        "is not correct. It should "
                        "be replaced by \n\'{}\'"
                        .format(num+1,
                                os.path.join(r, python_file_name),
                                lic_para))
                    msg_list = msg_list + msg

            print("\"{}\" License text approved".format(python_file))
            file_obj.close()
    return msg_list

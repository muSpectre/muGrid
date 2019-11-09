#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file python_muSpectre_license_test.py

@author Ali Falsafi<ali.falsafi @epfl.ch>

@date 18 Sep 2019

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
import sys
import python_license_test as lic_test

# muSpectre_sources = ["../../src/materials/", "../../src/cell",
#                      "../../src/common/", "../../src/projection/",
#                      "../../src/solver/", "../../tests/",
#                      "../../language_bindings/python/",
#                      "../../language_bindings/python/muSpectre"]

lic_paras = [" µSpectre is free software; you can redistribute it and/or "
             "modify it under the terms of the GNU Lesser General Public "
             "License as published by the Free Software Foundation, either"
             " version 3, or (at your option) any later version. ",
             " µSpectre is distributed in the hope that it will be useful,"
             " but WITHOUT ANY WARRANTY; without even the implied warranty of"
             " MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the"
             " GNU Lesser General Public License for more details. ",
             " You should have received a copy of the GNU Lesser General"
             " Public License along with µSpectre; see the file COPYING."
             " If not, write to the Free Software Foundation, Inc., 59"
             " Temple Place - Suite 330, Boston, MA 02111-1307, USA. ",
             " Additional permission under GNU GPL version 3 section 7 ",
             " If you modify this Program, or any covered work, by linking or"
             " combining it with proprietary FFT implementations or numerical"
             " libraries, containing parts covered by the terms of those"
             " libraries\' licenses, the licensors of this Program grant you"
             " additional permission to convey the resulting work. "]

py_lic_paras = ["µSpectre is free software; you can redistribute it and/or\n"
                "modify it under the terms of the GNU Lesser General Public"
                " License as\npublished by the Free Software Foundation,"
                " either version 3, or (at\nyour option) any later version.",
                "µSpectre is distributed in the hope that it will be useful,"
                " but\nWITHOUT ANY WARRANTY; without even the implied warranty"
                " of\nMERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See"
                " the GNU\nLesser General Public License for more details.",
                "You should have received a copy of the GNU Lesser General"
                " Public License\nalong with µSpectre; see the file COPYING."
                " If not, write to the\nFree Software Foundation, Inc.,"
                " 59 Temple Place - Suite 330,\nBoston, MA 02111-1307, USA.",
                'Additional permission under GNU GPL version 3 section 7',
                "If you modify this Program, or any covered work, by linking"
                " or combining it\nwith proprietary FFT implementations or"
                " numerical libraries, containing parts\ncovered by the terms"
                " of those libraries' licenses, the licensors of this\nProgram"
                " grant you additional permission to convey the resulting"
                " work.\n"]


class CheckMuSpectreHeaderFiles():

    def test_muSpectre_header_files(self, muSpectre_sources):
        lic_test.header_license_test(muSpectre_sources, lic_paras)


class CheckMuSpectreSourceFiles():

    def test_muSpectre_source_files(self, muSpectre_sources):
        lic_test.source_license_test(muSpectre_sources, lic_paras)


class CheckMuSpectrePythonFiles():

    def test_muSpectre_python_files(self, muSpectre_sources):
        lic_test.python_license_test(muSpectre_sources, py_lic_paras)


def main():
    muSpectre_sources = []
    muSpectre_sources = lic_test.arg_parser.parse_args(sys.argv[1:])
    header_test_case = CheckMuSpectreHeaderFiles
    header_test_case.test_muSpectre_header_files(header_test_case,
                                                 muSpectre_sources)
    source_test_case = CheckMuSpectreSourceFiles
    source_test_case.test_muSpectre_source_files(source_test_case,
                                                 muSpectre_sources)

    python_test_case = CheckMuSpectrePythonFiles
    python_test_case.test_muSpectre_python_files(python_test_case,
                                                 muSpectre_sources)


if __name__ == "__main__":
    main()

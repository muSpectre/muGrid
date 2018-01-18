#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
file   python_projection_tests.py

@author Till Junge <till.junge@altermail.ch>

@date   18 Jan 2018

@brief  compare µSpectre's projection operators to GooseFFT

@section LICENSE

Copyright © 2018 Till Junge

µSpectre is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µSpectre is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Emacs; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.
"""

import unittest
import numpy as np

from python_test_imports import µ

def build_test_classes(Proj, name):
    class Checker(unittest.TestCase):
        def test_failure(self):
            self.assertTrue(Proj)
    Checker.__name__ = name
    return Checker

trueClass = build_test_classes(True, "trueClass")
falseClass= build_test_classes(True, "falseClass")

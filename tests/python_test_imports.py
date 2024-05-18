#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file    python_test_imports.py

@author  Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date    16 Oct 2019

@brief   prepares sys.path to load muGrid

yCopyright © 2018 Till Junge

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

import sys
import os

project_home = os.path.join(os.getcwd(), '../..')

# Default path of the library
sys.path.insert(0, os.path.join(project_home,
                                "language_bindings/python"))
sys.path.insert(0, os.path.join(project_home,
                                "language_bindings/libmugrid/python"))

# Path of the library when compiling with Xcode
sys.path.insert(0, os.path.join(project_home,
                                "language_bindings/python/Debug"))
sys.path.insert(0, os.path.join(project_home,
                                "language_bindings/libmugrid/python/Debug"))

import muGrid

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_communicator_tests.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   21 May 2019

@brief  Test muFFT's wrapper of the MPI communicator

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

from python_test_imports import muFFT


class Communicator_Check(unittest.TestCase):
    def test_sum_default(self):
        # The default communicator is COMM_SELF, i.e. each process by itself
        comm = muFFT.Communicator()
        self.assertEqual(comm.sum(comm.rank+3), 3)

    @unittest.skipIf(not muFFT.has_mpi,
                     'muFFT was compiled without MPI support')
    def test_sum_comm_world(self):
        from mpi4py import MPI
        comm = muFFT.Communicator(MPI.COMM_WORLD)
        # 1 + 2 + 3 + ... + n = n*(n+1)/2
        self.assertEqual(comm.sum(comm.rank+3),
                         comm.size*(comm.size+1)/2 + 2*comm.size)


if __name__ == '__main__':
    unittest.main()

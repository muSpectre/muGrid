#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_communicator_tests.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   21 May 2019

@brief  Test muGrid's wrapper of the MPI communicator

Copyright © 2018 Till Junge

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

import numpy as np


class Communicator_Check(unittest.TestCase):
    def test_sum_default(self):
        # The default communicator is COMM_SELF, i.e. each process by itself
        comm = muGrid.Communicator()
        self.assertEqual(comm.sum(comm.rank+3), 3)

    @unittest.skipIf(not muGrid.has_mpi,
                     'muGrid was compiled without MPI support')
    def test_sum_comm_world(self):
        try:
            from mpi4py import MPI
            comm = muGrid.Communicator(MPI.COMM_WORLD)
        except ImportError:
            comm = muGrid.Communicator()
        # 1 + 2 + 3 + ... + n = n*(n+1)/2
        self.assertEqual(comm.sum(comm.rank+3),
                         comm.size*(comm.size+1)/2 + 2*comm.size)

    def test_cum_sum_comm_world(self):
        try:
            from mpi4py import MPI
            comm = muGrid.Communicator(MPI.COMM_WORLD)
        except ImportError:
            comm = muGrid.Communicator()
        # 1 + 2 + 3 + ... + n = n*(n+1)/2
        self.assertEqual(comm.cumulative_sum(comm.rank+1),
                         comm.rank*(comm.rank+1)/2 + comm.rank + 1)

    def test_bcast_1(self):
        # The default communicator is COMM_SELF, i.e. each process by itself
        comm = muGrid.Communicator()
        scalar_arg = comm.rank + 3
        res = comm.bcast(scalar_arg, 0)
        self.assertEqual(res, 3)

        scalar_arg = comm.rank + 1
        res = comm.bcast(scalar_arg=scalar_arg, root=comm.size - 1)
        self.assertEqual(res, comm.size)

    # TODO(RLeute): I think we need such a test
    # def test_get_comm(self):
    #     # test if the function get comm works and returns the mpi communicator
    #     from mpi4py import MPI
    #     mpi_communicator = MPI.COMM_WORLD
    #     comm = muGrid.Communicator(mpi_communicator)
    #     self.assertEqual(comm.mpi_comm, mpi_communicator)

    @unittest.skipIf(not muGrid.has_mpi,
                     'muFFT was compiled without MPI support')
    def test_bcast_2(self):
        try:
            from mpi4py import MPI
            comm = muGrid.Communicator(MPI.COMM_WORLD)
        except ImportError:
            comm = muGrid.Communicator()
        scalar_arg = comm.rank + 3
        res = comm.bcast(scalar_arg, 0)
        self.assertEqual(res, 3)

        scalar_arg = comm.rank + 1
        res = comm.bcast(scalar_arg=scalar_arg, root=comm.size - 1)
        self.assertEqual(res, comm.size)

    def test_gather(self):
        try:
            from mpi4py import MPI
            comm = muGrid.Communicator(MPI.COMM_WORLD)
        except ImportError:
            comm = muGrid.Communicator()

        # gather arrays "a" with different lengths on the ranks
        a = np.arange(comm.rank*2+4).reshape((-1, 2)).T

        a_gathered = comm.gather(a)

        # construct reference
        for i in range(comm.size):
            if i == 0:
                a_ref = np.arange(i*2+4).reshape((-1, 2))
            elif i >= 1:
                a_tmp = np.arange(i*2+4).reshape((-1, 2))
                a_ref = np.concatenate((a_ref, a_tmp), axis=0)

        self.assertTrue((a_gathered == a_ref.T).all())


if __name__ == '__main__':
    unittest.main()

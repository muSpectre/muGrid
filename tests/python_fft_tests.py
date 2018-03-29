#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
file   python_fft_tests.py

@author Till Junge <till.junge@altermail.ch>

@date   17 Jan 2018

@brief  Compare µSpectre's fft implementations to numpy reference

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

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None


class FFT_Check(unittest.TestCase):
    def setUp(self):
        self.resolution = [6, 4]
        self.dim = len(self.resolution)
        self.lengths = [3., 3.]
        self.engines = [('fftw', False),
                        ('fftwmpi', True),
                        ('pfft', True)]
        self.tol = 1e-14 * np.prod(self.resolution)

    def test_forward_transform(self):
        for engine_str, transposed in self.engines:
            try:
                engine = µ.fft.FFT(self.resolution, self.lengths,
                                   fft=engine_str)
            except KeyError:
                # This FFT engine has not been compiled into the code. Skip
                # test.
                continue
            engine.initialise()
            in_arr = np.random.random([*self.resolution, self.dim, self.dim])
            out_ref = np.fft.rfftn(in_arr, axes=(0, 1))
            if transposed:
                out_ref = out_ref.swapaxes(0, 1)
            out_msp = engine.fft(in_arr.reshape(-1, self.dim**2).T).T
            err = np.linalg.norm(out_ref -
                                 out_msp.reshape(out_ref.shape))
            self.assertTrue(err < self.tol)

    def test_reverse_transform(self):
        for engine_str, transposed in self.engines:
            try:
                engine = µ.fft.FFT(self.resolution, self.lengths,
                                   fft=engine_str)
            except KeyError:
                # This FFT engine has not been compiled into the code. Skip
                # test.
                continue
            engine.initialise()

            complex_res = µ.get_hermitian_sizes(self.resolution)
            in_arr = np.zeros([*complex_res, self.dim, self.dim],
                              dtype=complex)
            in_arr.real = np.random.random(in_arr.shape)
            in_arr.imag = np.random.random(in_arr.shape)

            out_ref = np.fft.irfftn(in_arr, axes=(0, 1))
            if transposed:
                in_arr = in_arr.swapaxes(0, 1)
            out_msp = engine.ifft(in_arr.reshape(-1, self.dim**2).T).T
            out_msp *= engine.normalisation()
            err = np.linalg.norm(out_ref -
                                 out_msp.reshape(out_ref.shape))
            self.assertTrue(err < self.tol)

if __name__ == '__main__':
    unittest.main()

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
modify it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µSpectre is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

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

from python_test_imports import muFFT


class FFT_Check(unittest.TestCase):
    def setUp(self):
        self.nb_grid_pts = [6, 4]
        self.dimx = 2
        self.dimy = 3

        self.tol = 1e-14 * np.prod(self.nb_grid_pts)

        if muFFT.has_mpi:
            from mpi4py import MPI
            self.communicator = MPI.COMM_WORLD
        else:
            self.communicator = None

        self.engines = [('fftwmpi', True),
                        ('pfft', True)]
        if self.communicator is None or self.communicator.Get_size() == 1:
            self.engines += [('fftw', False)]

    def test_constructor(self):
        """Check that engines can be initialized with either bare MPI
        communicator or muFFT communicators"""
        for engine_str, transposed in self.engines:
            try:
                engine = muFFT.FFT(self.nb_grid_pts, self.dimx * self.dimy,
                                   fft=engine_str,
                                   communicator=self.communicator)
            except KeyError:
                # This FFT engine has not been compiled into the code. Skip
                # test.
                continue

            engine = muFFT.FFT(self.nb_grid_pts, self.dimx * self.dimy,
                               fft=engine_str,
                               communicator=muFFT.Communicator(
                                   self.communicator)
                               )

    def test_forward_transform(self):
        for engine_str, transposed in self.engines:
            try:
                engine = muFFT.FFT(self.nb_grid_pts, self.dimx * self.dimy,
                                   fft=engine_str,
                                   communicator=self.communicator)
            except KeyError:
                # This FFT engine has not been compiled into the code. Skip
                # test.
                continue

            np.random.seed(1)
            global_in_arr = np.random.random(
                [*self.nb_grid_pts, self.dimx, self.dimy])
            global_out_ref = np.fft.rfftn(global_in_arr, axes=(0, 1))
            if transposed:
                global_out_ref = global_out_ref.swapaxes(0, 1)
            out_ref = global_out_ref[(*engine.fourier_slices, ...)]
            in_arr = global_in_arr[(*engine.subdomain_slices, ...)]

            # Test two-dimensional array input
            out_msp = engine.engine.fft(
                in_arr.reshape(-1, self.dimx * self.dimy).T).T
            err = np.linalg.norm(out_ref - out_msp.reshape(out_ref.shape))
            self.assertTrue(err < self.tol,"{}".format(err) )

            # Separately test input as fully flattened array
            out_msp = engine.engine.fft(in_arr.reshape(-1)).T
            err = np.linalg.norm(out_ref - out_msp.reshape(out_ref.shape))
            self.assertTrue(err < self.tol)

            # Separately test convenience interface
            out_msp = engine.fft(in_arr)
            err = np.linalg.norm(out_ref - out_msp)
            self.assertTrue(err < self.tol)

    def test_reverse_transform(self):
        for engine_str, transposed in self.engines:
            try:
                engine = muFFT.FFT(self.nb_grid_pts, self.dimx * self.dimy,
                                   fft=engine_str,
                                   communicator=self.communicator)
            except KeyError:
                # This FFT engine has not been compiled into the code. Skip
                # test.
                continue

            complex_res = muFFT.get_hermitian_sizes(self.nb_grid_pts)
            global_in_arr = np.zeros([*complex_res, self.dimx, self.dimy],
                                     dtype=complex)
            np.random.seed(1)
            global_in_arr.real = np.random.random(global_in_arr.shape)
            global_in_arr.imag = np.random.random(global_in_arr.shape)
            out_ref = np.fft.irfftn(
                global_in_arr,
                axes=(0, 1))[(*engine.subdomain_slices, ...)]

            if transposed:
                global_in_arr = global_in_arr.swapaxes(0, 1)
            in_arr = global_in_arr[(*engine.fourier_slices, ...)]
            out_msp = engine.engine.ifft(
                in_arr.reshape(-1, self.dimx * self.dimy).T).T
            out_msp *= engine.normalisation
            err = np.linalg.norm(out_ref - out_msp.reshape(out_ref.shape))
            self.assertTrue(err < self.tol)

            # Separately test convenience interface
            out_msp = engine.ifft(in_arr)
            out_msp *= engine.normalisation
            err = np.linalg.norm(out_ref - out_msp)
            self.assertTrue(err < self.tol)

    def test_nb_components1_forward_transform(self):
        for engine_str, transposed in self.engines:
            try:
                engine = muFFT.FFT(self.nb_grid_pts,
                                   fft=engine_str,
                                   communicator=self.communicator)
            except KeyError:
                # This FFT engine has not been compiled into the code. Skip
                # test.
                continue

            np.random.seed(1)
            global_in_arr = np.random.random(self.nb_grid_pts)
            global_out_ref = np.fft.rfftn(global_in_arr,axes=(0, 1))
            if transposed:
                global_out_ref = global_out_ref.swapaxes(0, 1)
            out_ref = global_out_ref[(*engine.fourier_slices, ...)]
            in_arr = global_in_arr[(*engine.subdomain_slices, ...)]

            # Separately test convenience interface
            out_msp = engine.fft(in_arr)
            assert out_msp.shape == engine.nb_fourier_grid_pts, \
                "{} not equal to {}".format(out_msp.shape,
                                            engine.nb_fourier_grid_pts)

    def test_nb_components1_reverse_transform(self):
        """
        asserts that the output is of shape ( , ) and not ( , , 1)
        """
        for engine_str, transposed in self.engines:
            try:
                engine = muFFT.FFT(self.nb_grid_pts,
                                   fft=engine_str,
                                   communicator=self.communicator)
            except KeyError:
                # This FFT engine has not been compiled into the code. Skip
                # test.
                continue

            complex_res = muFFT.get_hermitian_sizes(self.nb_grid_pts)
            global_in_arr = np.zeros(complex_res,
                                     dtype=complex)
            np.random.seed(1)
            global_in_arr.real = np.random.random(global_in_arr.shape)
            global_in_arr.imag = np.random.random(global_in_arr.shape)

            if transposed:
                global_in_arr = global_in_arr.swapaxes(0, 1)
            in_arr = global_in_arr[engine.fourier_slices]

            out_msp = engine.ifft(in_arr)
            assert out_msp.shape==engine.nb_subdomain_grid_pts, \
                "{} not equal to {}".format(out_msp.shape,
                                            engine.nb_subdomain_grid_pts)

    def test_communicator(self):
        for engine_str, transposed in self.engines:
            try:
                engine = muFFT.FFT(self.nb_grid_pts,
                                   fft=engine_str,
                                   communicator=self.communicator)
            except KeyError:
                # This FFT engine has not been compiled into the code. Skip
                # test.
                continue

            comm = engine.engine.get_communicator()
            self.assertEqual(comm.sum(comm.rank+4),
                             comm.size*(comm.size+1)/2 + 3*comm.size)


if __name__ == '__main__':
    unittest.main()

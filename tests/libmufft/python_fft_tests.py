#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_fft_tests.py

@author Till Junge <till.junge@altermail.ch>

@date   17 Jan 2018

@brief  Compare µSpectre's fft implementations to numpy reference

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


class FFT_Check(unittest.TestCase):
    def setUp(self):
        #               v- grid
        #                      v-components
        self.grids = [([8, 4], (2, 3)),
                      ([6, 4], (1,)),
                      ([6, 4, 5], (2, 3)),
                      ([6, 4, 4], (1,))]

        if muFFT.has_mpi:
            from mpi4py import MPI
            self.communicator = muFFT.Communicator(MPI.COMM_WORLD)
        else:
            self.communicator = muFFT.Communicator()

        self.engines = []
        if muFFT.has_mpi:
            self.engines = ['fftwmpi', 'pfft']
        if self.communicator.size == 1:
            self.engines += ['fftw']

    def test_constructor(self):
        """Check that engines can be initialized with either bare MPI
        communicator or muFFT communicators"""
        for engine_str in self.engines:
            if muFFT.has_mpi:
                # Check initialization with bare MPI communicator
                from mpi4py import MPI
                s = MPI.COMM_WORLD.Get_size()
                try:
                    engine = muFFT.FFT([6*s, 4*s], 6, fft=engine_str,
                                       communicator=MPI.COMM_WORLD)
                except AttributeError:
                    # This FFT engine has not been compiled into the code. Skip
                    # test.
                    continue

            s = self.communicator.size
            engine = muFFT.FFT([6*s, 4*s], 6,
                               fft=engine_str,
                               communicator=self.communicator)
            self.assertEqual(
                self.communicator.sum(np.prod(engine.nb_subdomain_grid_pts)),
                np.prod(engine.nb_domain_grid_pts),
                msg='{} engine'.format(engine_str))

            comm = engine.communicator
            self.assertEqual(comm.sum(comm.rank+4),
                             comm.size*(comm.size+1)/2 + 3*comm.size,
                             msg='{} engine'.format(engine_str))

    def test_forward_transform(self):
        for engine_str in self.engines:
            for nb_grid_pts, dims in self.grids:
                if np.prod(dims) == 1:
                    continue

                s = self.communicator.size
                nb_grid_pts = s*np.array(nb_grid_pts)

                try:
                    engine = muFFT.FFT(nb_grid_pts, np.prod(dims),
                                       fft=engine_str,
                                       communicator=self.communicator)
                except AttributeError:
                    # This FFT engine has not been compiled into the code. Skip
                    # test.
                    continue

                if len(nb_grid_pts) == 2:
                    axes = (0, 1)
                elif len(nb_grid_pts) == 3:
                    axes = (0, 1, 2)
                else:
                    raise RuntimeError('Cannot handle {}-dim transforms'
                                       .format(len(nb_grid_pts)))

                # We need to transpose the input to np.fft because muFFT
                # uses column-major while np.fft uses row-major storage
                np.random.seed(1)
                global_in_arr = np.random.random([*dims, *nb_grid_pts])
                global_out_ref = np.fft.fftn(global_in_arr.T, axes=axes).T
                out_ref = global_out_ref[(..., *engine.fourier_slices)]
                in_arr = global_in_arr[(..., *engine.subdomain_slices)]

                tol = 1e-14 * np.prod(nb_grid_pts)

                # Separately test convenience interface
                out_msp = engine.fft(in_arr)
                err = np.linalg.norm(out_ref - out_msp)
                self.assertLess(err, tol, msg='{} engine'.format(engine_str))

    def test_reverse_transform(self):
        for engine_str in self.engines:
            for nb_grid_pts, dims in self.grids:
                if np.prod(dims) == 1:
                    continue

                s = self.communicator.size
                nb_grid_pts = list(s*np.array(nb_grid_pts))

                try:
                    engine = muFFT.FFT(nb_grid_pts, np.prod(dims),
                                       fft=engine_str,
                                       communicator=self.communicator)
                except AttributeError:
                    # This FFT engine has not been compiled into the code. Skip
                    # test.
                    continue

                if len(nb_grid_pts) == 2:
                    axes = (0, 1)
                elif len(nb_grid_pts) == 3:
                    axes = (0, 1, 2)
                else:
                    raise RuntimeError('Cannot handle {}-dim transforms'
                                       .format(len(nb_grid_pts)))

                # We need to transpose the input to np.fft because muFFT
                # uses column-major while np.fft uses row-major storage
                np.random.seed(1)
                complex_res = engine.nb_fourier_grid_pts
                global_in_arr = np.zeros([*dims, *complex_res], dtype=complex)
                global_in_arr.real = np.random.random(global_in_arr.shape)
                global_in_arr.imag = np.random.random(global_in_arr.shape)
                in_arr = global_in_arr[(..., *engine.fourier_slices)]
                global_out_ref = np.fft.irfftn(global_in_arr.T, axes=axes).T
                out_ref = global_out_ref[(..., *engine.subdomain_slices)]

                tol = 1e-14 * np.prod(nb_grid_pts)

                # Separately test convenience interface
                out_msp = engine.ifft(in_arr)
                out_msp *= engine.normalisation
                err = np.linalg.norm(out_ref - out_msp)
                self.assertLess(err, tol, msg='{} engine'.format(engine_str))

    def test_nb_components1_forward_transform(self):
        for engine_str in self.engines:
            for nb_grid_pts, dims in self.grids:
                if np.prod(dims) != 1:
                    continue

                try:
                    engine = muFFT.FFT(nb_grid_pts,
                                       fft=engine_str,
                                       communicator=self.communicator)
                except AttributeError:
                    # This FFT engine has not been compiled into the code. Skip
                    # test.
                    continue

                np.random.seed(1)
                global_in_arr = np.random.random(nb_grid_pts)
                global_out_ref = np.fft.rfftn(global_in_arr)
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
        for engine_str in self.engines:
            for nb_grid_pts, dims in self.grids:
                if np.prod(dims) != 1:
                    continue

                try:
                    engine = muFFT.FFT(nb_grid_pts,
                                       fft=engine_str,
                                       communicator=self.communicator)
                except AttributeError:
                    # This FFT engine has not been compiled into the code. Skip
                    # test.
                    continue

                complex_res = engine.nb_fourier_grid_pts
                global_in_arr = np.zeros(complex_res, dtype=complex)
                np.random.seed(1)
                global_in_arr.real = np.random.random(global_in_arr.shape)
                global_in_arr.imag = np.random.random(global_in_arr.shape)

                in_arr = global_in_arr[engine.fourier_slices]

                out_msp = engine.ifft(in_arr)
                assert out_msp.shape == engine.nb_subdomain_grid_pts, \
                    "{} not equal to {}".format(out_msp.shape,
                                                engine.nb_subdomain_grid_pts)

    @unittest.skipIf(muFFT.has_mpi and muFFT.Communicator().size > 1,
                     'MPI parallel FFTs do not support 1D transforms')
    def test_1d_transform(self):
        nb_grid_pts = [128, ]

        # Only serial engines support 1d transforms
        engine = muFFT.FFT(nb_grid_pts, fft='fftw')

        arr = np.random.random(nb_grid_pts)
        fft_arr_ref = np.fft.rfft(arr)
        fft_arr = engine.fft(arr)
        self.assertTrue(np.allclose(fft_arr_ref, fft_arr))

        out_arr = engine.ifft(fft_arr) * engine.normalisation
        self.assertTrue(np.allclose(out_arr, arr))

    def test_fftfreq_numpy(self):
        for engine_str in self.engines:
            for nb_grid_pts, dims in self.grids:
                if np.prod(dims) == 1:
                    continue

                s = self.communicator.size
                nb_grid_pts = list(s*np.array(nb_grid_pts))

                try:
                    engine = muFFT.FFT(nb_grid_pts, np.prod(dims),
                                       fft=engine_str,
                                       communicator=self.communicator)
                except AttributeError:
                    # This FFT engine has not been compiled into the code. Skip
                    # test.
                    continue

                freq = np.array(
                    np.meshgrid(*(np.fft.fftfreq(n) for n in nb_grid_pts),
                        indexing='ij'))

                freq = freq[(..., *engine.fourier_slices)]
                assert np.allclose(engine.fftfreq, freq)

    def test_fftfreq(self):
        # Check that x and y directions are correct
        nb_grid_pts = [7, 4]
        nx, ny = nb_grid_pts
        engine = muFFT.FFT(nb_grid_pts, 1, fft='serial')
        qx, qy = engine.fftfreq

        qarr = np.zeros(engine.nb_fourier_grid_pts, dtype=complex)
        qarr[np.logical_and(np.abs(np.abs(qx)*nx - 1) < 1e-6,
                            np.abs(np.abs(qy)*ny - 0) < 1e-6)] = 0.5
        rarr = engine.ifft(qarr)
        assert np.allclose(rarr, rarr[:, 0].reshape(-1, 1))
        assert np.allclose(rarr[:, 0], np.cos(np.arange(nx)*2*np.pi/nx))

        qarr = np.zeros(engine.nb_fourier_grid_pts, dtype=complex)
        qarr[np.logical_and(np.abs(np.abs(qx)*nx - 0) < 1e-6,
                            np.abs(np.abs(qy)*ny - 1) < 1e-6)] = 0.5
        rarr = engine.ifft(qarr)
        assert np.allclose(rarr, rarr[0, :].reshape(1, -1))
        assert np.allclose(rarr[0, :], np.cos(np.arange(ny)*2*np.pi/ny))

if __name__ == '__main__':
    unittest.main()

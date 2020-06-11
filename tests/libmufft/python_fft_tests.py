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

import gc

import unittest
import numpy as np

from python_test_imports import muFFT, muGrid


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
                    nb_dof = 6
                    engine = muFFT.FFT([6*s, 4*s], fft=engine_str,
                                       communicator=MPI.COMM_WORLD)
                    engine.initialise(nb_dof)
                except AttributeError:
                    # This FFT engine has not been compiled into the code. Skip
                    # test.
                    continue

            s = self.communicator.size
            nb_dof = 6
            engine = muFFT.FFT([6*s, 4*s],
                               fft=engine_str,
                               communicator=self.communicator)
            engine.initialise(nb_dof)
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
                s = self.communicator.size
                nb_grid_pts = s*np.array(nb_grid_pts)

                try:
                    engine = muFFT.FFT(nb_grid_pts,
                                       fft=engine_str,
                                       communicator=self.communicator)
                    engine.initialise(np.prod(dims))
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
                out_msp = engine.register_fourier_space_field("out_msp",
                                                            np.prod(dims))
                engine.fft(in_arr, out_msp)
                out_array = np.array(out_msp).reshape(out_ref.shape, order='f')
                err = np.linalg.norm(out_ref - out_array)
                self.assertLess(err, tol, msg='{} engine'.format(engine_str))

    def test_reverse_transform(self):
        for engine_str in self.engines:
            for nb_grid_pts, dims in self.grids:
                s = self.communicator.size
                nb_grid_pts = list(s*np.array(nb_grid_pts))

                try:
                    engine = muFFT.FFT(nb_grid_pts,
                                       fft=engine_str,
                                       communicator=self.communicator)
                    engine.initialise(np.prod(dims))
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
                complex_res = list(engine.nb_domain_grid_pts)
                complex_res[0] = complex_res[0] // 2 + 1
                global_in_arr = np.zeros([*dims, *complex_res], dtype=complex)
                global_in_arr.real = np.random.random(global_in_arr.shape)
                global_in_arr.imag = np.random.random(global_in_arr.shape)
                in_arr = global_in_arr[(..., *engine.fourier_slices)]
                in_arr.shape = (*dims, *engine.nb_fourier_grid_pts)
                global_out_ref = np.fft.irfftn(global_in_arr.T, axes=axes).T
                out_ref = global_out_ref[(..., *engine.subdomain_slices)]

                tol = 1e-14 * np.prod(nb_grid_pts)

                # Separately test convenience interface
                out_msp = np.empty([*dims, *engine.nb_subdomain_grid_pts],
                                   dtype=float, order='f')
                try:
                    engine.ifft(in_arr, out_msp)
                except Exception as err:
                    self.assertEqual("a", in_arr.shape)

                out_msp *= engine.normalisation
                err = np.linalg.norm(out_ref - out_msp)
                self.assertLess(err, tol, msg='{} engine'.format(engine_str))

    def test_forward_transform_field_interface(self):
        for engine_str in self.engines:
            for nb_grid_pts, dims in self.grids:
                s = self.communicator.size
                nb_grid_pts = s*np.array(nb_grid_pts)

                try:
                    engine = muFFT.FFT(nb_grid_pts,
                                       fft=engine_str,
                                       communicator=self.communicator)
                    engine.initialise(np.prod(dims))
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

                fc = muGrid.GlobalFieldCollection(len(nb_grid_pts))
                fc.initialise(tuple(engine.nb_subdomain_grid_pts))
                in_field = fc.register_real_field('in_field', np.prod(dims))
                in_field.array(dims, muGrid.Pixel)[...] = global_in_arr[
                    (..., *engine.subdomain_slices)]

                tol = 1e-14 * np.prod(nb_grid_pts)

                # Separately test convenience interface
                out_field = engine.register_fourier_space_field("out_field",
                                                              np.prod(dims))
                engine.fft(in_field, out_field)
                err = np.linalg.norm(out_ref -
                                     out_field.array(dims, muGrid.Pixel))
                self.assertLess(err, tol, msg='{} engine'.format(engine_str))

    def test_reverse_transform_field_interface(self):
        for engine_str in self.engines:
            for nb_grid_pts, dims in self.grids:
                s = self.communicator.size
                nb_grid_pts = list(s*np.array(nb_grid_pts))

                try:
                    engine = muFFT.FFT(nb_grid_pts,
                                       fft=engine_str,
                                       communicator=self.communicator)
                    engine.initialise(np.prod(dims))
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
                complex_res = list(engine.nb_domain_grid_pts)
                complex_res[0] = complex_res[0] // 2 + 1
                global_in_arr = np.zeros([*dims, *complex_res], dtype=complex)
                global_in_arr.real = np.random.random(global_in_arr.shape)
                global_in_arr.imag = np.random.random(global_in_arr.shape)
                global_out_ref = np.fft.irfftn(global_in_arr.T, axes=axes).T
                out_ref = global_out_ref[(..., *engine.subdomain_slices)]

                tol = 1e-14 * np.prod(nb_grid_pts)

                fourier_field = engine.register_fourier_space_field(
                    "fourier_field", np.prod(dims))
                fourier_field.array(dims, muGrid.Pixel)[...] = \
                    global_in_arr[(..., *engine.fourier_slices)]

                fc = muGrid.GlobalFieldCollection(len(nb_grid_pts))
                fc.initialise(tuple(engine.nb_subdomain_grid_pts))
                out_field = fc.register_real_field('out_field', np.prod(dims))

                # Separately test convenience interface
                engine.ifft(fourier_field, out_field)
                err = np.linalg.norm(
                    out_ref -
                    out_field.array(dims, muGrid.Pixel)*engine.normalisation)
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
                    engine.initialise(np.prod(dims))
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
                out_msp = engine.register_fourier_space_field("out_msp",
                                                            np.prod(dims))
                engine.fft(in_arr, out_msp)

                # Check that the output array does not have a unit first dimension
                assert np.squeeze(out_msp).shape == engine.nb_fourier_grid_pts, \
                    "{} not equal to {}".format(out_msp.shape,
                                                engine.nb_fourier_grid_pts)


    def test_nb_components1_forward_transform(self):
        for engine_str in self.engines:
            for nb_grid_pts, dims in self.grids:
                if np.prod(dims) != 1:
                    continue

                try:
                    engine = muFFT.FFT(nb_grid_pts,
                                       fft=engine_str,
                                       communicator=self.communicator)
                    engine.initialise(np.prod(dims))
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
                out_msp = engine.register_fourier_space_field('out_msp',
                                                            np.prod(dims))
                engine.fft(in_arr, out_msp)

                # Check that the output array does not have a unit first dimension
                self.assertEqual(np.squeeze(out_msp).shape, engine.nb_fourier_grid_pts),# \
                    #                "{} not equal to {}".format(out_msp.shape,
                #                                            engine.nb_fourier_grid_pts)
                # TODO(pastewka): I think this test is out of date. the numpy
                # rfftn shortens a different dimension, and therefore gets a
                # different shape. can we ditch this?
                # self.assertEqual(out_msp.shape, global_out_ref.shape)
                # self.assertEqual(len(out_msp.shape), len(global_out_ref.shape))

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
                    engine.initialise(np.prod(dims))
                except AttributeError:
                    # This FFT engine has not been compiled into the code. Skip
                    # test.
                    continue

                complex_res = list(engine.nb_domain_grid_pts)
                complex_res[0] = complex_res[0] // 2 + 1
                global_in_arr = np.zeros(complex_res, dtype=complex)
                np.random.seed(1)
                global_in_arr.real = np.random.random(global_in_arr.shape)
                global_in_arr.imag = np.random.random(global_in_arr.shape)

                in_arr = global_in_arr[engine.fourier_slices]
                in_arr.shape =(*dims, *engine.nb_fourier_grid_pts)
                out_msp = np.empty((1, *engine.nb_subdomain_grid_pts),
                                   order="f")
                engine.ifft(in_arr, out_msp)
                assert out_msp.shape == (1, *engine.nb_subdomain_grid_pts), \
                    "{} not equal to {}".format(out_msp.shape,
                                                engine.nb_subdomain_grid_pts)

    @unittest.skipIf(muFFT.has_mpi and muFFT.Communicator().size > 1,
                     'MPI parallel FFTs do not support 1D transforms')
    def test_1d_transform(self):
        nb_grid_pts = [128, ]

        # Only serial engines support 1d transforms
        engine = muFFT.FFT(nb_grid_pts, fft='fftw')

        nb_dof = 1
        engine.initialise(nb_dof)
        arr = np.random.random(nb_grid_pts * nb_dof)

        fft_arr_ref = np.fft.rfft(arr)
        fft_arr = engine.register_fourier_space_field("fourier work space", nb_dof)
        engine.fft(arr, fft_arr)
        self.assertTrue(np.allclose(fft_arr_ref, fft_arr))

        out_arr = np.empty_like(arr)
        engine.ifft(fft_arr, out_arr)
        out_arr *= engine.normalisation
        self.assertTrue(np.allclose(out_arr, arr))

    def test_fftfreq_numpy(self):
        for engine_str in self.engines:
            for nb_grid_pts, dims in self.grids:
                if np.prod(dims) == 1:
                    continue

                s = self.communicator.size
                nb_grid_pts = list(s*np.array(nb_grid_pts))

                try:
                    engine = muFFT.FFT(nb_grid_pts,
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
        nb_dof = 1
        engine = muFFT.FFT(nb_grid_pts, fft='serial')
        engine.initialise(nb_dof)
        qx, qy = engine.fftfreq

        qarr = np.zeros(engine.nb_fourier_grid_pts, dtype=complex)
        qarr[np.logical_and(np.abs(np.abs(qx)*nx - 1) < 1e-6,
                            np.abs(np.abs(qy)*ny - 0) < 1e-6)] = 0.5

        rarr = np.zeros(nb_grid_pts, order='f')
        engine.ifft(qarr, rarr)
        assert np.allclose(rarr, rarr[:, 0].reshape(-1, 1))
        assert np.allclose(rarr[:, 0], np.cos(np.arange(nx)*2*np.pi/nx))

        qarr = np.zeros(engine.nb_fourier_grid_pts, dtype=complex)
        qarr[np.logical_and(np.abs(np.abs(qx)*nx - 0) < 1e-6,
                            np.abs(np.abs(qy)*ny - 1) < 1e-6)] = 0.5
        engine.ifft(qarr, rarr)
        assert np.allclose(rarr, rarr[0, :].reshape(1, -1))
        assert np.allclose(rarr[0, :], np.cos(np.arange(ny)*2*np.pi/ny))

    def test_buffer_lifetime(self):
        res = [2, 3]
        data = np.random.random(res)
        ref = np.fft.rfftn(data.T).T
        # Python will attempt to remove the muFFT.FFT temporary object here
        # right after the call to fft. However, since fft returns a pointer
        # to an *internal* buffer of the object, garbage collection should
        # be deferred until `tested` is destroyed.
        engine = muFFT.FFT(res, fft="serial")
        engine.initialise(1)
        f_data = engine.register_fourier_space_field("fourier work space", 1)
        engine.fft(data, f_data)
        tested = f_data.array()
        gc.collect()
        # It should not own the data, because it reference an internal buffer
        assert not tested.flags.owndata
        assert np.allclose(ref.real, tested.real)
        assert np.allclose(ref.imag, tested.imag)

    def test_strides(self):
        for engine_str in self.engines:
            try:
                engine = muFFT.FFT([3, 5, 7], fft=engine_str,
                                   communicator=self.communicator)
                engine.initialise(1)
            except AttributeError:
                # This FFT engine has not been compiled into the code. Skip
                # test.
                continue

            if engine_str == 'fftw':
                assert engine.subdomain_strides == (1, 3, 15) # column-major
                assert engine.fourier_strides == (1, 2, 10) # column-major
            elif engine_str == 'fftwmpi':
                assert engine.subdomain_strides == (1, 4, 20) # padding in first dimension
                assert engine.fourier_strides == (1, 14, 2) # transposed output


if __name__ == '__main__':
    unittest.main()

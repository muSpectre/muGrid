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


class Common_Check(unittest.TestCase):
    def test_fft_freqs_1d(self):
        nx = 5
        fftf = muFFT.FFTFreqs([nx-1])
        w = fftf.get_xi(np.arange(nx, dtype=np.int32).reshape(1, nx))
        self.assertTrue(np.allclose(w, [ 0., 1., -2., -1., 0.]))

    def test_normalized_fft_freqs_1d(self):
        nx = 5
        fftf = muFFT.FFTFreqs([nx], [nx])
        w = fftf.get_xi(np.arange(nx, dtype=np.int32).reshape(1, nx))
        self.assertTrue(np.allclose(w, [ 0./nx, 1./nx, 2./nx, -2./nx, -1./nx]))
        self.assertTrue(np.allclose(w, np.fft.fftfreq(nx)))

    def test_fft_freqs_2d(self):
        nx = 5
        ny = 3
        fftf = muFFT.FFTFreqs([nx, ny])
        wx, wy = fftf.get_xi(
            np.array([[1, 2], [2, 3], [3, 4]], dtype=np.int32).T.copy())
        self.assertTrue(np.allclose(wx, [ 1., 2., -2.]))
        self.assertTrue(np.allclose(wy, [ -1., 0., 1.]))

    def test_fft_freqs_3d(self):
        nx = 5
        ny = 3
        nz = 6
        fft = muFFT.FFT([nx, ny, nz])
        fftf = muFFT.FFTFreqs(fft.nb_domain_grid_pts, fft.nb_domain_grid_pts)
        grid_pts = np.mgrid[fft.fourier_slices].astype(np.int32)
        wx, wy, wz = fftf.get_xi(grid_pts)
        self.assertTrue(np.allclose(wx[:,0,0], np.fft.fftfreq(nx)[:nx//2+1]))
        self.assertTrue(np.allclose(wy[0,:,0], np.fft.fftfreq(ny)))
        self.assertTrue(np.allclose(wz[0,0,:], np.fft.fftfreq(nz)))

if __name__ == '__main__':
    unittest.main()

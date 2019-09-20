# !/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_derivative_tests.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   20 Jun 2019

@brief  test discrete derivative

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
import numpy as np

from python_test_imports import muFFT

class DerivativeCheck2d(unittest.TestCase):
    def setUp(self):
        self.nb_pts = [23, 27]
        np.random.seed(7)
        self.field = np.random.random(self.nb_pts)
        self.fft = muFFT.FFT(self.nb_pts)
        self.fourier_field = self.fft.fft(self.field)

    def test_fourier_derivative(self):
        diffop = muFFT.FourierDerivative(2, 0)
        q = self.fft.wavevectors()
        d = diffop.fourier(q)
        fourier_field_copy = np.copy(self.fourier_field)
        diff_field = self.fft.ifft(d * self.fourier_field) * \
            self.fft.normalisation
        ndiff = self.fft.ifft(1j*2*np.pi*q[0] * fourier_field_copy) * \
            self.fft.normalisation
        nx, ny = self.nb_pts
        for x in range(nx):
            for y in range(ny):
                self.assertAlmostEqual(diff_field[x,y], ndiff[x,y])

    def test_upwind_differences(self):
        diffop = muFFT.DiscreteDerivative([0, 0], [[-1, 1]])
        q = self.fft.wavevectors()
        d = diffop.fourier(q)
        diff_field = self.fft.ifft(d * self.fourier_field) * \
            self.fft.normalisation
        nx, ny = self.nb_pts
        for x in range(nx):
            for y in range(ny):
                ndiff = self.field[x, (y+1)%ny] - self.field[x, y]
                self.assertAlmostEqual(diff_field[x, y], ndiff)

    def test_averaged_upwind_differences(self):
        diffop = µ.DiscreteDerivative([0, 0], [[-0.5, -0.5],
                                               [ 0.5,  0.5]])
        q = self.fft.wavevectors()
        d = diffop.fourier(q)
        diff_field = self.fft.ifft(d * self.fourier_field) * \
            self.fft.normalisation
        nx, ny = self.nb_pts
        for x in range(nx):
            for y in range(ny):
                ndiff = (self.field[x, (y+1)%ny] - self.field[x, y] \
                         + self.field[(x+1)%nx, (y+1)%ny] - self.field[(x+1)%nx, y])/2
                self.assertAlmostEqual(diff_field[x, y], ndiff)

    def test_central_differences(self):
        diffop = muFFT.DiscreteDerivative([-1, 0], [[-0.5], [0], [0.5]])
        q = self.fft.wavevectors()
        d = diffop.fourier(q)
        diff_field = self.fft.ifft(d * self.fourier_field) * \
            self.fft.normalisation
        nx, ny = self.nb_pts
        for x in range(nx):
            for y in range(ny):
                ndiff = (self.field[(x+1)%nx, y] - self.field[(x-1)%nx, y])/2
                self.assertAlmostEqual(diff_field[x, y], ndiff)

    def test_sixth_order_central_differences(self):
        diffop = muFFT.DiscreteDerivative([0, -3],
                                          [[-1/60, 3/20, -3/4, 0,
                                            3/4, -3/20, 1/60]])
        q = self.fft.wavevectors()
        d = diffop.fourier(q)
        diff_field = self.fft.ifft(d * self.fourier_field) * \
            self.fft.normalisation
        nx, ny = self.nb_pts
        for x in range(nx):
            for y in range(ny):
                ndiff = -1/60*self.field[x, (y-3)%ny] \
                        +3/20*self.field[x, (y-2)%ny] \
                        -3/4*self.field[x, (y-1)%ny] \
                        +3/4*self.field[x, (y+1)%ny] \
                        -3/20*self.field[x, (y+2)%ny] \
                        +1/60*self.field[x, (y+3)%ny]
                self.assertAlmostEqual(diff_field[x, y], ndiff)

class DerivativeCheck3d(unittest.TestCase):
    def setUp(self):
        self.nb_pts = [23, 23, 17]
        self.field = np.random.random(self.nb_pts)
        self.fft = muFFT.FFT(self.nb_pts)
        self.fourier_field = self.fft.fft(self.field)

    def test_upwind_differences_x(self):
        diffop = µ.DiscreteDerivative([0, 0, 0], [[[-1, 1]]]) \
            .rollaxes(-1).rollaxes(-1)
        q = self.fft.wavevectors()
        d = diffop.fourier(q)
        diff_field = self.fft.ifft(d * self.fourier_field) * \
            self.fft.normalisation
        nx, ny, nz = self.nb_pts
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    ndiff = self.field[(x+1)%nx, y, z] - self.field[x, y, z]
                    self.assertAlmostEqual(diff_field[x, y, z], ndiff)

    def test_upwind_differences_y(self):
        diffop = µ.DiscreteDerivative([0, 0, 0], [[[-1, 1]]]).rollaxes(-1)
        q = self.fft.wavevectors()
        d = diffop.fourier(q)
        diff_field = self.fft.ifft(d * self.fourier_field) * \
            self.fft.normalisation
        nx, ny, nz = self.nb_pts
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    ndiff = self.field[x, (y+1)%ny, z] - self.field[x, y, z]
                    self.assertAlmostEqual(diff_field[x, y, z], ndiff)

    def test_upwind_differences_z(self):
        diffop = µ.DiscreteDerivative([0, 0, 0], [[[-1, 1]]])
        q = self.fft.wavevectors()
        d = diffop.fourier(q)
        diff_field = self.fft.ifft(d * self.fourier_field) * \
            self.fft.normalisation
        nx, ny, nz = self.nb_pts
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    ndiff = self.field[x, y, (z+1)%nz] - self.field[x, y, z]
                    self.assertAlmostEqual(diff_field[x, y, z], ndiff)

    def test_averaged_upwind_differences_x(self):
        diffop = µ.DiscreteDerivative([0, 0, 0],
                                      [[[-0.25, -0.25], [-0.25, -0.25]],
                                       [[ 0.25,  0.25], [ 0.25,  0.25]]]) \
            .rollaxes(-1).rollaxes(-1)
        q = self.fft.wavevectors()
        d = diffop.fourier(q)
        diff_field = self.fft.ifft(d * self.fourier_field) * \
            self.fft.normalisation
        nx, ny, nz = self.nb_pts
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    ndiff = (self.field[(x+1)%nx, y, z] - self.field[x, y, z] \
                        + self.field[(x+1)%nx, (y+1)%ny, z] - self.field[x, (y+1)%ny, z] \
                        + self.field[(x+1)%nx, y, (z+1)%nz] - self.field[x, y, (z+1)%nz] \
                        + self.field[(x+1)%nx, (y+1)%ny, (z+1)%nz] - self.field[x, (y+1)%ny, (z+1)%nz])/4
                    self.assertAlmostEqual(diff_field[x, y, z], ndiff)

    def test_averaged_upwind_differences_y(self):
        diffop = µ.DiscreteDerivative([0, 0, 0],
                                      [[[-0.25, -0.25], [-0.25, -0.25]],
                                       [[ 0.25,  0.25], [ 0.25,  0.25]]]) \
            .rollaxes(-1)
        q = self.fft.wavevectors()
        d = diffop.fourier(q)
        diff_field = self.fft.ifft(d * self.fourier_field) * \
            self.fft.normalisation
        nx, ny, nz = self.nb_pts
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    ndiff = (self.field[x, (y+1)%ny, z] - self.field[x, y, z] \
                        + self.field[(x+1)%nx, (y+1)%ny, z] - self.field[(x+1)%nx, y, z] \
                        + self.field[x, (y+1)%ny, (z+1)%nz] - self.field[x, y, (z+1)%nz] \
                        + self.field[(x+1)%nx, (y+1)%ny, (z+1)%nz] - self.field[(x+1)%nx, y, (z+1)%nz])/4
                    self.assertAlmostEqual(diff_field[x, y, z], ndiff)

    def test_averaged_upwind_differences_z(self):
        diffop = µ.DiscreteDerivative([0, 0, 0],
                                      [[[-0.25, -0.25], [-0.25, -0.25]],
                                       [[ 0.25,  0.25], [ 0.25,  0.25]]])
        q = self.fft.wavevectors()
        d = diffop.fourier(q)
        diff_field = self.fft.ifft(d * self.fourier_field) * \
            self.fft.normalisation
        nx, ny, nz = self.nb_pts
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    ndiff = (self.field[x, y, (z+1)%nz] - self.field[x, y, z] \
                        + self.field[(x+1)%nx, y, (z+1)%nz] - self.field[(x+1)%nx, y, z] \
                        + self.field[x, (y+1)%ny, (z+1)%nz] - self.field[x, (y+1)%ny, z] \
                        + self.field[(x+1)%nx, (y+1)%ny, (z+1)%nz] - self.field[(x+1)%nx, (y+1)%ny, z])/4
                    self.assertAlmostEqual(diff_field[x, y, z], ndiff)

if __name__ == "__main__":
    unittest.main()

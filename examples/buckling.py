#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   buckling.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   09 Aug 2020

@brief  buckling example demonstrating the trust-region Newton CG optimizer

Copyright © 2019 Till Junge

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

from mpi4py import MPI
import optparse
import sys
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    matplotlib_found = True
except ImportError:
    matplotlib_found = False

from python_example_imports import muSpectre as msp
from muFFT import Stencils2D

print(msp.__file__)

###

parser = optparse.OptionParser()

parser.add_option('-f', '--to-file',
                  action="store", dest="plot_file",
                  help="store plot to file instead of showing it on screen")

options, args = parser.parse_args()

###

def make_triangles(displ):
    displ_x, displ_y = displ
    nx, ny = displ_x.shape

    def i(x, y):
        return x + nx * y

    x, y = np.mgrid[:nx - 1, :ny - 1]
    upper_triangles = np.stack((i(x, y), i(x, y + 1), i(x + 1, y)), axis=2)
    lower_triangles = np.stack((i(x + 1, y), i(x, y + 1), i(x + 1, y + 1)),
                               axis=2)

    triangles = np.array([upper_triangles, lower_triangles]).reshape(-1, 3)

    return Triangulation(
        displ_x.reshape(-1, order='f'),
        displ_y.reshape(-1, order='f'),
        triangles)


###

nx, ny = 23, 127
mask = np.zeros((nx, ny), dtype=bool)
amplitude1 = 5
amplitude2 = 2
s = np.sin(2 * np.pi * np.arange(ny) / ny)
x, y = np.mgrid[:nx, :ny]
mask[np.logical_and(x > amplitude1 * (1 + s),
                    x < nx - amplitude2 * (1 + s))] = True

###

nb_domain_grid_pts = mask.shape
domain_lengths = [float(r) for r in nb_domain_grid_pts]

## define the convergence tolerance for the Newton-Raphson increment
newton_tol = 1e-6
equil_tol = newton_tol
inc_tr_tol = 1e-4
dec_tr_tol = 1e-2
## tolerance for the solver of the linear cell
cg_tol = 1e-6

## macroscopic strain
applied_strain = []
if len(args) > 0:
    strain_steps = [float(s) for s in args]
else:
    strain_steps = np.linspace(0, 0.05, 10)
for s in strain_steps:
    applied_strain += [[[2 * s, 0], [0, -s]]]

maxiter = 1000  # for linear cell solver
trust_region = 100.0

## numerical derivative, two quadrature points
gradient = Stencils2D.linear_finite_elements
weights = [1, 1]

###
rve = msp.Cell(nb_domain_grid_pts, domain_lengths,
               msp.Formulation.finite_strain, gradient, weights,
               fft='fftw', communicator=MPI.COMM_WORLD)
hard = msp.material.MaterialLinearElastic1_2d.make(rve, "hard", 1., .33)
vacuum = msp.material.MaterialLinearElastic1_2d.make(rve, "vacuum", 0.0, 0.33)
for pixel_index, pixel in enumerate(rve.pixels):
    if mask[tuple(pixel)]:
        hard.add_pixel(pixel_index)
    else:
        vacuum.add_pixel(pixel_index)
## we need the trust-region solver here because of the buckling instability where the
## Hessian becomes negative definite
solver = msp.solvers.KrylovSolverTrustRegionCG(
    rve, maxiter=maxiter, verbose=msp.Verbosity.Silent)

result = msp.solvers.trust_region_newton_cg(rve, applied_strain, solver,
                                            trust_region=trust_region,
                                            newton_tol=newton_tol,
                                            equil_tol=equil_tol,
                                            dec_tr_tol=dec_tr_tol,
                                            inc_tr_tol=inc_tr_tol,
                                            verbose=msp.Verbosity.Some)

###

if matplotlib_found and MPI.COMM_WORLD.Get_size() == 1:
    plt.figure()
    phase = np.vstack((mask, mask))
    for i, res in enumerate([result[0], result[-1]]):
        F = res.grad.reshape((2, 2, 2) + tuple(nb_domain_grid_pts))
        detF = np.linalg.det(F.T).T
        print(i, np.logical_and(detF < 0, mask).sum())

        strain = res.grad.reshape(rve.strain.shape, order='F')
        r, displ = msp.gradient_integration.get_complemented_positions('0p', rve, F0=None,
                                                                        periodically_complemented=True,
                                                                        strain_array=strain)

        tri = make_triangles(displ)
        # plt.subplot(1, len(result), i+1, aspect=1)
        plt.subplot(1, 2, i + 1, aspect=1)
        tpc = plt.tripcolor(tri, phase.reshape(-1), ec='black')
        plt.colorbar(tpc)

    plt.tight_layout()

    if options.plot_file is not None:
        plt.savefig(options.plot_file)
    else:
        plt.show()


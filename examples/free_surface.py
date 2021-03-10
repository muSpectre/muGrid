#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   free_surface.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   10 Feb 2021

@brief  Biaxial compression of a three-dimension volume with a free surface

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

import optparse
import random

import numpy as np

from mpi4py import MPI

try:
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    matplotlib_found = True
except ImportError:
    matplotlib_found = False

from python_example_imports import muSpectre as msp

print(msp.__file__)

###

parser = optparse.OptionParser()

parser.add_option('-d', '--strain-step',
                  action="store", dest="strain_step",
                  help="stepwise increase strain with this value",
                  type='float', default=0.01)
parser.add_option('-n', '--nb-steps',
                  action="store", dest="nb_steps",
                  help="number of strain steps",
                  type='int', default=10)
parser.add_option('-t', '--topography',
                  action="store", dest="topography_filename",
                  help="filename for surface topography visualization")
parser.add_option('-s', '--show-topography',
                  action="store_true", dest="show_topography",
                  help="filename for surface topography visualization")
parser.add_option('-f', '--to-file',
                  action="store", dest="filename",
                  help="filename for full output")
parser.add_option('-g', '--nb-grid-pts',
                  action='store', dest='nb_grid_pts',
                  help='number of grid points',
                  default='4,4,4')

options, args = parser.parse_args()

###

nx, ny, nz = nb_domain_grid_pts = [
    int(x) for x in options.nb_grid_pts.split(',')]
domain_lengths = [float(r) for r in nb_domain_grid_pts]

###

Youngs_modulus = 1
Poisson_ratio = 0.33
hardening_exponent = 0.1
yield_min = 0.1
yield_max = 0.2

###

## define the convergence tolerance for the Newton-Raphson increment
newton_tol = 1e-6
equil_tol = newton_tol
inc_tr_tol = 1e-4
dec_tr_tol = 1e-2
## tolerance for the solver of the linear cell
cg_tol = 1e-6

## macroscopic strain
s = options.strain_step
strain_step = np.array([[-s, 0, 0], [0, -s, 0], [0, 0, 2*s]])

maxiter = 1000  # for linear cell solver
trust_region = 100.0

## numerical derivative, six elements
gradient = msp.linear_finite_elements.gradient_3d

###

rve = msp.Cell(nb_domain_grid_pts, domain_lengths,
               msp.Formulation.finite_strain, gradient,
               fft='fftw', communicator=MPI.COMM_WORLD)
material = msp.material.MaterialHyperElastoPlastic2_3d.make(rve, "material")
vacuum = msp.material.MaterialLinearElastic1_3d.make(rve, "vacuum", 0.0, 0.33)
for pixel_index, pixel in enumerate(rve.pixels):
    if pixel[2] == nz-1:
        vacuum.add_pixel(pixel_index)
    else:
        yield_stress = yield_min+random.random()*(yield_max-yield_min)
        material.add_pixel(pixel_index, Youngs_modulus, Poisson_ratio,
                           yield_stress, hardening_exponent)

## use trust-region solver for stability
solver = msp.solvers.KrylovSolverCG(
    rve, cg_tol, maxiter=maxiter, verbose=msp.Verbosity.Silent)

for i in range(options.nb_steps):
    print('=== STEP {}/{} ==='.format(i+1, options.nb_steps))
    msp.solvers.newton_cg(
        rve, strain_step, solver,
        newton_tol=newton_tol,
        equil_tol=equil_tol,
        IsStrainInitialised=msp.solvers.IsStrainInitialised.No \
            if i == 0 else msp.solvers.IsStrainInitialised.Yes,
        verbose=msp.Verbosity.Detailed)

if options.filename is not None:
    msp.linear_finite_elements.write_3d(options.filename, rve)

positions = rve.projection.integrate(rve.strain).array()
surface_topography = -positions[2,0,:,:,0]
surface_topography -= surface_topography.mean()

if options.topography_filename is not None:
    np.savetxt(options.topography_filename, surface_topography)

if options.show_topography:
    ## show final surface
    plt.subplot(111, aspect=1)
    plt.pcolormesh(surface_topography)
    plt.colorbar()
    plt.show()

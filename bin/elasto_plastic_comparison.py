#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   odd_comparison.py

@author Till Junge <till.junge@altermail.ch>

@date   08 Dec 2018

@brief  comparison to GooseFFT's odd.py calculation

Copyright © 2018 Till Junge

µSpectre is free software; you can redistribute it and/or
modify it under the terms of the GNU General Lesser Public License as
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
import sys
import os
import numpy as np
from mpi4py import MPI

sys.path.append(os.path.join(os.getcwd(), "language_bindings/python"))
import muSpectre as µ


# set up of the microstructure (omit the slice operator at the end to
# run the full problem)
phase  = np.load('odd_image.npz')['phase'][:41, :41]
resolution = list(phase.shape)
dim = len(resolution)
lengths = resolution
formulation = µ.Formulation.finite_strain

# choose material (constitutive) model
Mat = µ.material.MaterialHyperElastoPlastic1_2d
K = 0.833  # bulk modulus
mu = 0.386  # shear modulus
# hardening moduli
H_soft, H_hard = 2000.0e6 / 200.0e9, 2. * 2000.0e6 / 200.0e9
# initial yield stresses
tauy0_soft, tauy0_hard = 600.0e6 / 200.0e9, 2. * 600.0e6 / 200.0e9

# MuSpectre uses Young's modulus and Poisson's ratio instead
Young = (9 * K * mu) / (3 * K + mu)
Poisson = (3 * K - 2 * mu) / (2 * (3 * K + mu))

# set up system
rve = µ.Cell(resolution, lengths, formulation, fft='fftwmpi',
             communicator=MPI.COMM_WORLD)
hard = Mat.make(rve, "hard", Young, Poisson, tauy0_hard, H_hard)
soft = Mat.make(rve, "soft", Young, Poisson, tauy0_soft, H_soft)


for pixel in rve:
    if phase[tuple(pixel)] == 1:
        hard.add_pixel(pixel)
    elif phase[tuple(pixel)] == 0:
        soft.add_pixel(pixel)
    else:
        raise Exception("phase '{}' should not exist".format(
            phase[tuple(pixel)]))
    pass

rve.initialise(flags=µ.FFT_PlanFlags.patient)

# number if load increments
ninc    = 250
epsbar  = np.linspace(0.0,0.1,ninc+1)[1:]
stretch = np.exp(np.sqrt(3.0)/2.0*epsbar)

ΔFbars = list()
for inc, lam  in zip(range(1,ninc+1),stretch[:40]):
    print('=============================')
    print('inc: {0:d}'.format(inc))

    ΔF = np.zeros((dim, dim))
    ΔF[0, 0] = lam-1.
    ΔF[1, 1] = 1./lam-1.
    print("ΔF =\n{}".format(ΔF))
    print("F =\n{}".format(ΔF+np.eye(dim)))

    ΔFbars.append(ΔF)
    pass


cg_tol = 1e-8
newton_tol = 1e-5
equil_tol = 1e-10
maxiter = 1000

solver = µ.solvers.SolverCG(rve, cg_tol, maxiter, verbose=False)

results = µ.solvers.de_geus(
    rve, ΔFbars, solver, newton_tol, equil_tol, verbose = 2)
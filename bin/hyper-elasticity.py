#!/usr/bin/env python3
"""
file   hyper-elasticity.py

@author Till Junge <till.junge@epfl.ch>

@date   16 Jan 2018

@brief  Recreation of GooseFFT's hyper-elasticity.py calculation

@section LICENCE

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

import sys
import os
import numpy as np
import argparse

sys.path.append(os.path.join(os.getcwd(), "language_bindings/python"))
import pyMuSpectre as µ

def parse_args():
    parser = argparse.ArgumentParser(description='hyper-elastic example')
    parser.add_argument("prefactor", metavar="fac", type=float,
                        help="factor")
    args = parser.parse_args()
    if args.prefactor <= 0:
        parser.error("prefactor must be a positive and non-zero float")
    return args


def compute(ex):
    N = [11, 11, 11]
    lens = [1., 1., 1.]
    incl_size = 3

    cell = µ.SystemFactory(N,
                           lens,
                           µ.Formulation.finite_strain)
    hard = µ.material.MaterialHooke3d.make(cell, "hard",
                                           210.*ex, .33)
    soft = µ.material.MaterialHooke3d.make(cell, "soft",
                                            70.*ex, .33)
    for  pixel in cell:
        if ((pixel[0] >= N[0]-incl_size) and
            (pixel[1] < incl_size) and
            (pixel[2] >= N[2]-incl_size)):
            hard.add_pixel(pixel)
        else:
            soft.add_pixel(pixel)

    print("{} pixels in the inclusion".format(hard.size()))
    cell.initialise();
    cg_tol, newton_tol = 1e-8, 1e-5
    maxiter = 200
    verbose = 1
    dF_bar = np.array([[0, 1., 0], [0, 0, 0], [0, 0, 0]])

    optimize_res = µ.solvers.de_geus(
        cell, dF_bar, cg_tol, newton_tol, maxiter, verbose)
    print("nb_cg: {}\n{}".format(optimize_res.nb_fev, optimize_res.grad.T[:2,:]))

def main():
    args = parse_args()
    compute(args.prefactor)

if __name__ == "__main__":
    main()

# !/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   compute_analytical_eshelby_solution.py

@author Richard Leute <richard.leute@imtek.uni-freiburg.de>

@date   12 Okt 2020

@brief  compute the stress and strain boundary conditions due to periodic
        eshelby inclusions by superposition of the analytic solution

Copyright © 2021 Till Junge

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


import numpy as np
import argparse
from helper_functions import look_up_eshelby_result, save_eshelby_results, \
     compute_eshelby_analytical, generate_eshelby_details_dic

parser = argparse.ArgumentParser()

parser.add_argument('-t', '--test_run',
                    action="store_true", dest="test_run",
                    help="run a small and fast test run which is checked "
                    "against values of an precomputed reference file "
                    "'file_name'. This is only needed for the automatic test "
                    "during running ctest.")
parser.add_argument('-s', '--silent',
                    action="store_true", dest="silent",
                    help="run the skript in silent mode with no verbosity.")

args = parser.parse_args()


# Dimensions of the grid of periodic Eshelby inhomogeneities to correct for PBC.
# Only odd numbers are allowed because pbc_grid counts the periodic cells, thus
# you always have to count the center cell and an even number would lead to an
# asymmetric constelation. If you dont want to correct for PBC set pbc_grid to
# pbc_grid = np.array([1, 1, 1])
if args.test_run:
    pbc_grid = np.array([1, 3, 3])
else:
    pbc_grid = np.array([1, 11, 11])

# grid resolution of the Eshelby computation
if args.test_run:
    nb_grid_pts = np.array([1, 11, 11], dtype=int)
else:
    nb_grid_pts = np.array([1, 151, 151], dtype=int)

# Domain dimensions
lengths = np.array([1, 100, 100])

# Macroscopic strain 1% (strain/stress at infinity)
eps_0 = np.array([[0.00, 0.00, 0.00],
                  [0.00, 0.01, 0.00],
                  [0.00, 0.00, 0.01]])

# Principal half axes of the ellipse a>b>c
a = 10000
b = 10.00001
c = 10.00000

# Youngs modulus and Poisson number of the matrix (E,nu)
# and the inhomogeneity (E_I, nu_I)
E = 1.
nu = 0.33
E_I = 0.1
nu_I = 0.33

eps_p_0 = None  # eigen strain of the inhomogeneity


# --- analytical computation or read in of a stored result --- #
eshelby_details = generate_eshelby_details_dic(nb_grid_pts, lengths, a, b, c,
                                               E, nu, E_I, nu_I, eps_0, eps_p_0,
                                               pbc_grid)

# check if you want to do an Eshelby computation or if there are stored results
stored_esh_res = look_up_eshelby_result(eshelby_details,
                                        folder="./stored_eshelby_results/")
if stored_esh_res is not None:
    if not args.silent:
        print("load stored Eshelby results from {}".format(stored_esh_res))
    ser = np.load(stored_esh_res, allow_pickle=True)
    if not args.silent:
        print("... loaded results.\n")
    esh_stress = ser["eshelby_stress"]
    esh_strain = ser["eshelby_strain"]

elif stored_esh_res is None:
    esh_stress, esh_strain = compute_eshelby_analytical(
        eshelby_details, pbc_grid, verbose=not args.silent)
    save_eshelby_results(eshelby_details, esh_stress, esh_strain,
                         results_folder="./stored_eshelby_results/")

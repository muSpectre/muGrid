# !/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   test_analytical_eshelby_solution.py

@author Richard Leute <richard.leute@imtek.uni-freiburg.de>

@date   03 Mai 2021

@brief  run the script 'compute_analytical_eshelby_solution.py' to keep it up
        to date and test its computed result against a reference file.

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

import os
import subprocess
import numpy as np

from helper_functions import find_eshelby_number, compare_dictionaries

# run the script 'compute_analytical_eshelby_solution.py'
n = find_eshelby_number(folder="./stored_eshelby_results/")
out = subprocess.call(["python3",
                       "./compute_analytical_eshelby_solution.py",
                       "--test_run", "--silent"])

# read in the wrote file and compare it against the reference ''
f_name = "./stored_eshelby_results/analytical_eshelby_solution_"+str(n)+".npz"
r_name = "./stored_eshelby_results/analytical_eshelby_reference.npz"

eshelby_solution = np.load(f_name, allow_pickle=True)
eshelby_reference = np.load(r_name, allow_pickle=True)

# compare Eshelby details
correct_dic = compare_dictionaries(eshelby_solution["eshelby_details"][()],
                                   eshelby_reference["eshelby_details"][()],
                                   verbose=False)
if not correct_dic:
    raise RuntimeError("The script 'compute_analytical_eshelby_solution.py' "
                       "does not produce the same results as stored in the "
                       "reference file. They differ in the 'eshelby_details' "
                       "dictionary.")

correct_strain = np.allclose(eshelby_solution["eshelby_strain"],
                             eshelby_reference["eshelby_strain"],
                             rtol=1e-14, atol=0)
if not correct_strain:
    raise RuntimeError("The script 'compute_analytical_eshelby_solution.py' "
                       "does not produce the same results as stored in the "
                       "reference file. They differ in the computed eshelby "
                       "strain.")

correct_stress = np.allclose(eshelby_solution["eshelby_stress"],
                             eshelby_reference["eshelby_stress"],
                             rtol=1e-14, atol=0)
if not correct_stress:
    raise RuntimeError("The script 'compute_analytical_eshelby_solution.py' "
                       "does not produce the same results as stored in the "
                       "reference file. They differ in the computed eshelby "
                       "stress.")

# delete the computed eshelby solution to prevent a mess of files if this test
# is run several times
if os.path.exists(f_name):
    os.remove(f_name)

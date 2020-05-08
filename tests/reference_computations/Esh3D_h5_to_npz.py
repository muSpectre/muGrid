# !/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   Esh3D_h5_to_npz.py

@author Richard Leute <richard.leute@imtek.uni-freiburg.de>

@date   14 Apr 2020

@brief  convert the *.h5 files produced by Esh3d into *.npz files to make them
        readable without the hdf5 package
        $ python3 Esh3D_h5_to_npz.py

Copyright © 2020 Till Junge

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
import h5py
import os

# inclusion
reference = h5py.File("eshelby_inclusion.ref.h5", 'r')
ellip = reference["ellip"][0]
ocoord = reference["ocoord"][:]
odat = reference["odat"]
np.savez("eshelby_inclusion.ref.npz", ellip=ellip, ocoord=ocoord, odat=odat)

# inhomogeneity
reference = h5py.File("eshelby_inhomogeneity.ref.h5", 'r')
ellip = reference["ellip"][0]
ocoord = reference["ocoord"][:]
odat = reference["odat"]
np.savez("eshelby_inhomogeneity.ref.npz",
         ellip=ellip, ocoord=ocoord, odat=odat)


# delete in- and out-put files from Esh3D
def silent_remove(filename):
    try:
        os.remove(filename)
    except OSError:
        pass


silent_remove("eshelby_inclusion.ref.h5")
silent_remove("eshelby_inhomogeneity.ref.h5")
silent_remove("eshelby_inclusion.ref.inp")
silent_remove("eshelby_inhomogeneity.ref.inp")

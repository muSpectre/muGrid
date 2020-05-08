#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   eshelby_inclusion_Esh3D.py

@author Richard Leute <richard.leute@imtek.uni-freiburg.de>

@date   09 Apr 2020

@brief  File to compute the reference eshelby inclusion data "" with the code
        Esh3D by Chunfang Meng (https://github.com/Chunfang/Esh3D).
        Meng, C., W. Heltsley, and D. Pollard, 2012, "Evaluation of the Eshelby
        Solution for the Ellipsoidal Inclusion and Heterogeneity", Computers &
        Geosciences, Vols. 40, pp. 40-48
        The file is inspired by the example skript "esh3d.py" of the Esh3D code.

        compute the results with:
        $ python3 eshelby_inclusion_Esh3D.ref.py
        $ ./esh3d -f eshelby_inclusion.ref.inp
        $ ./esh3d -f eshelby_inhomogeneity.ref.inp
        where ./esh3d is the binary executable of Esh3D git hash

        convert the *.h5 files into *.npz files to make them readable without
        the hdf5 package
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
import os

# ESHELBY INCLUSION #
fout = "eshelby_inclusion.ref.inp"
E = 210
nu = 0.3
# remote stress

# ellip(nellip,15): 1-3 ellipsoid centroid coordinate, 4-6 semi-axises, 7-9
# rotation angles around x,y and z axises, 10-15 eigen strain
ellip = np.array([[0., 0., 0.,
                   1., 0.75, 0.5,
                   0., 0., 0.,
                   3e-3, 1e-3, 1.5e-3, 2e-3, 0., 5e-4]],
                 dtype=np.float64)

# test positions (random positions in a box from x,y,z ∈ {-L/2,L/2})
lengths = np.array([4, 4, 4])  # Lx, Ly, Lz
np.random.seed(20200409)  # 9th of March 2020
x_ = (np.random.random((20, 3))-0.5) * (lengths/2)

if os.path.isfile(fout):
    os.remove(fout)
f = open(fout, 'a')
np.savetxt(f, ['full-incl'], fmt='%s')
np.savetxt(f, np.array([[len(ellip), len(x_)]], dtype=np.float64), fmt='%d '*2)
np.savetxt(f, np.array([[E, nu]], dtype=np.float64), fmt='%g '*2)
np.savetxt(f, ellip, delimiter=' ', fmt='%.17f '*15)
np.savetxt(f, x_, delimiter=' ', fmt='%g '*3)
f.close()


# ESHELBY INHOMOGENITY #
fout = "eshelby_inhomogeneity.ref.inp"
E = 210
nu = 0.3
# remote stress
rstress = np.array([[0.1, 0.05, 0.2, 0.15, 0.02, 0.08]], dtype=np.float64)

# ellip(nellip,17): 1-3 ellipsoid centroid coordinate, 4-6 semi-axises, 7-9
# rotation angles around x,y and z axises, 10,11 inclusion Young's modulus
# and Poisson's ratio, 12-17 eigen strain
E_I = 2*E
nu_I = 0.3
ellip = np.array([[1., -0.5, 0.5,
                   1., 0.75, 0.5,
                   0., 0., 0.,
                   E_I, nu_I,
                   2e-3, 1.5e-3, 1e-3, 4e-3, 1.2e-3, 2e-4]],
                 dtype=np.float64)

# test positions (random positions in a box from x,y,z ∈ {-L/2,L/2})
lengths = np.array([4, 4, 4])  # Lx, Ly, Lz
np.random.seed(20200408)  # 8th of March 2020
x_ = (np.random.random((20, 3))-0.5) * (lengths/2)

if os.path.isfile(fout):
    os.remove(fout)
f = open(fout, 'a')
np.savetxt(f, ['full-inho'], fmt='%s')
np.savetxt(f, np.array([[len(ellip), len(x_)]], dtype=np.float64), fmt='%d '*2)
np.savetxt(f, np.array([[E, nu]], dtype=np.float64), fmt='%g '*2)
np.savetxt(f, rstress, fmt='%g '*6)
np.savetxt(f, ellip, delimiter=' ', fmt='%.17f '*17)
np.savetxt(f, x_, delimiter=' ', fmt='%g '*3)
f.close()

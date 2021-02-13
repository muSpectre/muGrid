#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   linear_finite_elements.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   10 Feb 2021

@brief  Functions for the integration of periodic first- and second-rank
        tensor fields on an n-dimensional rectangular grid

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

import numpy as np

import muFFT.Stencils2D
import muFFT.Stencils3D

from .gradient_integration import get_complemented_positions


# Linear finite elements in 2D (each pixel is subdivided into two triangles)
gradient_2d = muFFT.Stencils2D.linear_finite_elements


# Linear finite elements in 3D (each voxel is subdivided into six tetrahedra)
gradient_3d = muFFT.Stencils3D.linear_finite_elements


def write_3d(fn, rve):
    """
    Write results of a 3D calculation that employs a decomposition of each
    voxel in six tetrahedra (using the `gradient_3d` stencil) to a file. The
    output is handled by `meshio`, which means all `meshio` formats are
    supported.

    More on `meshio` can be found here: https://github.com/nschloe/meshio

    fn  -- filename
    rve -- representative volume element, i.e. the `Cell` object
    """
    import meshio

    # Size of RVE
    nx, ny, nz = rve.nb_domain_grid_pts

    # Positions, periodically complemented
    x, y, z = get_complemented_positions(rve)

    # Global node indices
    def c2i(xp, yp, zp):
        return xp + (nx + 1) * (yp + (ny + 1) * zp)

    # Integer cell coordinates
    xc, yc, zc = np.mgrid[:nx, :ny, :nz]
    xc = xc.ravel(order='F')
    yc = yc.ravel(order='F')
    zc = zc.ravel(order='F')

    x = x.ravel(order='F')
    y = y.ravel(order='F')
    z = z.ravel(order='F')

    # Construct mesh
    points = np.transpose([x, y, z])
    cells = np.swapaxes(
        [[c2i(xc, yc, zc), c2i(xc+1, yc, zc),
          c2i(xc+1, yc+1, zc), c2i(xc+1, yc+1, zc+1)],
         [c2i(xc, yc, zc), c2i(xc+1, yc, zc),
          c2i(xc+1, yc, zc+1), c2i(xc+1, yc+1, zc+1)],
         [c2i(xc, yc, zc), c2i(xc, yc+1, zc),
          c2i(xc+1, yc+1, zc), c2i(xc+1, yc+1, zc+1)],
         [c2i(xc, yc, zc), c2i(xc, yc+1, zc),
          c2i(xc, yc+1, zc+1), c2i(xc+1, yc+1, zc+1)],
         [c2i(xc, yc, zc), c2i(xc, yc, zc+1),
          c2i(xc+1, yc, zc+1), c2i(xc+1, yc+1, zc+1)],
         [c2i(xc, yc, zc), c2i(xc, yc, zc+1),
          c2i(xc, yc+1, zc+1), c2i(xc+1, yc+1, zc+1)]],
        0, 1)

    cells = cells.reshape((4, -1), order='F').T

    # Get stress
    stress = rve.stress.array() \
        .reshape((3, 3, -1), order='F').T.swapaxes(1, 2)

    # Write mesh to file
    meshio.write_points_cells(
        fn,
        points,
        {'tetra': cells},
        cell_data={
            'stress': np.array([stress])
        })

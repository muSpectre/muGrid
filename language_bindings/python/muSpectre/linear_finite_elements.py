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
from .gradient_integration import get_complemented_positions_class_solver
from .gradient_integration import get_complemented_positions_fem


# Linear finite elements in 2D (each pixel is subdivided into two triangles)
gradient_2d = muFFT.Stencils2D.linear_finite_elements
gradient_2d_hexagonal = muFFT.Stencils2D.hexagonal_linear_finite_elements


# Linear finite elements in 3D (each voxel is subdivided into six tetrahedra)
gradient_3d = muFFT.Stencils3D.linear_finite_elements


def write_3d(file_name, rve, cell_data=None, point_data=None,
             F0=np.eye(3), displacement_field=False):
    """
    Write results of a 3D calculation that employs a decomposition of each
    voxel in six tetrahedra (using the `gradient_3d` stencil) to a file. The
    output is handled by `meshio`, which means all `meshio` formats are
    supported.

    More on `meshio` can be found here: https://github.com/nschloe/meshio

    file_name  -- filename
    rve        -- representative volume element, i.e. the `Cell` object
    cell_data  -- dictionary of cell data with corresponding keys
    point_data -- dictionary of point data with corresponding keys. If
                  displacement_field=True the key 'displacement' is reserved
                  for the displacement field and cannot be used.
    F0 -- F0 describes an affine deformation of the undeformed grid i.e.
            * rectangular grid: F0=np.eye(3) which is the default case and
                                corresponds to the undeformed grid.
    displacement_field -- False (default): The deformed structure is stored.
                          True: The undeformed structure and the displacement
                                field are stored. You can get the deformed
                                structure by adding the displacement field to
                                the undeformed structure. E.g. in ParaView you
                                can use the function 'Warp By Vector' (We
                                recommend the Xdmf3ReaderT, ParaView 5.7.0).
    """
    import meshio
    # Size of RVE
    nx, ny, nz = rve.nb_domain_grid_pts

    # Positions, periodically complemented
    if displacement_field:
        [x_0, y_0, z_0], [x_displ, y_displ, z_displ] \
            = get_complemented_positions("0d", rve, F0)
        x_0 = x_0.ravel(order='F')
        y_0 = y_0.ravel(order='F')
        z_0 = z_0.ravel(order='F')
        points = np.transpose([x_0, y_0, z_0])

        # Get displacements
        x_displ = x_displ.ravel(order='F')
        y_displ = y_displ.ravel(order='F')
        z_displ = z_displ.ravel(order='F')
        displacement = np.transpose([x_displ, y_displ, z_displ])
        if point_data is not None:
            if type(point_data) is not dict:
                raise RuntimeError("'point_data' has to be a dictionary but is"
                                   " of type '" + str(type(point_data)) + "'.")
            if 'displacement' in point_data.keys():
                raise RuntimeError("For displacement_field=True the point_data"
                                   " key 'displacement' is already reserved for"
                                   " the displacement field. Please use an"
                                   " other key for your field.")
            else:
                point_data['displacement'] = displacement
        else:
            point_data = {'displacement': displacement}
    else:
        [x_def, y_def, z_def] = get_complemented_positions("p", rve, F0)
        x_def = x_def.ravel(order='F')
        y_def = y_def.ravel(order='F')
        z_def = z_def.ravel(order='F')
        points = np.transpose([x_def, y_def, z_def])
        if point_data is None:
            point_data = {}

    # Global node indices
    def c2i(xp, yp, zp):
        return xp + (nx + 1) * (yp + (ny + 1) * zp)

    # Integer cell coordinates
    xc, yc, zc = np.mgrid[:nx, :ny, :nz]
    xc = xc.ravel(order='F')
    yc = yc.ravel(order='F')
    zc = zc.ravel(order='F')

    # Construct mesh
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

    # Get stress and strain
    stress = rve.stress.array() \
        .reshape((3, 3, -1), order='F').T.swapaxes(1, 2)
    strain = rve.strain.array() \
        .reshape((3, 3, -1), order='F').T.swapaxes(1, 2)

    # Write mesh to file
    if cell_data is None:
        meshio.write_points_cells(
            file_name,
            points,
            {"tetra": cells},
            point_data=point_data,
            cell_data={
                'stress': np.array([stress]),
                'strain': np.array([strain])
            })
    else:
        meshio.write_points_cells(
            file_name,
            points,
            {"tetra": cells},
            point_data=point_data,
            cell_data=cell_data
        )


def write_2d(file_name, rve, cell_data=None, point_data=None,
             F0=np.eye(2), displacement_field=False):
    """
    Write results of a 2D calculation that employs a decomposition of each
    voxel in two triangles (using the `gradient_2d` stencil) to a file. The
    output is handled by `meshio`, which means all `meshio` formats are
    supported.

    More on `meshio` can be found here: https://github.com/nschloe/meshio

    file_name  -- filename
    rve        -- representative volume element, i.e. the `Cell` object
    cell_data  -- dictionary of cell data with corresponding keys
    point_data -- dictionary of point data with corresponding keys. If
                  displacement_field=True the key 'displacement' is reserved
                  for the displacement field and cannot be used.
    F0 -- F0 describes an affine deformation of the undeformed grid i.e.
            * rectangular grid: F0=np.eye(2) which is the default case and
                                corresponds to the undeformed grid.
            * hexagonal grid: with dy = sqrt(3)/2*dx
                              F0 = np.array([[ 1, 1/sqrt(3)],
                                             [ 0,     1    ]])
    displacement_field -- False (default): The deformed structure is stored.
                          True: The undeformed structure and the displacement
                                field are stored. You can get the deformed
                                structure by adding the displacement field to
                                the undeformed structure. E.g. in ParaView you
                                can use the function 'Warp By Vector' (We
                                recommend the Xdmf3ReaderT, ParaView 5.7.0).
    """
    import meshio
    # Size of RVE
    nx, ny = rve.nb_domain_grid_pts

    # Positions, periodically complemented
    if displacement_field:
        [x_0, y_0], [x_displ, y_displ] \
            = get_complemented_positions("0d", rve, F0)
        x_0 = x_0.ravel(order='F')
        y_0 = y_0.ravel(order='F')
        points = np.transpose([x_0, y_0])

        # Get displacements
        x_displ = x_displ.ravel(order='F')
        y_displ = y_displ.ravel(order='F')
        z_displ = 0 * np.copy(x_displ)
        displacement = np.transpose([x_displ, y_displ, z_displ])
        if point_data is not None:
            if type(point_data) is not dict:
                raise RuntimeError("'point_data' has to be a dictionary but is"
                                   " of type '" + str(type(point_data)) + "'.")
            if 'displacement' in point_data.keys():
                raise RuntimeError("For displacement_field=True the point_data"
                                   " key 'displacement' is already reserved for"
                                   " the displacement field. Please use an"
                                   " other key for your field.")
            else:
                point_data['displacement'] = displacement
        else:
            point_data = {'displacement': displacement}
    else:
        [x_def, y_def] = get_complemented_positions("p", rve, F0)
        x_def = x_def.ravel(order='F')
        y_def = y_def.ravel(order='F')
        points = np.transpose([x_def, y_def])
        if point_data is None:
            point_data = {}

    # Global node indices
    def c2i(xp, yp):
        return xp + (nx + 1) * yp

    # Integer cell coordinates
    xc, yc = np.mgrid[:nx, :ny]
    xc = xc.ravel(order='F')
    yc = yc.ravel(order='F')

    # Construct mesh
    cells = np.swapaxes(
        [[c2i(xc, yc), c2i(xc+1, yc), c2i(xc, yc+1)],
         [c2i(xc, yc+1), c2i(xc+1, yc), c2i(xc+1, yc+1)]],
        0, 1)

    cells = cells.reshape((3, -1), order='F').T

    # Get stress and strain
    stress = rve.stress.array() \
        .reshape((2, 2, -1), order='F').T.swapaxes(1, 2)
    strain = rve.strain.array() \
        .reshape((2, 2, -1), order='F').T.swapaxes(1, 2)

    # Write mesh to file
    if cell_data is None:
        meshio.write_points_cells(
            file_name,
            points,
            {"triangle": cells},
            point_data=point_data,
            cell_data={
                'stress': np.array([stress]),
                'strain': np.array([strain])
            })
    else:
        meshio.write_points_cells(
            file_name,
            points,
            {"triangle": cells},
            point_data=point_data,
            cell_data=cell_data
        )


def write_2d_class(file_name, rve, solver, cell_data=None):
    """
    Write results of a 2D calculation that employs a decomposition of each
    voxel in two triangles (using the `gradient_2d` stencil) to a file. The
    output is handled by `meshio`, which means all `meshio` formats are
    supported.

    More on `meshio` can be found here: https://github.com/nschloe/meshio

    file_name  -- filename
    rve -- representative volume element, i.e. the `CellData` object
    solver -- The solver used to solve the CellData with applied load
    cell_data = dictionary of cell data with corresponding keys
    """
    import meshio
    # Size of RVE
    nx, ny = rve.nb_domain_grid_pts

    # Positions, periodically complemented
    [x_def, y_def], [x_undef, y_undef] \
        = get_complemented_positions_class_solver("pd", rve=rve,
                                                  solver=solver)

    # Global node indices
    def c2i(xp, yp):
        return xp + (nx + 1) * yp

    # Integer cell coordinates
    xc, yc = np.mgrid[:nx, :ny]
    xc = xc.ravel(order='F')
    yc = yc.ravel(order='F')

    x_def = x_def.ravel(order='F')
    y_def = y_def.ravel(order='F')

    # Construct mesh
    points = np.transpose([x_def,
                           y_def])
    cells = np.swapaxes(
        [[c2i(xc, yc), c2i(xc+1, yc), c2i(xc, yc+1)],
         [c2i(xc, yc+1), c2i(xc+1, yc), c2i(xc+1, yc+1)]],
        0, 1)
    cells = cells.reshape((3, -1), order='F').T
    # Get Stress
    stress = solver.flux.array() \
        .reshape((2, 2, -1), order='F').T.swapaxes(1, 2)
    # Get Strain
    strain = solver.grad.array() \
        .reshape((2, 2, -1), order='F').T.swapaxes(1, 2)

    if cell_data is None:
        # Write mesh to file
        meshio.write_points_cells(
            file_name,
            points,
            {"triangle": cells},
            cell_data={
                'stress': np.array([stress]),
                'strain': np.array([strain])
            })
    else:
        # Write mesh to file
        meshio.write_points_cells(
            file_name,
            points,
            {"triangle": cells},
            cell_data=cell_data
        )


def write_2d_fem(file_name, rve, solver, cell_data=None):
    """
    Write results of a 2D calculation that employs a decomposition of each
    voxel in two triangles to a file. The output is handled by `meshio`, which
    means all `meshio` formats are supported.

    More on `meshio` can be found here: https://github.com/nschloe/meshio

    file_name  -- filename
    rve -- representative volume element, i.e. the `CellData` object
    solver -- The solver used to solve the CellData with applied load
    cell_data = dictionary of cell data with corresponding keys
    """
    import meshio
    # Size of RVE
    nx, ny = rve.nb_domain_grid_pts

    # Positions, periodically complemented
    [x_def, y_def], [x_undef, y_undef] \
        = get_complemented_positions_fem("pd", rve=rve,
                                         solver=solver)

    # Global node indices
    def c2i(xp, yp):
        return xp + (nx + 1) * yp

    # Integer cell coordinates
    xc, yc = np.mgrid[:nx, :ny]
    xc = xc.ravel(order='F')
    yc = yc.ravel(order='F')

    x_def = x_def.ravel(order='F')
    y_def = y_def.ravel(order='F')

    # Construct mesh
    points = np.transpose([x_def,
                           y_def])
    cells = np.swapaxes(
        [[c2i(xc, yc), c2i(xc+1, yc), c2i(xc, yc+1)],
         [c2i(xc, yc+1), c2i(xc+1, yc), c2i(xc+1, yc+1)]],
        0, 1)
    cells = cells.reshape((3, -1), order='F').T
    # Get Stress
    stress = solver.flux.field.array() \
                              .reshape((2, 2, -1), order='F').T.swapaxes(1, 2)
    # Get Strain
    strain = solver.grad.array() \
                        .reshape((2, 2, -1), order='F').T.swapaxes(1, 2)

    if cell_data is None:
        # Write mesh to file
        meshio.write_points_cells(
            file_name, points,
            {"triangle": cells},
            cell_data={
                'stress': np.array([stress]),
                'strain': np.array([strain])
            })
    else:
        # Write mesh to file
        meshio.write_points_cells(
            file_name,
            points,
            {"triangle": cells},
            cell_data=cell_data
        )

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   vtk_export.py

@author Till Junge <till.junge@epfl.ch>

@date   22 Nov 2018

@brief  function for export of vtk files

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
from uvw import RectilinearGrid, DataArray


def vtk_export(fpath, x_n, placement, point_data=None, cell_data=None):
    """write a vtr file for visualisation of µSpectre results.

    Keyword Arguments:
    fpath        -- file name for output path *WITHOUT EXTENSION*
    x_n          -- nodal positions, as computed by
                    gradient_integration.make_grid or
                    gradient_integration.compute_placement
    placement    -- nodal deformed placement as computed by
                    gradient_integration.compute_placement
    point_data   -- (default None) dictionary of point data. These must have
                    either of the following shapes:
                    a) tuple(x_n.shape[1:])
                       interpreted as scalar field,
                    b) tuple([dim] + x_n.shape[1:])
                       interpreted as vector field, or
                    c) tuple([dim, dim] + x_n.shape[1:])
                       interpreted as second-rank tensor field
    cell_data    -- (default None) dictionary of cell data. These must have
                    either of the following shapes:
                    a) tuple(x_c.shape[1:])
                       interpreted as scalar field,
                    b) tuple([dim] + x_c.shape[1:])
                       interpreted as vector field, or
                    c) tuple([dim, dim] + x_c.shape[1:])
                       interpreted as second-rank tensor field
                    where x_c is the array of pixel/voxel positions as computed
                    by gradient_integration.make_grid

    Returns:
    uvw object with informations about the written vtr file.
    """
    dim = len(x_n.shape[:-1])
    if dim not in (2, 3):
        raise Exception(
            ("should be two- or three-dimensional, got positions for a {}-"
             "dimensional problem").format(dim))
    res_n = list(x_n.shape[1:])
    vtk_res = res_n if dim == 3 else res_n + [1]
    res_c = [max(1, r-1) for r in res_n]

    # setting up the geometric grid
    x_coordinates = x_n[0, :, 0] if dim == 2 else x_n[0, :, 0, 0]
    y_coordinates = x_n[1, 0, :] if dim == 2 else x_n[1, 0, :, 0]
    z_coordinates = np.zeros_like([1]) if dim == 2 else x_n[2, 0, 0, :]
    path = fpath + ".vtr"
    uvw_obj = RectilinearGrid(path, [np.copy(x_coordinates),
                                     np.copy(y_coordinates),
                                     np.copy(z_coordinates)])

    # displacements are mandatory, so they get added independently of
    # any additional point data
    disp = np.zeros([3] + vtk_res)
    disp[:dim, ...] = (placement - x_n).reshape((dim,)+tuple(vtk_res))
    uvw_obj.addPointData(DataArray(disp, np.arange(1, dim+1), "displacement"))

    # check name clashes:
    if point_data is None:
        point_data = dict()
    if cell_data is None:
        cell_data = dict()
    if "displacement" in point_data.keys():
        raise Exception("Name 'displacement' is reserved")
    if "displacement" in cell_data.keys():
        raise Exception("Name 'displacement' is reserved")

    # helper functions to add data
    def add_data(data_array, point=True):
        if point:
            uvw_obj.addPointData(data_array)
        elif not point:
            uvw_obj.addCellData(data_array)

    def add_scalar(value, name, point=True):
        data_array = DataArray(value, np.arange(0, len(value.shape)), name)
        add_data(data_array, point)

    def add_vector(value, name, point=True):
        data_array = DataArray(value, np.arange(1, len(value.shape)), name)
        add_data(data_array, point)

    def add_tensor(value, name, point=True):
        data_array = DataArray(value, np.arange(2, len(value.shape)), name)
        add_data(data_array, point)

    adders = {(): add_scalar,
              (dim,): add_vector,
              (dim, dim): add_tensor}

    def shape_checker(value, reference):
        """
        checks whether values have the right shape and determines the
        appropriate function to add them to the output file
        """
        res = value.shape[-dim:]
        shape = tuple(value.shape[:-dim])
        if not res == tuple(reference[:dim]):
            raise Exception(
                ("The last {} dimensions of dataset '{}' have the wrong size,"
                 " should be {}, but got {}").format(
                     dim, key, reference[:dim], res))
        if not shape in (adders.keys()):
            raise Exception(
                ("Can only handle scalar [{}], vectorial [{}],"
                 " or second order tensorial fields [{},{}], but "
                 "got a field of shape {}").format(1, dim, dim, dim, shape))
        return res, shape, adders[shape]

    # add point data
    for key, value in point_data.items():
        res, shape, adder = shape_checker(value, res_n)
        adder(value, key, point=True)

    # add cell data
    for key, value in cell_data.items():
        res, shape, adder = shape_checker(value, res_c)
        adder(value, key, point=False)

    uvw_obj.write()
    return uvw_obj

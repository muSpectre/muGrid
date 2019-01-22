#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   vtk_export.py

@author Till Junge <till.junge@epfl.ch>

@date   22 Nov 2018

@brief  function for export of vtk files

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

import numpy as np
from tvtk.api import tvtk, write_data

def vtk_export(fpath, x_n, placement, point_data=None, cell_data=None,
               legacy_ascii=False):
    """write a vtr (vtk for rectilinear grids) file for visualisation of
    µSpectre results. Optionally, a legacy ascii file can be written
    (useful for debugging)

    Keyword Arguments:
    fpath        -- file name for output path *WITHOUT EXTENSION*
    x_n          -- nodal positions, as computed by
                    gradient_integration.compute_grid
    placement    -- nodal deformed placement as computed by
                    gradient_integration.compute_placement
    point_data   -- (default None) dictionary of point data. These must have
                    either of the following shapes:
                    a) tuple(x_n.shape[:dim])
                       interpreted as scalar field,
                    b) tuple(x_n.shape[:dim] + [dim])
                       interpreted as vector field, or
                    c) tuple(x_n.shape[:dim] + [dim, dim])
                       interpreted as second-rank tensor field
    cell_data    -- (default None) dictionary of cell data. These must have
                    either of the following shapes:
                    a) tuple(x_c.shape[:dim])
                       interpreted as scalar field,
                    b) tuple(x_c.shape[:dim] + [dim])
                       interpreted as vector field, or
                    c) tuple(x_c.shape[:dim] + [dim, dim])
                       interpreted as second-rank tensor field
                    where x_c is the array of pixel/voxel positions as computed
                    by gradient_integration.compute_grid
    legacy_ascii -- (default False) If set to True, a human-readable, but larger
                     ascii file is written

    """
    dim = len(x_n.shape[:-1])
    if dim not in (2, 3):
        raise Exception(
            ("should be two- or three-dimensional, got positions for a {}-"
             "dimensional problem").format(dim))
    res_n = list(x_n.shape[:-1])
    vtk_res = res_n if dim == 3 else res_n + [1]
    res_c = [max(1, r-1) for r in res_n]

    # setting up the geometric grid
    vtk_obj = tvtk.RectilinearGrid()
    vtk_obj.dimensions = vtk_res
    vtk_obj.x_coordinates = x_n[:,0,0] if dim == 2 else x_n[:,0,0,0]
    vtk_obj.y_coordinates = x_n[0,:,1] if dim == 2 else x_n[0,:,0,1]
    vtk_obj.z_coordinates = np.zeros_like(
        vtk_obj.x_coordinates) if dim == 2 else x_n[0,0,:,2]

    # displacements are mandatory, so they get added independently of
    # any additional point data
    disp = np.zeros([np.prod(vtk_res), 3])
    disp[:,:dim] = (placement - x_n).reshape(-1, dim, order="F")
    vtk_obj.point_data.vectors = disp
    vtk_obj.point_data.vectors.name = "displacement"

    # check name clashes:
    if point_data is None:
        point_data = dict()
    if cell_data is None:
        cell_data = dict()
    if "displacement" in point_data.keys():
        raise Exception("Name 'displacement' is reserved")
    if "displacement" in cell_data.keys():
        raise Exception("Name 'displacement' is reserved")
    #clash_set = set(point_data.keys()) & set(cell_data.keys())
    #if clash_set:
    #    clash_names = ["'{}'".format(name) for name in clash_set]
    #    clash_names_fmt = ", ".join(clash_names)
    #    raise Exception(
    #        ("Only unique names are allowed, but the names {} appear in both "
    #         "point- and cell data").format(clash_names_fmt))
    #Note: is this necessary -> only same name in same dictionary doesn't make sense.
    #      I think it's ok to have e.g. the material phase as cell and point data?
    #Richard: I think the same, so I commented the check and would throw it away

    # helper functions to add data
    def add_data(value, name, point=True):
        data = vtk_obj.point_data if point else vtk_obj.cell_data
        data.add_array(value)
        data.get_array(data._get_number_of_arrays()-1).name = name
        return

    def add_scalar(value, name, point=True):
        add_data(value.reshape(-1, order="F"), name, point)
        return

    def add_vector(value, name, point=True):
        res = vtk_res if point else res_c
        vec = np.zeros([np.prod(res), 3])
        vec[:,:dim] = value.reshape(-1, dim, order="F")
        add_data(vec, name, point)
        return

    def add_tensor(value, name, point=True):
        res = vtk_res if point else res_c
        tens = np.zeros([np.prod(res), 3, 3])
        tens[:,:dim, :dim] = value.reshape(-1, dim, dim, order="F")
        add_data(tens.reshape(-1, 3*3), name, point)
        return

    adders = {()        : add_scalar,
              (dim,)    : add_vector,
              (dim, dim): add_tensor}

    def shape_checker(value, reference):
        """
        checks whether values have the right shape and determines the
        appropriate function to add them to the output file
        """
        res = value.shape[:dim]
        shape = value.shape[dim:]
        if not res == tuple(reference[:dim]):
            raise Exception(
                ("the first dim dimensions of dataset '{}' have the wrong size,"
                 " should be {}, but got {}").format(
                     key, reference[:dim], res))
        if not shape in (adders.keys()):
            raise Exception(
                ("Can only handle scalar, vectorial, and tensorial fields, but "
                 "got a field of shape {}").format(shape))
        return res, shape, adders[shape]

    # add point data
    for key, value in point_data.items():
        res, shape, adder = shape_checker(value, res_n)
        adder(value, key, point=True)

    # add cell data
    for key, value in cell_data.items():
        res, shape, adder = shape_checker(value, res_c)
        adder(value, key, point=False)

    path = fpath + (".vtk" if legacy_ascii else "")

    write_data(vtk_obj, path)
    return vtk_obj

# !/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   helper_functions.py

@author Richard Leute <richard.leute@imtek.uni-freiburg.de>

@date   29 Apr 2021

@brief  helper functions used in the paper_example scripts

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
import os
import glob
import itertools
from python_paper_example_imports import muSpectre as msp

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    # use Triangulation for triangles
    from matplotlib.tri import Triangulation
    # use patches.Polygon for squares
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    matplotlib_found = True

except ImportError:
    matplotlib_found = False


# --- Helper functions for plotting --- #
if matplotlib_found:
    # --- configure matplotlib parameters --- #
    def update_rc_parameters():
        # I want to thank here my coollege Wolfram Georg Nöhring for the
        # sharing of his useful matplotlib parameter settings
        matplotlib.rcParams.update(
            {'figure.subplot.left': 0.17,
             'figure.subplot.right': 0.9,
             'figure.subplot.bottom': 0.18,
             'figure.subplot.top': 0.88,
             'legend.labelspacing': 0.25,
             'legend.borderpad': 0.25,
             'legend.borderaxespad': 0.25,
             'axes.titlesize': 10,
             'axes.labelsize': 10,
             'axes.spines.top': False,
             'axes.spines.right': False,
             'lines.linewidth': 1,
             'lines.markersize': 2,
             'xtick.labelsize': 8,
             'ytick.labelsize': 8,
             'text.usetex': True,
             'font.family': "sans-serif",
             'font.sans-serif': "Arimo",
             'text.latex.preamble': r"\PassOptionsToPackage{sfdefault}{arimo}\usepackage{arimo}\usepackage{sansmath}\sansmath\usepackage{siunitx}",
             'mathtext.fontset': "dejavusans",
             'font.serif': "Tinos",
             'legend.fontsize': 8})

    # --- helper functions for plotting --- #
    # visualise with matplotlib triangles
    def make_triangles(displ):
        """
        (upper, lower, upper, lower ...)
        """
        displ_x, displ_y = displ
        nx, ny = displ_x.shape

        def i(x, y):
            return x + nx*y

        x, y = np.mgrid[:nx-1, :ny-1]
        upper_triangles = np.stack((i(x, y), i(x, y+1), i(x+1, y)), axis=2)
        lower_triangles = np.stack((i(x+1, y), i(x, y+1), i(x+1, y+1)), axis=2)

        # reorder triangles; (upper, lower, upper, lower, ...)
        triangles = np.hstack((upper_triangles.reshape(-1,3),
                               lower_triangles.reshape(-1,3))).reshape(-1,3)

        return Triangulation(
            displ_x.reshape(-1, order='f'),
            displ_y.reshape(-1, order='f'),
            triangles), triangles

    # visualise with matplotlib squares
    def make_squares(displ, cmap=matplotlib.cm.jet, lw=0.3, edge_c='k'):
        squares = []
        nx = displ.shape[1] - 1  # number of squares in x direction
        ny = displ.shape[2] - 1  # number of squares in y direction

        for x in range(nx):
            for y in range(ny):
                corner_1 = displ[:, x    , y    ]
                corner_2 = displ[:, x + 1, y    ]
                corner_3 = displ[:, x + 1, y + 1]
                corner_4 = displ[:, x    , y + 1]
                corners = np.stack([corner_1, corner_2, corner_3, corner_4],
                                   axis=1).T
                square = Polygon(corners, True)
                squares.append(square)

        return PatchCollection(squares, cmap=cmap,
                               linewidth=lw, edgecolor=edge_c)


# --- Helper functions to find plotting boundaries
def min_max_xy_vec(ndarray_data_dic):
    """
    for min max of placement dict in x and y
    """
    a_dic = ndarray_data_dic
    elements = a_dic.keys()
    x_min_l = []
    x_max_l = []
    y_min_l = []
    y_max_l = []

    for n in elements:
        ndarray = a_dic[n]
        shape = ndarray.shape
        if shape[0] == 2:
            tmp_x_min = np.amin(ndarray[0])
            tmp_x_max = np.amax(ndarray[0])
            tmp_y_min = np.amin(ndarray[1])
            tmp_y_max = np.amax(ndarray[1])
        else:
            raise ValueError("Wrong input in function "
                             "min_max_xy_vec(ndarray_data_dic)! The shape is "
                             + str(shape))
        x_min_l.append(tmp_x_min)
        x_max_l.append(tmp_x_max)
        y_min_l.append(tmp_y_min)
        y_max_l.append(tmp_y_max)

    x_min = min(x_min_l)
    x_max = max(x_max_l)
    y_min = min(y_min_l)
    y_max = max(y_max_l)

    return x_min, x_max, y_min, y_max


def min_max_xy_tens(ndarray_data_dic):
    """
    for min max of stress or strain tensor dict in xy direction
    """
    a_dic = ndarray_data_dic
    elements = a_dic.keys()
    xy_min_l = []
    xy_max_l = []

    for n in elements:
        ndarray = a_dic[n]
        shape = ndarray.shape
        if shape[1] == 2:
            # 1 quad point
            tmp_xy_min = np.amin(ndarray[0, 1])
            tmp_xy_max = np.amax(ndarray[0, 1])
        elif shape[1] == 4:
            # 2 quad points
            tmp_xy_min = min(np.amin(ndarray[0, 1]), np.amin(ndarray[0, 3]))
            tmp_xy_max = max(np.amax(ndarray[0, 1]), np.amax(ndarray[0, 3]))
        else:
            raise ValueError("Wrong input in function "
                             "min_max_xy_tens(ndarray_data_dic)! The shape is "
                             + str(shape))
        xy_min_l.append(tmp_xy_min)
        xy_max_l.append(tmp_xy_max)

    xy_min = min(xy_min_l)
    xy_max = max(xy_max_l)

    return xy_min, xy_max


# --- Helper functions for Eshelby example --- #
def find_eshelby_number(folder="./stored_eshelby_results/",
                        file_name="analytical_eshelby_solution_"):
    """
    return the next number of the eshelby storage file
    """
    storage_files = glob.glob(folder + file_name + "*.npz")
    if storage_files == []:
        n = 0
    else:
        # subtract prefix
        numbers = [f_name.replace(folder + file_name, "") for
                   f_name in storage_files]
        # subtract suffix
        numbers = [f_name.replace(".npz", "") for f_name in numbers]
        n = np.amax(np.array(numbers, dtype=int)) + 1

    return n


def compare_dictionaries(d1, d2, verbose=False):
    """
    compare d1 and d2 for same entries
    """
    same = False
    if len(d1.keys()) != len(d2.keys()):
        if verbose:
            print("Dictionaries have a different amount of keys!")
        return same

    common_keys = list(set(d1.keys()) & set(d2.keys()))
    if len(common_keys) != len(d1.keys()):
        if verbose:
            print("Dictionaries have partly different keys!")
        return same

    for k in common_keys:
        similar = d1[k] == d2[k]
        if type(similar) is not bool:
            similar = similar.all()
        if not similar and (d1[k].shape == d2[k].shape):
            # for float data one should check for close and not equal
            try:
                similar_close = np.allclose(d1[k], d2[k], rtol=1e-12, atol=0)
                similar = similar_close
            except TypeError:
                pass

        if not similar:
            if verbose:
                print("not equal in key '{}'".format(k))
            return same

    return True


def look_up_eshelby_result(eshelby_details,
                           folder="./stored_eshelby_results/",
                           file_name="analytical_eshelby_solution_",
                           verbose=False):
    """
    look for stored esheby results with same 'eshelby_details'
    """
    stored_eshelby_result = None

    # stored eshelby result candidates
    candidates = glob.glob(folder + file_name + "*.npz")
    for c in candidates:
        if verbose:
            print("\nI check ", os.path.basename(c))
        c_dic = np.load(c, allow_pickle=True)["eshelby_details"][()]
        same = compare_dictionaries(eshelby_details, c_dic, verbose)
        if same:
            return c

    return stored_eshelby_result


def generate_eshelby_details_dic(nb_grid_pts, lengths, a, b, c,
                                 E, nu, E_I, nu_I, eps_0, eps_p_0,
                                 pbc_grid=np.array([1, 1, 1])):
    if (pbc_grid % 2 == 0).any():
        raise ValueError("Only odd numbers are allowed for 'pbc_grid'."
                         "But your input was: " + str(pbc_grid))

    lens = lengths * pbc_grid
    nb_grid = nb_grid_pts * pbc_grid
    x_c = msp.gradient_integration.make_grid(lens, nb_grid)[1]
    # shift box by half length
    center_index = np.array([r//2 for r in nb_grid])
    center_shift = x_c[:, center_index[0], center_index[1], center_index[2]]\
        .reshape((3, 1, 1, 1))
    x_c = x_c - center_shift
    center_E = x_c[:, center_index[0], center_index[1], center_index[2]]

    # eshelby_details, dictionary describing all characteristics of an Eshelby
    # inclusion
    eshelby_details = {}
    eshelby_details["E"] = E
    eshelby_details["nu"] = nu
    eshelby_details["E_I"] = E_I
    eshelby_details["nu_I"] = nu_I
    eshelby_details["half_axes"] = np.array([a, b, c])
    eshelby_details["inclusion_center"] = center_E
    eshelby_details["external_strain"] = eps_0
    eshelby_details["eigen_strain"] = eps_p_0
    eshelby_details["nb_grid_pts"] = nb_grid_pts
    eshelby_details["lengths"] = lengths
    eshelby_details["grid_3-nx-ny-nz"] = x_c

    return eshelby_details


def compute_eshelby_analytical(eshelby_details,
                               pbc_grid=np.array([1, 1, 1]),
                               verbose=False):
    if (pbc_grid % 2 == 0).any():
        raise ValueError("Only odd numbers are allowed for 'pbc_grid'."
                         "But your input was: " + str(pbc_grid))

    E = eshelby_details["E"]
    nu = eshelby_details["nu"]
    E_I = eshelby_details["E_I"]
    nu_I = eshelby_details["nu_I"]
    a, b, c = eshelby_details["half_axes"]
    # center_E = eshelby_details["inclusion_center"]
    eps_0 = eshelby_details["external_strain"]
    eps_p_0 = eshelby_details["eigen_strain"]
    # nb_grid_pts = eshelby_details["nb_grid_pts"]
    # lengths = eshelby_details["lengths"]
    x_c = eshelby_details["grid_3-nx-ny-nz"]

    def compute_esh_point(pos):
        x, y, z = pos
        if (x/a)**2 + (y/b)**2 + (z/c)**2 <= 1:
            stress, strain = msp.eshelby_slow.get_stress_and_strain_in(
                E=E, nu=nu, E_I=E_I, nu_I=nu_I, a=a, b=b, c=c,
                eps_0=eps_0, eps_p_0=eps_p_0, return_eps_eq_eig=False)
        else:
            stress, strain = msp.eshelby_slow.get_stress_and_strain_out(
                x, y, z, E=E, nu=nu, E_I=E_I, nu_I=nu_I, a=a, b=b, c=c,
                eps_0=eps_0, eps_p_0=eps_p_0, return_eps_eq_eig=False)

        return stress, strain

    if verbose:
        nb_points = x_c.size // 3
        percent = nb_points // 100
        if percent == 0:
            percent = 1
        step = 0

    # ## compute esh solution
    esh_stress = np.empty(shape=(3,) + tuple(x_c.shape))
    esh_strain = np.empty(shape=(3,) + tuple(x_c.shape))

    if verbose:
        print("Compute Eshelby solution on the full domain")
        print("Properties of Eshelby problem:\n", eshelby_details)
    esh_stress = esh_stress.reshape(3, 3, -1)
    esh_strain = esh_strain.reshape(3, 3, -1)
    for i, pos in enumerate(x_c.reshape(3, -1).T):
        stress, strain = compute_esh_point(pos)
        esh_stress[:, :, i] = stress
        esh_strain[:, :, i] = strain
        if verbose:
            if step % percent == 0:
                print("{:>3.1f}%".format(step/percent))
            step += 1

    esh_stress = esh_stress.reshape((3,) + tuple(x_c.shape))
    esh_strain = esh_strain.reshape((3,) + tuple(x_c.shape))

    return esh_stress, esh_strain


def sum_up_PBC_eshelby(esh_strain, eps_0, nb_grid_pts, pbc_grid):
    # sum up periodic images
    nb_x, nb_y, nb_z = nb_grid_pts
    esh_strain_summed_up = np.zeros((3, 3) + tuple(nb_grid_pts))

    for nx, ny, nz in itertools.product(range(0, pbc_grid[0]),
                                        range(0, pbc_grid[1]),
                                        range(0, pbc_grid[2])):
        esh_strain_slice = esh_strain[:, :,
                                      nb_x * nx:nb_x * (nx + 1),
                                      nb_y * ny:nb_y * (ny + 1),
                                      nb_z * nz:nb_z * (nz + 1)]
        esh_strain_summed_up += esh_strain_slice

    # subtract strain at infinity for each added periodic image
    esh_strain_summed_up -= (eps_0 * (np.prod(pbc_grid) - 1))\
        [:, :, np.newaxis, np.newaxis, np.newaxis]
    average_summed_strain = np.average(esh_strain_summed_up, axis=(-3, -2, -1))

    return average_summed_strain, esh_strain_summed_up


def save_eshelby_results(eshelby_details, esh_stress, esh_strain,
                         results_folder="./stored_eshelby_results/",
                         results_file_name="analytical_eshelby_solution_"):
    # save eshelby computation with all conditions to later load and
    # prevent double computation
    n = find_eshelby_number(folder=results_folder)
    print("save the Eshelby results to\n" +
          results_folder + results_file_name + "{}.npz".format(n))
    np.savez(results_folder + results_file_name + "{}.npz".format(n),
             eshelby_details=eshelby_details,
             eshelby_stress=esh_stress,
             eshelby_strain=esh_strain,
             eshelby_displ=None)

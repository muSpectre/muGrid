#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   plot_tri.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
@author Ali Falsafi <ali.falsafi@epfl.ch>

@date   12 Mar 2021

@brief  Test muSpectre solution against traditional finite-elements

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

import sys
import numpy as np

def node_index(i, j, nb_grid_pts):
    """
    Turn node coordinates (i, j) into their global node index.

    Parameters
    ----------
    i : int
        x-coordinate (integer) of the node
    j : int
        y-coordinate (integer) of the node
    nb_grid_pts : tuple of ints
        Number of nodes in the Cartesian directions

    Returns
    -------
    g : int
        Global node index
    """
    Nx, Ny = nb_grid_pts
    return Ny*np.mod(i, Nx) + np.mod(j, Ny)


def derivative_matrix(dx, dy):
    """
    Compute matrix of shape function derivatives

    Parameters
    ----------
    dx : float
        Grid spacing in x-direction
    dy : float
        Grid spacing in y-direction
    """
    return np.array([[dy/dx, 1,      -dy/dx, 0, 0, -1],
                     [1,     dx/dy,  -1,     0, 0, -dx/dy],
                     [-dy/dx, -1,    dy/dx,  0, 0, 1],
                     [0,     0,      0,      0, 0, 0],
                     [0,     0,      0,      0, 0, 0],
                     [-1,    -dx/dy, 1,      0, 0, dx/dy]])/2


def load_vector(sigma_cc, dx, dy):
    xx = sigma_cc[0, 0]
    yy = sigma_cc[1, 1]
    xy = sigma_cc[0, 1]
    return np.array([-xx * dy - xy * dx,
                     -xy * dy - yy * dx,
                     xx * dy,
                     xy * dy,
                     xy * dx,
                     yy * dx])/2


def make_grid(nb_grid_pts, periodic=False):
    """
    Make an array that contains all elements of the grid. The
    elements are described by the global node indices of
    their corners. The order of the corners is in order of
    the local node index.

    They are sorted in geometric positive order and the first
    is the node with the right angle corner at the bottom
    left. Elements within the same box are consecutive.

    This is the first element per box:

        2
        | \
        |  \
    dy  |   \
        |    \
        0 --- 1

          dx

    This is the second element per box:

           dx
         1 ---0
          \   |
           \  |  dy
            \ |
             \|
              2

    Parameters
    ----------
    nb_grid_pts : tuple of ints
        Number of nodes in the Cartesian directions
    periodic : bool
        Periodic boundary conditions

    Returns
    -------
    triangles_el : numpy.ndarray
        Array containing the global node indices of the
        element corners. The first index (suffix _e)
        identifies the element number and the second index
        (suffix _l) the local node index of that element.
    """
    Nx, Ny = nb_grid_pts
    if periodic:
        nb_nodes = nb_grid_pts
    else:
        nb_nodes = Nx + 1, Ny + 1
    # These are the node position on a subsection of the grid
    # that excludes the rightmost and topmost nodes. The
    # suffix _G indicates this subgrid.
    y_G, x_G = np.mgrid[:Ny, :Nx]
    x_G.shape = (-1,)
    y_G.shape = (-1,)

    # List of triangles
    lower_triangles = np.vstack((node_index(x_G, y_G, nb_nodes),
                                 node_index(x_G + 1, y_G, nb_nodes),
                                 node_index(x_G, y_G + 1, nb_nodes)))
    upper_triangles = np.vstack((node_index(x_G + 1, y_G + 1, nb_nodes),
                                 node_index(x_G, y_G + 1, nb_nodes),
                                 node_index(x_G + 1, y_G, nb_nodes)))
    # Suffix _e indicates global element index
    return np.vstack(
        (lower_triangles, upper_triangles)).T.reshape(-1, 3)

def colors_to_cmap(colors):
    '''
    colors_to_cmap(nx3_or_nx4_rgba_array) yields a matplotlib colormap object that, when
    that will reproduce the colors in the given array when passed a list of n evenly
    spaced numbers between 0 and 1 (inclusive), where n is the length of the argument.

    Example:
      cmap = colors_to_cmap(colors)
      zs = np.asarray(range(len(colors)), dtype=np.float) / (len(colors)-1)
      # cmap(zs) should reproduce colors; cmap[zs[i]] == colors[i]
    '''
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    colors = np.asarray(colors)
    if colors.shape[1] == 3:
        colors = np.hstack((colors, np.ones((len(colors),1))))
    steps = (0.5 + np.asarray(range(len(colors)-1),
                              dtype=np.float))/(len(colors) - 1)
    return matplotlib.colors.LinearSegmentedColormap(
        'auto_cmap',
        {clrname: (
            [(0, col[0], col[0])] +
            [(step, c0, c1) for (step,c0,c1) in zip(steps,
                                                    col[:-1], col[1:])] +
                   [(1, col[-1], col[-1])])
         for (clridx,clrname) in enumerate(['red', 'green', 'blue', 'alpha'])
         for col in [colors[:,clridx]]},
        N=len(colors))

def plot_tri(nb_grid_pts, x_g=None, y_g=None, values_g=None, values_e=None,
             mesh_style=None, ax=None, periodic=False):
    """
    Plot results of a finite-element calculation on a
    two-dimensional structured grid using matplotlib.

    Parameters
    ----------
    nb_grid_pts : tuple of ints
        Number of nodes in the Cartesian directions
    x_g : array_like
        x-positions of the nodes
    y_g : array_like
        y-positions of the nodes
    values_g : array_like
        Expansion coefficients (values of the field) on the
        global nodes
    values_e : array_like
        Values on elements
    mesh_style : str, optional
        Will show the underlying finite-element mesh with
        the given style if set, e.g. 'ko-' to see edges
        and mark nodes by points
        (Default: None)
    ax : matplotlib.Axes, optional
        Axes object for plotting
        (Default: None)

    Returns
    -------
    trim : matplotlib.collections.Trimesh
        Result of tripcolor
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    # my_colors = np.array([[0.82745098, 0.82745098, 0.82745098, 1.        ],
    #                       [0.50196078, 0.50196078, 0.50196078, 1.        ],
    #                       [1.        , 0.64705882, 0.        , 1.        ]])
    # my_cmap = ListedColormap(my_colors)
    # newcmp = LinearSegmentedColormap.from_list("custom", my_colors)
    # newcmp = colors_to_cmap(my_colors)
    # plt.set_cmap(newcmp)
    Nx, Ny = nb_grid_pts

    # These are the node positions on the full global grid.
    if x_g is None and y_g is None:
        if periodic:
            y_g, x_g = np.mgrid[:Ny, :Nx]
        else:
            y_g, x_g = np.mgrid[:Ny+1, :Nx+1]
        x_g.shape = (-1,)
        y_g.shape = (-1,)
    elif not (x_g is not None and y_g is not None):
        raise ValueError('You need to specify both, x_g and y_g.')

    # Gouraud shading linearly interpolates the color between
    # the nodes
    if ax is None:
        ax = plt
    triangulation = matplotlib.tri.Triangulation(
        x_g, y_g, make_grid(nb_grid_pts, periodic=periodic))
    colormaps = ["Greys_r", "Oranges", "viridis"]
    alphas = [1.0,1.0, 0.1]
    vs = [[-4.0, 1.0],
          [1.0, 3.0],
          [0.0, 1.0]]
    if values_e is not None:
        for val_e, cmap, alpha, v in zip(values_e, colormaps, alphas, vs):
            print(f"{val_e.shape=}")
            c = ax.tripcolor(triangulation, facecolors=val_e, cmap=cmap,
                             vmin = v[0], vmax = v[1], alpha=alpha)
    elif values_g is not None:
        print(f"{values_g.shape=}")
        c = ax.tripcolor(triangulation, values_g)
    else:
        c = ax.tripcolor(triangulation, np.zeros_like(x_g))
    if mesh_style is not None:
        ax.triplot(triangulation, mesh_style)
    return c


def unit_vector(n, i):
    """
    Returns a vector of zeros with a single entry that is 1

    Parameters
    ----------
    n : int
        Length of the vector
    i : int
        Position where the 1 should be placed
    """
    a = np.zeros((n))
    a[i] = 1
    return a

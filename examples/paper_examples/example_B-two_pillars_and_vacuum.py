# !/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   example_B-two_pillars_and_vacuum.py

@author Richard Leute <richard.leute@imtek.uni-freiburg.de>

@date   14 Aug 2020

@brief  plot script for example B "Two pillars and vacuum" of the paper

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

import sys
import argparse
from python_paper_example_imports import muSpectre as msp
import muSpectre.gradient_integration as gi
from muFFT import Stencils2D

import numpy as np

try:
    from helper_functions import matplotlib, plt, gridspec, make_triangles,\
        make_squares, update_rc_parameters

    update_rc_parameters()

    matplotlib_found = True
except ImportError:
    matplotlib_found = False


parser = argparse.ArgumentParser()

parser.add_argument('-f', '--to-file',
                    action="store", dest="plot_file",
                    help="store plot to file instead of showing it on screen")

args = parser.parse_args()


# --- simulation parameters --- #
nb_domain_grid_pts = [17, 17]
nx, ny = nb_domain_grid_pts
domain_lengths = [float(L) for L in nb_domain_grid_pts]
dim = len(nb_domain_grid_pts)

newton_tol = 1e-6  # 1e-8
equil_tol = newton_tol
cg_tol = 1e-8  # 1e-14

# macroscopic strain
applied_strain = np.array([[0.0, 0.0],
                           [0.0, 0.1]])

maxiter = 1000
verbose = msp.Verbosity.Silent
formulation = msp.Formulation.finite_strain

# options: "f_q1", "cd_q1", "fw_q1", "afw_q1", "f_q2", "fw_q2"
grad_types = ["f_q1", "cd_q1", "fw_q1", "afw_q1", "f_q2", "fw_q2"]
grad_name = {"f_q1": "Fourier",
             "cd_q1": "central diff.",
             "fw_q1": "forward diff.",
             "afw_q1": "least squares",
             "f_q2": "Fourier 2 p.",
             "fw_q2": "linear FE 2 el."}


# --- gradient operators --- #
fourier_gradient = [msp.FourierDerivative(dim, i) for i in range(dim)]
# central difference derivative
cd_gradient = [Stencils2D.central_x, Stencils2D.central_y]
# forward difference derivative, the upwind differences stencil
fw_gradient = [Stencils2D.upwind_x, Stencils2D.upwind_y]
# averaged forward difference derivative, the upwind differences stencil
av_fw_gradient = [Stencils2D.averaged_upwind_x, Stencils2D.averaged_upwind_y]
# fourier grad 2 quad point, lower-left corner + upper-right corner
fourier_grad_lower = [msp.FourierDerivative(dim, i, [-1/6, -1/6])
                      for i in range(dim)]
fourier_grad_upper = [msp.FourierDerivative(dim, i, [ 1/6,  1/6])
                      for i in range(dim)]
fourier_gradient_2 = fourier_grad_lower + fourier_grad_upper
fw_gradient_2 = [Stencils2D.d_10_00, Stencils2D.d_01_00,
                 Stencils2D.d_11_01, Stencils2D.d_11_10]


# --- setup rve geometry --- #
left_vac = 1
center_vac = 1
right_vac = 0
vac = left_vac + center_vac + right_vac
half_inc_hight = 2
pillars = nx - vac
pillar_1 = pillars // 2  # odd width pillar
pillar_2 = pillars - pillar_1  # even width pillar


def set_geometry(pillar, inclusion, vacuum):
    """
    define the two pillar and vacuum geometry
    """
    phase = -np.ones(nb_domain_grid_pts, dtype=int)
    for pixel_index, pixel in enumerate(rve.pixels):
        if pixel[0] < left_vac:
            # vacuum left
            vacuum.add_pixel(pixel_index)
            phase[pixel[0], pixel[1]] = 0

        elif pixel[0] >= left_vac and \
             pixel[0] < left_vac + pillar_1:
            if pixel[0] == left_vac + pillar_1 // 2 and \
               pixel[1] > ny//2 - half_inc_hight and \
               pixel[1] < ny//2 + half_inc_hight:
                # inclusion
                inclusion.add_pixel(pixel_index)
                phase[pixel[0], pixel[1]] = 5
            else:
                # pillar 1
                pillar.add_pixel(pixel_index)
                phase[pixel[0], pixel[1]] = 1

        elif pixel[0] >= left_vac + pillar_1 and \
             pixel[0] < left_vac + pillar_1 + center_vac:
            # vacuum center
            vacuum.add_pixel(pixel_index)
            phase[pixel[0], pixel[1]] = 2

        elif pixel[0] >= left_vac + pillar_1 + center_vac and \
             pixel[0] < left_vac + pillar_1 + center_vac + pillar_2:
            # pillar 2
            pillar.add_pixel(pixel_index)
            phase[pixel[0], pixel[1]] = 3

        elif pixel[0] >= left_vac + pillar_1 + center_vac + pillar_2 and \
             pixel[0] < left_vac + pillar_1 + center_vac + pillar_2 + right_vac:
            # vacuum right
            vacuum.add_pixel(pixel_index)
            phase[pixel[0], pixel[1]] = 4

    return phase


# --- µSpectre computation --- #
stress = {}
grad = {}
placement = {}
x = {}

for i, grad_type in enumerate(grad_types):
    if grad_type == "f_q1":
        gradient = fourier_gradient
    elif grad_type == "cd_q1":
        gradient = cd_gradient
    elif grad_type == "fw_q1":
        gradient = fw_gradient
    elif grad_type == "afw_q1":
        gradient = av_fw_gradient
    elif grad_type == "fw_q2":
        gradient = fw_gradient_2
    elif grad_type == "f_q2":
        gradient = fourier_gradient_2
    else:
        raise RuntimeError("The given gradient '{}' type is not supported!"
                           .format(grad_type))

    rve = msp.Cell(nb_domain_grid_pts, domain_lengths,
                   formulation, gradient)
    pillar = msp.material.MaterialLinearElastic1_2d.make(
        rve, "pillar",   1.0, 0.33)
    inclusion = msp.material.MaterialLinearElastic1_2d.make(
        rve, "inclusion", 0.1, 0.33)
    vacuum = msp.material.MaterialLinearElastic1_2d.make(
        rve, "vacuum",    0.0, 0.00)

    phase = set_geometry(pillar, inclusion, vacuum)

    solver = msp.solvers.KrylovSolverCG(rve, cg_tol, maxiter,
                                        verbose=msp.Verbosity.Silent)
    result = msp.solvers.newton_cg(rve, applied_strain, solver,
                                   newton_tol=newton_tol, equil_tol=equil_tol,
                                   verbose=verbose)

    stress[grad_type] = result.stress.reshape(
        (dim, len(gradient), *nb_domain_grid_pts), order='F')
    grad[grad_type] = result.grad.reshape(
        (dim, len(gradient), *nb_domain_grid_pts), order='F')

    placement[grad_type], x[grad_type] = \
        gi.compute_placement(result, domain_lengths,
                             nb_domain_grid_pts, gradient)


# --- Plotting --- #
# the plotting is only possible if matplotlib was found
if matplotlib_found is False:
    print("matplotlib was not found wherefore the plotting is not possible.")
    sys.exit(0)

# adjust parameters
w_ratios = [1.2, 1, 1, 1, 1, 1, 1][:len(grad_types) + 1]
fig = plt.figure(figsize=(2*(len(grad_types) + 1), 4))
gs1 = gridspec.GridSpec(2, len(grad_types) + 1,
                        wspace=0.025, hspace=0.05,
                        width_ratios=w_ratios)
cmap = plt.get_cmap('seismic')

vmin = min([np.amin(stress[gt][0, 1]) for gt in grad_types])
vmax = max([np.amax(stress[gt][0, 1]) for gt in grad_types])
maximum = max(-vmin, vmax)

xmin = min([np.amin(placement[gt][0]) for gt in grad_types])
xmax = max([np.amax(placement[gt][0]) for gt in grad_types])
ymin = min([np.amin(placement[gt][1]) for gt in grad_types])
ymax = max([np.amax(placement[gt][1]) for gt in grad_types])

# cut hights
center = np.array([left_vac + pillar_1 // 2, ny//2])
cuts = [center[1], center[1]-1, center[1]-2, center[1]-3]
colors = ['#d7191c', '#fdae61', '#008837', '#abd9e9', '#2c7bb6']
markers = ['.', '.', '.', '.', '.']  # [".", 7, "1", "+", "x"]
markersize = 3.5
linestyles = ['-', '-', '-', '-', '-']

fig_labels = ["   ", "(b.1)", "(c.1)", "(d.1)", "(e.1)", "(f.1)", "(g.1)",
              "(a)", "(b.2)", "(c.2)", "(d.2)", "(e.2)", "(f.2)", "(g.2)"]

tx = 0.06  # text position x
ty = 0.90  # text position y
# background box for text
bbox = dict(facecolor='w', alpha=0.8, linewidth=0)


def write_label_text(ax, index):
    ax.text(tx, ty, fig_labels[index], transform=ax.transAxes, bbox=bbox)


# ## phase setup
cmaplist = []
cmaplist.append((178/255, 223/255, 138/255, 1.0))  # color for material
cmaplist.append((166/255, 206/255, 227/255, 1.0))  # color for vacuum
cmaplist.append((228/255, 26/255, 28/255, 1.0))  # color for inhomogeneity
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    'Phase_colors', cmaplist, len(cmaplist))

# make phase information clear, pillar = 0, vac = 1, inclusion = 2
phase[np.where(phase == 0)] = 7  # left vac
phase[np.where(phase == 2)] = 7  # center vac
phase[np.where(phase == 4)] = 7  # right vac
phase[np.where(phase == 1)] = 8  # left pillar
phase[np.where(phase == 3)] = 8  # right pillar
phase[np.where(phase == 5)] = 2  # inclusion

phase[np.where(phase == 7)] = 1  # vacuum
phase[np.where(phase == 8)] = 0  # pillars

ax1 = plt.subplot(gs1[len(grad_types) + 1], aspect=1)
# use x to have the undeformed specimen
p = make_squares(x[grad_types[0]], cmap=cmap)
color = phase.flatten()
p.set_array(color)
ax1.add_collection(p)
ax1.set_xlim(xmin, xmax)
ax1.set_xlabel(r"$x$-position")
ax1.set_ylim(ymin, ymax)
ax1.set_ylabel(r"$y$-position")
ax1.text(tx, ty, fig_labels[len(grad_types)+1],
         transform=ax1.transAxes, bbox=bbox)
ax1.tick_params(axis='both', left=False, bottom=False,
                labelleft=False, labelbottom=False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)

del_x = x['f_q1'][0, 1, 0] - x['f_q1'][0, 0, 0]  # grid spacing in x-direction

# --- actual plotting loop
for i, grad_type in enumerate(grad_types):
    displ = placement[grad_type]

    if grad_type[-2:] == "q1":  # one quadrature point
        # plot stress field
        ax2 = plt.subplot(gs1[i+1], aspect=1)
        write_label_text(ax2, i+1)
        pk1_matplot = stress[grad_type][0, 1, :, :]

        p = make_squares(displ, matplotlib.cm.seismic)
        color = pk1_matplot.flatten()
        p.set_array(color)
        ax2.add_collection(p)
        ax2.set_xlim(xmin, xmax)
        ax2.set_ylim(ymin, ymax)

        p.set_clim(-maximum, +maximum)

        ax2.tick_params(axis='both', left=False, bottom=False,
                        labelleft=False, labelbottom=False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)

        plt.title(grad_name[grad_type])

        # add colorbar
        if i == 0:
            cbaxes = fig.add_axes([0.24, 0.59, 0.02, 0.25])
            plt.colorbar(p, cax=cbaxes,
                         label=r"Stress  $P_{xy}$ $(E_{\text{hard}})$",
                         orientation='vertical')
            cbaxes.yaxis.set_ticks_position('left')
            cbaxes.yaxis.set_label_position('left')

        # add y-position label to axis
        if i == 0:
            ax2.set_ylabel(r"$\chi_y$-position")

        # plot stress cut
        shift = (vmax - vmin) * 0.14
        left_off = (displ[0, -1, 0] - displ[0, 0, 0]) * 0.05
        ax2 = plt.subplot(gs1[len(grad_types) + 2 + i])
        write_label_text(ax2, len(grad_types) + 2 + i)
        ax2.set_xlabel(r"$\chi_x$-position")
        if i == 0:
            ax2.set_ylabel(r"Stress  $P_{xy}$ $(E_{\text{hard}})$",
                           labelpad=2.0)
        if i > 0:
            ax2.spines['left'].set_visible(False)
        ax2.tick_params(axis='both', left=False, bottom=False,
                        labelleft=False, labelbottom=False)
        for n, cut in enumerate(cuts):
            downshift = (len(cuts)//2 - n - 1) * shift
            plt.hlines(downshift,
                       displ[0, 0, 0] - left_off, displ[0, -1, 0],
                       color='gray', alpha=0.8)
        for n, cut in enumerate(cuts):
            downshift = (len(cuts)//2 - n - 1) * shift
            plt.plot(displ[0, :-1, 0],
                     (stress[grad_type][0, 1])[:, cut] + downshift,
                     color=colors[n],
                     marker=markers[n],
                     markersize=markersize,
                     linestyle=linestyles[n],
                     linewidth=1.0)

        plt.xlim(xmin=displ[0, 0, 0] - left_off,
                 xmax=displ[0, -1, 0] + left_off)
        plt.ylim(ymin=-(maximum*1.3), ymax=+(maximum*1.04))

    elif grad_type[-2:] == "q2":  # two quadrature points
        # plot stress field
        ax2 = plt.subplot(gs1[i+1], aspect=1)
        write_label_text(ax2, i+1)
        pk1_matplot = np.transpose(
            stress[grad_type][0, [1, 3], :, :], (1, 2, 0)).flatten()

        tri, triangles = make_triangles(displ)

        triplot = plt.tripcolor(tri,
                                pk1_matplot,
                                edgecolors='k',
                                shading='flat',
                                linewidth=0.3, cmap="seismic")
        triplot.set_clim(-maximum, +maximum)
        ax2.set_xlim(xmin, xmax)
        ax2.set_ylim(ymin, ymax)
        ax2.tick_params(axis='both', left=False, bottom=False,
                        labelleft=False, labelbottom=False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)

        plt.title(grad_name[grad_type])

        # plot stress cut
        shift = (vmax - vmin) * 0.14
        left_off = (displ[0, -1, 0] - displ[0, 0, 0]) * 0.05
        ax2 = plt.subplot(gs1[len(grad_types) + 2 + i])
        write_label_text(ax2, len(grad_types) + 2 + i)
        ax2.set_xlabel(r"$\chi_x$-position")
        if i == 0:
            ax2.set_ylabel(r"Stress  $P_{xy}$ $(E_{\text{hard}})$",
                           labelpad=2.0)
        if i > 0:
            ax2.spines['left'].set_visible(False)
        ax2.tick_params(axis='both', left=False, bottom=False,
                        labelleft=False, labelbottom=False)
        for n, cut in enumerate(cuts):
            downshift = (len(cuts)//2 - n - 1) * shift
            plt.hlines(downshift,
                       displ[0, 0, 0] - left_off, displ[0, -1, 0],
                       color='gray', alpha=0.8)
        for n, cut in enumerate(cuts):
            downshift = (len(cuts)//2 - n - 1) * shift
            plt.plot(displ[0, :-1, 0]-1/6*del_x,
                     (stress[grad_type][0, 1])[:, cut] + downshift,
                     color=colors[n],
                     marker=markers[n],
                     markersize=markersize,
                     linestyle=linestyles[n],
                     linewidth=1.0)
            plt.plot(displ[0, :-1, 0]+1/6*del_x,
                     (stress[grad_type][0, 3])[:, cut] + downshift,
                     color=colors[n],
                     marker='+',
                     markersize=markersize+1,
                     linestyle='--',
                     linewidth=1.0)
        plt.xlim(xmin=displ[0, 0, 0] - left_off,
                 xmax=displ[0, -1, 0] + left_off)
        plt.ylim(ymin=-(maximum*1.3), ymax=+(maximum*1.04))


if args.plot_file is not None:
    plt.savefig(args.plot_file, dpi=600, bbox_inches='tight')
else:
    plt.show()

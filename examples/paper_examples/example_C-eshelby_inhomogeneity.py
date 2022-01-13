# !/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   example_C-eshelby_inhomogeneity.py

@author Richard Leute <richard.leute@imtek.uni-freiburg.de>

@date   19 Jan 2021

@brief  plot script for example C "Eshelby inhomogeneity" of the paper

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
from helper_functions import look_up_eshelby_result, \
    generate_eshelby_details_dic, sum_up_PBC_eshelby, \
    min_max_xy_vec, min_max_xy_tens
import muSpectre.gradient_integration as gi
from muFFT import Stencils2D

import numpy as np

try:
    from helper_functions import matplotlib, plt, gridspec, make_triangles,\
        make_squares, update_rc_parameters

    update_rc_parameters()
    matplotlib.rcParams.update({'axes.labelsize': 12})

    matplotlib_found = True
except ImportError:
    matplotlib_found = False


parser = argparse.ArgumentParser()

parser.add_argument('-f', '--to-file',
                    action="store", dest="plot_file",
                    help="store plot to file instead of showing it on screen")
parser.add_argument('-s', '--small_simulation',
                    action="store_true", dest="small_simulation",
                    help="running a smaller faster simulation")

args = parser.parse_args()


# --- simulation parameters --- #
if args.small_simulation:
    nb_grid_pts = [31, 31]
else:
    nb_grid_pts = [151, 151]
lengths = [100, 100]  # Domain dimensions
nx, ny = nb_grid_pts
dim = len(nb_grid_pts)

formulation = msp.Formulation.small_strain

newton_tol = 1e-14
equil_tol = 1e-8
cg_tol = -1
maxiter = 1000
verbose = msp.Verbosity.Silent

# macroscopic strain
applied_strain = np.array([[0.01, 0.00],
                           [0.00, 0.01]])

# finer mesh for all muSpectre computations, default = 1
n_times_finer = 1

# finer grid for liner finite elements Eshelby computation inset (panel h.3)
if args.small_simulation:
    single_finer = 3
else:
    single_finer = 11
make_finer = True

# periodic eshelby
# number of periodic images to correct the analytical Eshelby result for
# periodic boundary conditions
if args.small_simulation:
    pbc_grid = np.array([1, 3, 3])
else:
    pbc_grid = np.array([1, 11, 11])

# options: "f_q1", "cd_q1", "fw_q1", "afw_q1", "f_q2", "lfe_q2, "lfe_finer_q2"
grad_types = ["f_q1", "cd_q1", "fw_q1", "afw_q1", "f_q2", "lfe_q2", "lfe_finer_q2"]
grad_name = {"f_q1": "Fourier",
             "cd_q1": "central diff.",
             "fw_q1": "forward diff.",
             "afw_q1": "least squares",
             "f_q2": "Fourier 2 p.",
             "lfe_q2": "linear FE 2 el.",
             "lfe_finer_q2": "finer inset for lFE"}


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
fw_gradient_2 = Stencils2D.linear_finite_elements


# --- setup rve geometry --- #
# --- Inclusion geometry for analytical Eshelby computation
# We use the general analytical Eshelby solution to an ellipsoidal inclusion
# with a>b>c to havethe possibility to easily change the inclusion geometry.
# The general solution converges to the special solutions for e.g. an
# cylindrical inclusion or a spherical inclusion. However, it is not possible
# to break the rule a>b>c for the half axes in this implementation because
# otherwise some of the numeric expressions diverge.
center_index = np.array([r//2 for r in nb_grid_pts])
a = 10000  # half-axis in x-direction (only for analytical Eshelby computation)
b = 10.00001  # half-axis in y-direction
c = 10.00000  # half-axis in z-direction

E = 1.  # Youngs modulus matrix
nu = 0.33  # Poisson ratio matrix
E_I = 0.1  # Youngs modulus inhomogeneity
nu_I = 0.33  # Poisson ratio inhomogeneity

eps_0 = np.zeros((3, 3))
eps_0[1:, 1:] = applied_strain
eps_p_0 = None  # no eigen strain in the inhomogeneity


# --- Analytical eshelby solution (not PBC and PBC corrected) --- #
# By default the analytical results are read from a precomputed file
# --- Non PBC corrected Eshelby solution
lengths_E = np.array([1] + lengths)
nb_grid_pts_E = np.array([1] + nb_grid_pts)
del_x_E = lengths_E / nb_grid_pts_E
x_n_E, x_c_E = msp.gradient_integration.make_grid(lengths_E, nb_grid_pts_E)


eshelby_details = generate_eshelby_details_dic(
    nb_grid_pts_E, lengths_E, a, b, c,
    E, nu, E_I, nu_I, eps_0, eps_p_0, pbc_grid)

stored_esh_res_pbc = look_up_eshelby_result(
    eshelby_details, folder="./stored_eshelby_results/", verbose=True)

if stored_esh_res_pbc is not None:
    ser = np.load(stored_esh_res_pbc)
    esh_stress_pbc = ser["eshelby_stress"]
    esh_strain_pbc = ser["eshelby_strain"]
else:
    raise RuntimeError(
        "I can not find the proper result file for the periodic Eshelby "
        "inclusion! Use the script 'compute_analytic_eshelby_solution.py'. "
        "If you want to reproduce the example C 'Eshelby inhomogeneity' from "
        "the paper you can just run it. Otherwise you have to modify the "
        "parameters as you need.")

average_summed_strain, esh_strain_summed_up = \
    sum_up_PBC_eshelby(esh_strain_pbc, eps_0, nb_grid_pts_E, pbc_grid)


# --- µSpectre computation --- #
# --- adjust and initialise parameters
# shift inclusion parameters because we dont simulate axis a,
# so a=b b=c
a = b
b = c

# correct the applied strain by the analytic periodic Eshelby superposition
# correction (you have to subtract eps_0 because it is already included in the
# correction)
applied_strain += average_summed_strain[1:, 1:] - eps_0[1:, 1:]

# do a simulation on a finer grid than the eshelby solution
nb_grid_pts_mu = np.array(nb_grid_pts) * n_times_finer + (1*((n_times_finer+1)%2))
nb_grid_pts_mu_fine = np.array(nb_grid_pts) \
    * single_finer + (1*((single_finer+1) % 2))

del_x_mu = np.array(lengths)/nb_grid_pts_mu
center = np.array(lengths) / 2
center_index_mu = np.array([r//2 for r in nb_grid_pts_mu])
x_n_mu, x_c_mu = msp.gradient_integration.make_grid(lengths, nb_grid_pts_mu)
x_c_mu_f = msp.gradient_integration.make_grid(lengths, nb_grid_pts_mu_fine)[1]

stress = {}
grad = {}
grad_sym = {}
placement = {}
x = {}
phase = -np.ones(nb_grid_pts_mu, dtype=int)

# --- actual computation
for i, grad_type in enumerate(grad_types):
    nb_grid_pts_mu_int = nb_grid_pts_mu
    if grad_type == "f_q1":
        gradient = fourier_gradient
    elif grad_type == "cd_q1":
        gradient = cd_gradient
    elif grad_type == "fw_q1":
        gradient = fw_gradient
    elif grad_type == "afw_q1":
        gradient = av_fw_gradient
    elif grad_type == "lfe_q2":
        gradient = fw_gradient_2
    elif grad_type == "f_q2":
        gradient = fourier_gradient_2
    elif grad_type == "lfe_finer_q2":
        gradient = fw_gradient_2
        nb_grid_pts_mu_int = nb_grid_pts_mu_fine
        if not make_finer:
            # finer computation not needed
            continue
    else:
        raise RuntimeError("The given gradient '{}' type is not supported!"
                           .format(grad_type))

    del_x_mu_int = np.array(lengths)/nb_grid_pts_mu_int
    rve = msp.Cell(nb_grid_pts_mu_int, lengths, formulation,
                   gradient)
    matrix = msp.material.MaterialLinearElastic1_2d.make(
        rve, "matrix", E, nu)
    inhomogeneity = msp.material.MaterialLinearElastic1_2d.make(
        rve, "inhomogeneity", E_I, nu_I)
    for pixel_index, pixel in enumerate(rve.pixels):
        if ((((pixel[0]+0.5) * del_x_mu_int[0] - center[0])/a)**2 +
            (((pixel[1]+0.5) * del_x_mu_int[1] - center[1])/b)**2 <= 1):
            inhomogeneity.add_pixel(pixel_index)
            if i == 0:
                phase[tuple(pixel)] = 1
        else:
            matrix.add_pixel(pixel_index)
            if i == 0:
                phase[tuple(pixel)] = 0

    solver = msp.solvers.KrylovSolverCG(rve, cg_tol, maxiter,
                                        verbose=msp.Verbosity.Silent)
    result = msp.solvers.newton_cg(rve, applied_strain, solver,
                                   newton_tol=newton_tol, equil_tol=equil_tol,
                                   verbose=verbose)
    stress[grad_type] = \
        result.stress.reshape(*nb_grid_pts_mu_int, len(gradient), dim).T
    grad[grad_type] = \
        result.grad.reshape(*nb_grid_pts_mu_int, len(gradient), dim).T

    placement[grad_type], x[grad_type] = gi.get_complemented_positions(
        'p0', rve, periodically_complemented=True)

    if formulation == msp.Formulation.finite_strain:
        # correct the gradient by subtracting the unit gradient
        if grad_type[-2:] == "q2":
            unit_grad = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
        else:
            unit_grad = np.array([[1, 0],
                                  [0, 1]])
        grad[grad_type] = grad[grad_type] - unit_grad[:, :, np.newaxis, np.newaxis]

        # compute the symmetrized F
        if grad_type[-2:] == "q2":
            grad_sym[grad_type] = grad[grad_type]
            grad_sym[grad_type][0, [1, 3]] = 1/2 * (grad[grad_type][0, [1, 3]]
                                                    + grad[grad_type][1, [0, 2]])
            grad_sym[grad_type][1, [0, 2]] = grad_sym[grad_type][0, [1, 3]]
        else:
            grad_sym[grad_type] = grad[grad_type]
            grad_sym[grad_type][0, 1] = 1/2 * (grad[grad_type][0, 1]
                                               + grad[grad_type][1, 0])
            grad_sym[grad_type][1, 0] = grad_sym[grad_type][0, 1]
    elif formulation == msp.Formulation.small_strain:
        # in small strain case the Epsilon is already symmetrized
        grad[grad_type] = np.copy(grad[grad_type])
        grad_sym[grad_type] = grad[grad_type]
        pass
    else:
        raise RuntimeError("The case fomulation={} is not handled!"
                           .format(formulation))


# --- Plotting --- #
# the plotting is only possible if matplotlib was found
if matplotlib_found is False:
    print("matplotlib was not found wherefore the plotting is not possible.")
    sys.exit(0)

# --- adjust parameters
# switch to percent in strain measure
esh_strain_summed_up *= 100
for gt in grad_types:
    if not make_finer and gt == "lfe_finer_q2":
        continue
    grad[gt] *= 100

# min max values for plotting
xy_min_tens, xy_max_tens = min_max_xy_tens(grad_sym)
xy_maxmin_tens = max(-xy_min_tens, xy_max_tens)
x_min_vec, x_max_vec, y_min_vec, y_max_vec = min_max_xy_vec(placement)
xy_min_vec = min(x_min_vec, y_min_vec)
xy_max_vec = max(x_max_vec, y_max_vec)

esh_types = ["pbc_esh"]  # ["esh", "pbc_esh"]
if make_finer:
    # remove finer from list if exists
    if "lfe_finer_q2" in grad_types:
        grad_types.remove("lfe_finer_q2")

types = grad_types + esh_types
names = grad_name
names["pbc_esh"] = "Eshelby "+str(pbc_grid[1])+'x'+str(pbc_grid[2])

# add undeformed eshelby placements (x_n and x_n_pbc) to placement dic
placement[esh_types[0]] = x_n_E[1:, 0, :, :]
# add undeformed eshelby placements (x_n and x_n_pbc) to x dic
x[esh_types[0]] = x_n_E[1:, 0, :, :]

w_ratios = [1.3, 1, 1, 1, 1, 1, 1, 1, 1][:len(types) + 1]
h_ratios = [1, 1, 1, 1]
fig = plt.figure(figsize=(2*(len(types) + 1), 8))
gs1 = gridspec.GridSpec(4, len(types) + 1,
                        wspace=0.025, hspace=0.05,
                        width_ratios=w_ratios,
                        height_ratios=h_ratios)

# cut planes
center_esh = x[esh_types[0]].shape[-1]//2 - 1
shift_esh = int((b / 2.) / del_x_E[2])
center_mu = x[grad_types[0]].shape[-1]//2 - 1
shift_mu = shift_esh * n_times_finer
if make_finer:
    center_mu_f = x["lfe_finer_q2"].shape[-1]//2 - 1
    shift_mu_f = shift_esh * single_finer

colors = ['#6a3d9a', '#33a02c', '#ff7f00']
markers = ['.', '.', '.', '.', '.']
linestyles = ['-', '-', '-', '-', '-']

fig_labels = \
    ["   ", "(c.1)", "(d.1)", "(e.1)", "(f.1)", "(g.1)", "(h.1)", "(i.1)",
     "   ", "(c.2)", "(d.2)", "(e.2)", "(f.2)", "(g.2)", "(h.2)", "(i.2)",
     "(a)", "(c.3)", "(d.3)", "(e.3)", "(f.3)", "(g.3)", "(h.3)", "(i.3)",
     "(b)", "(c.4)", "(d.4)", "(e.4)", "(f.4)", "(g.4)", "(h.4)", "(i.4)"]

marker = ""

tx = 0.05  # text position x
ty = 0.88  # text position y
# background box for text
bbox = dict(facecolor='w', alpha=0.8, linewidth=0)


def write_label_text(ax, index):
    ax.text(tx, ty, fig_labels[index],
            transform=ax.transAxes, bbox=bbox, fontsize=script_size)


# --- Eshelby phase setup
cmaplist = []
cmaplist.append((178/255, 223/255, 138/255, 1.0))  # color for material
cmaplist.append((166/255, 206/255, 227/255, 1.0))  # color for vacuum
cmaplist.append((228/255, 26/255, 28/255, 1.0))  # color for inhomogeneity
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    'Phase_colors', cmaplist, len(cmaplist))

ax1 = plt.subplot(gs1[2*len(types) + 2], aspect=1)
# use x to have the undeformed specimen
x_local = x[types[0]]
p = make_squares(x_local, cmap=cmap, lw=0.0, edge_c='None')
color = phase.flatten()
p.set_array(color)
ax1.add_collection(p)
ax1.set_xlim(x_local[0, 0, 0], x_local[0, -1, 0])
ax1.set_ylim(x_local[1, 0, 0], x_local[0, -1, -1])
ax1.set_ylabel(r"$y$-position")
ax1.text(tx, ty, fig_labels[2*len(types)+2],
         transform=ax1.transAxes, bbox=bbox)
ax1.tick_params(axis='both', left=False, bottom=False,
                labelleft=False, labelbottom=False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)

# plot colored lines through phase to indicate the strain cross sections
ax1.axhline(y=x[types[0]][1, 0, center_mu - shift_mu] + del_x_mu[1]/2,
            linestyle='-', color=colors[0], linewidth=1.0)
ax1.axhline(y=x[types[0]][1, 0, center_mu] + del_x_mu[1]/2,
            linestyle='-', color=colors[1], linewidth=1.0)
ax1.axhline(y=x[types[0]][1, 0, center_mu + shift_mu] + del_x_mu[1]/2,
            linestyle='-', color=colors[2], linewidth=1.0)


# --- Eshelby pbc phase setup
cmaplist = []
cmaplist.append((178/255, 223/255, 138/255, 1.0))  # color for material
cmaplist.append((178/255, 223/255, 138/255, 0.5))  # color for outer material
cmaplist.append((228/255, 26/255, 28/255, 1.0))  # color for inhomogeneity
cmaplist.append((228/255, 26/255, 28/255, 0.5))  # color for outer inhomogen.
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    'Phase_colors', cmaplist, len(cmaplist))
# pbc phase
n_pbc_p = 3
x_n_pbc_p, x_c_pbc_p = msp.gradient_integration.make_grid(
    lengths_E[1:] * np.array([n_pbc_p, n_pbc_p]),
    nb_grid_pts_E[1:] * np.array([n_pbc_p, n_pbc_p]))
# shift box by half length
center_index_pbc_p = np.array(
    [r//2 for r in nb_grid_pts_E[1:] * np.array([n_pbc_p, n_pbc_p])])
center_pbc_p = x_c_pbc_p[:, center_index_pbc_p[0], center_index_pbc_p[1]]
x_n_pbc_p -= center_pbc_p[:, np.newaxis, np.newaxis]
x_c_pbc_p -= center_pbc_p[:, np.newaxis, np.newaxis]

# --- periodic phase ---
pbc_phase = np.zeros_like(x_c_pbc_p[0], order='C')
Lx_p, Ly_p = lengths_E[1:]
# outer pbc region
outer_region = np.where(((np.abs(x_c_pbc_p[0])-Lx_p/2) > 0)
                        | ((np.abs(x_c_pbc_p[1])-Ly_p/2) > 0))
pbc_phase[outer_region] = 1
# inhomogeneities
for i in range(-n_pbc_p//2, n_pbc_p//2+1, 1):
    for j in range(-n_pbc_p//2, n_pbc_p//2+1, 1):
        inhomo_region = np.where(
            ((x_c_pbc_p[0]+i*Lx_p)/b)**2 + ((x_c_pbc_p[1]+j*Ly_p)/c)**2 <= 1)
        if i == 0 and j == 0:
            pbc_phase[inhomo_region] = 2
        else:
            pbc_phase[inhomo_region] = 3

ax1 = plt.subplot(gs1[3*len(types) + 3], aspect=1)
p = make_squares(x_n_pbc_p, cmap=cmap, lw=0.0, edge_c='none')
color = pbc_phase.flatten()
p.set_array(color)
ax1.add_collection(p)
ax1.set_xlim(x_c_pbc_p[0, 0, 0], x_c_pbc_p[0, -1, 0])
ax1.set_xlabel(r"$x$-position")
ax1.set_ylim(x_c_pbc_p[1, 0, 0], x_c_pbc_p[0, -1, -1])
ax1.set_ylabel(r"$y$-position")
ax1.text(tx, ty, fig_labels[3*len(types)+3],
         transform=ax1.transAxes, bbox=bbox)
ax1.tick_params(axis='both', left=False, bottom=False,
                labelleft=False, labelbottom=False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(False)

# black lines to indicate grid of pbc cell
nx_pbc = x_c_pbc_p.shape[1]//n_pbc_p
ny_pbc = x_c_pbc_p.shape[2]//n_pbc_p
ax1.axvline(x=x_c_pbc_p[0, nx_pbc, 0],
            linestyle='-', color='k', linewidth=1.0)
ax1.axvline(x=x_c_pbc_p[0, 2*nx_pbc, 0],
            linestyle='-', color='k', linewidth=1.0)
ax1.axhline(y=x_c_pbc_p[1, 0, ny_pbc],
            linestyle='-', color='k', linewidth=1.0)
ax1.axhline(y=x_c_pbc_p[1, 0, 2*ny_pbc],
            linestyle='-', color='k', linewidth=1.0)

# zoom in third and fourth row to region of second row
x_min_z = 37
x_max_z = 63
y_min_z = 37
y_max_z = 63

script_size = 12

# --- actual plotting loop
for i, t in enumerate(types):
    # plot against undeformed pos x[t] or deformed pos placement[t]
    displ = x[t]

    # --- First row --- #
    if i == 0:
        ax01 = plt.subplot(gs1[1 + i], aspect=1)
        plt.setp(ax01.get_xticklabels(), visible=False)
        ax2 = ax01
    else:
        ax2 = plt.subplot(gs1[1 + i], aspect=1, sharey=ax01, sharex=ax01)
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)

    plt.title(names[t], fontsize=script_size)
    write_label_text(ax2, 1 + i)
    ax2.tick_params(axis='both',
                    left=False, right=False,
                    top=False, bottom=False,
                    labelleft=False, labelright=False,
                    labeltop=False, labelbottom=False)
    ax2.spines['left'].set_visible(True)
    ax2.spines['right'].set_visible(True)
    ax2.spines['top'].set_visible(True)
    ax2.spines['bottom'].set_visible(True)
    if i == 0:
        ax2.set_ylabel(r"$y$-position")

    if t[-2:] == "q1":
        F_matplot = grad_sym[t][0, 1]
        p = make_squares(displ, matplotlib.cm.seismic, lw=0.0, edge_c='none')
        color = F_matplot.flatten()
        p.set_array(color)
        p.set_clim(-xy_maxmin_tens, +xy_maxmin_tens)
        ax2.add_collection(p)
        ax2.set_xlim(xy_min_vec, xy_max_vec)
        ax2.set_ylim(xy_min_vec, xy_max_vec)

        # add colorbar
        if i == 0:
            cbaxes = fig.add_axes([0.22, 0.625, 0.012, 0.17])
            plt.colorbar(p, cax=cbaxes,
                         label=r"Strain  $\varepsilon_{xy}$ $(\%)$",
                         orientation='vertical')
            cbaxes.yaxis.set_ticks_position('left')
            cbaxes.yaxis.set_label_position('left')

    elif t[-2:] == "q2":
        F_matplot = np.transpose(grad_sym[t][0, [1, 3], :, :], (1, 2, 0))\
                      .flatten()
        tri, triangles = make_triangles(displ)

        triplot = plt.tripcolor(tri,
                                F_matplot,
                                edgecolors="none",
                                shading='flat',
                                linewidth=0.0, cmap="seismic")
        triplot.set_clim(-xy_maxmin_tens, +xy_maxmin_tens)
        ax2.set_xlim(xy_min_vec, xy_max_vec)
        ax2.set_ylim(xy_min_vec, xy_max_vec)

    elif t == "pbc_esh":
        F_matplot = esh_strain_summed_up[1, 2, 0].flatten()
        p = make_squares(displ, matplotlib.cm.seismic, lw=0.0, edge_c='none')
        color = F_matplot.flatten()
        p.set_array(color)
        ax2.add_collection(p)
        p.set_clim(-xy_maxmin_tens, +xy_maxmin_tens)
        ax2.set_xlim(xy_min_vec, xy_max_vec)
        ax2.set_ylim(xy_min_vec, xy_max_vec)

    # --- Second Row --- #
    if i == 0:
        ax02 = plt.subplot(gs1[len(types) + 2 + i], aspect=1)
        plt.setp(ax02.get_xticklabels(), visible=False)
        ax2 = ax02
    else:
        ax2 = plt.subplot(gs1[len(types) + 2 + i],
                          aspect=1, sharey=ax02, sharex=ax02)
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)

    write_label_text(ax2, len(types) + 2 + i)
    ax2.tick_params(axis='both',
                    left=False, right=False,
                    top=False, bottom=False,
                    labelleft=False, labelright=False,
                    labeltop=False, labelbottom=False)
    ax2.spines['left'].set_visible(True)
    ax2.spines['right'].set_visible(True)
    ax2.spines['top'].set_visible(True)
    ax2.spines['bottom'].set_visible(True)
    if i == 0:
        ax2.set_ylabel(r"$y$-position")

    if t[-2:] == "q1":
        F_matplot = grad_sym[t][0, 1]
        p = make_squares(displ, matplotlib.cm.seismic, lw=0.0, edge_c='none')
        color = F_matplot.flatten()
        p.set_array(color)
        p.set_clim(-xy_maxmin_tens, +xy_maxmin_tens)
        ax2.add_collection(p)
        ax2.set_xlim(x_min_z, x_max_z)
        ax2.set_ylim(y_min_z, y_max_z)

    elif t[-2:] == "q2":
        F_matplot = np.transpose(grad_sym[t][0, [1, 3], :, :], (1, 2, 0))\
                      .flatten()
        tri, triangles = make_triangles(displ)

        triplot = plt.tripcolor(tri,
                                F_matplot,
                                edgecolors="none",
                                shading='flat',
                                linewidth=0.0, cmap="seismic")
        triplot.set_clim(-xy_maxmin_tens, +xy_maxmin_tens)
        ax2.set_xlim(x_min_z, x_max_z)
        ax2.set_ylim(y_min_z, y_max_z)

    elif t == "pbc_esh":
        F_matplot = esh_strain_summed_up[1, 2, 0].flatten()
        p = make_squares(displ, matplotlib.cm.seismic, lw=0.0, edge_c='none')
        color = F_matplot.flatten()
        p.set_array(color)
        ax2.add_collection(p)
        p.set_clim(-xy_maxmin_tens, +xy_maxmin_tens)
        ax2.set_xlim(x_min_z, x_max_z)
        ax2.set_ylim(y_min_z, y_max_z)

    # --- Third row --- #
    if i == 0:
        ax03 = plt.subplot(gs1[2*len(types) + 3 + i])
        plt.setp(ax02.get_xticklabels(), visible=False)
        ax2 = ax03
    else:
        ax2 = plt.subplot(gs1[2*len(types) + 3 + i], sharey=ax03, sharex=ax03)
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)

    write_label_text(ax2, 2*len(types) + 3 + i)
    if i == 0:
        ax2.set_ylabel(r"Strain  $\varepsilon_{xx}$ $(\%)$", labelpad=2.0)
    if i > 0:
        ax2.spines['left'].set_visible(False)
    ax2.tick_params(axis='both', left=False, bottom=False,
                    labelleft=False, labelbottom=False)

    if t[-2:] == "q1":  # 1-quads
        plt.plot(x_c_mu[0, :, center_mu],
                 grad_sym[t][0, 0, :, center_mu], '-',
                 marker=marker, color=colors[1])

    elif t[-2:] == "q2":  # 2-quads
        if t == "lfe_q2" and make_finer:  # finer FE on 2-quads
            # add the finer simulation
            plt.plot(x_c_mu_f[0, :, center_mu_f],
                     grad_sym["lfe_finer_q2"][0, 0, :, center_mu_f],
                     '-', marker=" ", color='gray', alpha=0.7)
        plt.plot(x_c_mu[0, :, center_mu],
                 grad_sym[t][0, 0, :, center_mu],
                 '-', marker=marker, color=colors[1])

    elif t == "pbc_esh":  # eshelby pbc
        plt.plot(x_c_E[1, 0, :, center_esh],
                 esh_strain_summed_up[1, 1, 0, :, center_esh],
                 '-', marker=marker, color=colors[1])

    # zoom
    plt.xlim(x_min_z, x_max_z)
    if args.small_simulation:
        plt.ylim(2.57, 3.6)
    else:
        plt.ylim(2.57, 3.26)

    # --- Fourth row --- #
    if i == 0:
        ax04 = plt.subplot(gs1[3*len(types) + 4 + i])
        ax2 = ax04
    else:
        ax2 = plt.subplot(gs1[3*len(types) + 4 + i], sharey=ax04, sharex=ax04)
        plt.setp(ax2.get_yticklabels(), visible=False)

    write_label_text(ax2, 3*len(types) + 4 + i)
    ax2.set_xlabel(r"$x$-position")
    if i == 0:
        ax2.set_ylabel(r"Strain  $\varepsilon_{xy}$ $(\%)$", labelpad=2.0)
    if i > 0:
        ax2.spines['left'].set_visible(False)
    ax2.tick_params(axis='both', left=False, bottom=False,
                    labelleft=False, labelbottom=False)

    if t[-2:] == "q1":
        plt.plot(x_c_mu[0, :, center_mu - shift_mu],
                 grad_sym[t][0, 1, :, center_mu - shift_mu],
                 '-', marker=marker, color=colors[0])
        plt.plot(x_c_mu[0, :, center_mu + shift_mu],
                 grad_sym[t][0, 1, :, center_mu + shift_mu],
                 '-', marker=marker, color=colors[2])

    elif t[-2:] == "q2":
        # take each triangle separate
        plt.plot(x_c_mu[0, :, center_mu - shift_mu],
                 grad_sym[t][0, 1, :, center_mu - shift_mu],
                 '-', marker=marker, color=colors[0])
        plt.plot(x_c_mu[0, :, center_mu + shift_mu],
                 grad_sym[t][0, 1, :, center_mu + shift_mu],
                 '-', marker=marker, color=colors[2])

    elif t == "pbc_esh":
        plt.plot(x_c_E[1, 0, :, center_esh - shift_esh],
                 esh_strain_summed_up[1, 2, 0, :, center_esh - shift_esh],
                 '-', marker=marker, color=colors[0])
        plt.plot(x_c_E[1, 0, :, center_esh + shift_esh],
                 esh_strain_summed_up[1, 2, 0, :, center_esh + shift_esh],
                 '-', marker=marker, color=colors[2])
    # zoom
    plt.xlim(x_min_z, x_max_z)
    plt.ylim(-0.71, 0.71)


if args.plot_file is not None:
    plt.savefig(args.plot_file, dpi=600, bbox_inches='tight')
else:
    plt.show()

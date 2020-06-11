#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file  circle_gibbs_ringing_simple_split_laminate.py

@author Ali Falsafi <ali.falsafi@epfl.ch>

@DATE   12 Jun 2019

@brief  This is a working example of how to use CellSPlit / MaterialLaminate
for a simple precipitate problem which is here is circular stiff inclusion
inside a periodic media

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
from python_example_imports import muSpectre_gradient_integration as gi
from python_example_imports import muSpectre_vtk_export as vt_ex
import sys
import numpy as np
from python_example_imports import muSpectre as msp
np.set_printoptions(linewidth=500, precision=2)


nb_grid_pt = 17
length = 12
num_vertices = 20
radius = 4.23
conv = nb_grid_pt / length

nb_grid_pts = [nb_grid_pt, nb_grid_pt]
lengths = [length, length]

center = np.array([r//2 for r in lengths])
incl = nb_grid_pt//4

formulation = msp.Formulation.finite_strain
tol = 1e-5
equil_tol = 1e-7
cg_tol = 1e-7
maxiter = 4010
verbose = 2


def rve_constructor(res, lens, form, inclusion, Del0, maxiter=401,
                    verbose=2, solver_type=msp.solvers.KrylovSolverCGEigen,
                    contrast=10, cg_tol=1e-8):
    rve = msp.Cell(res,
                   lengths,
                   formulation)
    e = 70e9
    hard = msp.material.MaterialLinearElastic1_2d.make(
        rve, "hard", contrast*e, .33)
    soft = msp.material.MaterialLinearElastic1_2d.make(
        rve, "soft",  e, .33)
    for i, pixel in rve.pixels.enumerate():
        if inclusion(pixel):
            hard.add_pixel(i)
        else:
            soft.add_pixel(i)

    if verbose:
        print("{} pixels in the inclusion".format(hard.size()))
        print("{} pixels out of the inclusion".format(soft.size()))

    tol = 1e-5
    if formulation == msp.Formulation.small_strain:
        Del0 = .5*(Del0 + Del0.T)

    solver = solver_type(rve, cg_tol, maxiter, verbose=msp.Verbosity.Silent)
    r = msp.solvers.newton_cg(rve, Del0, solver, tol, 1)
    if verbose:
        print("{} pixels in the inclusion initilised".format(hard.size()))
        print("{} pixels out of the inclusion initilised".format(soft.size()))
    return r


def split_rve_constructor(res, lens, form, inclusions, Del0, maxiter=401,
                          verbose=2, solver_type=msp.solvers.KrylovSolverCGEigen,
                          contrast=10, cg_tol=1e-8):
    rve = msp.Cell(res,
                   lengths,
                   formulation, None, 'fftw', None,
                   msp.SplitCell.split)
    # rve.set_splitness(msp.SplitCell.split)
    e = 70e9
    hard = msp.material.MaterialLinearElastic1_2d.make(
        rve, "hard", contrast*e, .33)

    for i, inclusion in enumerate(inclusions):
        rve.make_precipitate(hard, inclusion)

    soft = msp.material.MaterialLinearElastic1_2d.make(
        rve, "soft",  e, .33)
    rve.complete_material_assignment(soft)
    if formulation == msp.Formulation.small_strain:
        Del0 = .5*(Del0 + Del0.T)

    solver = solver_type(rve, cg_tol, maxiter, verbose=msp.Verbosity.Silent)
    r = msp.solvers.newton_cg(rve, Del0, solver, tol, equil_tol,
                              verbose=msp.Verbosity.Silent)
    return r


def lam_mat_rve_constructor(res, lens, form, inclusions,
                            Del0, maxiter=401, verbose=2,
                            solver_type=msp.solvers.KrylovSolverCGEigen,
                            contrast=10, cg_tol=1e-8):
    e = 70e9
    rve = msp.Cell(res,
                   lengths,
                   formulation)

    mat_hard_laminate = msp.material.MaterialLinearElastic1_2d.make_free(
        rve, "hard_free", contrast * e, 0.33)

    mat_soft_laminate = msp.material.MaterialLinearElastic1_2d.make_free(
        rve, "soft_free", e, 0.33)

    mat_hard = msp.material.MaterialLinearElastic1_2d.make(
        rve, "hard", contrast*e, .33)
    mat_soft = msp.material.MaterialLinearElastic1_2d.make(
        rve, "soft",  e, .33)

    mat_lam = msp.material.MaterialLaminate_fs_2d.make(rve, "laminate")

    for i, inclusion in enumerate(inclusions):
        rve.make_precipitate_laminate(mat_lam, mat_hard,
                                      mat_hard_laminate,
                                      mat_soft_laminate,
                                      inclusion)
    rve.complete_material_assignemnt_simple(mat_soft)

    if formulation == msp.Formulation.small_strain:
        Del0 = .5*(Del0 + Del0.T)

    solver = solver_type(rve, cg_tol, maxiter, verbose=msp.Verbosity.Silent)
    r = msp.solvers.newton_cg(rve, Del0, solver, tol, equil_tol,
                              verbose=msp.Verbosity.Silent)
    return r


points = np.ndarray(shape=(num_vertices-1, 2))
for j, tetha in enumerate(np.linspace(0, 2*np.pi, num_vertices)):
    if j != num_vertices-1:
        points[j, 0] = center[0] + radius*np.cos(tetha)
        points[j, 1] = center[1] + radius*np.sin(tetha)

points_list = points.tolist()
a = [points_list]

Del0 = np.array([[1e-3, .0],
                 [0,  1e-3]])


print(center*0.5)


def circle(pixel): return \
    (np.linalg.norm(center*conv-np.array(pixel), 2) < radius*conv)


circle_r = rve_constructor(nb_grid_pts, lengths,
                           formulation, circle, Del0)


circle_r_split = split_rve_constructor(nb_grid_pts, lengths, formulation,
                                       a, Del0, cg_tol=1e-8, maxiter=2000)


circle_r_laminate = lam_mat_rve_constructor(nb_grid_pts, lengths,
                                            formulation, a, Del0,
                                            cg_tol=1e-8, maxiter=2000)


def comp_von_mises(r):
    res2 = r.stress.size/4
    res = int(round(np.sqrt(res2), 0))
    stress = r.stress.T.reshape(res, res, 2, 2)
    out_arr = np.zeros((res, res))
    s11 = stress[:, :, 0, 0]
    s22 = stress[:, :, 1, 1]
    s21_2 = stress[:, :, 0, 1]*stress[:, :, 1, 0]

    out_arr[:] = np.sqrt(s11**2 + s22**2 - s11 * s22 + 3*s21_2)
    return out_arr


def comp_component(r, component):
    res2 = r.stress.size/4
    res = int(round(np.sqrt(res2), 0))
    stress = r.stress.T.reshape(res, res, 2, 2)
    return stress[:, :, component[0], component[1]]


def mises_plot(r, zoom=[0, 0, 1, 1]):
    vm = comp_von_mises(r)*1e-6
    figure_size = [6, 6]
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111, aspect='equal')
    xmax, ymax = vm.shape
    xslice = slice(int(xmax * zoom[0]), int(xmax * (zoom[0] + zoom[2])))
    yslice = slice(int(ymax * zoom[1]), int(ymax * (zoom[1] + zoom[3])))
    x, y = np.mgrid[xslice, yslice]
    C = ax.pcolormesh(x, y, vm[xslice, yslice], cmap=cm.get_cmap("Greys"))
    C.set_edgecolor('face')
    C.set_clim(160, 270)
    bar = plt.colorbar(C)
    bar.ax.set_ylabel("von Mises (GPa)")
    # ax.xaxis.set_major_locator(plt.MaxNLocator(nb_grid_pt))
    # ax.yaxis.set_major_locator(plt.MaxNLocator(nb_grid_pt))
    plt.tight_layout()
    return fig


def mises_plot_contour(r, zoom=[0, 0, 1, 1]):
    vm = comp_von_mises(r)*1e-6

    figure_size = [6, 6]
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111, aspect='equal')
    xmax, ymax = vm.shape
    xslice = slice(int(xmax * zoom[0]), int(xmax * (zoom[0] + zoom[2])))
    yslice = slice(int(ymax * zoom[1]), int(ymax * (zoom[1] + zoom[3])))
    x, y = np.mgrid[xslice, yslice]
    ax.locator_params(nbins=4)
    # C = ax.contour(x, y, vm[xslice, yslice], colors='k', origin='lower')
    C = ax.contour(x, y, vm[xslice, yslice])
    plt.clabel(C, inline=True, fontsize=8)
    # C.set_clim(160, 270)
    # bar = plt.colorbar(C)
    # bar.ax.set_ylabel("von Mises (GPa)")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    return fig


def component_plot_contour(r, zoom=[0, 0, 1, 1], component=[0, 0]):
    vm = comp_component(r, component)*1e-6
    figure_size = [6, 6]
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111, aspect='equal')
    xmax, ymax = vm.shape
    xslice = slice(int(xmax * zoom[0]), int(xmax * (zoom[0] + zoom[2])))
    yslice = slice(int(ymax * zoom[1]), int(ymax * (zoom[1] + zoom[3])))
    x, y = np.mgrid[xslice, yslice]
    ax.locator_params(nbins=4)
    # C = ax.contour(x, y, vm[xslice, yslice], colors='k', origin='lower')
    if component[0] == component[1]:
        C = ax.contour(x, y, vm[xslice, yslice],
                       levels=[150.0, 160.0, 170.0, 180.0,
                               190.0, 200.0, 210.0, 220.0])
        plt.clabel(C, inline=True, fontsize=8)
    elif component[0] != component[1]:
        C = ax.contour(x, y, vm[xslice, yslice],
                       levels=[5.0, 10.0, 15.0, 20.0,
                               25.0, 30.0, 35.0])
        plt.clabel(C, inline=True, fontsize=8)
    # C.set_clim(160, 270)
    # bar = plt.colorbar(C)
    # bar.ax.set_ylabel("von Mises (GPa)")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    return fig


# prevent visual output during ctest
if len(sys.argv[:]) == 2:
    if sys.argv[1] != 1:
        pass
else:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    mises_plot(circle_r).savefig("gibbs_circle_simple.pdf", dpi=300)
    mises_plot(circle_r,
               zoom=[.55, .55, .25, .25]).savefig("gibbs_circle_simple_zoom_1.pdf",
                                                  dpi=300)
    mises_plot(circle_r,
               zoom=[.59, .59, .15, .15]).savefig("gibbs_circle_simple_zoom_2.pdf",
                                                  dpi=300)

    mises_plot_contour(circle_r).savefig("gibbs_circle_contour.pdf", dpi=300)
    component_plot_contour(
        circle_r, component=[0, 0]).savefig("gibbs_circle_simple_contour"
                                            "_normal.pdf",
                                            dpi=300)

    component_plot_contour(
        circle_r, component=[0, 1]).savefig("gibbs_circle_simple_contour"
                                            "_shear.pdf",
                                            dpi=300)
    mises_plot_contour(circle_r,
                       zoom=[.55, .55, .25, .25]).savefig(
                           "gibbs_circle_simple_zoom_1_contour.pdf", dpi=300)
    mises_plot_contour(circle_r,
                       zoom=[.59, .59, .15, .15]).savefig(
                           "gibbs_circle_simple_zoom_2_contour.pdf", dpi=300)

    mises_plot(circle_r_split).savefig("gibbs_circle_split.pdf", dpi=300)
    mises_plot(circle_r_split,
               zoom=[.55, .55, .25, .25]).savefig("gibbs_circle_zoom_1_split"
                                                  ".pdf",
                                                  dpi=300)
    mises_plot(circle_r_split,
               zoom=[.59, .59, .15, .15]).savefig("gibbs_circle_zoom_2_split.pdf",
                                                  dpi=300)

    mises_plot_contour(circle_r_split).savefig("gibbs_circle_split_contour.pdf",
                                               dpi=300)

    component_plot_contour(
        circle_r_split, component=[0, 0]).savefig("gibbs_circle_split_contour_normal.pdf",
                                                  dpi=300)

    component_plot_contour(
        circle_r_split, component=[0, 1]).savefig("gibbs_circle_split_contour_shear.pdf",
                                                  dpi=300)

    mises_plot_contour(circle_r_split,
                       zoom=[.55, .55, .25, .25]).savefig(
                           "gibbs_circle_zoom_1_split_contour.pdf", dpi=300)
    mises_plot_contour(circle_r_split,
                       zoom=[.59, .59, .15, .15]).savefig(
                           "gibbs_circle_zoom_2_split_contour.pdf", dpi=300)

    mises_plot(circle_r_laminate).savefig("gibbs_circle_laminate.pdf", dpi=300)
    mises_plot(circle_r_laminate,
               zoom=[.55, .55, .25, .25]).savefig(
                   "gibbs_circle_zoom_1_laminate.pdf", dpi=300)
    mises_plot(circle_r_laminate,
               zoom=[.59, .59, .15, .15]).savefig(
                   "gibbs_circle_zoom_2_laminate.pdf", dpi=300)

    mises_plot_contour(circle_r_laminate).savefig(
        "gibbs_circle_laminate_contour.pdf", dpi=300)
    component_plot_contour(
        circle_r_laminate, component=[0, 0]).savefig("gibbs_circle_laminate_contour_normal.pdf",
                                                     dpi=300)

    component_plot_contour(
        circle_r_laminate, component=[0, 1]).savefig("gibbs_circle_laminate_contour_shear.pdf",
                                                     dpi=300)
    mises_plot_contour(circle_r_laminate,
                       zoom=[.55, .55, .25, .25]).savefig(
                           "gibbs_circle_zoom_1_laminate_contour.pdf", dpi=300)
    mises_plot_contour(circle_r_laminate,
                       zoom=[.59, .59, .15, .15]).savefig(
                           "gibbs_circle_zoom_2_laminate_contour.pdf", dpi=300)

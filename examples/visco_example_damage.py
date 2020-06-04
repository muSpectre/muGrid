#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file visco_example_damage.py

@author Ali Falsafi <afalsafi@epfl.ch>

@date   25 Feb 2020

@brief this file is an example showing how to use the damage
material viscoelatic in deviatoric and elastic in bulk loading

Copyright © 2019 Till Junge

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
import argparse

from python_example_imports import muSpectre_vtk_export as vt_ex
from python_example_imports import muSpectre_gradient_integration as gi
from python_example_imports import muSpectre as µ


def comp_component(r, component):
    res2 = r.stress.size/4
    res = int(round(np.sqrt(res2), 0))
    stress = r.stress.T.reshape(res, res, 2, 2)
    return stress[:, :, component[0], component[1]]


def compute_visco_elastic_damage(N, lens, max_iter, cg_tol, newton_tol,
                                 equil_tol, nb_steps, Ev, Einf, eta, kappa,
                                 alpha, beta, dt, dF_bar):
    formulation = µ.Formulation.small_strain
    cell = µ.Cell(N, lens, formulation)
    soft = µ.material.MaterialViscoElasticDamageSS_2d.make(
        cell, "hard", Einf, Ev, eta, .33, kappa, alpha, beta, dt)
    hard = µ.material.MaterialViscoElasticSS_2d.make(
        cell, "hard", Einf, Ev, eta, .33, dt)

    for pixel_id, pixel_coord in cell.pixels.enumerate():
        if pixel_id == 60 or pixel_id == 61 or pixel_id == 59:
            soft.add_pixel(pixel_id)
        else:
            hard.add_pixel(pixel_id)

    cell.initialise()

    dF_steps = [np.copy(dF_bar)] * nb_steps
    for i in range(len(dF_steps)):
        dF_steps[i] = dF_steps[i] * i / len(dF_steps)

    if formulation == µ.Formulation.small_strain:
        for i in range(len(dF_steps)):
            dF_steps[i] = .5 * (dF_steps[i] + dF_steps[i].T)

    solver = µ.solvers.KrylovSolverCG(cell, cg_tol, max_iter,
                                      verbose=µ.Verbosity.Silent)
    res = µ.solvers.de_geus(cell, dF_steps, solver, newton_tol,
                            equil_tol, verbose=µ.Verbosity.Silent)
    print("nb_cg: {}\nF:\n{}".format(res[-1].nb_fev,
                                     µ.gradient_integration.reshape_gradient(res[-1].grad,
                                                                             cell.nb_domain_grid_pts)[:, :, 0, 0]))
    xy = np.zeros((len(res)))
    xx = np.zeros((len(res)))
    yy = np.zeros((len(res)))
    for i, r in enumerate(res):
        xy[i] = comp_component(r, [0, 1])[5, 5]
        xx[i] = comp_component(r, [0, 0])[5, 5]
        yy[i] = comp_component(r, [1, 1])[5, 5]
    dim = 2
    fourier_gradient = [µ.FourierDerivative(dim, i) for i in range(dim)]
    for i, re in enumerate(res):
        placement_n, x = gi.compute_placement(re, lens,
                                              N, fourier_gradient,
                                              formulation=formulation)
        PK1 = gi.reshape_gradient(re.stress, N)
        F = gi.reshape_gradient(re.grad, N)
        c_data = {"PK1 stress": PK1,
                  "F deformation gradient": F}
        p_data = {}
        vt_ex.vtk_export(fpath="vico_damage_example_{}".format(i),
                         x_n=x,
                         placement=placement_n,
                         point_data=p_data,
                         cell_data=c_data)
    return xx, yy, xy


def compute_visco_elastic(N, lens, max_iter, cg_tol, newton_tol,
                          equil_tol, nb_steps, Ev, Einf, eta, dt, dF_bar):
    formulation = µ.Formulation.small_strain
    cell = µ.Cell(N, lens, formulation)
    hard = µ.material.MaterialViscoElasticSS_2d.make(
        cell, "hard", Einf, Ev, eta, .33, dt)
    for pixel_id, pixel_coord in cell.pixels.enumerate():
        hard.add_pixel(pixel_id)

    cell.initialise()

    dF_steps = [np.copy(dF_bar)] * nb_steps
    for i in range(len(dF_steps)):
        dF_steps[i] = dF_steps[i] * i / len(dF_steps)

    if formulation == µ.Formulation.small_strain:
        for i in range(len(dF_steps)):
            dF_steps[i] = .5 * (dF_steps[i] + dF_steps[i].T)
    # dF_steps = dF_steps[1:]

    solver = µ.solvers.KrylovSolverCG(cell, cg_tol, max_iter,
                                      verbose=µ.Verbosity.Silent)
    res = µ.solvers.de_geus(cell, dF_steps, solver, newton_tol,
                            equil_tol, verbose=µ.Verbosity.Silent)
    print("nb_cg: {}\nF:\n{}".format(
        res[-1].nb_fev,
        µ.gradient_integration.reshape_gradient(
            res[-1].grad,
            cell.nb_domain_grid_pts)[:, :, 0, 0]))
    xy = np.zeros((len(res)))
    xx = np.zeros((len(res)))
    yy = np.zeros((len(res)))
    for i, r in enumerate(res):
        xy[i] = comp_component(r, [0, 1])[2, 2]
        xx[i] = comp_component(r, [0, 0])[2, 2]
        yy[i] = comp_component(r, [1, 1])[2, 2]
    return xx, yy, xy


def compute():
    dF_bar = np.array([[-0.1005179, +0.500],
                       [+0.500, +0.3005179]]) * 4e-1
    # a = np.sqrt(1. / np.linalg.det(dF_bar+np.identity(2)) )
    a = 1.0
    F_bar = (dF_bar+np.identity(2)) * a
    J = np.linalg.det(F_bar)
    dF_bar = F_bar - np.identity(2)
    dE = 0.5 * (((dF_bar+np.identity(2)).T).dot(dF_bar + np.identity(2))
                - np.identity(2))
    print("dE is\n {}".format(dE))
    print("J is\n {}".format(J))
    N = [11, 11]
    lens = [1., 1.]
    nb_steps = 10
    dt = 1.0e-5
    Ev = 2.859448e5
    Einf = 1.2876e5
    eta = 1.34e3
    kappa = 1.0
    alpha = 1.0
    beta = 0.1
    cg_tol, newton_tol, equil_tol = 1e-8, 1e-5, 1e-10
    max_iter = 40
    xx_vis, yy_vis, xy_vis = \
        compute_visco_elastic(N, lens, max_iter, cg_tol, newton_tol, equil_tol,
                              nb_steps, Ev, Einf, eta,
                              dt, dE)

    xx_vis_dam, yy_vis_dam, xy_vis_dam = \
        compute_visco_elastic_damage(N, lens, max_iter, cg_tol, newton_tol,
                                     equil_tol, nb_steps, Ev, Einf, eta, kappa,
                                     alpha, beta, dt, dE)
    print(len(sys.argv[:]))
    if len(sys.argv[:]) == 2:
        print(sys.argv[1])
        if sys.argv[1] == 1:
            pass
    else:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib import rc
        font = {'size': 16}
        rc('font', **font)
        rc('text', usetex=True)
        fig = plt.figure()
        ax = plt.subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        ax.plot(+xy_vis, 'b-.', linewidth=3, label='$P_{xy}^{vis}$')
        ax.plot(+xx_vis, 'b-', linewidth=3, label='$P_{xx}^{vis}$')
        ax.plot(+yy_vis, 'b--', linewidth=3, label='$P_{yy}^{vis}$')

        ax.plot(+xy_vis_dam, 'y-.', linewidth=2, label='$P_{xy}^{vis}$')
        ax.plot(+xx_vis_dam, 'y-', linewidth=2, label='$P_{xx}^{vis}$')
        ax.plot(+yy_vis_dam, 'y--', linewidth=2, label='$P_{yy}^{vis}$')

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=2)

        ax.set_xlabel('time')
        ax.set_ylabel('stress')
        plt.show()


def main():
    compute()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file strain_stress_control_example_fem_proj.py

@author Till Junge <till.junge@altermail.ch>

@date   23 Jun 2021

@brief

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
from python_example_imports import muSpectre as msp
import muSpectre.gradient_integration as gi
from muFFT import Stencils2D


def compute_func(control, Del0, E0, formulation):
    if control == msp.solvers.MeanControl.stress_control:
        print("solve for mean STRESS control")
    elif control == msp.solvers.MeanControl.strain_control:
        print("solve for mean STRAIN control")

    nb_grid_pts = [11, 11]
    nb_domain_grid_pts = nb_grid_pts
    center = np.array([r // 2 for r in nb_grid_pts])
    incl = nb_grid_pts[0] // 5

    # Domain dimensions
    domain_lengths = [7., 5.]
    gradient = Stencils2D.linear_finite_elements
    # build a computational domain
    cell = msp.cell.CellData.make(nb_grid_pts,
                                  domain_lengths)
    cell.nb_quad_pts = 2

    # define the material properties of the matrix and inclusion
    hard = msp.material.MaterialLinearElastic1_2d.make(
        cell, "hard", 10.*E0, .3)
    soft = msp.material.MaterialLinearElastic1_2d.make(
        cell, "soft", 0.1*E0, .3)

    # assign each pixel to exactly one material
    material_geometry = np.ndarray(nb_grid_pts)
    for i, pixel in cell.pixels.enumerate():
        if np.linalg.norm(center - np.array(pixel), 2) < incl:
            material_geometry[np.array(pixel)[0], np.array(pixel)[1]] = 1
            soft.add_pixel(i)
        else:
            material_geometry[np.array(pixel)[0], np.array(pixel)[1]] = 0
            hard.add_pixel(i)

    # define the convergence tolerance for the Newton-Raphson increment
    tol = 2e-8
    # tolerance for the solver of the linear cell
    cg_tol = 2e-8
    equi_tol = 0.0
    maxiter = 500  # for linear cell solver

    verbose_krylov = msp.Verbosity.Silent
    krylov_solver = msp.solvers.KrylovSolverCG(cg_tol, maxiter,
                                               verbose_krylov)

    verbose = msp.Verbosity.Silent
    newton_solver = msp.solvers.SolverNewtonCG(
        cell, krylov_solver,
        verbose, tol, equi_tol, maxiter,
        gradient, control)
    newton_solver.formulation = formulation
    newton_solver.initialise_cell()
    newton_solver.evaluate_stress_tangent()

    reference_material_stiffness = newton_solver.tangent.map.mean()
    result = newton_solver.solve_load_increment(Del0)

    stress = result.stress
    grad = result.grad
    stress = stress.reshape(2, 2, 2, *nb_grid_pts)
    grad = grad.reshape(2, 2, 2, *nb_grid_pts)

    if len(sys.argv[:]) == 2:
        if sys.argv[1] == '1':
            pass
        else:
            pk1_matplot =\
                stress[1, 1, :, :, :].flatten()

            placement, x = \
                gi.compute_placement(result, domain_lengths,
                                     nb_domain_grid_pts, gradient)
            displ = placement
            tri, triangles = make_triangles(displ)
            triplot = plt.tripcolor(tri,
                                    pk1_matplot,
                                    edgecolors='k',
                                    shading='flat',
                                    linewidth=0.3, cmap="seismic")

    return stress, grad, reference_material_stiffness[0, 0]/E0


def main():
    # Macroscopic strain
    del0 = 1.e-6
    strain0 = np.array([[del0/2.512, del0/6.157],
                        [del0/6.157, del0/1.247]])
    strain0 = .5 * (strain0 + strain0.T)
    E0 = 1.e+1

    # formulation (small_strain or finite_strain)
    formulation = msp.Formulation.finite_strain

    const1 = 25
    print("Starting with strain control")
    print("strain input=\n{}".format(strain0))
    stress_strain_control, grad_strain_control, ratio = \
        compute_func(msp.solvers.MeanControl.strain_control,
                     strain0, E0, formulation)
    mean_stress_strain_control = stress_strain_control.mean(axis=(2, 3, 4))
    print("stress input: {}".format(mean_stress_strain_control))
    print("grad_strain_control=\n{}".format(
        grad_strain_control.mean(axis=(2, 3, 4))))
    print(
        "stress_strain_control=\n{}".format(
            stress_strain_control.mean(axis=(2, 3, 4))
        ))
    print("Feed the output of strain control to solve stress control")
    stress_flux_control, grad_flux_control, _ =\
        compute_func(msp.solvers.MeanControl.stress_control,
                     mean_stress_strain_control, E0, formulation)

    print("grad_flux_control=\n{}".format(
        (grad_flux_control.mean(axis=(2, 3, 4)))))
    print("stress_flux_control=\n{}".format(
        stress_flux_control.mean(axis=(2, 3, 4))))

    print("Δgrad:\n{}".format(grad_strain_control.mean(
        axis=(2, 3, 4))-grad_flux_control.mean(axis=(2, 3, 4))))

    # starting with stress control
    stress0 = ratio * strain0 * E0
    print("stress input=\n{}".format(stress0))
    print("Starting with stress control")
    stress_stress_control, grad_stress_control, _ = \
        compute_func(msp.solvers.MeanControl.stress_control,
                     stress0, E0, formulation)
    mean_grad_stress_control = grad_stress_control.mean(axis=(2, 3, 4))
    print("grad_stress_control=\n{}".format(
        (grad_stress_control.mean(axis=(2, 3, 4)))))
    print(
        "stress_stress_control=\n{}".format(
            stress_stress_control.mean(axis=(2, 3, 4))))
    apply_strain_2 = mean_grad_stress_control

    if formulation == msp.Formulation.finite_strain:
        apply_strain_2 -= np.identity(2)

    print("apply_strain_2=\n{}".format(apply_strain_2))
    print("Feed the output of strain control to solve stress control")
    stress_grad_control_2, grad_grad_control_2, _ = \
        compute_func(msp.solvers.MeanControl.strain_control,
                     apply_strain_2, E0, formulation)
    print("grad_grad_control=\n{}".format(
        grad_grad_control_2.mean(axis=(2, 3, 4))-np.identity(2)))
    print("stress_grad_control=\n{}".format(
        stress_grad_control_2.mean(axis=(2, 3, 4))))

    print("Δflux:\n{}".format(
        (stress_stress_control.mean(axis=(2, 3, 4))-stress0)/stress0))


if __name__ == "__main__":
    main()

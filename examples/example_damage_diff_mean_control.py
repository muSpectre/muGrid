#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   example_damage_diff_mean_control.py

@author Ali Falsafi<ali.falsafi@epfl.ch>

@date   01 Sep 2021

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

from python_example_imports import muSpectre as msp
from python_example_imports import muSpectre as µ
from python_example_imports import muSpectre_gradient_integration as gi
from python_example_imports import muSpectre_vtk_export as vt_ex
from muFFT import Stencils2D
from enum import Enum
import argparse
import random
import os
import sys
import traceback
import numpy as np
np.set_printoptions(precision=3, linewidth=130)

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


comm = MPI.COMM_WORLD
rank = comm.Get_rank()


class MakeInclusion:
    def make_inclusion(self, pixel, center, radius):
        raise NotImplementedError


class MakeCricle(MakeInclusion):
    def make_inclusion(self, pixel, center, radius):
        return np.linalg.norm(center - np.array(pixel), 2) < radius


class MakeSquare(MakeInclusion):
    def make_square_inclusion(self, pixel, center, side):
        return (np.abs(center[0]-pixel[0]) < side
                and np.abs(center[1]-pixel[1]) < side)


class MakeTiltSquare(MakeInclusion):
    def make_inclusion(self, pixel, center, side):
        rot_matrix = np.array([[+np.sqrt(2)/2, -np.sqrt(2)/2],
                               [+np.sqrt(2)/2, +np.sqrt(2)/2]])
        rot_diff = np.matmul(rot_matrix, np.array(center) - np.array(pixel))
        return (np.abs(rot_diff[0]) < side and
                np.abs(rot_diff[1]) < side)


class InclusionShape(Enum):
    circle = 1
    square = 2
    tilt = 3


def load_random_field():
    return np.load("random_field_11.npy")


class EigenStrain:
    """
    Class whose eigen_strain_func function is used to apply eigen strains
    (gel expansion)
    """

    def __init__(self, eigen_pixels_in_structure, pixels,
                 nb_subdomain_grid_pts, sub_domain_loc,
                 eigenstrain_init,
                 eigenstrain_step,
                 eigenstrain_shape):
        self.eigen_pixels_in_structure_eigen = np.full(
            tuple(eigen_pixels_in_structure.shape), False, dtype=bool)
        self.pixels = pixels
        self.nb_sub_grid = nb_subdomain_grid_pts
        self.sub_loc = sub_domain_loc
        self.eigenstrain_step = eigenstrain_step
        self.eigenstrain_init = eigenstrain_init
        self.eigenstrain_shape = eigenstrain_shape
        self.step_nb = 0
        self.count_pixels = 0
        for i in range(self.nb_sub_grid[0]):
            for j in range(self.nb_sub_grid[1]):
                pixel_coord = np.zeros([2], dtype=int)
                pixel_coord[0] = self.sub_loc[0] + i
                pixel_coord[1] = self.sub_loc[1] + j
                if eigen_pixels_in_structure[pixel_coord[0],
                                             pixel_coord[1]] > 0:
                    self.eigen_pixels_in_structure_eigen[pixel_coord[0],
                                                         pixel_coord[1]] = True
                    self.count_pixels += 1

    def __call__(self, strain_field):
        self.eigen_strain_func(strain_field)

    def eigen_strain_func(self, strain_field):
        strain_step = self.step_nb
        strain_to_apply = (self.eigenstrain_init +
                           (self.eigenstrain_step * (strain_step+1)) *
                           self.eigenstrain_shape)
        for i in range(self.nb_sub_grid[0]):
            for j in range(self.nb_sub_grid[1]):
                pixel_coord = np.zeros([2], dtype=int)
                pixel_coord[0] = self.sub_loc[0] + i
                pixel_coord[1] = self.sub_loc[1] + j
                if self.eigen_pixels_in_structure_eigen[pixel_coord[0],
                                                        pixel_coord[1]]:
                    strain_field[:, :, 0, i, j] -= strain_to_apply
                    strain_field[:, :, 1, i, j] -= strain_to_apply
        comm.Barrier()


def compute_rve_fem_solver(control_mean,
                           incl_shape=InclusionShape.circle):
    nb_steps = 10
    nb_grid_pts = [11, 11]
    center = np.array([r // 2 for r in nb_grid_pts])
    incl = nb_grid_pts[0] // 5

    # Domain dimensions
    lengths = [5., 5.]

    eigen_pixels = np.zeros(nb_grid_pts)
    eigen_pixels[center[0], center[1]] = 1

    cell_tmp = µ.cell.CellData.make(nb_grid_pts, lengths)
    cell_tmp.nb_quad_pts = 2

    eigenstrain_shape = np.identity(2)

    # eigen strain initial and final amplitude
    eigenstrain_init = 1e-6
    eigenstrain_final = 2e-5

    eigenstrain_step = ((eigenstrain_final -
                         eigenstrain_init) /
                        nb_steps)

    eigen_class = EigenStrain(eigen_pixels,
                              cell_tmp.pixels,
                              cell_tmp.nb_subdomain_grid_pts,
                              cell_tmp.subdomain_locations,
                              eigenstrain_init,
                              eigenstrain_step,
                              eigenstrain_shape)
    print("{eigen_class.count_pixels=}")
    print("{}".format(eigen_class.count_pixels))

    ## formulation (small_strain or finite_strain)
    formulation = msp.Formulation.small_strain

    # build a computational domain
    cell = msp.cell.CellData.make(nb_grid_pts, lengths)

    # Determining the number of quad_pts in each pixel of solution domain
    # might not be necessary for some of the solvers as they use their specific
    # discretisation and determined the nb_quad_pts automatically
    cell.nb_quad_pts = 2

    # elastic material parameters
    E_matrix = 4.0e9
    E_inclusion = 2.0e10

    # damage material parameter
    kappa_var = 1e-5
    kappa = 1.0e-4

    random_field = load_random_field()
    print(random_field.shape)

    alpha = 2.0

    # making materials (paste and aggregate)
    phase_matrix = μ.material.MaterialLinearElastic1_2d.make(
        cell, "Phase_Matrix", E_matrix, .33)

    phase_matrix_dam = μ.material.MaterialDunant_2d.make(
        cell, "Phase_Matrix_dam",  E_matrix, .33, kappa, alpha)

    phase_inclusion = µ.material.MaterialLinearElastic1_2d.make(
        cell, "soft_ela", E_inclusion, .33)

    phase_inclusion_dam = µ.material.MaterialDunant_2d.make(
        cell, "soft_dam", E_inclusion, .33, kappa, alpha)

    phase_gel_dam = µ.material.MaterialDunant_2d.make(
        cell, "soft_dam", 0.1 * E_matrix, .33, kappa, alpha)

    # assign each pixel to exactly one material
    material_geometry = np.ndarray(nb_grid_pts)
    inclusion_maker = MakeInclusion()
    if incl_shape == InclusionShape.circle:
        inclusion_maker = MakeCricle()
    elif incl_shape == InclusionShape.square:
        inclusion_maker = MakeSquare()
    else:
        inclusion_maker = MakeTiltSquare()

    for pixel_id, pixel in cell.pixels.enumerate():
        if inclusion_maker.make_inclusion(pixel, center, incl):
            material_geometry[np.array(pixel)[0], np.array(pixel)[1]] = 1
            # phase_inclusion.add_pixel(
            # pixel_id, kappa_var * random_field[pixel[0],
            #                                    pixel[1]])
            phase_inclusion.add_pixel(pixel_id)
        else:
            material_geometry[np.array(pixel)[0], np.array(pixel)[1]] = 0
            phase_matrix_dam.add_pixel(
                pixel_id, kappa_var * random_field[pixel[0],
                                                   pixel[1]])
            # phase_matrix.add_pixel(
            #     pixel_id)

    # define the convergence tolerance for the Newton-Raphson increment
    newton_tol = 1e-10

    # tolerance for the solver of the linear cell
    cg_tol = 1e-8
    equil_tol = 1e-7

    # Macroscopic strain
    Del0 = np.array([[1.e-6, 0.000],
                     [0.000, 1e-6]])
    Del0 = .5 * (Del0 + Del0.T)
    if  control_mean == µ.solvers.MeanControl.stress_control:
        Del0 = Del0 * E_matrix

    maxiter = 500  # for linear cell solver
    maxiter_linear = 40000  # for linear cell solver
    maxiter_newton = 2000  # for linear cell solver
    trust_region = 5.0e-1
    eta_solver = 1.e-4

    verbosity_cg = msp.Verbosity.Silent
    verbosity_newton = msp.Verbosity.Silent

    reset_cg = msp.solvers.ResetCG.gradient_orthogonality
    reset_count = 0

    krylov_solver = µ.solvers.KrylovSolverTrustRegionCG(
        cg_tol, maxiter_linear,
        trust_region,
        verbosity_cg,
        reset_cg, reset_count)

    gradient = Stencils2D.linear_finite_elements

    newton_solver = µ.solvers.SolverTRNewtonCG(
        cell, krylov_solver,
        verbosity_newton,
        newton_tol,
        equil_tol, maxiter_newton,
        trust_region, eta_solver,
        gradient, control_mean)

    newton_solver.formulation = msp.Formulation.small_strain
    newton_solver.initialise_cell()

    for i in range(nb_steps):
        print("{}th load step: ".format(i))
        # print(np.linalg.norm((i + 1) * Del0))
        result = newton_solver.solve_load_increment(
            (i + 1) * Del0,
            eigen_class.eigen_strain_func)
        eigen_class.step_nb = eigen_class.step_nb + 1

        print(result)

        if control_mean == µ.solvers.MeanControl.stress_control:
            print("expected mean flux:")
            print("{}".format((i + 1) * Del0))
            print("newton_solver.flux.get_sub_pt_map().mean()=")
            print("{}".format(
                newton_solver.flux.field.get_sub_pt_map().mean().reshape([2, 2])))
        elif control_mean == µ.solvers.MeanControl.strain_control:
            print("expected mean grad :")
            print("{}".format((i+1) * Del0))
            print("newton_solver.grad.get_sub_pt_map().mean()=")
            print("{}".format(
                newton_solver.grad.field.get_sub_pt_map().mean().reshape([2, 2])))


    if len(sys.argv[:]) == 2:
        print(sys.argv[1])
        if sys.argv[1] == 1:
            pass
        else:
            import matplotlib.pyplot as plt
            from matplotlib import rc
            import matplotlib
            import matplotlib.pyplot as plt
            matplotlib.use('tkagg')
            # visualise e.g., stress in y-direction
            stress = result.stress
            strain = result.grad
            # stress is stored in a flatten stress tensor per pixel, i.e., a
            # dim^2 × prod(nb_grid_pts_i) array, so it needs to be reshaped
            fig = plt.figure()
            stress = stress.reshape(2, 2, 2,  *nb_grid_pts)
            CS = plt.pcolormesh(np.array(newton_solver.flux).reshape(
                2, 2, 2, *nb_grid_pts)[0, 0, 0, ...])
            plt.title("Stress")
            fig.savefig("phase.png")

            fig = plt.figure()
            strain = strain.reshape(2, 2, 2,  *nb_grid_pts)
            plt.title("Strain")
            CS = plt.pcolormesh(np.array(newton_solver.grad).reshape(
                2, 2, 2, *nb_grid_pts)[0, 0, 0, ...])
            fig.savefig("strain.png")

            fig = plt.figure()
            plt.pcolormesh(material_geometry)
            plt.title("Phase")
            CS = plt.pcolormesh(material_geometry)
            fig.savefig("stress.png")

            plt.show()


def main():
    print("SOLVING A SMALL DAMAGE PROBLEM WITH **ZERO** MEAN STRAIN CONTROL")
    compute_rve_fem_solver(control_mean=µ.solvers.MeanControl.strain_control)
    print("SOLVING A SMALL DAMAGE PROBLEM WITH **ZERO** MEAN STRESS CONTROL")
    compute_rve_fem_solver(control_mean=µ.solvers.MeanControl.stress_control)


if __name__ == "__main__":
    main()

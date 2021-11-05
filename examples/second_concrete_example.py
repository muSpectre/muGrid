#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   second_concrete_example.py

@author Ali Falsafi <ali.falsafi@epfl.ch>

@date   05 May 2020

@brief  The copy of the first concrete example with considering
        the random as the strngth of the material

Copyright © 2020 Ali Falsafi

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
import random

from python_example_imports import muSpectre_vtk_export as vt_ex
from python_example_imports import muSpectre_gradient_integration as gi
from python_example_imports import muSpectre as µ


class EigenStrain:

    def __init__(self, structure, pixels, probability, eigen, coeff=1.0):
        self.structure_eigen = np.full(
            tuple(structure.shape), False, dtype=bool)
        self.pixels = pixels
        self.probability = probability
        self.eigen = eigen
        self._coeff = coeff
        for pixel_id, pixel_coord in self.pixels.enumerate():
            if structure[pixel_coord[0], pixel_coord[1]] < 0:
                if (random.random() < self.probability):
                    self.structure_eigen[pixel_coord[0], pixel_coord[1]] = True

    def __call__(self, nb_steps, strain_field):
        self.eigen_strain_func(step_nb, strain_field)

    def eigen_strain_func(self, step_nb, strain_field):
        for pixel_id, pixel_coord in self.pixels.enumerate():
            if self.structure_eigen[pixel_coord[0], pixel_coord[1]]:
                strain_field[:, :, 0, pixel_coord[0],
                             pixel_coord[1]] -= (step_nb *
                                                 self.eigen *
                                                 self._coeff)


def read_concrete_micro_structure(N):
    a = np.load("concrete_micro_structure.npy")
    return a


def make_random_field(N, lens, hurst, rms_height):
    a = np.load("random_field.npy")
    return a


def compute_response(N, lens, max_iter, cg_tol, newton_tol,
                     equil_tol, nb_steps, Ev, Einf, eta, kappa,
                     alpha, beta, dt, dE, formulation,
                     paste_in_structure, random_field, eigen_strain_step,
                     verbose, coeff, eigen_class):

    # making cell
    cell = µ.Cell(N, lens, formulation)

    # making materials (paste and aggregate)
    agg = μ.material.MaterialLinearElasticDamage2_2d.make(
        cell, "Agg", 1.5 * (Einf+Ev), .33, kappa, alpha, beta)

    paste = µ.material.MaterialViscoElasticDamageSS2_2d.make(
        cell, "Paste", Einf, Ev, eta, .33, kappa, alpha, beta, dt)

    # adding pixels to materials
    for pixel_id, pixel_coord in cell.pixels.enumerate():
        if paste_in_structure[pixel_coord[0], pixel_coord[1]] < 0:
            agg.add_pixel(pixel_id,
                          random_field[tuple(pixel_coord)])
        else:
            paste.add_pixel(pixel_id,
                            random_field[tuple(pixel_coord)])

    # initialising the cell
    cell.initialise()

    # making load steps to be passed to newton_cg solver
    dF_steps = [np.copy(dE)] * nb_steps
    for i in range(len(dF_steps)):
        dF_steps[i] = dF_steps[i] * i / len(dF_steps)

    if formulation == µ.Formulation.small_strain:
        for i in range(len(dF_steps)):
            dF_steps[i] = .5 * (dF_steps[i] + dF_steps[i].T)

    # making krylov solver and calling newton_cg solver
    solver = µ.solvers.KrylovSolverCG(cell, cg_tol, max_iter,
                                      verbose)
    res = µ.solvers.newton_cg(cell, dF_steps, solver, newton_tol,
                              equil_tol, verbose,
                              μ.solvers.IsStrainInitialised.No,
                              µ.StoreNativeStress.No,
                              eigen_class.eigen_strain_func)

    nb_iters = []
    nb_fevs = []
    for re in res:
        nb_iters.append(re.nb_it)
        nb_fevs.append(re.nb_fev)
    print(nb_iters)
    print(nb_fevs)
    np.savetxt("sec_nb_iterations_coeff_{}.csv".format(coeff),
               nb_iters, delimiter=',')
    np.savetxt("sec_nb_evaluations_coeff_{}.csv".format(coeff),
               nb_fevs, delimiter=',')

    # extracting fields
    glo_damage_agg = cell.get_globalised_current_real_field(
        "Agg_child::strain measure")

    glo_damage_paste = cell.get_globalised_current_real_field(
        "Paste_child::strain measure")

    glo_dam_np_agg = np.array(glo_damage_agg).flatten()
    glo_dam_np_paste = np.array(glo_damage_paste).flatten()
    glo_dam_np = np.fmax(glo_dam_np_agg, glo_dam_np_paste)
    glo_dam_np_exp = 1.0 - (beta + ((1-beta) *
                                    np.divide(
                                        (1-np.exp(-(glo_dam_np-kappa)/alpha)),
                                        ((glo_dam_np-kappa)/alpha))))
    dim = len(N)
    glo_dam = np.reshape(glo_dam_np, (N))
    glo_dam_exp = np.reshape(glo_dam_np_exp, (N))
    paste_in_structure[paste_in_structure >= 0] = 1.0
    paste_in_structure[paste_in_structure < 0] = np.NaN

    # making vtk output for paraview
    fourier_gradient = [µ.FourierDerivative(dim, i) for i in range(dim)]
    for i, re in enumerate(res):
        placement_n, x = gi.compute_placement(
            re, lens, N, fourier_gradient, formulation=formulation)

        PK1 = re.stress.reshape((dim, dim) + tuple(N), order='f')
        F = re.grad.reshape((dim, dim) + tuple(N), order='f')

        c_data = {"σ": PK1,
                  "ε": F,
                  "measure": glo_dam,
                  "damage": glo_dam_exp,
                  "phase": paste_in_structure}
        p_data = {}
        vt_ex.vtk_export(fpath="sec_conc_example_{}_{}_{}".format(coeff,
                                                                  N[0], i),
                         x_n=x,
                         placement=placement_n,
                         point_data=p_data, cell_data=c_data)


def compute(coeff, formulation, N, lens, paste_in_structure, eigen_class,
            dE, eigen_strain_step, random_field):
    nb_steps = 2
    dt = 1.0e-3
    Ev = 2.859448e10
    Einf = 1.2876e10
    eta = 1.34e8
    kappa = 1.0
    alpha = 0.14
    beta = 0.30
    cg_tol, newton_tol, equil_tol = 1e-8, 1e-5, 1e-10
    max_iter = 200
    verbose = µ.Verbosity.Silent

    compute_response(N, lens, max_iter, cg_tol, newton_tol,
                     equil_tol, nb_steps, Ev, Einf, eta, kappa,
                     alpha, beta, dt, dE, formulation,
                     paste_in_structure, random_field, eigen_strain_step,
                     verbose, coeff,
                     eigen_class)


def main():
    formulation = µ.Formulation.small_strain
    N = [51, 51]
    lens = [1., 1.]

    dE = np.zeros((2, 2))
    eigen_strain_step = np.array([[1.0e-5, 0.0],
                                  [0.0, 1.0e-5]])

    if formulation == μ.Formulation.small_strain:
        dE = 0.5 * (dE + dE.T)
        eigen_strain_step = 0.5 * (eigen_strain_step + eigen_strain_step.T)

    # Random field specifications
    hurst = 0.8
    rms_height = 1.0e-3

    # making a tmp cell
    cell_tmp = µ.Cell(N, lens, formulation)

    paste_in_structure = read_concrete_micro_structure(N)

    random_field = make_random_field(N, lens, hurst, rms_height)

    # making the eigen strain class that wraps the function which adds
    # eigen strain to cell strain based on the load step number
    eigen_class = EigenStrain(paste_in_structure, cell_tmp.pixels, 0.2,
                              eigen_strain_step, random_field)

    coeffs = [0.01]
    for coeff in coeffs:
        eigen_class._coeff = coeff
        print("coeff is {}".format(eigen_class._coeff))
        compute(coeff, formulation, N, lens, paste_in_structure, eigen_class,
                dE, eigen_strain_step, random_field)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   field_extraction_example.py

@author Ali Falsafi <ali.falsafi@epfl.ch>

@date   26 Jun 2020

@brief  This example is meant to examine the possibility to extract fields
        several times in different states of simulation

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
import sys
import numpy as np
import argparse

from python_example_imports import muSpectre_vtk_export as vt_ex
from python_example_imports import muSpectre_gradient_integration as gi
from python_example_imports import muSpectre as µ


def compute_response(N, lens, max_iter, cg_tol, newton_tol, equil_tol,
                     Ev, Einf, eta, dt, dE, formulation,
                     verbose):
    # making cell
    cell = µ.Cell(N, lens, formulation)

    # making material
    mat = μ.material.MaterialViscoElasticSS_2d.make(cell, "material", Einf,
                                                    Ev, eta, 0.33, dt)

    for pixel_id, pixel_coord in cell.pixels.enumerate():
        mat.add_pixel(pixel_id)

    # initialising the cell
    cell.initialise()

    # extracting fields before solution
    history_before = cell.get_globalised_current_real_field(
        "history integral").clone("history_before",
                                  allow_overwrite=True)
    nb_steps = 100
    dF_steps = [np.copy(dE)] * nb_steps

    # making krylov solver and calling newton_cg solver
    solver = µ.solvers.KrylovSolverCG(cell, cg_tol, max_iter,
                                      verbose)
    res = µ.solvers.newton_cg(cell, dF_steps, solver, newton_tol,
                              equil_tol, verbose,
                              μ.solvers.IsStrainInitialised.No)

    fourier_gradient = [µ.FourierDerivative(2, i) for i in range(2)]

    # integration of the deformation gradient field
    placement_n, x = gi.compute_placement(res[-1], lens,
                                          N, fourier_gradient,
                                          formulation=formulation)

    # extracting fields after solution
    history_after = cell.get_globalised_current_real_field(
        "history integral").clone("history_after", allow_overwrite=True)

    history = cell.get_globalised_current_real_field(
        "history integral")

    h1 = history_before.array().reshape([len(N), len(N)] +
                                        list(reversed(N)))
    h2 = history_after.array().reshape([len(N), len(N)] +
                                       list(reversed(N)))

    h0 = history.array().reshape([len(N), len(N)] +
                                 list(reversed(N)))
    new_order = [3, 2, 1, 0]
    PK1_reshape = (res[-1].stress.reshape(list(reversed(N)) + [len(N), len(N)]
                                          )).transpose(*new_order)

    dim = len(N)
    PK1_gi = res[-1].stress.reshape((dim, dim) + tuple(N), order='f')
    c_data = {"history": h0,
              "history_before": h1,
              "history_after": h2,
              "stress gi": PK1_gi,
              "stress reshape": PK1_reshape}
    p_data = {}
    vt_ex.vtk_export(fpath="field_extraction_example",
                     x_n=x,
                     placement=placement_n,
                     point_data=p_data, cell_data=c_data)


def compute():
    formulation = μ.Formulation.small_strain
    dE = np.array([[+0.1005179, +0.200],
                   [+0.200, +0.3005179]]) * 2.5e-3

    if formulation == μ.Formulation.small_strain:
        dE = 0.5 * (dE + dE.T)

    N = [3, 3]
    lens = [1., 1.]
    dt = 1.0e-3
    Ev = 2.859448e10
    Einf = 1.2876e10
    eta = 1.34e8
    verbose = µ.Verbosity.Silent
    cg_tol, newton_tol, equil_tol = 1e-8, 1e-5, 1e-10
    max_iter = 100

    compute_response(N, lens, max_iter, cg_tol, newton_tol, equil_tol,
                     Ev, Einf, eta, dt, dE, formulation,
                     verbose)


def main():
    compute()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   netcdf_io_use_case.py

@author Richard Leute <richard.leute@imtek.uni-freiburg.de>
@author Ali Falsafi <ali.falsafi@epfl.ch>

@date   09 Nov 2020

@brief  Show how to use µSpectres NetCDF-IO interface to read and write fields
        of a µSpectre simulation. This example shows three different cases:
        A) A linear elastic material with no internal variables
        B) A linear elastic material with internal variables
        C) A hyper elasto plastic material with internal variables and state
           fields that represent variables with a history
        You can switch between the different cases by setting the variable
        "case" at the beginning of this script to "A", "B" or "C".

Copyright © 2020 Till Junge

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

import os
import numpy as np
from python_example_imports import muSpectre as msp
import muGrid
from muFFT import Stencils2D
from _muSpectre.solvers import IsStrainInitialised as ISI


# parallelization
try:
    from mpi4py import MPI
    # comm = MPI.COMM_WORLD
    comm = muGrid.Communicator(MPI.COMM_WORLD)
    fft = "mpi"  # 'fftwmpi'
except ImportError:
    MPI = None
    comm = muGrid.Communicator()
    fft = "serial"

# initialise material with a cell
# volume conserving deformation
a = 0.99
b = 1/a  # (a*l) * (b*l) = l^2  ==> b = 1/a
F_tot = np.array([[a, 0],
                  [0, b]])  # final deformation
DelF_tot = F_tot - np.eye(2)

# cell
nb_grid_pts = [3, 3]  # resolution of the box
lens = nb_grid_pts  # lengths of the box
dim = len(nb_grid_pts)  # dimension of the problem

# 2D
fd_gradient_2 = Stencils2D.linear_finite_elements
gradient_op = fd_gradient_2


# materials
Young = 1
Poisson = 0.30


def run_cell(case):
    if case in ("A", "B"):
        formulation = msp.Formulation.small_strain
    elif case == "C":
        formulation = msp.Formulation.finite_strain
    cell = msp.Cell(nb_grid_pts, lens, formulation, gradient_op, fft, comm)
    mat_vac = msp.material.MaterialLinearElastic1_2d.make(
        cell, "2d-vacuum", 0.0, 0)
    if case == "A":
        # A) material linear elastic 1
        mat_le1 = msp.material.MaterialLinearElastic1_2d.make(
            cell, "2d-le1-mat",
            Young, Poisson)
    elif case == "B":
        # B) material linear elatic 4
        mat_le4 = msp.material.MaterialLinearElastic4_2d.make(
            cell, "2d-le4-mat")

    elif case == "C":
        # C) material hyper elasto plastic 2
        mu = Young / (2*(1+Poisson))
        yield_crit = mu * (0.025 + 0.01 * np.random.random(nb_grid_pts))
        hardening = 1e-2
        mat_hpl = msp.material.MaterialHyperElastoPlastic2_2d.make(
            cell, "2d-hpl")

    else:
        raise RuntimeError(
            "The selected case '{}' is not implemented. Choose one "
            "of the cases 'A', 'B' or 'C'.".format(case))

    for pixel_id, pixel in cell.pixels.enumerate():
        if np.array(pixel)[1] == nb_grid_pts[1]-1:
            # top layer in z-direction is vacuum
            mat_vac.add_pixel(pixel_id)
        else:
            if case == "A":
                mat_le1.add_pixel(pixel_id)
            elif case == "B":
                mat_le4.add_pixel(pixel_id, Young, Poisson)
            elif case == "C":
                mat_hpl.add_pixel(pixel_id, Young, Poisson,
                                  yield_crit[tuple(pixel)], hardening)

    # solver
    newton_tol = 1e-8  # tolerance for newton algo
    cg_tol = 1e-8  # tolerance for cg algo
    equil_tol = 1e-8  # tolerance for equilibrium
    maxiter = 10000
    verbose = msp.Verbosity.Full

    if case in ("A", "B"):
        solver = msp.solvers.KrylovSolverCG(cell, cg_tol, maxiter, verbose)
    elif case == "C":
        trust_region = 100.0
        dec_tr_tol = 10e-2
        inc_tr_tol = 10e-4
        solver = msp.solvers.KrylovSolverTrustRegionCG(
            cell=cell,
            tol=cg_tol,
            maxiter=maxiter,
            trust_region=trust_region,
            verbose=verbose)
    cell.initialise()

    # stepwise deformation
    n_steps = 5
    steps = 1./n_steps * np.arange(0, n_steps+1)
    DelF_list = DelF_tot[np.newaxis, :] * steps[:, np.newaxis, np.newaxis]

    ### Start muSpectre ###
    #---------------------#
    # create file io object
    file_name = "netcdf-use-case-{}.nc".format(case)
    if os.path.exists(file_name):
        if comm.rank == 0:
            os.remove(file_name)
    # wait for rank 0 to delete the old netcdf file
    MPI.COMM_WORLD.Barrier()

    file_io_object = muGrid.FileIONetCDF(
        file_name, muGrid.FileIONetCDF.OpenMode.Write, comm)
    # register global fields of the cell which you want to write
    print("cell field names:\n", cell.get_field_collection_field_names())
    cell_field_names = cell.get_field_collection_field_names()
    file_io_object.register_field_collection(
        field_collection=cell.get_field_collection(),
        field_names=["strain"])

    # register internal fields of materials
    if case == "A":
        mat_le1_field_names = mat_le1.list_fields()
        # print("material le1 field names:\n", mat_le1_field_names)
        file_io_object.register_field_collection(
            field_collection=mat_le1.collection)

    elif case == "B":
        mat_le4_field_names = mat_le4.list_fields()
        print("material le4 field names:\n", mat_le4_field_names)
        file_io_object.register_field_collection(
            field_collection=mat_le4.collection)

    elif case == "C":
        mat_hpl_field_names = mat_hpl.list_fields()
        mat_hpl_state_field_names = mat_hpl.list_state_fields()
        print("material hpl field names:\n", mat_hpl_field_names)
        print("material hpl state field names:\n", mat_hpl_state_field_names)
        file_io_object.register_field_collection(
            field_collection=mat_hpl.collection)

    # init zero strain (identity matrix for finit strain formulation)
    cell.set_uniform_strain(np.eye(dim))
    for frame, DelF in enumerate(DelF_list):
        if case in ("A", "B"):
            result = msp.solvers.newton_cg(cell, DelF, solver,
                                           newton_tol=newton_tol,
                                           equil_tol=equil_tol,
                                           verbose=verbose,
                                           IsStrainInitialised=ISI.Yes)
        elif case == "C":
            result = msp.solvers.trust_region_newton_cg(
                cell, DelF, solver,
                trust_region=trust_region,
                newton_tol=newton_tol,
                equil_tol=equil_tol,
                dec_tr_tol=dec_tr_tol,
                inc_tr_tol=inc_tr_tol,
                verbose=verbose,
                IsStrainInitialised=ISI.Yes)

        # example of selecting fields by name
        # If you do not want to write all fields it is recommended to also NOT
        # register those fields because otherwise they are empty in each frame.

        file_io_object.append_frame().write(["strain"])
        # file_io_object.append_frame().write([cell_field_names[0],
        #                                      mat_le4_field_names[0]])
        # file_io_object.append_frame().write([mat_hpl_state_field_names[0]])

        # save all fields
        # file_io_object.append_frame()
        # file_io_object.write(frame)
    file_io_object.close()


def run_cell_data(case):
    if case in ("A", "B"):
        formulation = msp.Formulation.small_strain
    elif case == "C":
        formulation = msp.Formulation.finite_strain

    if fft == "serial":
        cell = msp.cell.CellData.make_parallel(nb_grid_pts, lens, comm)
    elif fft == "mpi":
        cell = msp.cell.CellData.make(nb_grid_pts, lens)

    cell.nb_quad_pts = 2

    mat_vac = msp.material.MaterialLinearElastic1_2d.make(cell,
                                                          "2d-vacuum",
                                                          0.0,
                                                          0)
    if case == "A":
        # A) material linear elastic 1
        mat_le1 = msp.material.MaterialLinearElastic1_2d.make(cell, "2d-le1-mat",
                                                              Young, Poisson)
    elif case == "B":
        # B) material linear elatic 4
        mat_le4 = msp.material.MaterialLinearElastic4_2d.make(
            cell, "2d-le4-mat")

    elif case == "C":
        # C) material hyper elasto plastic 2
        mu = Young / (2*(1+Poisson))
        yield_crit = mu * (0.025 + 0.01 * np.random.random(nb_grid_pts))
        hardening = 1e-2
        mat_hpl = msp.material.MaterialHyperElastoPlastic2_2d.make(
            cell, "2d-hpl")

    else:
        raise RuntimeError("The selected case '{}' is not implemented. Choose one "
                           "of the cases 'A', 'B' or 'C'.".format(case))

    for pixel_id, pixel in cell.pixels.enumerate():
        if np.array(pixel)[1] == nb_grid_pts[1]-1:
            # top layer in z-direction is vacuum
            mat_vac.add_pixel(pixel_id)
        else:
            if case == "A":
                mat_le1.add_pixel(pixel_id)
            elif case == "B":
                mat_le4.add_pixel(pixel_id, Young, Poisson)
            elif case == "C":
                mat_hpl.add_pixel(pixel_id, Young, Poisson,
                                  yield_crit[tuple(pixel)], hardening)

    # solver
    newton_tol = 1e-8  # tolerance for newton algo
    cg_tol = 1e-8  # tolerance for cg algo
    equil_tol = 1e-10
    maxiter = 10000
    verbose = msp.Verbosity.Full

    if case in ("A", "B"):
        krylov_solver = msp.solvers.KrylovSolverCG(cg_tol, maxiter, verbose)
    elif case == "C":
        trust_region = 100.0
        dec_tr_tol = 10e-2
        inc_tr_tol = 10e-4
        krylov_solver = msp.solvers.KrylovSolverTrustRegionCG(
            tol=cg_tol,
            maxiter=maxiter,
            trust_region=trust_region,
            verbose=verbose)

    # stepwise deformation
    n_steps = 5
    steps = 1./n_steps * np.arange(0, n_steps+1)
    DelF_list = DelF_tot[np.newaxis, :] * steps[:, np.newaxis, np.newaxis]

    # stepwise deformation
    ### Start muSpectre ###
    #---------------------#
    # create file io object
    file_name = "netcdf-use-case-{}-cell-data.nc".format(case)
    if os.path.exists(file_name):
        if comm.rank == 0:
            os.remove(file_name)
    # wait for rank 0 to delete the old netcdf file
    MPI.COMM_WORLD.Barrier()

    file_io_object = muGrid.FileIONetCDF(
        file_name, muGrid.FileIONetCDF.OpenMode.Write, comm)
    # register global fields of the cell which you want to write
    print("cell field names:\n", cell.get_field_names())
    cell_field_names = cell.get_field_names()
    file_io_object.register_field_collection(
        field_collection=cell.fields)

    if case in ("A", "B"):
        solver = msp.solvers.SolverNewtonCG(cell,
                                            krylov_solver,
                                            verbose,
                                            newton_tol,
                                            equil_tol,
                                            maxiter,
                                            fd_gradient_2)
        solver.formulation = msp.Formulation.small_strain
    elif case == "C":
        solver = msp.solvers.SolverTRNewtonCG(cell,
                                              krylov_solver,
                                              verbose,
                                              newton_tol,
                                              equil_tol,
                                              maxiter,
                                              trust_region,
                                              1e-3,
                                              fd_gradient_2)
        solver.formulation = msp.Formulation.finite_strain
    cell.get_fields().set_nb_sub_pts("quad", 2)
    solver.initialise_cell()

    field_names = ["grad", "flux"]

    # register internal fields of materials
    if case == "A":
        mat_le1_field_names = mat_le1.list_fields()
        print("material le1 field names:\n", mat_le1_field_names)
        file_io_object.register_field_collection(
            field_collection=mat_le1.collection)

    elif case == "B":
        mat_le4_field_names = mat_le4.list_fields()
        print("material le4 field names:\n", mat_le4_field_names)
        file_io_object.register_field_collection(
            field_collection=mat_le4.collection)

    elif case == "C":
        mat_hpl_field_names = mat_hpl.list_fields()
        mat_hpl_state_field_names = mat_hpl.list_state_fields()
        print("material hpl field names:\n", mat_hpl_field_names)
        print("material hpl state field names:\n", mat_hpl_state_field_names)
        file_io_object.register_field_collection(
            field_collection=mat_hpl.collection)

    # init zero strain (identity matrix for finit strain formulation)
    # cell.set_uniform_strain(np.eye(dim))
    for frame, DelF in enumerate(DelF_list):
        result = solver.solve_load_increment(DelF)

        # example of selecting fields by name
        # If you do not want to write all fields it is recommended to also NOT
        file_io_object.append_frame().write()


def main():
    # choose the example case described in @brief to: "A", "B" or "C"
    case = "B"

    # run the examples with the cell and solver function
    run_cell(case)

    # run the examples with the cell_data and solver classed:
    run_cell_data(case)


if __name__ == "__main__":
    main()

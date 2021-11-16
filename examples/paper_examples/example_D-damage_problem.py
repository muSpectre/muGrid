#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   example_D-damage_problem.py

@author Ali Falsafi <ali.falsafi@epfl.ch>

@date   19 Mar 2021

@brief  description

Copyright © 2021 Ali Falsafi

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


# The outputs of this file is saved in a sub-folder in the example F_old_average
# one can open and visualize them using paraview

# For obtaining the results similar to figure in the paper this script_size
# should run with the following parameters:

# python3 example_D-damage_problem.py -N 201 -n 1000 -d 10 -a 10 -s 10 -r 2 - -g 0.02 -f 2.0e-3 -FEM

import os
import numpy as np
import argparse

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


from python_paper_example_imports import muSpectre_vtk_export as vt_ex
from python_paper_example_imports import muSpectre_gradient_integration as gi
from python_paper_example_imports import muSpectre as µ
from python_paper_example_imports import muSpectre as msp
from muFFT import Stencils2D

parser = argparse.ArgumentParser(
    description="Solve a very small damage problem example")

parser.add_argument("-n", "--nb_steps", type=int,
                    required=True, help="The number of solution steps")

parser.add_argument("-d", "--dump_freq", type=int,
                    required=True, help="The frequency of dumping stuff")

parser.add_argument("-g", "--gel_percentage", type=float,
                    required=True,
                    help="The percentage of gel pockets in inclusions")

parser.add_argument("-a", "--alpha", type=float,
                    required=True,
                    help=("The slop of the damage part of the cons. law" +
                          " of damage material"))

parser.add_argument("-v", "--verbosity", type=int, default=0,
                    required=False,
                    help=("verbosity of solvers (0,1,2) the greater"
                          + " the more verbose"))

parser.add_argument("-f", "--eigenstrain_final", type=float,
                    required=True,
                    help="The final eigen strain of the gel pockets")

parser.add_argument("-s", "--seed", type=int,
                    required=True,
                    help="The seed number for making gel pockets")

parser.add_argument("-r", "--reset", type=int, default=0,
                    help=("Reset criterion for cg solver"))

# parser.add_argument("-c", "--control", type=int, default=0,
#                     help=("Mean strain/stress control"))


group_proj = parser.add_mutually_exclusive_group()

group_proj.add_argument("-FOU", "--fourier",
                        action="store_true", help="The projection choice")

group_proj.add_argument("-FEM", "--fem_proj",
                        action="store_true", help="The projection choice")


control_proj = parser.add_mutually_exclusive_group()

control_proj.add_argument("-SNC", "--strain_control",
                          action="store_true", help="The mean control choice")

control_proj.add_argument("-SSC", "--stress_control",
                          action="store_true", help="The mean control choice")

args = parser.parse_args()

dim = 2
fourier_gradient = [µ.FourierDerivative(dim, i) for i in range(dim)]

if args.fourier:
    gradient = fourier_gradient
    grad_name = "fourier"
elif args.fem_proj:
    gradient = Stencils2D.linear_finite_elements
    grad_name = "fem"
else:
    raise Exception("projection not defined")

if args.strain_control:
    control_mean = µ.solvers.MeanControl.strain_control
elif args.stress_control:
    control_mean = µ.solvers.MeanControl.stress_control
else:
    raise Exception("mean control not defined")

# saving args parameters in local variables:
nb_steps = args.nb_steps
dump_freq = args.dump_freq
gel_percentage = args.gel_percentage
alpha = args.alpha
verbose = args.verbosity
eigenstrain_final = args.eigenstrain_final
seed_no = args.seed
seed_no = int(seed_no)
reset = args.reset
reset = int(reset)

reset_cg = µ.solvers.ResetCG.no_reset
reset_count = 0
reset_cg_str = ""

if reset == 0:
    reset_cg = µ.solvers.ResetCG.no_reset
    reset_count = 0
    reset_cg_str = "no_reset"
elif reset == 1:
    reset_cg = µ.solvers.ResetCG.iter_count
    reset_count = 0
    reset_cg_str = "fixed_iter_count"
elif reset == 2:
    reset_cg = µ.solvers.ResetCG.gradient_orthogonality
    reset_count = 0
    reset_cg_str = "gradient_orthogonality"
elif reset == 3:
    reset_cg = µ.solvers.ResetCG.valid_direction
    reset_count = 0
    reset_cg_str = "valid_direction"
else:
    raise ValueError


verbosity_cg = µ.Verbosity.Silent
verbosity_newton = µ.Verbosity.Silent
if verbose == 1:
    verbosity_cg = µ.Verbosity.Silent
    verbosity_newton = µ.Verbosity.Full
elif verbose == 2:
    verbosity_cg = µ.Verbosity.Full
    verbosity_newton = µ.Verbosity.Full

dim = 2
Nx = 201
N = [Nx, Nx]

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

directory = "./"
name = "_conc_new" + "_f_" + str(eigenstrain_final) + "_g_" + \
    str(gel_percentage) + "_α_" + str(alpha) + "_g_" + str(gel_percentage) + \
    "_seed_" + str(seed_no) + "_control_" + str(control_mean) + "_"  + reset_cg_str
print(name)

output_name = grad_name + name + "len_" + str(Nx) + "_steps_" + str(nb_steps)
if not os.path.exists("./" + output_name):
    os.makedirs("./" + output_name)

# solver parameters
cg_tol, newton_tol, equil_tol = 1e-8, 1e-8, 0.0
if control_mean == µ.solvers.MeanControl.strain_control:
    cg_tol, newton_tol, equil_tol = 1e-8, 1e-8, 0.0
elif control_mean == µ.solvers.MeanControl.stress_control:
    cg_tol, newton_tol, equil_tol = 1e-9, 1e-9, 0.0

maxiter_linear = 40000  # for linear cell solver
maxiter_newton = 2000  # for linear cell solver
trust_region = 5.0e-1  # initial trust region radius
eta_solver = 1.e-4  # trust region solver parameter


# cell parameters
N = [Nx, Nx]
lens = [1.0, 1.0]

# formulation (finite of small strain)
formulation = µ.Formulation.small_strain

# elastic material parameters
E1 = 4.0e9
E2 = 2.0e10

# damage material parameter
kappa_var = 1e-6
kappa = 1.0e-4

# eigen strain initial and final amplitude
eigenstrain_init = 1e-6


def load_random_field():
    return np.load("random_field_201.npy")


def load_micro_structure():
    return np.load("concrete_micro_structure_201.npy")


def make_phase():
    """
     function used to create the material phases in the cell
     phase0: matrix (soft elastic)
     phase1: matrix damage (soft damage, cement paste)
     phase2: inclusion (hard elastic)
     phase3: inclusion damage (hard damage, aggregate)
     phase4: gel elastic (super soft elastic, gel) randomly placed
    """
    # for defining the concrete micro-structure
    # from the field constructed with LSA
    treshold = -0.2

    # the field containing the phases
    phase = np.ones(N)
    micro_struct = load_micro_structure()
    eigen_pixels = np.zeros(N)
    np.random.seed(seed_no)
    count_agg = 0
    for i in range(N[0]):
        for j in range(N[1]):
            if (micro_struct[i, j] < treshold):
                count_agg += 1
    my_rands = np.random.rand(count_agg)
    k = 0
    for i in range(N[0]):
        for j in range(N[1]):
            if (micro_struct[i, j] < treshold):
                if my_rands[k] > gel_percentage:
                    phase[i, j] = 3
                else:
                    phase[i, j] = 2
                    eigen_pixels[i, j] = 1
                k += 1
    np.save("phase_N_" + str(Nx) + "_g_" + str(gel_percentage) + "_s_" +
            str(seed_no) + ".npy", phase)
    return phase, eigen_pixels


class CellEXtractionDunant:
    """
    class used for extracting the internal fields of materials
    in the intermediate steps of the solution
    """

    def __init__(self, alpha, kappa,
                 dump_freq=1, name=None):
        self.alpha = alpha
        self.kappa = kappa
        self.kappas = (load_random_field() * kappa_var) + self.kappa
        self.name = name
        self.freq = dump_freq
        self.step_nb = 0

    def dam_calc(self, kap):
        if kap > 0.0:
            dam =\
                ((1.0 + self.alpha) *
                 (self.kappa / kap)) - self.alpha
        else:
            dam = 0.0
        return dam

    def dam_calc(self, kap, kap_init):
        if kap > 0.0:
            dam =\
                ((1.0 + self.alpha) *
                 (kap_init / kap)) - self.alpha
        else:
            dam = 0.0
        return dam

    def cell_extract_func(self, solver, cell):
        step_nb = self.step_nb
        if (step_nb % self.freq == 0):
            # if(True):
            glo_kappa = cell.get_globalised_old_real_field(
                "strain measure")
            glo_stress = solver.grad.field
            glo_strain = solver.flux.field
            glo_tan = solver.tangent
            with open("./" + output_name + "/stress_strain_"+self.name +
                      "_{}_{}.npz".format(Nx, step_nb), 'wb') as f:
                np.savez(f,
                         stress=glo_stress,
                         strain=glo_strain)
            dim = cell.dims
            N = tuple(cell.nb_domain_grid_pts)
            glo_kap_np = np.array(glo_kappa)
            glo_dam_np = np.zeros_like(glo_kap_np)
            for pixel_id, pixel_coord in cell.pixels.enumerate():
                i = pixel_coord[0]
                j = pixel_coord[1]
                if grad_name == "fem":
                    for k in [0, 1]:
                        # glo_dam_np[:, k, i, j] = \
                        #     self.dam_calc(glo_kap_np[:, k, i, j])
                        glo_dam_np[:, k, i, j] = \
                            self.dam_calc(glo_kap_np[:, k, i, j],
                                          self.kappas[i, j])
                else:
                    glo_dam_np[i, j] = self.dam_calc(glo_kap_np[i, j],
                                                     self.kappas[i, j])

            glo_dam_np = np.nan_to_num(glo_dam_np)
            with open("./" + output_name + "/" + grad_name +
                      "kap_tr_{}".format(trust_region) +
                      self.name+"_{}_{}.npz".format(Nx, step_nb), 'wb') as f:
                np.savez(f, glo_kap=glo_kap_np)
            glo_dam_np[glo_dam_np < 0] = 0.
            with open("./" + output_name + "/" +
                      grad_name + "dam_tr_{}".format(
                          trust_region)
                      + self.name+"_{}_{}.npz".format(Nx, step_nb), 'wb') as f:
                np.savez(f, glo_dam=glo_dam_np)


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
                    self.count_pixels = self.count_pixels + 1

    def __call__(self, strain_field):
        self.eigen_strain_func(strain_field)

    def eigen_strain_func(self, strain_field):
        strain_step = self.step_nb
        strain_to_apply = (self.eigenstrain_init +
                           (self.eigenstrain_step * (strain_step+1)) *
                           self.eigenstrain_shape)
        # print("Eigen Strain:\n{}".format(strain_to_apply))
        for i in range(self.nb_sub_grid[0]):
            for j in range(self.nb_sub_grid[1]):
                pixel_coord = np.zeros([2], dtype=int)
                pixel_coord[0] = self.sub_loc[0] + i
                pixel_coord[1] = self.sub_loc[1] + j
                if self.eigen_pixels_in_structure_eigen[pixel_coord[0],
                                                        pixel_coord[1]]:
                    strain_field[:, :, 0, i, j] -= strain_to_apply
                    if grad_name == "fem":
                        strain_field[:, :, 1, i, j] -= strain_to_apply
        comm.Barrier()


def make_vtk_single(re, cell, solver,
                    phase_in_structure,
                    eigen_pixels_in_structure,
                    step):
    """
    function used to make vtk files
    """
    i = step
    # making vtk output for paraview
    fourier_gradient = [µ.FourierDerivative(dim, i) for i in range(dim)]
    step_nb = i + 1
    if ((step_nb % dump_freq) == 0):
        # if(True):
        if grad_name in ["fourier", "upwind", "central"]:
            grad2 = re.grad
            PK1 = re.stress.reshape((dim, dim) + tuple(N), order='f')
            F = re.grad.reshape((dim, dim) + tuple(N), order='f')
        elif grad_name == "fem":
            grad = np.reshape(np.array(re.grad), [2, 2, 2, Nx, Nx])
            grad_shape = grad.shape
            grad1 = np.average(grad, 2)
            grad2 = np.reshape(grad1,
                               (grad_shape[0],
                                grad_shape[1],
                                1,
                                *tuple(grad_shape[3:])))
            PK1 = 0.5 * (re.stress.reshape((dim, dim, 2) + tuple(N),
                                           order='f')[:, :, 0, ...] +
                         re.stress.reshape((dim, dim, 2) + tuple(N),
                                           order='f')[:, :, 1, ...])
            F = 0.5 * (re.grad.reshape((dim, dim, 2) + tuple(N),
                                       order='f')[:, :, 0, ...] +
                       re.grad.reshape((dim, dim, 2) + tuple(N),
                                       order='f')[:, :, 1, ...])
            PK1_2 = re.stress.reshape((dim, dim, 2) + tuple(N),
                                      order='f')[:, :, :, ...]
            F_2 = re.grad.reshape((dim, dim, 2) + tuple(N),
                                  order='f')[:, :, :, ...]
        placement_n, x = gi.compute_placement(
            grad2, lens,
            N, fourier_gradient,
            formulation=μ.Formulation.small_strain)
        kap_file = ("./" + output_name + "/" + grad_name +
                    "kap_tr_{}".format(trust_region) +
                    name+"_{}_{}.npz".format(Nx, step_nb))
        kap_data = np.load(kap_file)
        kap_saved = kap_data['glo_kap']
        kap_shape = kap_saved.shape
        file_dam = ("./" + output_name + "/" +
                    grad_name + "dam_tr_{}".format(
                        trust_region)
                    + name+"_{}_{}.npz".format(Nx, step_nb))
        data_dam = np.load(file_dam)
        dam_saved = data_dam['glo_dam']
        for j in range(dam_saved.shape[-1]):
            for k in range(dam_saved.shape[-2]):
                if (phase_in_structure[j, k] == 0 or
                    phase_in_structure[j, k] == 2 or
                        phase_in_structure[j, k] == 4):
                    dam_saved[..., j, k] = np.nan
        c_data = {"σ": PK1,
                  "ε": F,
                  "κ": kap_saved,
                  "Reduction": dam_saved,
                  "eigen": eigen_pixels_in_structure,
                  "phase": phase_in_structure}

        p_data = {}
        out_file_name = \
            ("./"+output_name+"/"+output_name +
             "len_{}".format(step_nb))
        if grad_name == "fem":
            stre = PK1_2.reshape((2, 2, -1), order='F').T.swapaxes(1, 2)
            stra = F_2.reshape((2, 2, -1), order='F').T.swapaxes(1, 2)
            kap = kap_saved.reshape((1, 1, -1), order='F').T.swapaxes(1, 2)
            dam = dam_saved.reshape((1, 1, -1), order='F').T.swapaxes(1, 2)
            phase_2 = np.zeros_like(dam_saved)
            eigen_2 = np.zeros_like(dam_saved)
            for k in [0, 1]:
                phase_2[0, k, ...] = phase_in_structure
                eigen_2[0, k, ...] = eigen_pixels_in_structure
            phase_2 = phase_2.reshape((1, 1, -1),
                                      order='F').T.swapaxes(1, 2)
            eigen_2 = eigen_2.reshape((1, 1, -1),
                                      order='F').T.swapaxes(1, 2)
            c_data_2 = {"stress": np.array([stre]),
                        "strain": np.array([stra]),
                        "kap": np.array([kap]),
                        "dam": np.array([dam]),
                        "phase": np.array([phase_2]),
                        "eigen": np.array([eigen_2])}
            msp.linear_finite_elements.write_2d_class(
                out_file_name + ".xdmf",
                cell, solver, c_data_2)
        else:
            vt_ex.vtk_export(
                fpath=out_file_name,
                x_n=x, placement=placement_n,
                point_data=p_data, cell_data=c_data)


def main():
    """
    main function in which the load steps are applied
    and the solvers are constructed and called to solve equilibrium
    """
    # external load
    dE = np.array([[0, 0],
                   [0, 0]])

    cell_tmp = µ.cell.CellData.make(N, lens)
    print("grad_name=", grad_name)
    if grad_name == "fem":
        cell_tmp.nb_quad_pts = 2
    else:
        cell_tmp.nb_quad_pts = 1

    eigenstrain_step = ((eigenstrain_final -
                         eigenstrain_init) /
                        nb_steps)
    print("eigenstrain_step=", eigenstrain_step)
    eigenstrain_shape = np.identity(2)

    phase_in_structure, eigen_pixels_in_structure = make_phase()

    # constructing the eigen strain application class
    eigen_class = EigenStrain(eigen_pixels_in_structure,
                              cell_tmp.pixels,
                              cell_tmp.nb_subdomain_grid_pts,
                              cell_tmp.subdomain_locations,
                              eigenstrain_init,
                              eigenstrain_step,
                              eigenstrain_shape)

    # constructing the cell extraction class
    cell_extract_class = CellEXtractionDunant(alpha, kappa,
                                              dump_freq, name)

    # making the cell
    cell = µ.cell.CellData.make(N, lens)
    if grad_name == "fem":
        cell.nb_quad_pts = 2
    else:
        cell.nb_quad_pts = 1

    # making materials (paste and aggregate)
    phase_matrix = μ.material.MaterialLinearElastic1_2d.make(
        cell, "Phase_Matrix", E1, .33)

    phase_matrix_dam = μ.material.MaterialDunant_2d.make(
        cell, "Phase_Matrix_dam",  E1, .33, kappa, alpha)

    phase_inclusion = µ.material.MaterialLinearElastic1_2d.make(
        cell, "soft_ela", E2, .33)

    phase_inclusion_dam = µ.material.MaterialDunant_2d.make(
        cell, "soft_dam", E2, .33, kappa, alpha)

    phase_gel_dam = µ.material.MaterialDunant_2d.make(
        cell, "soft_dam", 0.1 * E1, .33, kappa, alpha)

    random_field = load_random_field()

    # Adding pixels to materials
    for pixel_id, pixel_coord in cell.pixels.enumerate():
        if phase_in_structure[pixel_coord[0],
                              pixel_coord[1]] == 0:
            phase_matrix.add_pixel(pixel_id)
        elif phase_in_structure[pixel_coord[0],
                                pixel_coord[1]] == 1:
            # print(1e-8 * random_field[pixel_coord[0],
            #                    pixel_coord[1]])
            phase_matrix_dam.add_pixel(
                pixel_id,
                kappa_var * random_field[pixel_coord[0],
                                         pixel_coord[1]])
        elif phase_in_structure[pixel_coord[0],
                                pixel_coord[1]] == 2:
            phase_inclusion.add_pixel(pixel_id)
        elif phase_in_structure[pixel_coord[0],
                                pixel_coord[1]] == 3:
            phase_inclusion_dam.add_pixel(
                pixel_id,
                kappa_var * random_field[pixel_coord[0],
                                         pixel_coord[1]])
        elif phase_in_structure[pixel_coord[0],
                                pixel_coord[1]] == 4:
            phase_gel_dam.add_pixel(pixel_id)
        else:
            raise Exception("Phase Not Defined")

    # making load steps to be passed to newton_cg solver
    dF_steps = [np.copy(dE)] * nb_steps
    for i in range(1, len(dF_steps)):
        dF_steps[i] = dF_steps[i] * i / len(dF_steps)
    if formulation == µ.Formulation.small_strain:
        for i in range(len(dF_steps)):
            dF_steps[i] = .5 * (dF_steps[i] + dF_steps[i].T)
    applying_strain = dF_steps

    # making Krylov solver
    krylov_solver = µ.solvers.KrylovSolverTrustRegionCG(
        cg_tol, maxiter_linear,
        trust_region,
        verbosity_cg,
        reset_cg, reset_count)

    # creating the trust region solver
    solver = µ.solvers.SolverTRNewtonCG(
        cell, krylov_solver,
        verbosity_newton,
        newton_tol,
        equil_tol, maxiter_newton,
        trust_region, eta_solver,
        gradient, control_mean)

    solver.formulation = msp.Formulation.small_strain
    solver.initialise_cell()

    # calling the solver's solver_load_increment
    results = []
    done_steps = 0
    for i, strain_to_apply in enumerate(applying_strain):
        print("load step number: {}".format(i))
        cell_extract_class.step_nb = cell_extract_class.step_nb + 1
        res = solver.solve_load_increment(
            np.array(strain_to_apply),
            eigen_class.eigen_strain_func,
            cell_extract_class.cell_extract_func)
        print(res.message)
        if res.success:
            if (True):
                print(" making vtk for ", i, "th step")
                make_vtk_single(res, cell, solver,
                                phase_in_structure,
                                eigen_pixels_in_structure, i)
                print("mean stress:\n {}".format(
                    solver.flux.field.get_sub_pt_map().mean()))
                print("mean strain:\n {}".format(
                    solver.grad.field.get_sub_pt_map().mean()))
            results.append(res)
            eigen_class.step_nb = eigen_class.step_nb + 1
            done_steps = done_steps + 1
        else:
            print("The last step was not successful")
            break


if __name__ == "__main__":
    main()

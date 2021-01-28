#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   sensitivity_analysis.py

@author Indre Joedicke <indre.joedicke@imtek.uni-freiburg.de>

@date   22 Apr 2020

@brief  Function to perform a sensitivity analysis

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

import numpy as np
import muSpectre as µ

### ----- Helper function ----- ###
def calculate_dstress_dphase(cell, strains, Young, delta_Young, Poisson,
                              delta_Poisson, gradient=None):
    """
    Function to calculate the partial derivative of the stress with respect
    to the phase for a linear interpolation of the Youngs modulus and the
    Poisson ratios.

    Parameters
    ----------
    cell: object
        muSpectre cell object
    strains: list of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
        List of microscopic equilibrium strains in column-major order
    Young: np.ndarray(nb_pixels)
        Youngs modulos at the pixels in column-major order
    delta_Young: float
        Young(phase=1) - Young(phase=0)
    Poisson: np.ndarray(nb_pixels)
        Poissons ratios at the pixels in column-major order
    delta_Poisson: float
        Poisson(phase=1) - Poisson(phase=0)
    gradient: list of subclasses of DerivativeBase
        Type of the derivative operator used for the projection for each
        Cartesian direction. Default is FourierDerivative for each direction.

    Returns
    -------
    dstress_dphase: List of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
        List of the partial derivatives of the stress with respect to the strains.
    """
    dim = cell.nb_domain_grid_pts.dim
    nb_grid_pts = [*cell.nb_domain_grid_pts]
    nb_quad_pts = cell.nb_quad_pts

    # Derivatives of Poissons ratio and Youngs modulo with respect to the phase
    nu = Poisson
    E = Young
    lame1_deriv = (delta_Poisson*(1+2*nu**2)*E +
                       nu*(1-2*nu)*(1+nu)*delta_Young)
    lame1_deriv = lame1_deriv / ((1-2*nu)**2 * (1+nu)**2)
    lame2_deriv = 0.5*((1+nu)*delta_Young - E*delta_Poisson)
    lame2_deriv = lame2_deriv / (1+nu)**2
    Young_deriv = lame2_deriv*(3*lame1_deriv+2*lame2_deriv)
    Young_deriv = Young_deriv / (lame1_deriv+lame2_deriv)
    Poisson_deriv = 0.5 * lame1_deriv / (lame1_deriv+lame2_deriv)

    # Helper cell construction
    helper_cell = µ.Cell(cell.nb_domain_grid_pts, cell.domain_lengths,
                         cell.formulation, gradient)
    LinMat = µ.material.MaterialLinearElastic4_2d
    helper_material = LinMat.make(helper_cell, "helper_material")
    for pixel_id, pixel in helper_cell.pixels.enumerate():
        helper_material.add_pixel(pixel_id, Young_deriv[pixel_id],
                                  Poisson_deriv[pixel_id])

    # Calculate dstress_dphase
    dstress_dphase = []
    for strain in strains:
        strain = strain.reshape([dim, dim, nb_quad_pts, *nb_grid_pts], order='F')
        dstress_dphase.append(helper_cell.evaluate_stress(strain).copy())

    return dstress_dphase

### ----- Main function for sensitivty analysis ----- ###
def sensitivity_analysis(f_deriv_strains, f_deriv_phase, phase, Young1,
                         Poisson1, Young2, Poisson2, cell, krylov_solver,
                         strains, stresses, equil_tol=1e-8, gradient=None,
                         args=()):
    """
    Function to perform a sensitivity analysis based on the discrete
    adjoint method. The two materials of the problem must both be linear
    elastic. A linear interpolation for the material parameters is used in
    the diffuse interface.

    Parameters
    ----------
    f_deriv_strains: callable function
       function taking the phase, the strain, the stress, the cell and args as
       arguments and returning the partial derivative of the aim function with
       respect to the strain in form of a np.ndarray(dim**2*number of pixels).
       The iteration order of the pixels must be column-major.
    f_deriv_phase: callable function
       function taking the phase, the strain, the stress, the Young's moduli,
       the delta of the Young's modulos, the Poisson's ratios, the delta of
       the Poisson's ratios and args as arguments and returning the partial
       derivative of the aim function with respect to the phase in form of a
       np.ndarray(number of pixels). The iteration order of the pixels must be
       column-major.
    phase: np.ndarray(nb_grid_pts) of floats between 0 and 1
        Describes the material distribution. For each pixel, phase=0 corresponds
        to material 1, phase=1 corresponds to material 2. The iteration order of
        the pixels must be column-major.
    Young1: float
        Young's modulo of the first material
    Poisson1: float
        Poisson's ratio of the first material
    Young2: float
        Young's modulo of the second material
    Poisson2: float
        Poisson's ratio of the second material
    cell: object
        muSpectre Cell object
    solver: object
        muSpectre KrylovSolver object
    equil_tol: float
        tolerance for the stress in the newton step. Default is 1e-8
    strains: list of np.ndarray(dim**2 * nb_pixels) of floats
        List of microscopic equilibrium strains
    stresses: list of np.ndarray(dim**2 * nb_pixels) of floats
        List of microscopic equilibrium stresses
    args: list
        list containing additional parameters for the calculation of the partial
        derivatives. Default is args=()

    Returns
    -------
    S: np.ndarray(nb_grid_pts)
        Sensitivity at each pixel.
    """

    phase = phase.flatten(order='F')

    # Check the dimension
    dim = len(cell.nb_domain_grid_pts)
    if dim != 2:
        raise Exception("The sensitivity analysis is only implemented for 2D.")

    nb_grid_pts = cell.nb_domain_grid_pts
    nb_quad_pts = cell.nb_quad_pts
    shape = [dim, dim, nb_quad_pts, *nb_grid_pts]

    # Adjoint equation G:K:adjoint = -G:f_deriv_strain
    rhs_list = f_deriv_strains(phase, strains, stresses, cell, args)
    adjoint_list = []
    for i in range(len(strains)):
        strain = strains[i].reshape(shape, order='F')
        cell.evaluate_stress_tangent(strain)
        rhs = rhs_list[i]
        if np.linalg.norm(rhs) > equil_tol:
            rhs = rhs.reshape(shape, order='F')
            rhs = - cell.project(rhs).flatten(order='F')
            adjoint = krylov_solver.solve(rhs)
            adjoint = adjoint.reshape(shape, order='F')
            adjoint_list.append(adjoint.copy())
        else:
            adjoint_list.append(np.zeros(shape))

    # Sensitivity equation S = dfdrho + dKdrho:F adjoint
    delta_Young = Young2 - Young1
    delta_Poisson = Poisson2 - Poisson1
    Young = delta_Young*phase + Young1
    Poisson = delta_Poisson*phase + Poisson1
    dstress_dphase_list = calculate_dstress_dphase(cell, strains, Young,
                                                   delta_Young, Poisson,
                                                   delta_Poisson, gradient)
    f_deriv_phase_array = f_deriv_phase(phase, strains, stresses, cell, Young,
                                        delta_Young, Poisson, delta_Poisson,
                                        dstress_dphase_list, args)
    S = f_deriv_phase_array.flatten(order='F')

    for i in range(len(strains)):
        dstress_dphase = dstress_dphase_list[i]
        adjoint = adjoint_list[i]
        S += np.sum(adjoint*dstress_dphase, axis=(0, 1, 2)).flatten(order='F')

    return S.reshape(nb_grid_pts, order='F')

### ----- Testing the partial derivatives ----- ###
def partial_derivatives_finite_diff(aim_function, phase_ini, Young1, Poisson1,
                                    Young2, Poisson2, nb_grid_pts, lengths,
                                    formulation, DelFs, gradient=None,
                                    krylov_solver_type = µ.solvers.KrylovSolverCG,
                                    krylov_solver_args = (1e-8, 100),
                                    solver = µ.solvers.newton_cg,
                                    solver_args = (1e-6, 1e-6),
                                    nb_strain_steps=1,
                                    delta = 10e-8,
                                    args=()):
    """
    Function to calculate the partial derivatives of an aim function with
    respect to the phase and the strain using finite differences.
    Only to test the analytical calculation of the partial derivatives in small
    systems.

    Parameters
    ----------
    aim_function: callable function
       Aim function. Must take the following arguments: phase, strains,
       stresses, cell, args
    phase_ini: np.ndarray(nb_grid_pts) of floats between 0 and 1
        Describes the material distribution. For each pixel, phase=0 corresponds
        to material 1, phase=1 corresponds to material 2. The iteration order of
        the pixels must be column-major.
    Young1: float
        Young's modulo of the first material
    Poisson1: float
        Poisson's ratio of the first material
    Young2: float
        Young's modulo of the second material
    Poisson2: float
        Poisson's ratio of the second material
    nb_grid_pts: list of ints
        number of grid points in each spatial dimension
    lengths: list of floats
        length of the considered cell in each spatial dimension
    formulation: object
        µSpectre formulation object
    DelFs: list of np.ndarray(dim, dim) of floats
        List of prescribed macroscopic strain
    gradient: list of subclasses of DerivativeBase
        Type of the derivative operator used for the projection for each
        Cartesian direction. Default is FourierDerivative for each direction.
    krylov_solver_type: callable µSpectre krylov solver
        Default is µ.solvers.KrylovSolverCG
    krylov_solver_args: List of additional arguments for the krylov_solver
        krylov_solver is called with (cell, *krylov_solver_args). The
        default is (1e-8, 100).
    solver: callable µSpectre solver
        Default is µ.solvers.newton_cg
    solver_args: List of additional arguments for the solver
        krylov_solver is called with (cell, DelF, krylov_solver, *solver_args).
        The default is (1e-6, 1e-6).
    nb_strain_steps: int
        The prescribed macroscopic strains are applied in nb_strain_steps
        uniform intervalls. Default is 1.
    delta: float
        Disturbance for the finite difference calculations. Default is 10e-8
    args: list
        list containing additional parameters for the calculation of the aim
        function. Default is args=()

    Returns
    -------
    df_dstrains_list: list of np.ndarray(dim, dim, nb_quad_pts, nb_grid_pts) of
                      floats
        Partial derivative of the aim function with respect to the strain,
        calculated with finite differences.
    df_dphase: np.ndarray(nb_grid_pts) of floats
        Partial derivative of the aim function with respect to the phase,
        calculated with finite differences.

    """

    phase_ini = phase_ini.flatten(order='F')

    # Check the dimension
    dim = len(nb_grid_pts)
    if dim != 2:
        raise Exception("The sensitivity analysis is only implemented for 2D.")

    # Construct cell with initial phase distribution
    cell_ini = µ.Cell(nb_grid_pts, lengths, formulation, gradient)
    krylov_solver = krylov_solver_type(cell_ini, *krylov_solver_args)
    mat = µ.material.MaterialLinearElastic4_2d.make(cell_ini, "material")
    Young = (Young2 - Young1)*phase_ini + Young1
    Poisson = (Poisson2 - Poisson1)*phase_ini + Poisson1
    for pixel_id, pixel in cell_ini.pixels.enumerate():
        mat.add_pixel(pixel_id, Young[pixel_id], Poisson[pixel_id])

    # Shape of strains and stresses
    shape = [dim, dim, cell_ini.nb_quad_pts, *nb_grid_pts]

    # Initial aim function
    stress_ini_list = []
    strain_ini_list = []
    for i, DelF in enumerate(DelFs):
        applied_strain = []
        for s in range(nb_strain_steps+1):
            applied_strain.append( s / nb_strain_steps * DelF)
        result = solver(cell_ini, applied_strain, krylov_solver, *solver_args)
        strain = result[nb_strain_steps].grad.reshape(shape, order='F').copy()
        strain_ini_list.append(strain)
        stress = cell_ini.evaluate_stress(strain)
        stress_ini_list.append(stress.copy())

    f_ini = aim_function(phase_ini, strain_ini_list, stress_ini_list, cell_ini,
                         args)

    # Finite difference: partial derivative with respect to the phase
    phase_dist = phase_ini.copy()
    df_dphase = np.empty(phase_ini.size)
    for i in range(phase_ini.size):
        phase_dist[i] = phase_dist[i] + delta
        # Construct cell with disturbed phase distribution
        cell_dist = µ.Cell(nb_grid_pts, lengths, formulation, gradient)
        mat_dist = µ.material.MaterialLinearElastic4_2d.make(cell_dist,
                                                             "material")
        Young = (Young2 - Young1)*phase_dist + Young1
        Poisson = (Poisson2 - Poisson1)*phase_dist + Poisson1
        for pixel_id, pixel in cell_dist.pixels.enumerate():
            mat_dist.add_pixel(pixel_id, Young[pixel_id], Poisson[pixel_id])
        # Disturbed aim function
        stress_dist_list = []
        for strain_ini in strain_ini_list:
            stress_dist = cell_dist.evaluate_stress(strain_ini).copy()
            stress_dist_list.append(stress_dist)
        f_dist = aim_function(phase_dist, strain_ini_list, stress_dist_list,
                             cell_dist, args)
        # Partial derivative
        df_dphase[i] = (f_dist - f_ini) / delta
        phase_dist[i] = phase_ini[i]

    # Finite difference: partial derivatives with respect to the strains
    df_dstrains_list = []
    for index, strain_ini in enumerate(strain_ini_list):
        df_dstrain = np.empty((strain_ini.size))
        strain_dist = strain_ini.flatten(order='F')
        for i in range(strain_ini.size):
            # Disturbed strains
            strain_dist[i] += delta
            strain_dist_list = []
            for inner_index in range(len(strain_ini_list)):
                if inner_index == index:
                    strain_dist_list.append(strain_dist.reshape(shape,
                                                                order='F'))
                else:
                    strain_dist_list.append(strain_ini_list[inner_index])
            # Disturbed aim function
            stress_dist_list = []
            for inner_strain_dist in strain_dist_list:
                stress_dist = cell_ini.evaluate_stress(inner_strain_dist).copy()
                stress_dist_list.append(stress_dist)
            f_dist = aim_function(phase_ini, strain_dist_list, stress_dist_list,
                                  cell_ini, args)
            # Partial derivative
            df_dstrain[i] = (f_dist - f_ini) / delta
            strain_dist[i] -= delta

        df_dstrain = df_dstrain.reshape(shape, order='F')
        df_dstrains_list.append(df_dstrain)

    return df_dstrains_list, df_dphase

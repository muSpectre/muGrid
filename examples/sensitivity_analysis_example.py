#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   sensitivity_analysis_example.py

@author Indre Joedicke <indre.joedicke@imtek.uni-freiburg.de>

@date   12 May 2020

@brief  Example for muSpectre sensitivity analysis

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

from python_example_imports import muSpectre as µ
from python_example_imports import muFFT
from python_example_imports import muSpectre_sensitivity_analysis as sa

import numpy as np
import time

try:
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    matplotlib_found = True
except ImportError:
    matplotlib_found = False

### ----- Function definitions ----- ###
# Aim function 1 = int(phase²(1-phase)²)
def aim_function_1(phase, lengths):
    f1 = phase**2 * (1-phase)**2
    f1 = np.average(f1) * lengths[0] * lengths[1]

    return f1

# Partial derivatives of aim function 1
def f1_deriv_strain(phase, strains, stresses, cell, args):
    return [np.zeros(strains[0].size)]

def f1_deriv_phase(phase, strains, stresses, cell, E, delta_E, nu, delta_nu,
                   dstress_dstrain, args):
    lengths = cell.domain_lengths
    deriv = 2*phase*(1-phase)*(1-2*phase)
    deriv = deriv * lengths[0] * lengths[1] / cell.nb_pixels
    return deriv.flatten(order='F')

# Helper function for aim functions who make a call to muSpectre
def aim_function_helper(function_call, phase, Young1, Poisson1, Young2,
                         Poisson2, nb_grid_pts, lengths, formulation, cg_tol,
                         maxiter, DelFs, newton_tol, equil_tol, verbose, args):
    phase = phase.flatten(order='F')
    Young = (Young2 - Young1)*phase + Young1
    Poisson = (Poisson2 - Poisson1)*phase + Poisson1

    # Solve the equilibrium equations
    cell = µ.Cell(nb_grid_pts, lengths, formulation)
    solver = µ.solvers.KrylovSolverCG(cell, cg_tol, maxiter, verbose)
    mat = µ.material.MaterialLinearElastic4_2d.make(cell, "material")
    for pixel_id, pixel in cell.pixels.enumerate():
        mat.add_pixel(pixel_id, Young[pixel_id], Poisson[pixel_id])
    strains = []
    stresses = []
    for DelF in DelFs:
        result = µ.solvers.newton_cg(cell, DelF, solver, newton_tol,
                                     equil_tol, verbose=verbose)
        strain = result.grad.reshape([dim, dim, 1, *nb_grid_pts], order='F')
        stress = cell.evaluate_stress(strain).copy()
        strains.append(strain)
        stresses.append(stress)

    # Calculate the aim function
    aim_function = function_call(phase, strains, stresses, cell, args)

    return aim_function

# Aim function 2 = 1/(Lx*Ly) * int(stress_00)
def aim_function_2(phase, strains, stresses, cell, args):
    stress = stresses[0]
    f2 = np.average(stress[0, 0])

    return f2

# Partial derivatives of aim function 2
def f2_deriv_strain(phase, strains, stresses, cell, args):
    dim = cell.nb_domain_grid_pts.dim
    strain = strains[0].reshape([dim, dim, 1, *cell.nb_domain_grid_pts],
                                order='F')
    stress, tangent = cell.evaluate_stress_tangent(strain)
    deriv = tangent[0, 0] / cell.nb_pixels
    return [deriv.flatten(order='F')]

def f2_deriv_phase(phase, strains, stresses, cell, Young, delta_Young, Poisson,
                   delta_Poisson, dstress_dphase, args):
    deriv = dstress_dphase[0][0, 0] / cell.nb_pixels
    return deriv.flatten(order='F')

### ----- Parameter definitions ----- ###
# Geometry
nb_grid_pts = [3, 5]
lengths = [1, 2]

dim = len(nb_grid_pts)
hx = lengths[0]/nb_grid_pts[0]
hy = lengths[1]/nb_grid_pts[1]

# Formulation
formulation = µ.Formulation.finite_strain

# Macroscopic strain
DelF = np.zeros((dim, dim))
DelF[0, 0] = 0.1
# List containing all macroscopic strains of which the aim function depends
DelFs = [DelF]

# Material parameter
Young1 = 10
Young2 = 30
Poisson1 = 0
Poisson2 = 0.3

# Phase distribution (sinus in x-direction, constant in y-direction)
x = np.linspace(0, lengths[0], nb_grid_pts[0], endpoint=False)
x = x + hx/2
phase = np.empty(nb_grid_pts)
for j in range(nb_grid_pts[1]):
    phase[:, j] = 0.5*np.sin(2*np.pi/lengths[0]*x) + 0.5

# muSpectre solver parameters
newton_tol       = 1e-6
cg_tol           = 1e-8 # tolerance for cg algo
equil_tol        = 1e-6 # tolerance for equilibrium
maxiter          = 100
verbose          = µ.Verbosity.Silent

# disturbance of phase
delta_phase = 1e-6

### ----- Equilibrium equation ----- ###
args = ()
krylov_solver_args = (cg_tol, maxiter, verbose)
solver_args = (newton_tol, equil_tol, verbose)
delta_Young = Young2 - Young1
delta_Poisson = Poisson2 - Poisson1

# Construct cell
phase = phase.flatten(order='F')
Young = delta_Young*phase + Young1
Poisson = delta_Poisson*phase + Poisson1
cell = µ.Cell(nb_grid_pts, lengths, formulation)
mat = µ.material.MaterialLinearElastic4_2d.make(cell, "material")
for pixel_id, pixel in cell.pixels.enumerate():
    mat.add_pixel(pixel_id, Young[pixel_id], Poisson[pixel_id])

# Solve equilibrium equation
solver = µ.solvers.KrylovSolverCG(cell, *krylov_solver_args)
strains = []
stresses = []
for DelF in DelFs:
    result = µ.solvers.newton_cg(cell, DelF, solver, *solver_args)
    strain = result.grad.reshape([dim, dim, 1, *nb_grid_pts], order='F')
    stress = cell.evaluate_stress(strain).copy()
    strains.append(strain)
    stresses.append(stress)

### ----- Test partial derivatives of 2. aim function ----- ###
# Finite difference calculation of the derivatives with µSpectre
derivatives = sa.partial_derivatives_finite_diff(aim_function_2, phase,
                                                 Young1, Poisson1,
                                                 Young2, Poisson2,
                                                 nb_grid_pts, lengths,
                                                 formulation, DelFs,
                                                 krylov_solver_args=
                                                 krylov_solver_args,
                                                 solver_args=solver_args)
df2_dstrain = derivatives[0][0].flatten(order='F')
df2_dphase = derivatives[1]

# Analytical derivatives
dstress_dphase = sa.calculate_dstress_dphase(cell, strains, Young, delta_Young,
                                             Poisson, delta_Poisson)
df2_dstrain_ana = f2_deriv_strain(phase, strains, stresses, cell, args)[0]
df2_dphase_ana = f2_deriv_phase(phase, strains, stresses, cell, Young,
                                delta_Young, Poisson, delta_Poisson,
                                dstress_dphase, args)

# Comparison
if np.allclose(df2_dstrain, df2_dstrain_ana, rtol=1e-5, atol=1e-5):
    print('The partial derivative of f2 with respect to the strain is correct')
else:
    message = 'ERROR: False calculation of the partial derivative of f2'
    message = message + ' with respect to the strain'
    print(message)
    error = np.linalg.norm(df2_dstrain_ana - df2_dstrain)
    error = error/np.linalg.norm(df2_dstrain_ana)*100
    print('Error [%]', error)
if np.allclose(df2_dphase_ana, df2_dphase, rtol=1e-5, atol=1e-5):
    print('The partial derivative of f2 with respect to phase is correct')
else:
    message = 'ERROR: False calculation of the partial derivative of f2'
    message = message + ' with respect to the phase'
    print(message)
    error = np.linalg.norm(df2_dphase_ana - df2_dphase)
    error = error/np.linalg.norm(df2_dphase_ana)*100
    print('  Error [%]', error)

### ----- Sensitivity analysis with finite difference ----- ###
time_fd = time.time()

# First cost funtion
phase = phase.reshape(nb_grid_pts, order='F')
S1_fd = np.empty(phase.shape)
f1_ini = aim_function_1(phase, lengths)
for i in range(nb_grid_pts[0]):
    for j in range(nb_grid_pts[1]):
        phase[i, j] = phase[i, j] + delta_phase
        f1_dist = aim_function_1(phase, lengths)
        S1_fd[i, j] = (f1_dist - f1_ini) / delta_phase
        phase[i, j] = phase[i, j] - delta_phase

# Second cost function
S2_fd = np.empty(phase.shape)
f2_ini = aim_function_2(phase, strains, stresses, cell, args)
for i in range(nb_grid_pts[0]):
    for j in range(nb_grid_pts[1]):
        phase[i, j] = phase[i, j] + delta_phase
        f2_dist = aim_function_helper(aim_function_2, phase, Young1, Poisson1,
                                      Young2, Poisson2, nb_grid_pts, lengths,
                                      formulation, cg_tol, maxiter, DelFs,
                                      newton_tol, equil_tol, verbose, args)
        S2_fd[i, j] = (f2_dist - f2_ini) / delta_phase
        phase[i, j] = phase[i, j] - delta_phase

time_fd = time.time() - time_fd

### ----- muSpectre definitions ----- ###
# Cell
cell = µ.Cell(nb_grid_pts, lengths, formulation)

# Material
phase = phase.flatten(order='F')
delta_Young = Young2 - Young1
delta_Poisson = Poisson2 - Poisson1
Young = delta_Young*phase + Young1
Poisson = delta_Poisson*phase + Poisson1
mat = µ.material.MaterialLinearElastic4_2d.make(cell, "material")
for pixel_id, pixel in cell.pixels.enumerate():
    mat.add_pixel(pixel_id, Young[pixel_id], Poisson[pixel_id])

cell.initialise()

# Solver
krylov_solver = µ.solvers.KrylovSolverCG(cell, cg_tol, maxiter, verbose)

# Solution of the equilibrium problem
strains = []
stresses = []
for DelF in DelFs:
    result = µ.solvers.newton_cg(cell, DelF, krylov_solver, newton_tol, equil_tol,
                                 verbose=verbose)
    strains.append(result.grad.reshape((dim, dim, 1, nb_grid_pts[0],
                                            nb_grid_pts[1]), order='F'))
    stresses.append(result.stress)

### ----- Sensitivity analysis with muSpectre ----- ###
time_mu = time.time()

# First cost function
S1_mu = sa.sensitivity_analysis(f1_deriv_strain, f1_deriv_phase, phase, Young1,
                                Poisson1, Young2, Poisson2, cell,
                                krylov_solver, strains, stresses,
                                equil_tol=equil_tol)

# Second cost function
S2_mu = sa.sensitivity_analysis(f2_deriv_strain, f2_deriv_phase, phase, Young1,
                                Poisson1, Young2, Poisson2, cell,
                                krylov_solver, strains, stresses,
                                equil_tol=equil_tol)

time_mu = time.time() - time_mu

### ----- Plot results ----- ###
if matplotlib_found:
    # Plot phase
    phase = phase.reshape(nb_grid_pts[0], nb_grid_pts[1], order='F')
    fig_phase_2D, ax = plt.subplots()
    title = 'Phase distribution'
    fig_phase_2D.suptitle(title)
    c = ax.imshow(np.flip(phase, axis=1).T, extent=[0, lengths[0], 0, lengths[1]])
    cbar = fig_phase_2D.colorbar(c)
    ax.set(xlabel='x', ylabel='y')

    fig_phase, ax = plt.subplots()
    title = 'Phase distribution'
    fig_phase.suptitle(title, fontsize=22)
    ax.set_xlabel(xlabel='x', fontsize=18)
    ax.set_ylabel('phase', fontsize=18)
    ax.plot(x, phase[:, 0], linewidth=3)

    # Sensitivities of the first cost function
    fig_1, ax = plt.subplots()
    title = 'Comparison of muSpectre sensitivity analysis with finite difference'
    title = title + '\n f=int(phase²(1-phase)²)'
    fig_1.suptitle(title)
    ax.set(xlabel='x', ylabel='sensitivity')
    ax.grid()

    ax.plot(x, S1_mu[:, 0], label='muSpectre')
    ax.plot(x, S1_fd[:, 0], '--', label='finite diff')

    ax.legend()

    # Sensitivities of the second cost function
    fig_2, ax = plt.subplots()
    title = 'Comparison of muSpectre sensitivity analysis with finite difference'
    title = title + '\n f=1/Lx/Ly*int(stress)'
    fig_2.suptitle(title)
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('sensitivity', fontsize=16)
    ax.grid()

    ax.plot(x, S2_mu[:, 0], label='muSpectre', linewidth=3.5)
    ax.plot(x, S2_fd[:, 0], '--', label='finite diff', linewidth=3)

    ax.legend(fontsize=16)

    # Sensitivities of the second cost function (2D)
    fig_2_2D, ax = plt.subplots(1, 2)
    title = 'Comparison of muSpectre sensitivity analysis with finite difference'
    title = title + '\n f=1/Lx/Ly*int(stress)'
    fig_2_2D.suptitle(title)
    ax[0].set(title='muSpectre')
    ax[1].set(title='finite diff')

    c = ax[0].imshow(np.flip(S2_mu, axis=1).T, extent=[0, lengths[0], 0, lengths[1]])
    c2 = ax[1].imshow(np.flip(S2_fd, axis=1).T, extent=[0, lengths[0], 0, lengths[1]])
    cbar = fig_2_2D.colorbar(c, ax=ax[0])
    cbar2 = fig_2_2D.colorbar(c2, ax=ax[1])

    # plt.show()

# Print error between finite difference calculation and adjoint calculation
error_1 = np.linalg.norm(S1_fd - S1_mu)
if np.linalg.norm(S1_mu) > 1e-9:
    print('Relative error for f1 [%] =', error_1/np.linalg.norm(S1_mu)*100)
else:
    print('Absolute error for f1 =', error_1)

error_2 = np.linalg.norm(S2_fd - S2_mu)
if np.linalg.norm(S2_mu) > 1e-9:
    print('Relative error for f2 [%] =', error_2/np.linalg.norm(S2_mu)*100)
else:
    print('Absolute error for f2 =', error_2)

# Print time
print('Time of the finite difference calculation =', time_fd)
print('Time of the muSpectre calculation =', time_mu)

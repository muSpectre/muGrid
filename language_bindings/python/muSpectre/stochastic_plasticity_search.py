#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   stochastic_plasticity_search.py

@author Richard Leute <richard.leute@imtek.uni-freiburg.de>

@date   13 Mär 2019

@brief  A search algorithm to simulate stochastic plasticity

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

import numpy as np
import warnings
import sys
sys.path.append('/home/rl/Work/muSpectre/build_sp/language_bindings/python')
import muSpectre as µ
import muSpectre.gradient_integration as gi
from _muSpectre.solvers import IsStrainInitialised as ISI

from timeit import default_timer as timer

### For parallel version
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

### ----- Helper functions ----- ###
def compute_deviatoric_stress(stress, dim):
    """computes deviatoric stress tensor σ^{dev}=σ-\frac{1}{dim} tr(σ) I"""
    return stress - 1/dim * np.trace(stress) * np.eye(dim)

def compute_equivalent_von_Mises_stress(stress, dim):
    """computes σ_{eq}
    3D              : σ_{eq} = sqrt{3/2 · σᵈᵉᵛ:σᵈᵉᵛ}
    2D(plane stress): σ_{eq} = sqrt{σ₁₁² + σ₂₂² - σ₁₁σ₂₂ + 3·σ₁₂²}
    1D              : σ_{eq} = σ₁₁
    """
    if dim == 3:
        sigma_dev = compute_deviatoric_stress(stress, dim)
        sigma_eq  = np.sqrt(3/2 * np.tensordot(sigma_dev, sigma_dev))
    elif dim == 2:
        sigma_eq = np.sqrt(stress[0,0]**2 + stress[1,1]**2
                           - stress[0,0]*stress[1,1] + 3 * stress[0,1]**2)
    elif dim == 1:
        sigma_eq = stress[0,0]
    else:
        raise RuntimeError("The von Mises equivalent stress is not defined for "
                           "{}D".format(dim))
    return sigma_eq

def compute_strain_direction(stress, dim):
    """Strain direction which maximizes the locally dissipated energy
    strain_direction = deviatoric_stress / equivalent_von_Mises_stress"""
    sigma_dev = compute_deviatoric_stress(stress, dim)
    sigma_eq  = compute_equivalent_von_Mises_stress(stress, dim)
    return sigma_dev / sigma_eq

def update_eigen_strain(mat, pixel, stress, dim):
    """Updates the eigen_strain on one pixel."""
    strain_direction = compute_strain_direction(stress, dim)
    strain_increment = mat.get_plastic_increment(pixel) * strain_direction
    old_eigen_strain = mat.get_eigen_strain(pixel)
    new_eigen_strain = old_eigen_strain + strain_increment
    mat.set_eigen_strain(pixel, np.asfortranarray(new_eigen_strain))

def set_new_threshold(mat, pixel, inverse_cumulative_dist_func):
    if inverse_cumulative_dist_func is not None:
        new_threshold = inverse_cumulative_dist_func(np.random.random())
        mat.set_stress_threshold(pixel, new_threshold)
    else:
        pass

def propagate_avalanche_step(mat, cell, dim, solver, newton_tol,
                             newton_equil_tol, newton_verbose):
    """
    Compute elastic response and resulting overloaded pixels for zero
    deformation (DelF = 0). Thus there is only a new stress distribution because
    the plastic strain was updated and there are new stress thresholds from the
    previous avalanche part.

    Keyword Arguments:
    material         -- µSpectre material object
    cell             -- µSpectre cell object
    solver           -- µSpectre solver object
    newton_tol       -- float, tolerance for the increment in the newton step
    newton_equil_tol -- float, tolerance for the stress in the newton step
    newton_verbose   -- int (default 0), verbosity of bracket search, 0 no
                        messages, 1 print messages

    Returns:
    overloaded_pixels ndarray, Each rank returns its list of overloaded pixels.
    """
    zero_DelF = np.zeros((dim,)*2)
    µ.solvers.newton_cg(cell, zero_DelF, solver,
                        newton_tol, newton_equil_tol,
                        newton_verbose, IsStrainInitialised = ISI.Yes)
    mat.reset_overloaded_pixels()
    overloaded_pixels = mat.identify_overloaded_pixels(
        cell, np.copy(cell.stress.array((dim, dim)).reshape(-1,1)))
    return overloaded_pixels

def reshape_avalanche_history(avalanche_history_list, dim):
    """
    Reshape the nested lists of the avalanche history by filling it with NaNs in
    such a way that the nested lists can be written as one np.ndarray(), with a
    shape: (#avalanche time steps, maximum avalanche in one time step, dim)

    Keyword Arguments:
    avalanche_history_list
    dim

    Returns:
    """
    ava_hl = avalanche_history_list
    length = len(max(avalanche_history_list, key=len))
    fill = [[np.nan]*dim]
    return np.array([ava_i.tolist()+fill*(length-len(ava_i)) for ava_i in ava_hl])

def compute_average_deformation_gradient(F_field, dim):
    return np.average(F_field, axis = tuple(range(2, dim+2)))


### ----- Main functions ----- ###
def bracket_search(material, cell, solver, newton_tol, newton_equil_tol,
                   yield_surface_accuracy, n_max_bracket_search, DelF_initial,
                   verbose = 0, test_mode=False):
    """
    Tries to find the yield surface of the first breaking pixel in an material
    to at least the accuracy 'yield_surface_accuracy' by a bisection method.

    Keyword Arguments:
    material               -- µSpectre material object
    cell                   -- µSpectre cell object
    solver                 -- µSpectre solver object
    newton_tol             -- float, tolerance for the increment in the newton step
    newton_equil_tol       -- float, tolerance for the stress in the newton step
    yield_surafce_accuracy -- float, determines the accuracy of the bisection
                              methode in the sense of:
                              |F_true - F_num| < yield_surface_accuracy
                              where F_true is the theoretic value and F_num the
                              numeric estimate for the deformation gradient and
                              |·| is the Frobenius matrix norm.
    n_max_bracket_search   -- int, the maximum number of bisection steps.
    DelF_initial           -- initial step to increase the deformation gradient
                              F. It defines the deformation "direction" thus the
                              total deformation F_tot should be given as F_tot =
                              a*DelF_initial for a real number a.
    verbose                -- int (default 0), verbosity of bracket search, 0 no
                              messages, 1 print messages
    test_mode              -- bool (default False), if True bracket_search()
                              returs (next_DelF_guess, PK2, F, breacking_pixel)
                              For False it returns only (next_DelF_guess, PK2, F)

    Returns:
    next_DelF_guess np.ndarray (dim, dim), guess for the search step size DelF
                    of the next plastically deforming pixel (dtype = float)
    PK2             np.ndarray (nx*ny*nz, dim*dim), Piola Kirchhoff stress 2
                    (dtype = float)
    F               np.ndarray (nx*ny*nz, dim*dim), deformation gradient
                    (dtype = float)
    Only for "test_mode = True":
    breaking_pixel  np.array, index of pixel(s) which was(were) found to break
                    at 'F_num' within an accuracy of yield_surface_accuracy
                    (yield_surface_accuracy² for more than one pixel)
                    (|F_true - F_num| < yield_surface_accuracy)
    """
    #initialize parameters
    mat         = material
    ys_acc      = yield_surface_accuracy
    comm = cell.communicator
    dim         = len(cell.nb_domain_grid_pts)
    DelF_final  = np.copy(DelF_initial)
    DelF        = np.copy(DelF_initial)
    DelF_step_size  = np.copy(DelF_initial)
    next_DelF_guess = np.copy(DelF_initial)
    newton_verbose  = 0
    previous_step_was = "initial_step"
    n_bracket = 0
    first_time = True #do I overcome exactly one pixel the first time or not?

    #Small intial step size error
    if np.linalg.norm(DelF_initial) < (1 if ys_acc > 1 else ys_acc**2)/1e4:
        raise ValueError("Your initial step size DelF_initial seems to be "
                         "verry small or even zero!\nDelF_initial:\n{}"\
                         .format(DelF_initial))

    # TODO rleute Bracket search goes forever
    # start = timer()
    # while (n_bracket < n_max_bracket_search):
    #     if timer()-start > 1:
    #         raise Exception("bracket search wouldn't finish within a second")
    #     ### compute elastic response ------------------------------------------#
    #     result = µ.solvers.newton_cg(cell, DelF, solver,
    #                                  newton_tol, newton_equil_tol,
    #                                  newton_verbose,
    #                                  IsStrainInitialised = ISI.Yes)
    #     PK2 = np.copy(result.stress)
    #     mat.reset_overloaded_pixels()
    #     overloaded_pixels = mat.identify_overloaded_pixels(cell, PK2)
    #     if not overloaded_pixels:
    #         #prevent empty list which leads to incompatible function argument in
    #         #comm.gather()
    #         overloaded_pixels = np.empty((0, dim))
    #     n_pix = comm.sum(len(overloaded_pixels))

    #     ### Decide "elastic" or "too_large_step" ?-----------------------------#
    #     if n_pix == 0:
    #         if previous_step_was == "elastic":
    #             #go twice the step forward
    #             DelF_step_size *= 2
    #             DelF = +DelF_step_size
    #         elif previous_step_was == "too_large_step":
    #             #go half the step forward
    #             DelF_step_size /= 2
    #             DelF = +DelF_step_size
    #         elif previous_step_was == "initial_step":
    #             #keep everything as it is initialized
    #             pass

    #         previous_step_was = "elastic"

    #     elif n_pix >= 1:
    #         if n_pix == 1 and first_time:
    #             #safe the current deformation step as guess for the next step
    #             next_DelF_guess = np.copy(DelF_step_size)
    #             first_time = False

    #         DelF_step_size /= 2
    #         DelF = - DelF_step_size
    #         previous_step_was = "too_large_step"

    #     else:
    #         if comm.rank == 0:
    #             n_pix_error = \
    #                 ("\nSomething went totally wrong!\nThe number of overloaded"
    #                  "pixels 'n_pix' is "+str(n_pix)+"\nThis case is not "
    #                  "handeled and probably doesn't make sense.")
    #             raise RuntimeError(n_pix_error)

    #     ### Check convergence + RETURN STATEMENT --------------------------- ###
    #     reached_accuracy = (2 * np.linalg.norm(DelF_step_size) < ys_acc)
    #     if reached_accuracy:
    #         if n_pix == 1:
    #             F = result.grad
    #             if test_mode or verbose:
    #                 breaking_pixel = comm.gather(np.array(overloaded_pixels,
    #                                     dtype=np.int32).reshape((-1)).T).T
    #                 if verbose and comm.rank == 0:
    #                     print("I needed {} bracket search steps."
    #                           .format(n_bracket))
    #                     print("The following pixel broke: ", breaking_pixel)
    #             ### RETURNS:
    #             if test_mode:
    #                 return next_DelF_guess, PK2, F, breaking_pixel
    #             else:
    #                 return next_DelF_guess, PK2, F #<== RETURN

    #         reached_npixel_accuracy = (2 * np.linalg.norm(DelF_step_size) <
    #                                    (1 if ys_acc > 1 else ys_acc**2))
    #         if reached_npixel_accuracy and (n_pix > 1):
    #             warnings.warn("The avalanche starts with {} initially "
    #                 "plastically deforming pixels, instead of starting with one"
    #                 " pixel!".format(n_pix), RuntimeWarning)
    #             F = result.grad
    #             if test_mode or verbose:
    #                 breaking_pixel = comm.gather(np.array(overloaded_pixels,
    #                                     dtype=np.int32).reshape((-1)).T).T
    #                 if verbose and comm.rank == 0:
    #                     print("I needed {} bracket search steps."
    #                           .format(n_bracket))
    #                     print("The following pixels broke simultaneously",
    #                           breaking_pixel)
    #             ### RETURNS:
    #             if test_mode:
    #                 return next_DelF_guess, PK2, F, breaking_pixel
    #             else:
    #                 return next_DelF_guess, PK2, F #<== RETURN

    #     ### Update parameters before loop ends --------------------------------#
    #     DelF_final += DelF
    #     n_bracket += 1

    # if n_bracket == n_max_bracket_search:
    #     n_max_error = \
    #         ("The maximum number, n_max_bracket_search={}, of bracket search "
    #          "steps was reached.\nEither there is a problem or you should "
    #          "increase the maximum number of allowed bracket search steps."
    #          .format(n_max_bracket_search))
    #     F_ava = compute_average_deformation_gradient(cell.strain, dim) \
    #         * cell.nb_subdomain_grid_pts[-1] / cell.nb_domain_grid_pts[-1]
    #     F_ava_tot = comm.sum(F_ava)

    #     if comm.rank == 0 and verbose:
    #         print("Last stepsize DelF_step_size:\n", DelF_step_size)
    #         print("computed average deformation:\n", F_av_tot.T)
    #     raise RuntimeError(n_max_error)


def propagate_avalanche(material, cell, solver, newton_tol, newton_equil_tol,
                        PK2, F_initial, n_max_avalanche, verbose = 0,
                        inverse_cumulative_dist_func = None,
                        save_avalanche = None, n_strain_loop = 0):
    """
    Starting from an initially plastically deforming pixel it updates the
    strain_field and thresholds of all pixels that undergoe a plastic
    deformation during the avalanche process.

    Keyword Arguments:
    material         -- µSpectre material object
    cell             -- µSpectre cell object
    solver           -- µSpectre solver object
    newton_tol       -- float, tolerance for the increment in the newton step
    newton_equil_tol -- float, tolerance for the stress in the newton step
    PK2              -- np.ndarray (nx*ny*nz, dim*dim), Piola Kirchhoff stress 2
                        as given from µSpectre::solvers::OptimizeResult.stress
                        or bracket_search (dtype = float)
    F_initial        -- np.ndarray (nx*ny*nz, dim*dim), deformation gradient
                        as given from µSpectre::solvers::OptimizeResult.grad
                        or bracket_search (dtype = float)
    n_max_avalanche  -- int, maximum number of avalanche time steps
    verbose          -- int, verbosity of propagate_avalanche, 0 no messages,
                        1 print messages (default 0)
    inverse_cumulative_dist_func -- lambda function, inverse cummulative distri-
                        bution function of the yield threshold distribution
                        which is used to evaluate new random yield thresholds.
                        The lambda function gets as input a random number from a
                        uniform distribution in the interval [0,1) produced by
                        np.random.random(). (default None, no new thresholds are
                        drawn, thus constant yield thresholds in the simulation)
    save_avalanche   -- call back function, which gets the ordered parameters:
                         n_strain_loop -- int, number of the strain loop step
                         ava_history   -- np.array of shape(avalanche timesteps,
                                          #plastic deformations, dim) containing
                                          the plastic deformed pixel indices.
                          PK2_initial  -- PK2-stress before avalanche
                          F_initial    -- deformation gradient before avalanche
                          PK2_final    -- PK2-stress after avalanche
                          F_final      -- deformation gradient after avalanche
                          comm         -- muFFT::Communicator object
                        (default None, the avalanche is not saved)
    n_strain_loop    -- int, number of the actual strain loop step is only used
                        for documentation of the avalanche and therefore passed
                        to the call back function save_avalanche (default = 0).

    Returns:
        No return except of the call back function "save_avalanche".
    """
    mat  = material
    comm = cell.communicator
    dim  = len(cell.nb_domain_grid_pts) #dimension
    if comm is None:
        comm.size = 0
    mat.reset_overloaded_pixels()
    overloaded_pixels = mat.identify_overloaded_pixels(cell, PK2)
    if not overloaded_pixels:
        #prevent empty list which leads to incompatible function argument in
        #comm.gather()
        overloaded_pixels = np.empty((0, dim))
    n_pix_avalanche = comm.sum(len(overloaded_pixels))

    n_avalanche = 0
    newton_verbose = 0
    # TODO (rleute)
    # PK2_pixel = gi.reshape_gradient(PK2, cell.nb_subdomain_grid_pts)

    # if save_avalanche is not None:
    #     ava_history = []
    #     PK2_initial   = np.copy(PK2_pixel)
    #     F_initial     = F_initial

    # while (n_pix_avalanche >= 1) and (n_avalanche < n_max_avalanche):
    #     if save_avalanche is not None:
    #         ava_t = comm.gather(np.array(overloaded_pixels,
    #                             dtype=np.int32).reshape((-1, dim)).T).T
    #         if comm.rank == 0:
    #             ava_history.append(ava_t)

    #     for pixel in overloaded_pixels:
    #         if comm.size > 1:
    #             #shift pixel by subdomain location because each core has only
    #             #its part of the stress tensor, but the pixel index is global!
    #             index = tuple(np.array(pixel) -
    #                           np.array(cell.subdomain_locations)) + (...,)
    #         else:
    #             index = tuple(pixel)+(...,)
    #         update_eigen_strain(mat, pixel, PK2_pixel[index], dim)
    #         set_new_threshold(mat, pixel, inverse_cumulative_dist_func)
    #     overloaded_pixels =\
    #         propagate_avalanche_step(mat, cell, dim, solver, newton_tol,
    #                                  newton_equil_tol, newton_verbose)
    #     n_pix_avalanche = comm.sum(len(overloaded_pixels))

    #     n_avalanche += 1

    # if verbose:
    #     print("I needed {} avalanche time steps".format(n_avalanche))

    # if (n_avalanche == n_max_avalanche): #recognise if avalanche overflows!
    #     max_avalanche_error = \
    #     ("\n"+str(n_avalanche)+" avalanche steps!\nYou have reached the"
    #      " maximum allowed avalanche size of " + str(n_max_avalanche) +
    #      " internal steps.\nIncrease the maximum allowed avalanche "
    #      "steps 'n_max_avalanche' if it does make sense.")
    #     raise RuntimeError(max_avalanche_error)

    # if save_avalanche is not None:
    #     if comm.rank == 0:
    #         PK2_final = cell.stress
    #         F_final   = cell.strain
    #         ava_history = reshape_avalanche_history(ava_history, dim)
    #         save_avalanche(n_strain_loop, ava_history,
    #                        PK2_initial, F_initial,
    #                        PK2_final, F_final, comm)

def strain_cell(material, cell, solver, newton_tol, newton_equil_tol,
                DelF, F_tot, yield_surface_accuracy, n_max_strain_loop,
                n_max_bracket_search, n_max_avalanche, verbose = 0,
                inverse_cumulative_dist_func = None, save_avalanche = None,
                is_strain_initialised = False):
    """
    Strain the cell stepwise to the maximum deformation of F_tot. By default the
    cell is initialised with zero deformation (finite strain: np.eye(dim)).

    Keyword Arguments:
    material         -- µSpectre material object
    cell             -- µSpectre cell object
    solver           -- µSpectre solver object
    newton_tol       -- float, tolerance for the increment in the newton step
    newton_equil_tol -- float, tolerance for the stress in the newton step
    DelF             -- np.ndarray (dim, dim), step to increase the deformation
                        gradient F. It defines the deformation "direction" thus
                        the total deformation F_tot should be given as F_tot =
                        a*DelF for a real number a. (dtype float)
    F_tot            -- np.ndarray (dim, dim), required final deformation (dtype
                        float)
    yield_surface_accuracy -- float, determines the accuracy of the bisection
                              methode in the sense of:
                              |F_true - F_num| < yield_surface_accuracy
                              where F_true is the theoretic value and F_num the
                              numeric estimate for the deformation gradient at
                              which the first pixel deforms plastic and |·| is
                              the Frobenius matrix norm.
    n_max_strain_loop      -- int, maximum number of strain loop steps
    n_max_bracket_search   -- int, the maximum number of bisection steps.
    n_max_avalanche        -- int, maximum number of avalanche time steps
    verbose                -- int, verbosity of strain cell, 0 no messages,
                              1 print messages (default 0)
    inverse_cumulative_dist_func -- lambda function, inverse cummulative distri-
                        bution function of the yield threshold distribution
                        which is used to evaluate new random yield thresholds.
                        The lambda function gets as input a random number from a
                        uniform distribution in the interval [0,1) produced by
                        np.random.random(). (default None, no new thresholds are
                        drawn, thus constant yield thresholds in the simulation)
    save_avalanche   -- call back function, which gets the ordered parameters:
                         n_strain_loop -- int, number of the strain loop step
                         ava_history   -- np.array of shape(avalanche timesteps,
                                          #plastic deformations, dim) containing
                                          the plastic deformed pixel indices.
                         PK2_initial  -- PK2-stress before avalanche
                         F_initial    -- deformation gradient before avalanche
                         PK2_final    -- PK2-stress after avalanche
                         F_final      -- deformation gradient after avalanche
                         comm         -- muFFT::Communicator object
                        (default None, the avalanche is not saved)
    is_strain_initialised  -- (default False) characterises wheather the strain
                              is initialised by the user (True) or automatic to
                              zero deformation (finite strain: np.eye(dim))
                              (dtype = bool)

    Returns:
        F_ava_tot, np.ndarray of shape (dim, dim) containing the final reached
        average deformation gradient (dtype float).
    """
    #---------------------------- while loop init stuff -----------------------#
    comm = cell.communicator
    dim     = len(cell.nb_domain_grid_pts)
    n_strain_loop = 0
    DelF = np.copy(DelF)

    F_old_average = np.eye(dim) #grid averaged F_old (zero deformation at start)
    test_mask = np.where(DelF != 0)

    #if not initialised by user, initialise unit-matrix deformation gradient
    if not is_strain_initialised:
        cell_strain = cell.strain.array((dim, dim))
        np.squeeze(cell_strain)[:] = np.tensordot(
            np.eye(dim), np.ones(cell.nb_subdomain_grid_pts), axes=0)

    #--------------------- cross check input on consistency -------------------#
    if yield_surface_accuracy <=0 :
        raise ValueError("The yield_surface_accuracy should be an non zero "
                         "positive value!\nYour input was: {}"\
                         .format(yield_surface_accuracy))
    if n_max_strain_loop <= 0 or not isinstance(n_max_strain_loop, int):
        raise ValueError("The maximum number of strain steps "
                         "'n_max_strain_loop' should be an non zero positive "
                         "integer value!\nYour input was: {}"\
                         .format(n_max_strain_loop))
    if n_max_bracket_search <= 0  or not isinstance(n_max_bracket_search, int):
        raise ValueError("The maximum number of bracket search steps "
                         "'n_max_bracket_search' should be an non zero positive"
                         " integer value!\nYour input was: {}"\
                         .format(n_max_bracket_search))
    if n_max_avalanche <= 0  or not isinstance(n_max_avalanche, int):
        raise ValueError("The maximum number of time steps during an avalanche "
                         "'n_max_avalanche' should be an non zero positive "
                         "integer value!\nYour input was: {}"\
                         .format(n_max_avalanche))
    #Check if the deformation steps DelF can lead to the required deformation
    deformation_mask = np.where(F_tot - F_old_average != 0)
    if not (np.array(test_mask) == np.array(deformation_mask)).all():
        raise ValueError("\nProbably you can not reach the desired deformation "
                         "F_tot:\n{}\n\nby steps of DelF:\n{}\n\nstarting from"
                         " an initial deformation F_0:\n{}\n\nTherefore choose "
                         "proper steps DelF!".format(F_tot, DelF, F_old_average))
    if inverse_cumulative_dist_func is None:
        warnings.warn("You have not given an inverse cummulative distribution "
                      "function 'inverse_cummulativ_dist_func'. Thus the yield "
                      "thresholds are held constant during the simulation and "
                      "are not updated after a plastic deformation!")

    #------------------------ strain cell while loop --------------------------#
    while n_strain_loop < n_max_strain_loop:
        DelF, PK2, F = \
            bracket_search(material, cell, solver, newton_tol, newton_equil_tol,
                           yield_surface_accuracy, n_max_bracket_search,
                           DelF_initial = DelF, verbose = verbose)
        propagate_avalanche(material, cell, solver, newton_tol,
            newton_equil_tol, PK2 = PK2, F_initial = F,
            n_max_avalanche = n_max_avalanche, verbose = verbose,
            inverse_cumulative_dist_func = inverse_cumulative_dist_func,
            save_avalanche = save_avalanche, n_strain_loop = n_strain_loop)


    # TODO (rleute)
    #     F_ava = compute_average_deformation_gradient(cell.strain, dim) \
    #         * cell.nb_subdomain_grid_pts[-1] / cell.nb_domain_grid_pts[-1]
    #     F_ava_tot = comm.sum(F_ava)
    #     if ((F_ava_tot - F_tot)[test_mask] >= 0).all() :
    #         # reached/overcome the required deformation
    #         break

    #     n_strain_loop += 1

    # if ((F_ava_tot - F_tot)[test_mask] >= 0).all() and (comm.rank == 0):
    #     if verbose:
    #         print("\nReached the required deformation!")
    #         print("The required average deformation gradient was:\n", F_tot)
    #         print("The reached final average deformation gradient is:\n",
    #               F_ava_tot)

    # #give warnings if the while loop breakes without having converged
    # if (n_strain_loop == n_max_strain_loop) and (comm.rank == 0):
    #     warnings.warn("Not converged!\nReached the maximum number of "
    #         "deformation steps: n_strain_loop ({}) = n_max_strain_loop ({})"
    #         "The reached final average deformation gradient is:\n{}"\
    #         .format(n_strain_loop, n_max_strain_loop, F_ava_tot),
    #         RuntimeWarning)

    # return F_ava_tot

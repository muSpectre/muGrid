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
from inspect import signature
import sys
sys.path.append('/home/rl/Work/muSpectre/build/language_bindings/python')
import muSpectre as µ
import muSpectre.gradient_integration as gi
from _muSpectre.solvers import IsStrainInitialised as ISI

# For parallel version
try:
    from mpi4py import MPI
except ImportError:
    MPI = None


# ----- Helper functions ----- #
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
        sigma_eq = np.sqrt(3/2 * np.tensordot(sigma_dev, sigma_dev))
    elif dim == 2:
        sigma_eq = np.sqrt(stress[0, 0]**2 + stress[1, 1]**2
                           - stress[0, 0]*stress[1, 1] + 3 * stress[0, 1]**2)
    elif dim == 1:
        sigma_eq = stress[0, 0]
    else:
        raise RuntimeError("The von Mises equivalent stress is not defined "
                           "for {}D".format(dim))
    return sigma_eq


def compute_strain_direction(stress, dim):
    """Strain direction which maximizes the locally dissipated energy
    strain_direction = deviatoric_stress / equivalent_von_Mises_stress"""
    sigma_dev = compute_deviatoric_stress(stress, dim)
    sigma_eq = compute_equivalent_von_Mises_stress(stress, dim)
    return sigma_dev / sigma_eq


def update_eigen_strain(mat, quad_pt_id, stress, dim):
    """Updates the eigen_strain on one quad point."""
    strain_direction = compute_strain_direction(stress, dim)
    strain_increment = mat.get_plastic_increment(quad_pt_id) * strain_direction
    old_eigen_strain = mat.get_eigen_strain(quad_pt_id)
    new_eigen_strain = old_eigen_strain + strain_increment
    mat.set_eigen_strain(quad_pt_id, np.asfortranarray(new_eigen_strain))


def set_new_threshold(mat, quad_pt_id, inverse_cumulative_dist_func):
    if inverse_cumulative_dist_func is not None:
        new_threshold = inverse_cumulative_dist_func(np.random.random())
        mat.set_stress_threshold(quad_pt_id, new_threshold)
    else:
        pass


def propagate_avalanche_step(mat, cell, dim, solver, newton_tol,
                             newton_equil_tol,
                             newton_verbose=µ.Verbosity.Silent):
    """
    Compute elastic response and resulting overloaded quad points for zero
    deformation (DelF = 0). Thus there is only a new stress distribution
    because the plastic strain was updated and there are new stress thresholds
    from the previous avalanche part.

    Keyword Arguments:
    material         -- µSpectre material object
    cell             -- µSpectre cell object
    solver           -- µSpectre solver object
    newton_tol       -- float, tolerance for the increment in the newton step
    newton_equil_tol -- float, tolerance for the stress in the newton step
    newton_verbose   -- µSpectre Verbosity (default µ.Verbosity.Silent),
                        verbosity of bracket search, "Silent" no messages,
                        "Full" print messages.

    Returns:
    overloaded_quad_pts ndarray, Each rank returns its list of overloaded quad
                                 points.
    """
    zero_DelF = np.zeros((dim,)*2)
    µ.solvers.newton_cg(cell, zero_DelF, solver,
                        newton_tol, newton_equil_tol,
                        newton_verbose, IsStrainInitialised=ISI.Yes)
                        # for finite_strain with PK2
                        # store_native_stress=µ.StoreNativeStress.Yes)
    mat.reset_overloaded_quad_pts()
    overloaded_quad_pts = mat.identify_overloaded_quad_pts(
        cell, cell.stress.array().T.reshape(-1, 1))
    # for finite_strain with PK2
    # overloaded_quad_pts = mat.identify_overloaded_quad_pts(cell)

    return overloaded_quad_pts


def reshape_avalanche_history(avalanche_history_list, dim):
    """
    Reshape the nested lists of the avalanche history by filling it with NaNs
    in such a way that the nested lists can be written as one np.ndarray(),
    with a shape:
    (#avalanche time steps, maximum avalanche in one time step, dim)

    Keyword Arguments:
    avalanche_history_list
    dim

    Returns:
    """
    ava_hl = avalanche_history_list
    length = len(max(avalanche_history_list, key=len))
    fill = [np.nan]
    return np.array(
        [ava_i.tolist() + fill*(length-len(ava_i)) for ava_i in ava_hl])


def compute_average_deformation_gradient(F_field, dim):
    return np.average(F_field, axis=tuple(range(2, 2+1+dim)))


def compute_global_quad_pt_id_from_pixel(global_pixel, nb_domain_grid_pts,
                                         nb_quad_pts, pixel_quad_pt):
    dim = len(nb_domain_grid_pts)

    if dim == 3:
        nx, ny, nz = nb_domain_grid_pts
        px, py, pz = global_pixel
        return (px + py * nx + pz * nx * ny) * nb_quad_pts + pixel_quad_pt

    elif dim == 2:
        nx, ny = nb_domain_grid_pts
        px, py = global_pixel
        return (px + py * nx) * nb_quad_pts + pixel_quad_pt

    elif dim == 1:
        nx = nb_domain_grid_pts
        px = global_pixel
        return px * nb_quad_pts + pixel_quad_pt

    else:
        raise ValueError("Not supported dimension D={}. The function "
                         "'compute_quad_pt_id_from_pixel' is only available "
                         "for one two or three dimensional systems."
                         .format(dim))


def compute_pixel_from_quad_pt_id(
        quad_pt_id, nb_domain_grid_pts, nb_quad_pts_per_pixel):
    dim = len(nb_domain_grid_pts)
    nb_qp = nb_quad_pts_per_pixel
    if dim == 3:
        nx, ny, nz = nb_domain_grid_pts
        p_qp = quad_pt_id % nb_qp
        id_red = quad_pt_id // nb_qp
        pz = id_red // (nx * ny)
        py = (id_red - (nx * ny * pz)) // nx
        px = id_red - (nx * ny * pz + nx * py)

        return p_qp, [px, py, pz]

    elif dim == 2:
        nx, ny = nb_domain_grid_pts
        p_qp = quad_pt_id % nb_qp
        id_red = quad_pt_id // nb_qp
        py = id_red // nx
        px = id_red - (nx * py)

        return p_qp, [px, py]

    elif dim == 1:
        nx = nb_domain_grid_pts
        p_qp = quad_pt_id % nb_qp
        id_red = quad_pt_id // nb_qp
        px = id_red

        return p_qp, [px]

    else:
        raise ValueError("Not supported dimension D={}. The function "
                         "'compute_pixel_from_quad_pt' is only available for "
                         "one two or three dimensional systems.".format(dim))


# ----- Main functions ----- #
def bracket_search(material, cell, solver, newton_tol, newton_equil_tol,
                   yield_surface_accuracy, n_max_bracket_search, DelF_initial,
                   verbose=0):
    """
    Tries to find the yield surface of the first breaking quad point in a
    material to at least the accuracy 'yield_surface_accuracy' by a bisection
    method.

    Keyword Arguments:
    material               -- µSpectre material object
    cell                   -- µSpectre cell object
    solver                 -- µSpectre solver object
    newton_tol             -- float, tolerance for the increment in the newton
                              step
    newton_equil_tol       -- float, tolerance for the stress in the newton
                              step
    yield_surafce_accuracy -- float, determines the accuracy of the bisection
                              methode in the sense of:
                              |F_true - F_num| < yield_surface_accuracy
                              where F_true is the theoretic value and F_num the
                              numeric estimate for the deformation gradient and
                              |·| is the Frobenius matrix norm.
    n_max_bracket_search   -- int, the maximum number of bisection steps.
    DelF_initial           -- initial step to increase the deformation gradient
                              F. It defines the deformation "direction" thus
                              the total deformation F_tot should be given as
                              F_tot = a * DelF_initial for a real number a.
    verbose                -- int (default 0), verbosity of bracket search,
                              0 no messages, 1 print messages

    Returns:
    next_DelF_guess np.ndarray (dim, dim), guess for the search step size DelF
                    of the next plastically deforming quad point
                    (dtype = float)
    F               np.ndarray (nx*ny*nz*dim*dim, 1), deformation gradient at
                    which the weakest quad point undergoes a plastic
                    deformation. (nx,ny,nz) is the subdomain resolution given
                    by cell.nb_subdomain_grid_pts. (dtype = float)
    breaking_quad_pts np.array, index of quad point(s) which was(were) found to
                      break at 'F_num' within an accuracy of
                      yield_surface_accuracy (yield_surface_accuracy² for more
                      than one quad point)
                      (|F_true - F_num| < yield_surface_accuracy)
    """
    # initialize parameters
    mat = material
    ys_acc = yield_surface_accuracy
    comm = cell.communicator
    dim = len(cell.nb_domain_grid_pts)
    DelF_final = np.copy(DelF_initial)
    DelF = np.copy(DelF_initial)
    DelF_step_size = np.copy(DelF_initial)
    next_DelF_guess = np.copy(DelF_initial)
    newton_verbose = µ.Verbosity.Silent
    previous_step_was = "initial_step"
    n_bracket = 0
    first_time = True  # do I overcome exactly one quad point
                       # the first time or not?

    # Small intial step size error
    if np.linalg.norm(DelF_initial) < (1 if ys_acc > 1 else ys_acc**2)/1e4:
        raise ValueError("Your initial step size DelF_initial seems to be "
                         "verry small or even zero!\nDelF_initial:\n{}"
                         .format(DelF_initial))

    while (n_bracket < n_max_bracket_search):
        # compute elastic response ------------------------------------------ #
        result = µ.solvers.newton_cg(cell, DelF, solver,
                                     newton_tol, newton_equil_tol,
                                     newton_verbose,
                                     IsStrainInitialised=ISI.Yes)
                                     # for finite_strain with PK2
                                     # store_native_stress=µ.StoreNativeStress.Yes)
        PK2 = np.copy(result.stress)
        mat.reset_overloaded_quad_pts()
        overloaded_quad_pts = mat.identify_overloaded_quad_pts(cell, PK2)
        # for finite_strain with PK2
        # overloaded_quad_pts = mat.identify_overloaded_quad_pts(cell)
        n_pix = comm.sum(len(overloaded_quad_pts))

        # Decide "elastic" or "too_large_step" ?---------------------------- #
        if n_pix == 0:
            if previous_step_was == "elastic":
                # go twice the step forward
                DelF_step_size *= 2
                DelF = +DelF_step_size
            elif previous_step_was == "too_large_step":
                # go half the step forward
                DelF_step_size /= 2
                DelF = +DelF_step_size
            elif previous_step_was == "initial_step":
                # keep everything as it is initialized
                pass

            previous_step_was = "elastic"

        elif n_pix >= 1:
            if n_pix == 1 and first_time:
                # safe the current deformation step as guess for the next step
                next_DelF_guess = np.copy(DelF_step_size)
                first_time = False

            DelF_step_size /= 2
            DelF = - DelF_step_size
            previous_step_was = "too_large_step"

        else:
            if comm.rank == 0:
                n_pix_error = \
                    ("\nSomething went totally wrong!\nThe number of overloaded"
                     "quad points 'n_pix' is "+str(n_pix)+"\nThis case is not "
                     "handeled and probably doesn't make sense.")
                raise RuntimeError(n_pix_error)

        # Check convergence + RETURN STATEMENT ------------------------------ #
        reached_accuracy = (2 * np.linalg.norm(DelF_step_size) < ys_acc)
        if reached_accuracy:
            if n_pix == 1:
                F = result.grad
                breaking_quad_pts = \
                    comm.gather(np.array(overloaded_quad_pts, dtype=np.int64)
                                .reshape((-1, 1)).T).reshape(-1)
                if verbose and comm.rank == 0:
                    print("I needed {} bracket search steps."
                          .format(n_bracket))
                    print("The following quad point broke: ",
                          breaking_quad_pts)
                # ### RETURN:
                return next_DelF_guess, PK2, F, breaking_quad_pts

            reached_nquad_pt_accuracy = (2 * np.linalg.norm(DelF_step_size) <
                                         (1 if ys_acc > 1 else ys_acc**2))
            if reached_nquad_pt_accuracy and (n_pix > 1):
                warnings.warn(
                    "bracket_search found {} quad points overcoming their "
                    "yield threshold for the final deformation. To initialise "
                    "the avalanche think about using the parameter "
                    "'single_quad_pt_start' of 'propagate_avalanche()' to "
                    "start the avalanche from a single quad point!"
                    .format(n_pix), RuntimeWarning)
                F = result.grad
                breaking_quad_pts = \
                    comm.gather(np.array(overloaded_quad_pts, dtype=np.int64)
                                .reshape((-1, 1)).T).reshape(-1)
                if verbose and comm.rank == 0:
                    print("I needed {} bracket search steps."
                          .format(n_bracket))
                    print("The following quad points broke simultaneously",
                          breaking_quad_pts)
                # ### RETURN:
                return next_DelF_guess, PK2, F, breaking_quad_pts

        # Update parameters before loop ends -------------------------------- #
        DelF_final += DelF
        n_bracket += 1

    if n_bracket == n_max_bracket_search:
        n_max_error = \
            ("The maximum number, n_max_bracket_search={}, of bracket search "
             "steps was reached.\nEither there is a problem or you should "
             "increase the maximum number of allowed bracket search steps."
             .format(n_max_bracket_search))
        # F_ava = 0 if processor is empty
        factor = \
            cell.nb_subdomain_grid_pts[dim-1] / cell.nb_domain_grid_pts[dim-1]
        if factor != 0:
            F_ava = compute_average_deformation_gradient(
                cell.strain, dim) * factor
        else:
            # processor is empty
            F_ava = np.zeros((dim, dim), order="F")
        F_ava_tot = comm.sum(F_ava)

        if comm.rank == 0 and verbose:
            print("Last stepsize DelF_step_size:\n", DelF_step_size)
            print("computed average deformation:\n", F_ava_tot)
        raise RuntimeError(n_max_error)


def propagate_avalanche(material, cell, solver, newton_tol, newton_equil_tol,
                        PK2, F_initial, n_max_avalanche, verbose=0,
                        initially_overloaded_quad_pts=None,
                        single_quad_pt_start=False,
                        inverse_cumulative_dist_func=None,
                        save_avalanche=None, save_fields=None,
                        n_strain_loop=0):
    """
    Starting from an initial plastic deforming quad point/s it updates the
    strain_field and thresholds of all quad points that undergo a plastic
    deformation during the avalanche process.

    Keyword Arguments:
    material         -- µSpectre material object
    cell             -- µSpectre cell object
    solver           -- µSpectre solver object
    newton_tol       -- float, tolerance for the increment in the newton step
    newton_equil_tol -- float, tolerance for the stress in the newton step
    PK2              -- np.ndarray (nx*ny*nz*dim*dim, 1), Piola Kirchhoff
                        stress 2 as given from µSpectre::solvers::Optimize-
                        Result.stress or bracket_search, where (nx,ny,nz) is
                        the subdomain resolution given by cell.nb_subdomain_-
                        grid_pts. PK2 is the stress at which the first quad
                        point of an avalanche undergoes a plastic deformation.
                        (dtype = float)
    F_initial        -- np.ndarray (nx*ny*nz, dim*dim), deformation gradient
                        as given from µSpectre::solvers::OptimizeResult.grad
                        or bracket_search, where (nx,ny,nz) is the subdomain
                        resolution given by cell.nb_subdomain_grid_pts.
                        F_initial is the strain at which the first quad point
                        of an avalanche undergoes a plastic deformation.
                        (dtype = float)
    n_max_avalanche  -- int, maximum number of avalanche time steps
    verbose          -- int, verbosity of propagate_avalanche, 0 no messages,
                        1 print messages (default 0)
    initially_overloaded_quad_pts -- np.ndarray (# overloaded quad points, dim)
                        List of the overloaded quad point/s which initialise
                        the avalanche. Ideally it should match with the
                        overloaded quad point/s one gets for a computation with
                        the input, material, cell, F_initial and PK2, but it
                        does not have to. For example it can make sense to
                        choose only one out of several overloaded quad points,
                        for an initial configuration, to start an avalanche.
                        (default None, the overloaded quad points are computed
                        from the input configuration) (dtype = int)
    single_quad_pt_start -- bool; True: if the avalanche wants to start with
                        more than one quad point a single quad point out of the
                        overloaded quad points is randomly, with equal weight
                        for each quad point, choosen to initialise the
                        avalanche. False: The avalanche starts with the given
                        or computed overloaded quad point/s, i.e. it can also
                        start with more than one plastic deforming quad point.
                        (dtype = bool) (default False)
    inverse_cumulative_dist_func -- lambda function, inverse cummulative dist-
                        ribution function of the yield threshold distribution
                        which is used to evaluate new random yield thresholds.
                        The lambda function gets as input a random number from
                        a uniform distribution in the interval [0,1) produced
                        by np.random.random(). (default None, no new thresholds
                        are drawn, thus constant yield thresholds in the
                        simulation)
    save_avalanche   -- call back function, executed only on rank 0 and gets
                        the ordered parameters:
                         n_strain_loop -- int, number of the strain loop step
                         ava_history   -- np.array of shape(avalanche timesteps
                                          , #plastic deformations, dim)
                                          containing the plastic deformed quad
                                          point indices.
                        (default None, the avalanche is not saved)
    save_fields      -- call back function, executed in parallel and gets the
                        ordered parameters:
                         cell             -- muSpectre cell object
                         n_strain_loop    -- int, number of the strain loop step
                         before_avalanche -- bool, indicates if the function is
                                             executed before (True) or after
                                             (False) the avalanche.
                        You can use this function to call the write function of
                        µSpectres IO interface, i.e. prepare an µGrid NetCDf io
                        object and call write(["stress", "strain"])
                        (default None, the function is not called)
    n_strain_loop    -- int, number of the actual strain loop step is only used
                        for documentation of the avalanche and therefore passed
                        to the call back functions (default = 0).

    Returns:
        No return except of the call back functions "save_avalanche" and
        "save_fields".
    """
    mat = material
    cell = cell
    nb_quad_pts = cell.nb_quad_pts
    comm = cell.communicator
    dim = len(cell.nb_domain_grid_pts)  # dimension
    iop = initially_overloaded_quad_pts
    if comm is None:
        comm.size = 0

    if iop is None:
        # compute the overloaded quad points from the given input configuration
        mat.reset_overloaded_quad_pts()
        overloaded_quad_pts = mat.identify_overloaded_quad_pts(cell, PK2)
        # for finite_strain with PK2
        # overloaded_quad_pts = mat.identify_overloaded_quad_pts(cell)
        if not overloaded_quad_pts:
            # prevent empty list which leads to incompatible function argument
            # in comm.gather()
            overloaded_quad_pts = []
        if single_quad_pt_start:
            # select on rank 0 random a single quad point at which the
            # avalanche starts
            iop = comm.gather(np.array(overloaded_quad_pts, dtype=np.int64)
                              .reshape((-1, 1)).T).reshape(-1)
            random_index = len(iop) + 1  # invalid index if not updated
            if comm.rank == 0:
                random_index = np.random.randint(len(iop), size=1)[0]
            random_index = comm.bcast(scalar_arg=random_index, root=0)
            iop = np.array([iop[random_index]], dtype=np.int64)
            # write only overloaded quad points into the list if they exist in
            # the subdomain of the processor
            loc = np.array(cell.subdomain_locations)
            nb_pts = np.array(cell.nb_subdomain_grid_pts)
            io_pixel = [np.array(
                compute_pixel_from_quad_pt_id(quad_pt_id,
                                              cell.nb_domain_grid_pts,
                                              cell.nb_quad_pts)[1]).flatten()
                        for quad_pt_id in iop]
            quad_pt_in_domain = np.logical_and((io_pixel >= loc).all(axis=1),
                                               (io_pixel < loc + nb_pts)
                                               .all(axis=1))
            overloaded_quad_pts = iop[quad_pt_in_domain].tolist()

    else:
        if single_quad_pt_start:
            # select on rank 0 random a single quad point at which the
            # avalanche starts
            random_index = len(iop) + 1  # invalid index if not updated
            if comm.rank == 0:
                random_index = np.random.randint(len(iop), size=1)[0]
            random_index = comm.bcast(scalar_arg=random_index, root=0)
            iop = np.array([iop[random_index]], dtype=np.int64)

        # write only overloaded quad points into the list if they exist in the
        # subdomain of the processor. Therefore each quad point id is computed
        # back to its pixel coordinates and then checked whether it lies in the
        # subdomain
        loc = np.array(cell.subdomain_locations)
        nb_pts = np.array(cell.nb_subdomain_grid_pts)
        io_pixel = [np.array(
            compute_pixel_from_quad_pt_id(quad_pt_id,
                                          cell.nb_domain_grid_pts,
                                          cell.nb_quad_pts)[1]).flatten()
                    for quad_pt_id in iop]
        quad_pt_in_domain = np.logical_and((io_pixel >= loc).all(axis=1),
                                           (io_pixel < loc + nb_pts)
                                           .all(axis=1))
        overloaded_quad_pts = iop[quad_pt_in_domain].flatten().tolist()

    n_pix_avalanche = comm.sum(len(overloaded_quad_pts))

    if n_pix_avalanche != 1 and single_quad_pt_start:
        RuntimeError("You have initialised propagate_avalanche with "
                     "'single_quad_pt_start = True' but nevertheless "
                     "propagate_avalanche wants to start the avalanche with "
                     "{} quad points! Probably there is a Bug in the code, "
                     "please inform the developers.".format(n_pix_avalanche))

    n_avalanche = 0
    newton_verbose = µ.Verbosity.Silent
    # for finite_strain with PK2
    # PK2_quad_pt = \
    #     mat.native_stress.array().reshape((dim, dim, nb_quad_pts)
    #                                       + tuple(cell.nb_subdomain_grid_pts),
    #                                       order='f')
    PK2_quad_pt = PK2.reshape((dim, dim, nb_quad_pts) +
                              tuple(cell.nb_subdomain_grid_pts), order='F')

    if save_avalanche is not None:
        ava_history = []
    if save_fields is not None:
        # call save_fields(cell, n_strain_loop, before_avalanche=True/False)
        # which by itself can call file_io_object.append_frame().write() with
        # the required field names. The fiel_io_object has to be set  up in the
        # corect way outside of the function.
        save_fields(cell, n_strain_loop, before_avalanche=True)

    subdomain_quad_pt_offset = compute_global_quad_pt_id_from_pixel(
        cell.subdomain_locations, cell.nb_domain_grid_pts, cell.nb_quad_pts, 0)

    while (n_pix_avalanche >= 1) and (n_avalanche < n_max_avalanche):
        if save_avalanche is not None:
            ava_t = comm.gather(np.array(overloaded_quad_pts, dtype=np.int64)
                                .reshape((-1, 1)).T).reshape(-1)
            if comm.rank == 0:
                ava_history.append(ava_t.flatten())

        for quad_pt_id in overloaded_quad_pts:
            quad_pt, pixel = compute_pixel_from_quad_pt_id(
                quad_pt_id, cell.nb_domain_grid_pts, nb_quad_pts)
            if comm.size > 1:
                # shift pixel by subdomain location because each core has only
                # its part of the stress tensor, but the pixel index is global!
                local_index = np.s_[:, :, quad_pt] \
                    + tuple(np.array(pixel) - cell.subdomain_locations)
            else:
                local_index = np.s_[:, :, quad_pt] + tuple(pixel)
            # here I need the per processor, local_quad_pt_id
            local_quad_pt_id = quad_pt_id - subdomain_quad_pt_offset
            update_eigen_strain(mat, local_quad_pt_id,
                                PK2_quad_pt[local_index], dim)
            set_new_threshold(mat, local_quad_pt_id,
                              inverse_cumulative_dist_func)
        overloaded_quad_pts =\
            propagate_avalanche_step(mat, cell, dim, solver, newton_tol,
                                     newton_equil_tol, newton_verbose)
        n_pix_avalanche = comm.sum(len(overloaded_quad_pts))

        n_avalanche += 1

    if verbose and comm.rank == 0:
        print("I needed {} avalanche time steps".format(n_avalanche))

    if (n_avalanche == n_max_avalanche):  # recognise if avalanche overflows!
        max_avalanche_error = \
            ("\n"+str(n_avalanche)+" avalanche steps!\nYou have reached the"
             " maximum allowed avalanche size of " + str(n_max_avalanche) +
             " internal steps.\nIncrease the maximum allowed avalanche "
             "steps 'n_max_avalanche' if it does make sense.")
        raise RuntimeError(max_avalanche_error)

    if save_avalanche is not None:
        if comm.rank == 0:
            ava_history = reshape_avalanche_history(ava_history, dim)
            save_avalanche(n_strain_loop, ava_history)

    if save_fields is not None:
        # call save_fields(cell, n_strain_loop, before_avalanche=True/False)
        # which by itself can call file_io_object.append_frame().write() with
        # the required field names. The fiel_io_object has to be set  up in the
        # corect way outside of the function.
        save_fields(cell, n_strain_loop, before_avalanche=False)


def strain_cell(material, cell, solver, newton_tol, newton_equil_tol,
                DelF, F_tot, yield_surface_accuracy, n_max_strain_loop,
                n_max_bracket_search, n_max_avalanche, verbose=0,
                single_quad_pt_start=False, inverse_cumulative_dist_func=None,
                save_avalanche=None, save_fields=None,
                is_strain_initialised=False):
    """
    Strain the cell stepwise to the maximum deformation of F_tot. By default
    the cell is initialised with zero deformation (finite strain: np.eye(dim)).

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
    F_tot            -- np.ndarray (dim, dim), required final deformation
                        (dtype float)
    yield_surface_accuracy -- float, determines the accuracy of the bisection
                              methode in the sense of:
                              |F_true - F_num| < yield_surface_accuracy
                              where F_true is the theoretic value and F_num the
                              numeric estimate for the deformation gradient at
                              which the first quad point deforms plastic and
                              |·| is the Frobenius matrix norm.
    n_max_strain_loop      -- int, maximum number of strain loop steps
    n_max_bracket_search   -- int, the maximum number of bisection steps.
    n_max_avalanche        -- int, maximum number of avalanche time steps
    verbose                -- int, verbosity of strain cell, 0 no messages,
                              1 print messages (default 0)
    single_quad_pt_start -- bool; True: if the avalanche wants to start with
                        more than one quad point a single quad point out of the
                        overloaded quad points is randomly, with equal weight
                        for each quad point, choosen to initialise the
                        avalanche. False: The avalanche starts with the given
                        or computed overloaded quad point/s, i.e. it can also
                        start with more than one plastic deforming quad point.
                        (dtype = bool) (default False)
    inverse_cumulative_dist_func -- lambda function, inverse cummulative dist-
                        ribution function of the yield threshold distribution
                        which is used to evaluate new random yield thresholds.
                        The lambda function gets as input a random number from
                        a uniform distribution in the interval [0,1) produced
                        by np.random.random(). (default None, no new thresholds
                        are drawn, thus constant yield thresholds in the
                        simulation)
    save_avalanche   -- call back function, executed only on rank 0 and gets
                        the ordered parameters:
                         n_strain_loop -- int, number of the strain loop step
                         ava_history   -- np.array of shape(avalanche timesteps
                                          , #plastic deformations, dim)
                                          containing the plastic deformed quad
                                          point indices.
                        (default None, the avalanche is not saved)
    save_fields      -- call back function, executed in parallel and gets the
                        ordered parameters:
                         cell             -- muSpectre cell object
                         n_strain_loop    -- int, number of the strain loop step
                         before_avalanche -- bool, indicates if the function is
                                             executed before (True) or after
                                             (False) the avalanche.
                        You can use this function to call the write function of
                        µSpectres IO interface, i.e. prepare an µGrid NetCDf io
                        object and call write(["stress", "strain"])
    is_strain_initialised  -- (default False) characterises wheather the strain
                              is initialised by the user (True) or automatic to
                              zero deformation (finite strain: np.eye(dim))
                              (dtype = bool)

    Returns:
        F_ava_tot, np.ndarray of shape (dim, dim) containing the final reached
        average deformation gradient (dtype float).
    """
    # --------------------------- while loop init stuff --------------------- #
    comm = cell.communicator
    dim = len(cell.nb_domain_grid_pts)
    n_strain_loop = 0
    DelF = np.copy(DelF)

    # grid averaged F_old (zero deformation at start)
    if cell.formulation == µ.Formulation.finite_strain:
        F_old_average = np.eye(dim)
    elif cell.formulation == µ.Formulation.small_strain:
        F_old_average = np.zeros((dim, dim))
    test_mask = np.where(DelF != 0)

    # if not initialised by user, initialise unit-matrix deformation gradient
    if not is_strain_initialised:
        cell_strain = cell.strain.array()
        cell_strain[:] = np.tensordot(
            F_old_average, np.ones((cell.nb_quad_pts,)
                                   + tuple(cell.nb_subdomain_grid_pts)), axes=0)

    # -------------------- cross check input on consistency ----------------- #
    if yield_surface_accuracy <= 0:
        raise ValueError("The yield_surface_accuracy should be an non zero "
                         "positive value!\nYour input was: {}"
                         .format(yield_surface_accuracy))
    if n_max_strain_loop <= 0 or not isinstance(n_max_strain_loop, int):
        raise ValueError("The maximum number of strain steps "
                         "'n_max_strain_loop' should be an non zero positive "
                         "integer value!\nYour input was: {}"
                         .format(n_max_strain_loop))
    if n_max_bracket_search <= 0 or not isinstance(n_max_bracket_search, int):
        raise ValueError("The maximum number of bracket search steps "
                         "'n_max_bracket_search' should be an non zero "
                         "positive integer value!\nYour input was: {}"
                         .format(n_max_bracket_search))
    if n_max_avalanche <= 0 or not isinstance(n_max_avalanche, int):
        raise ValueError("The maximum number of time steps during an avalanche"
                         " 'n_max_avalanche' should be an non zero positive "
                         "integer value!\nYour input was: {}"
                         .format(n_max_avalanche))
    # Check if the deformation steps DelF can lead to the required deformation
    deformation_mask = np.where(F_tot - F_old_average != 0)
    if not (np.equal(np.array(test_mask), np.array(deformation_mask)).all()) \
       or (np.sign(F_tot - F_old_average) != np.sign(DelF)).any():
        raise ValueError("\nProbably you can not reach the desired deformation"
                         " F_tot:\n{}\n\nby steps of DelF:\n{}\n\nstarting "
                         "from an initial deformation F_0:\n{}\n\nTherefore "
                         "choose proper steps DelF!"
                         .format(F_tot, DelF, F_old_average))
    if inverse_cumulative_dist_func is None:
        warnings.warn("You have not given an inverse cummulative distribution "
                      "function 'inverse_cummulativ_dist_func'. Thus the yield"
                      " thresholds are held constant during the simulation and"
                      " are not updated after a plastic deformation!")
    # Check if the call back functions have the correct signature
    # if they are not None
    if save_avalanche is not None:
        n_para = len(signature(save_avalanche).parameters)
        str_para = str(signature(save_avalanche))
        if n_para != 2:
            raise ValueError("The signature of save_avalanche should be:\n"
                             "save_avalance(n_strain_loop, ava_history)\n"
                             "You instead have given n={} keyword parameters"
                             .format(n_para))
        if str_para != "(n_strain_loop, ava_history)":
            raise ValueError("The signature of save_avalanche should be:\n"
                             "save_avalance(n_strain_loop, ava_history)\n"
                             "You instead have given:\nsave_avalanche{}"
                             .format(str_para))
    if save_fields is not None:
        n_para = len(signature(save_fields).parameters)
        str_para = str(signature(save_fields))
        if n_para != 3:
            raise ValueError("The signature of save_fields should be:\n"
                             "save_fields(cell, n_strain_loop, before_avalanche)\n"
                             "You instead have given n={} keyword parameters"
                             .format(n_para))
        if str_para != "(cell, n_strain_loop, before_avalanche)":
            raise ValueError("The signature of save_fields should be:\n"
                             "save_fields(cell, n_strain_loop, before_avalanche)\n"
                             "You instead have given:\nsave_stress_strain{}"
                             .format(str_para))

    # ----------------------- strain cell while loop ------------------------ #
    while n_strain_loop < n_max_strain_loop:
        DelF, PK2, F, breaking_quad_pts = \
            bracket_search(material, cell, solver, newton_tol,
                           newton_equil_tol, yield_surface_accuracy,
                           n_max_bracket_search, DelF_initial=DelF,
                           verbose=verbose)
        propagate_avalanche(
            material, cell, solver, newton_tol, newton_equil_tol, PK2,
            F_initial=F, n_max_avalanche=n_max_avalanche, verbose=verbose,
            initially_overloaded_quad_pts=breaking_quad_pts,
            single_quad_pt_start=single_quad_pt_start,
            inverse_cumulative_dist_func=inverse_cumulative_dist_func,
            save_avalanche=save_avalanche, save_fields=save_fields,
            n_strain_loop=n_strain_loop)

        # F_ava = 0 if processor is empty
        factor = \
            cell.nb_subdomain_grid_pts[dim-1] / cell.nb_domain_grid_pts[dim-1]
        if factor != 0:
            F_ava = compute_average_deformation_gradient(
                cell.strain.array(), dim) * factor
        else:
            # processor is empty
            F_ava = np.zeros((dim, dim), order="F")
        F_ava_tot = np.copy(comm.sum(F_ava))

        if (((F_ava_tot - F_tot)*np.sign(DelF))[test_mask] >= 0).all():
            # reached/overcome the required deformation. By the test_mask the
            # convergence is only checked in the deformation directions. The
            # sign of DelF is needed to handle increase and decrease of the
            # deformation gradient in a single case.
            break

        n_strain_loop += 1

    if (((F_ava_tot - F_tot)*np.sign(DelF))[test_mask] >= 0).all() \
       and (comm.rank == 0):
        if verbose:
            print("\nReached the required deformation!")
            print("The required average deformation gradient was:\n", F_tot)
            print("The reached final average deformation gradient is:\n",
                  F_ava_tot)

    # give warnings if the while loop breakes without having converged
    if (n_strain_loop == n_max_strain_loop) and (comm.rank == 0):
        warnings.warn("Not converged!\nReached the maximum number of "
                      "deformation steps: n_strain_loop ({}) = n_max_strain"
                      "_loop ({}) The reached final average deformation "
                      "gradient is:\n{}"
                      .format(n_strain_loop, n_max_strain_loop, F_ava_tot),
                      RuntimeWarning)

    return F_ava_tot

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   projection_comparison.py

@author Till Junge <till.junge@epfl.ch>

@date   12 Jan 2018

@brief  small case for debugging

Copyright © 2018 Till Junge

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
import numpy as np


def material_type(dim):
    return µ.material.MaterialLinearElastic1_2d if (dim == 2) else µ.material.MaterialLinearElastic1_3d


def example_problem(microstructure, mat1_prop, mat2_prop, projector):
    """

    :param microstructure: array of T/F to mark material 1 and 2, can be 2D or 3D
    :param mat1_prop: Young modulus and Poisson ratio of mat 1
    :param mat2_prop: Young modulus and Poisson ratio of mat 2
    :param projector: Projection operator to be used
    :return: dictionary containing : number of CG iterations and execution time in secs

    """
    rve = µ._muSpectre.cell.Cell(projector)
    dim = len(microstructure.shape)
    hard = material_type(dim).make(
        rve, "hard", mat1_prop[0], mat1_prop[1])
    soft = material_type(dim).make(
        rve, "soft", mat2_prop[0], mat2_prop[1])

    for i, pixel in rve.pixels.enumerate():

        if microstructure[tuple(pixel)]:

            hard.add_pixel(i)
        else:
            soft.add_pixel(i)

    tol = 1e-5
    cg_tol = 1e-8
    eq_tol = 1e-8

    Del0 = np.zeros((dim, dim))
    Del0[0, 1] = Del0[1, 0] = 0.03

    maxiter = 1001
    verbose = µ.Verbosity.Silent

    solver = µ.solvers.KrylovSolverCG(rve, cg_tol, maxiter, verbose)

    def return_function():
        r = µ.solvers.de_geus(rve, Del0, solver, tol, eq_tol, verbose)
        return r.nb_fev

    return return_function


def get_projectors(microstructure, mat1_prop, mat2_prop, ):
    nb_grid_pts = microstructure.shape
    lengths = np.ones_like(nb_grid_pts)

    dim = len(nb_grid_pts)

    # FFTEngine= muFFT.FFT(nb_grid_pts, nb_dof_per_pixel=dim*dim)
    gradient = [µ.FourierDerivative(dim, i) for i in range(dim)]
    if dim == 2:
        projection_green_type = µ.ProjectionApproxGreenOperator_2d
        projection_classic_type = µ.ProjectionSmallStrain_2d
    else:
        projection_green_type = µ.ProjectionApproxGreenOperator_3d
        projection_classic_type = µ.ProjectionSmallStrain_3d

    projection_classic = projection_classic_type(
        muFFT.FFT(nb_grid_pts), lengths.tolist())

    rve = µ._muSpectre.cell.Cell(projection_classic)

    mat1 = material_type(dim).make(rve, "refrence", mat1_prop[0], mat1_prop[1])
    mat2 = material_type(dim).make(rve, "refrence2", mat2_prop[0], mat2_prop[1])
    volume_fraction = microstructure.astype(float).mean()
    C_ref = volume_fraction * mat1.C + (1 - volume_fraction) * mat2.C

    projection_green = projection_green_type(
        muFFT.FFT(nb_grid_pts), lengths.tolist(),
        C_ref)

    return projection_classic, projection_green


if __name__ == '__main__':
    Young = 1
    Poisson = 0.3
    contrast = 31

    mat1_prop = contrast * Young, Poisson
    mat2_prop = Young, Poisson
    d = 35
    microstructure = np.random.random((d, d)) < 0.7
    projector_classic, projector_green = get_projectors(microstructure, mat1_prop, mat2_prop)

    projector1 = example_problem(microstructure, mat1_prop, mat2_prop, projector_classic)
    projector2 = example_problem(microstructure, mat1_prop, mat2_prop, projector_green)

    print("Preconditioned scheme needs {} steps".format(projector2()))
    print("Classic scheme needs {} steps".format(projector1()))

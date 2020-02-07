#!/usr/bin/env python3

import sys

import os

from python_test_imports import µ
from python_test_imports import muFFT
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
    verbose = 0

    solver = µ.solvers.KrylovSolverCG(rve, cg_tol, maxiter, verbose=False)

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

    projection_classic = projection_classic_type(muFFT.FFT(nb_grid_pts, nb_dof_per_pixel=dim * dim), lengths.tolist())

    rve = µ._muSpectre.cell.Cell(projection_classic)

    mat1 = material_type(dim).make(rve, "refrence", mat1_prop[0], mat1_prop[1])
    mat2 = material_type(dim).make(rve, "refrence2", mat2_prop[0], mat2_prop[1])
    volume_fraction = microstructure.astype(float).mean()
    C_ref = volume_fraction * mat1.C + (1 - volume_fraction) * mat2.C

    projection_green = projection_green_type(muFFT.FFT(nb_grid_pts, nb_dof_per_pixel=dim * dim), lengths.tolist(),
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

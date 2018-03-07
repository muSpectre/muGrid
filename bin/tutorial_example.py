#!/usr/bin/env python3

import numpy as np
import muSpectre as msp
import matplotlib.pyplot as plt


## currently, muSpectre is restricted to odd-numbered resolutions for
## reasons explained in T.W.J. de Geus, J. Vondřejc, J. Zeman,
## R.H.J. Peerlings, M.G.D. Geers, Finite strain FFT-based non-linear
## solvers made simple, Computer Methods in Applied Mechanics and
## Engineering, Volume 318, 2017
## https://doi.org/10.1016/j.cma.2016.12.032
resolution = [51, 51]
center = np.array([r//2 for r in resolution])
incl = resolution[0]//5


## Domain dimensions
lengths = [7., 5.]
## formulation (small_strain or finite_strain)
formulation = msp.Formulation.small_strain

## build a computational domain
rve = msp.CellFactory(resolution,
                      lengths,
                      formulation)

## define the material properties of the matrix and inclusion
hard = msp.material.MaterialLinearElastic1_2d.make(
    rve, "hard", 10e9, .33)
soft = msp.material.MaterialLinearElastic1_2d.make(
    rve, "soft",  70e9, .33)

## assign each pixel to exactly one material
for i, pixel in enumerate(rve):
    if np.linalg.norm(center - np.array(pixel),2)<incl:
        hard.add_pixel(pixel)
    else:
        soft.add_pixel(pixel)

## define the convergence tolerance for the Newton-Raphson increment
tol = 1e-5
## tolerance for the solver of the linear cell
cg_tol = 1e-8


## Macroscopic strain
Del0 = np.array([[.0, .0],
                 [0,  .03]])
Del0 = .5*(Del0 + Del0.T)


maxiter = 50 ## for linear cell solver

## Choose a solver for the linear cells. Currently avaliable:
## SolverCG, SolverCGEigen, SolverBiCGSTABEigen, SolverGMRESEigen,
## SolverDGMRESEigen, SolverMINRESEigen.
## See Reference for explanations
solver = msp.solvers.SolverCGEigen(rve, cg_tol, maxiter, verbose=True)


## Verbosity levels:
## 0: silent,
## 1: info about Newton-Raphson loop,
verbose = 1

## Choose a solution strategy. Currently available:
## de_geus: is discribed in de Geus et al. see Ref above
## newton_cg: classical Newton-Conjugate Gradient solver. Recommended
result = msp.solvers.newton_cg(rve, Del0, solver, tol, verbose)

print(result)

## visualise e.g., stress in y-direction
stress = result.stress
## stress is stored in a flatten stress tensor per pixel, i.e., a
## dim^2 × prod(resolution_i) array, so it needs to be reshaped
stress = stress.T.reshape(*resolution, 2, 2)

plt.pcolormesh(stress[:, :, 1, 1])
plt.colorbar()
plt.show()


#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   visualisation_example_finite_strain.py

@author Richard Leute <richard.leute@imtek.uni-freiburg.de>

@date   14 Jan 2019

@brief  small example of how one can use the visualisation tools:
        gradient_integration() and vtk_export()

Copyright © 2019 Till Junge, Richard Leute

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

from python_example_imports import muSpectre_vtk_export as vt_ex
from python_example_imports import muSpectre_gradient_integration as gi
from python_example_imports import muSpectre as µ
import numpy as np
import sys
import os
sys.path.append("language_bindings/python/")


### Input parameters ###
#----------------------#
# general
if len(sys.argv) == 4:
    nb_grid_pts = [int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])]
    print("\nYour input grid is: ", nb_grid_pts)
elif len(sys.argv) == 1:
    nb_grid_pts = [51, 51, 3]
    print("\nYou can give a different grid size as input parameters:\n"
          "$ python3 visualisation_example_finite_strain.py nx ny nz\n"
          "Where nx, ny and nz are the number of gridpoints in the spatial"
          "directions x,y,z.\nCaution nx, ny > 5 is required for this example.")
else:
    print("\nYou can either run this example with no input parameters or\n"
          "you can give a 3D grid size as input parameters:\n"
          "$ python3 visualisation_example_finite_strain.py nx ny nz\n"
          "e.g.:\n",
          "$ python3 visualisation_example_finite_strain.py 51 51 3\n"
          "Where nx, ny and nz are the number of gridpoints in the spatial"
          "directions x,y,z.\nCaution nx, ny > 9 is required for this example.")
lengths = [1.0, 1.0, 1.0]
dim = len(nb_grid_pts)
Nx, Ny, Nz = nb_grid_pts
formulation = µ.Formulation.finite_strain
Young = [10, 20]  # Youngs modulus for each phase
Poisson = [0.3, 0.4]  # Poissons ratio for each phase

# solver
newton_tol = 1e-6  # tolerance for newton algo
cg_tol = 1e-6  # tolerance for cg algo
equil_tol = 1e-6  # tolerance for equilibrium
maxiter = 100
verbose = 0

# sinusoidal bump
d = int(nb_grid_pts[1]*0.2) #thickness of the phase with higher elasticity in pixles
l = Nx//3  # length of bump
h = Ny//5  # height of bump

low_y = (Ny-d)//2  # lower y-boundary of phase
high_y = low_y+d  # upper y-boundary of phase

left_x = (Nx-l)//2  # boundaries in x direction left
right_x = left_x + l  # boundaries in x direction right
x = np.arange(l)
p_y = h*np.sin(np.pi/(l-1)*x)  # bump function

xy_bump = np.ones((l, h, Nz))  # grid representing bumpx
for i, threshold in enumerate(np.round(p_y)):
    xy_bump[i, int(threshold):, :] = 0

phase = np.zeros(nb_grid_pts, dtype=int)  # 0 for surrounding matrix
phase[:, low_y:high_y, :] = 1
phase[left_x:right_x, high_y:high_y+h, :] = xy_bump


### Run muSpectre ###
#-------------------#
fourier_gradient = [µ.FourierDerivative(dim , i) for i in range(dim)]
cell = µ.Cell(nb_grid_pts, lengths, formulation, fourier_gradient)
mat = µ.material.MaterialLinearElastic4_3d.make(cell, "material")

for pixel_id, pixel in cell.pixels.enumerate():
    # add Young and Poisson depending on the material index
    m_i = phase.flatten(order='F')[pixel_id]  # m_i = material index/phase index
    mat.add_pixel(pixel_id, Young[m_i], Poisson[m_i])

cell.initialise()  # initialization of fft to make faster fft
DelF = np.array([[0, 0.7, 0],
                 [0, 0, 0],
                 [0, 0, 0]])

solver_newton = µ.solvers.KrylovSolverCG(cell, cg_tol, maxiter,
                                         verbose=µ.Verbosity.Silent)
result = µ.solvers.newton_cg(cell, DelF, solver_newton,
                             newton_tol, equil_tol,
                             verbose=µ.Verbosity.Silent)

# print solver results
print('\nstatus messages of optimizeresults:')
print('-----------------------------------')
print('success:            ', result.success)
print('status:             ', result.status)
print('message:            ', result.message)
print('# iterations:       ', result.nb_it)
print('# cell evaluations: ', result.nb_fev)
print('formulation:        ', result.formulation)


### visualisation ###
#-------------------#
# integration of the deformation gradient field
placement_n, x = gi.compute_placement(result, lengths,
                                      nb_grid_pts, fourier_gradient,
                                      formulation = formulation)

# some fields which can be added to the visualisation
# 2-tensor field containing the first Piola Kirchhoff stress
PK1 = result.stress.reshape((dim, dim) + tuple(nb_grid_pts), order='f')
F   = result.grad.reshape((dim, dim) + tuple(nb_grid_pts), order='f')

# scalar field containing the distance to the origin O
distance_O = np.linalg.norm(placement_n, axis=0)

# random fields
# shape of the center point grid
center_shape = tuple(np.array(x.shape[1:])-1)
scalar_c = np.random.random(center_shape)
vector_c = np.random.random((dim,) + center_shape)
tensor2_c = np.random.random((dim, dim) + center_shape)
scalar_n = np.random.random(x.shape[1:])
vector_n = np.random.random((dim,) + x.shape[1:])
tensor2_n = np.random.random((dim, dim) + x.shape[1:])

# write dictionaries with cell data and point data
c_data = {"scalar field": scalar_c,
          "vector field": vector_c,
          "2-tensor field": tensor2_c,
          "PK1 stress": PK1,
          "F deformation gradient": F,
          "phase": phase}
p_data = {"scalar field": scalar_n,
          "vector field": vector_n,
          "2-tensor field": tensor2_n,
          "distance_O": distance_O}

vt_ex.vtk_export(fpath="visualisation_example_finite_strain",
                 x_n=x,
                 placement=placement_n,
                 point_data=p_data,
                 cell_data=c_data)

print("\nThe file 'visualisation_example_finite_strain.vtr' was successfully"
      " written!\n"
      "You can open it for example with paraview or some other software:\n"
      "$ paraview visualisation_example_finite_strain.vtr")

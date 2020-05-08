# µSpectre

Copyright © 2018 Till Junge <till.junge@epfl.ch>

µSpectre is free software; you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free Software
Foundation, either version 3, or (at your option) any later version.

µSpectre is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with µSpectre; see the file COPYING. If not, write to the Free Software
Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.

Additional permission under GNU GPL version 3 section 7

If you modify this Program, or any covered work, by linking or combining it with
proprietary FFT implementations or numerical libraries, containing parts covered
by the terms of those libraries' licenses, the licensors of this Program grant
you additional permission to convey the resulting work.


This README contains only a small quick start guide. Please refer to the [full
documentation](https://muspectre.gitlab.io/muspectre) for more help.

## Building µSpectre
For the first installation of µSpectre, please refer to the 
[full documentation - getting started](https://muspectre.gitlab.io/muspectre/GettingStarted.html#)
for more information.

µSpectre is a CMake project that uses C++14. It depends on the Boost unit test
framework for testing and uses uses python3 as secondary API. You will need a
modern C++ compiler (µSpectre was tested with gcc-6 , gcc-7, clang-4 and clang5)
and CMake version 3.5.0 or higher.

### Compilation

  1. git clone https://gitlab.com/muspectre/muspectre.git
  2. cd build
  3. cmake -DCMAKE_BUILD_TYPE=Release ..
  4. make

µSpectre makes use of expression templates, as a result, production code
**must** be compiled with the `CMAKE_BUILD_TYPE` flag set to `Release` in order
to get performance (under gcc, non-optimised code is about 50 times slower than
the release). Be careful with parallel compilation: compiling µSpectre is quite
memory-hungry because of the use of expression templates, a parallel `make -j`
requires currently about 10 GB of RAM under GCC.

## Simple usage example
The following is a simple example for using µSpectre through its convenient
Python interface

    #!/usr/bin/env python3

    import numpy as np
    import muSpectre as µ

    # setting the geometry
    nb_grid_pts = [51, 51]
    center = np.array([r//2 for r in nb_grid_pts])
    incl = nb_grid_pts[0]//5

    lengths = [7., 5.]
    formulation = µ.Formulation.small_strain

    # creating the periodic cell
    rve = µ.SystemFactory(nb_grid_pts,
                          lengths,
                          formulation)
    hard = µ.material.MaterialLinearElastic1_2d.make(
        rve, "hard", 10e9, .33)
    soft = µ.material.MaterialLinearElastic1_2d.make(
        rve, "soft",  70e9, .33)


    # assign a material to each pixel
    for i, pixel in enumerate(rve):
        if np.linalg.norm(center - np.array(pixel),2)<incl:
            hard.add_pixel(pixel)
        else:
            soft.add_pixel(pixel)

    tol = 1e-5
    cg_tol = 1e-8

    # set macroscopic strain
    Del0 = np.array([[.0, .0],
                     [0,  .03]])
    if formulation == µ.Formulation.small_strain:
        Del0 = .5*(Del0 + Del0.T)
    maxiter = 401
    verbose = 2

    solver = µ.solvers.SolverCG(rve, cg_tol, maxiter, verbose=False)
    r = µ.solvers.newton_cg(rve, Del0, solver, tol, verbose)
    print("nb of {} iterations: {}".format(solver.name(), r.nb_fev))

You can find more examples using both the python and the c++ interface in the
[`examples`](./examples) and [`tests`](./tests) folder.

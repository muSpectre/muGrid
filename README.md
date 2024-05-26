# µGrid

µGrid is a library for discrete representation of fields on structured grids.
A *field* is a physical quantity that varies in space. µGrid makes it easy to
implement algorithms that operate on fields, such as solving partial
differential equations. It supports parallelization using domain decomposition
implemented using the Message Passing Interface (MPI).

µGrid is written in C++ and currently has language bindings for
[Python](https://www.python.org/).

This README contains only a small quick start guide. Please refer to the
[full documentation](https://muspectre.github.io/muGrid/) for more help.

## Quick start

To install µGrid, run

    pip install muGrid

Note that on most platforms this will install a binary wheel, that was
compiled with a minimal configuration. To compile for your specific platform
use

    pip install -v --no-binary muGrid muGrid

which will compile the code. µGrid will autodetect
[MPI](https://www.mpi-forum.org/).
For I/O, it will try to use
[Unidata NetCDF](https://www.unidata.ucar.edu/software/netcdf/)
for serial builds and
[PnetCDF](https://parallel-netcdf.github.io/) for MPI-parallel builds.
Monitor output to see which of these options were automatically detected.

## Simple usage example

The following is a simple example for using µGrid through its convenient
Python interface

    #!/usr/bin/env python3

    import numpy as np
    import muGrid

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

You can find more examples using both the Python and the C++ interface in the
[`examples`](./examples) and [`tests`](./tests) folder.

## Funding

This development has received funding from the
[Swiss National Science Foundation](https://www.snf.ch/en)
within an Ambizione Project and by the
[European Research Council](https://erc.europa.eu) within
[Starting Grant 757343](https://cordis.europa.eu/project/id/757343).

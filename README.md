# µGrid

Project µSpectre aims at providing an **open-source platform for efficient
FFT-based continuum mesoscale modelling**. This README contains only a small
quick start guide. Please refer to the
[full documentation](https://muspectre.gitlab.io/muspectre) for more help.

## Quick start

To install µSpectre, run

    pip install muGrid

Note that on most platforms this will install a binary wheel, that was
compiled with a minimal configuration. To compile for your specific platform
use

    pip install -v --no-binary muGrid muGrid

which will compile the code. Monitor output for the compilation options
printed on screen. µSpectre will autodetect various options and report
which ones were enabled.

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

## Funding

This development is funded by the
[Swiss National Science Foundation](https://www.snf.ch/en)
within an Ambizione Project and by the
[European Research Council](https://erc.europa.eu) within
[Starting Grant 757343](https://cordis.europa.eu/project/id/757343).

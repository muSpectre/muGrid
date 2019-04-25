.. _Summary:

Summary
-------

Project *µ*\Spectre aims at providing an open-source platform for efficient FFT-based continuum mesoscale modelling. It's development is funded by the `Swiss National Science Foundation <snf.ch>`_ within an Ambizione Project.

Computational continuum mesoscale modelling (or computational homogenisation) involves computing the overall response of a periodic unit cell of material, a so-called representative volume element (RVE), to a given average (i.e., macroscale) strain. Typically, this is done using the finite-element method, even though it is neither able to leverage its main strength, the trivial handling of complex geometries, nor otherwise particularly well suited for periodic problems. An alternative method for modelling periodic RVE, developed by `Moulinec and Suquet`_, is based on the fast Fourier transform (FFT). This method has evolved substantially over the last two decades, with particularly important and currently underused improvements in the last two years, see `Zeman et al`_, `de Geus et al`_.

This new method for the solution of the core problem of computational homogenisation is significantly  superior to the FEM  in terms of computational cost and memory footprint for most applications, but has not been exploited to its full potential. One major obstacle to the wide adoption of the method is the lack of a robust, validated, open-source code. Hence, researchers choose the well-known and tested FEM that has numerous commercial, open-source or legacy in-house FEM codes.

The goal of this project is to develop *µ*\Spectre, an open-source platform for efficient FFT-based continuum mesoscale modelling, which will overcome this obstacle. The project is designed to
i)
propose a de facto standard implementation for the spectral RVE method that subsequent implementations can be compared to, in order to concentrate the development effort of all interested parties in the field,
ii)
make *µ*\Spectre widely accessible for users by providing language bindings for virtually all relevant popular computing platforms and comprehensive user's manuals in order to help widespread adoption, and, finally
iv)
make *µ*\Spectre eminently modifiable for developers by developing it in the open, with a clean architecture and extensive developer's documentation in order to maximise outside contributions.

Furthermore, this project places great importance on truly reproducible and verifiable science with a credible open data strategy in the firm belief that these qualifiers help to reach and guarantee a high level of scientific quality, difficult to reach otherwise, and to attract outside collaborations and contributions that help boost the scientific output beyond what can be achieved by a single team.


.. _`Moulinec and_Suquet` :

`H. Moulinec and P. Suquet <https://doi.org/10.1016/S0045-7825(97)00218-1>`_. A numerical method for computing the overall response of nonlinear composites with complex microstructure. *Computer Methods in Applied Mechanics and Engineering*, 157(1):69–94, 1998. doi: 10.1016/S0045-7825(97) 00218-1.

.. _`Zeman et al` :

`J. Zeman, T. W. J. de Geus, J. Vondřejc, R. H. J. Peerlings, and M. G. D. Geers <https://dx.doi.org/10.1002/nme.5481>`_. A finite element perspective on non- linear FFT-based micromechanical simulations. *International Journal for Numerical Methods in Engineering*, 2017. doi: 10.1002/nme.5481.


.. _`de Geus et al` :

`T.W.J. de Geus, J. Vondřejc, J. Zeman, R.H.J. Peerlings, M.G.D. Geers <https://doi.org/10.1016/j.cma.2016.12.032>`_. Finite strain FFT-based non-linear solvers made simple, *Computer Methods in Applied Mechanics and Engineering* (318, pp. 412-430), 2017

Change log for µSpectre
=======================

0.11.0 (07Sep20)
----------------

- Trust-Region Newton-CG solver for nonlinear problems with instabilities
- Updated Eigen archive URL which broke installation via pip

0.10.0 (22Jul20)
----------------
- Support for more flexible strideste in fields: full column-major, row-major and
  strided pixels portion (#103)
- User control over buffer copies in muFFT with option to avoid them completely 
- Gradient integration for multiple quadrature points

0.9.3 (28Jun20)
---------------
- Bug fix: Packaging with sdist did not remove dirty flag which lead to a
  broken PyPI package

0.9.2 (28Jun20)
---------------
- Bug fix: operator= of TypedFieldBase passed wrong strides during strided
  copy; this broke the MPI-parallel FFTW forward transform (#130)

0.9.1 (17Jun20)
---------------
- Bug fix: Packaging with sdist only included FFT engines present during
  the packaging process

0.9.0 (17Jun20)
---------------
- Initial release of µSpectre
  * FFT based micromechanical homogeneization
  * Arbitrary constitutive laws in small and finite strain
  * Krylov and Newton solver suite
  * MPI parallelization
- Initial release of µFFT
  * Generic wrapper for MPI-parallel FFT libraries
- Initial release of µGrid
  * Generic library for managing regular grids

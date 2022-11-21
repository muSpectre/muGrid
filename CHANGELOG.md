Change log for µSpectre
=======================

0.24.0 (not yet released)
-------------------------

- muFFT: Changed Python install procedure; default to no MPI and MPI now needs
  to be explicitly enabled

0.23.1 (24Mar2022)
-------------------------
- muSpectre: A bug fixed in the call of the constructor of FieldCollection
- muFFT: Fixed `pip install muFFT` on macOS


0.23.0 (15Oct2021)
-------------------------
- muSpectre: making vector operation methods in solver classed with communicating inside them
- muSpectre: added logical reduction on was_last_step_nonlinear evaluation in CellData 


0.22.0 (24Sep2021)
-------------------------
- muSpectre: mean stress control bugs resolved
- muSpectre: mean stress control is now usable in MPI
- muSpectre: examples using mean stress control added


0.21.0 (26Aug2021)
-------------------------
- CI: Refactoring of the CI with addition of coverage and ccache


0.20.2 (04Aug2021)
-------------------------
- muSpectre: deleted an unnecssary parameter from write_2d_class function in iinear_finite_elements.py


0.20.1 (03Aug2021)
-------------------------
- muSpectre: improve write_2d and write_3d functions, 2D stencil for hexagonal grid


0.20.0 (15Jul2021)
-------------------------
- muSpectre: capability to apply mean stress (instead of mean strain added)


0.19.2 (28Jun2021)
-------------------------
- muGrid: correct bugs in the FileIONetCDF


0.19.1 (23Jun2021)
-------------------------
- muSpectre: Fixed a minor array reshape bug in the tutorial_example_new.py that was jeopardizing the output stress plot


0.19.0 (16Jun2021)
-------------------------
- muSpectre: added material_dunnat_tc (bilinear elastic- linear strain softening with tensile-compressive wiegthed norm as strain measure)
- muSpectre: added material_dunnat_t (bilinear elastic- linear strain softening with maiximum tensile principal strain as strain measure)


0.18.2 (10Jun2021)
-------------------------
- muSpectre: Small bugfix and addition of regularization for slightly non-pd Hessians in phase field fracture material
- muGrid: Added functions for reporting version
- muGrid: Added global attributes to FileIONetCDF

0.18.1 (02Jun21)
----------------
- muSpectre: Fix, changed the reset criterion for gradient orthogonality in
  FEM trust region precondtioned Krylov solvers
- muSpectre: Fix, added calling clear_was_last_step_nonlinear in
  fem_newton_trust_region_pc solver
- muSpectre: Fixed get_complemented_positions
- muFFT: Fixed large transforms
- muFFT: Fixed segfault when input buffer had wrong shape


0.18.0 (10May21)
----------------
- muSpectre: Added trust fem region solver + ability to handle precondtioner
- muSpectre: re-organized the krylov solver hierarchy to circumvent diamond
  inheritance by introducing KrylovSolverXXXTraits classes


0.17.0 (24Apr21)
----------------
- muSpectre: Added trust region solver class and a simplistic damage material
- muSpectre: physics have their specific name that might be used later for outputs
- muSpectre: trsut region krylov solver has different resetart strategies available
- muSpectre: gradient integration for solver class + cell data is now available


0.16.0 (31Mar21)
----------------

- µSpectre: The ProjectionGradient works for vectorial and rank-two-tensor
  gradient fields and replaces ProjectionFiniteStrainFast
- µSpectre: The SolverNewtonCG can now handle scalar problems (e.g., diffusion
  equation, heat equation, etc.)
- clang: No more warnings are emitted during compilation


0.15.1 (30Mar21)
-------------------------

- muSpectre: Added material for phase field fracture simulations
- muGrid: Fix support for fields with >= 2^32 elements


0.15.0 (12Mar21)
----------------

- muSpectre: All projection operators now have a gradient argument and can
  work with discrete derivatives
- muSpectre: Projection classes now have an `integrate` method that
  reconstructs the node positions
- muSpectre: Added `linear_finite_elements` to stencil database
- muSpectre: Enabled even number of grid points for discrete stencils
- muFFT: PFFT engine works with pencil decomposition
- all: Fixed installation via CMake and `make install`
- all: Fixed cross platform install of NetCDF I/O


0.14.0 (04Feb21)
----------------

- muFFT: implement serial wrapper to FFTW hcfft


0.13.0 (28Jan21)
----------------

- muSpectre: CellData and Solver classes for multiphysics calculations
- muSpectre: Sensitivity analysis
- Bug fix (muFFT): Handle cases where MPI processes have no grid points


0.12.0 (19Nov20)
----------------

- muGrid: Parallel I/O via NetCDF
- muFFT: Derivatives in 1D
- muFFT: Second derivatives


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

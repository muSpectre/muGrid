<!--
Style for this changelog (keep to it so entries stay uniform):

- Newest version first. Heading `vX.Y.Z (DDMonYY)`, underlined with dashes.
  A version not yet released is headed `vX.Y.Z (not yet released)`.
- One bullet per user-visible change, at most two lines (about 200
  characters). Say what changed for the user and, if useful, why or by how
  much. Omit internals: no function, class, file or variable names, no
  environment variables, no commit or PR references.
- Start every bullet with one of: `ENH:` (new feature), `BUG:` (fix),
  `API:` (interface change), `PERF:` (speed or memory), `MAINT:`, `TST:`,
  `DOC:`, `BUILD:`, `CI:`.
- Small related fixes may share one bullet. No trailing period on bullets.
-->

Change log for µGrid
====================

v1.5.2 (not yet released)
-------------------------

- BUG: GPU-aware MPI is detected with any MPI library that can report it, including
  MPICH and its derivatives, not only Open MPI; others still stage through the host
- BUG: On unified-memory APUs, plain device memory is staged through the host for a
  non-GPU-aware MPI and for NetCDF instead of failing; managed memory is used in place
- BUILD: Configuring reports which GPU-support queries the MPI library provides
- ENH: The communicator reduces, gathers and exchanges NumPy and CuPy arrays between
  ranks, in C++ and Python, staging device arrays through the host when MPI needs it
- MAINT: Ghost exchange, FFT transposes and Python reductions share one decision on
  whether MPI takes device buffers, instead of each making its own

v1.5.1 (18Oct26)
----------------

- BUG: ROCm builds of 1.5.0 failed to import (undefined symbol): the GPU kernel of
  the Green preconditioner's block product was compiled for CUDA only
- BUILD: Builds from a source archive, e.g. an offline pip install of the sdist, report
  their release version (and the commit of a development version) instead of "unknown"

v1.5.0 (18Oct26)
----------------

- PERF: The hybrid preconditioner's GPU sweep is compiled into the library, so it no
  longer compiles a kernel at run time; it is 1.2-2x faster in 2D, unchanged in 3D
- PERF: The Green preconditioner's per-mode block product on GPUs is compiled into the
  library instead of generated at run time, and 1.1-4.6x faster
- ENH: The Laplace operator takes a grid spacing per axis, so it is correct on
  anisotropic grids. Without one it behaves as before
- ENH: Per-frame NetCDF variables are written as keyword arguments of a frame,
  are validated, can be read back, and carry a fill value for frames without a value
- PERF: The hybrid FFT/tridiagonal preconditioner is up to 5x faster on GPUs:
  transforms in memory order, a retuned sweep kernel, and direct GPU-to-GPU exchange
- BUG: Both conjugate-gradient solvers confirm convergence on the true residual,
  raise when it stagnates, and detect indefinite operators and preconditioners
- BUG: The pipelined conjugate-gradient solver no longer returns drifted iterates:
  it periodically recomputes its residual, at about 10% more operator applications
- BUG: The single-precision Green-function preconditioner lost long-wavelength modes
  on large grids; its symbol is now built in double where needed, at float32 speed
- BUG: The FEM gradient transpose was wrong at subdomain boundaries. It now writes
  only owned nodes: communicate the input's ghosts first, never reduce the output
- BUG: Solvers use private work fields per call, so nested solves and solves of
  different precision on one field collection no longer interfere
- BUG: The stored reference preconditioner for single-precision solves is built in
  double and stored at half the memory
- BUG: Requesting an existing single-precision field with another precision, shape
  or sub-point raises instead of returning the wrong field
- BUG: Operators and kernels validate their inputs: sub-point counts, padded storage,
  transpose weight counts, and outputs that would have been silently copied
- BUG: Stencil coefficients are dropped relative to the largest one, not below an
  absolute threshold, so operators in SI units at small scales no longer vanish
- BUG: NetCDF reads into a narrower type (double into float, float into integer)
  raise instead of rounding silently
- BUG: MPI reductions: uneven gathers are rejected on every rank, integer min/max no
  longer deadlock on empty ranks, sums keep complex parts and large integers
- BUG: Fields on several GPUs of one process: every operation runs on the field's
  own device. Python GPU reductions stage through the host without GPU-aware MPI
- BUG: The pixel view of a field with several sub-points raises on device storage,
  where its ordering differs from the host; use the sub-point view instead
- BUG: The 3D Laplace stencil reported to Python had x and z swapped; in-place
  addition on float64 GPU fields was missing
- BUG: Serial single-precision FFTs work with every backend; the Nyquist frequency
  follows numpy; cuFFT plans beyond the 32-bit range raise; no rank is left empty
- BUG: Smaller fixes: ghost exchange refuses non-column-major fields, a decomposition
  without subdivisions splits the ranks itself, the deviatoric projector is right in 2D
- API: All boolean environment switches accept 1/0, yes/no, true/false and on/off,
  and reject anything else
- API: A single-precision FEM gradient operator is selectable by dtype; the nodal
  moment operator is exported from the package
- DOC: The 3D linear-tetrahedron element is the five-tetrahedra voxel of Leute's
  thesis, not a Kuhn triangulation; tests pin its tables and the stencil it builds
- DOC: Fourier-space ghost padding, inverse-transform input requirements, operator
  sign conventions, coordinate semantics and MPI initialisation on import

v1.4.1 (07Oct26)
----------------

- BUG: Replacing or clearing the device allocator no longer frees memory of live
  device fields; each block is freed by the allocator that made it
- API: The C++ device-allocator hook takes a context and a release callback, so
  stateful allocators need no global state; callbacks run outside muGrid's lock
- BUG: The GPU stiffness operator's stress average uses a cached scratch buffer
  instead of allocating raw device memory on every call

v1.4.0 (25Sep26)
----------------

- ENH: New hybrid preconditioner: FFT in the rank-local axes and a tridiagonal solve
  in the distributed one, so no all-to-all. Exact, and it runs on GPUs
- PERF: The hybrid's tridiagonal sweep is one fused GPU kernel, 50-120x faster, with
  compressed factors taking 7x less memory
- ENH: The Green preconditioner can evaluate its symbol per mode instead of storing
  it: 2.3 GB less at 512³ and 2.7x faster per apply on GPUs, where it is the default
- ENH: New multigrid reference preconditioner (serial for now) with derived Jacobi
  damping, and multilinear grid transfers between nested grids with exact adjoints
- ENH: One public reference stencil defines the uniform reference operator for all
  preconditioners; the assembled route no longer needs an impulse response
- ENH: The homogenization example gained multigrid, synchronised timers and a
  decomposition option; new examples price V-cycles against FFTs and the FFT split
- PERF: The reference symbol is assembled at the solve precision and inverted slab
  by slab, cutting peak memory by a quarter
- BUG: The hybrid preconditioner computes on its decomposition's GPU, not the
  current one; multi-GPU runs faulted and deadlocked
- BUG: Reductions of single-precision fields return double, so large values no
  longer overflow and stall the solver; squares are widened before multiplying
- BUG: Both solvers reject non-finite residuals and curvatures with a clear error; a
  float32 tolerance below the accuracy floor warns
- BUG: FFT transposes and GPU kernels use range-checked counts, so oversized
  transforms raise instead of wrapping
- BUG: The hybrid deflates its null space exactly and uses precision-aware
  tolerances; its single-precision path is tested end to end
- BUG: Multigrid coarse levels are built on the fine level's device; its failures
  say what is wrong
- BUILD: HIP builds on EasyBuild toolchains find the right GCC installation
- TST: The solver test file was never collected because of its name; fixed

v1.3.0 (16Sep26)
----------------

- ENH: New nodal moment operator: cell moments of a nodal field's FE interpolant and
  their gradients in one fused host/GPU pass, for polynomial phase-field energies
- ENH: Elements gain shape-function values and a moment quadrature that is exact
  for quartic integrands with positive weights
- TST: The simplex decompositions are stored as data and tied to the gradient
  tables by compile-time checks

v1.2.0 (14Sep26)
----------------

- PERF: Device-to-device copies use a kernel (30x faster on managed memory) and
  transforms block the host only when needed: 30-40% per CG iteration
- BUG: GPU builds no longer lose their Python bindings to link-time optimisation;
  a CUDA/HIP runtime older than the headers is detected at first use

v1.1.1 (12Sep26)
----------------

- PERF: The fused stiffness kernels keep their geometry at working precision in
  constant memory and tile through shared memory: up to 15x faster in float32
- PERF: The block-Fourier preconditioner applies in one fused GPU kernel; GPU
  reductions cap their grid size
- MAINT: Serial builds no longer initialise MPI on import

v1.1.0 (04Sep26)
----------------

- ENH: Ghost reduction supports halos wider than a rank's interior and is the exact
  adjoint of ghost communication for any decomposition
- ENH: The FFT engine's field helpers accept sub-points
- BUG: FFTs of fields with several sub-points returned garbage; mismatched field
  pairs and fields from foreign collections are rejected
- TST: An environment switch exercises the axis-by-axis FFT paths in the test suite

v1.0.0 (16Jul26)
----------------

- ENH: NetCDF output can be flushed to disk for incremental checkpointing
- BUG: GPU architecture autodetection, guards against GPU API failures and against
  mismatched NetCDF reads, float32 NetCDF I/O, rejection of GPU backends not built

v0.112.0 (07Jul26)
------------------

- ENH: Double-precision accumulators even in single-precision runs
- ENH: Per-frame, grid-less quantities can be stored in NetCDF output next to fields
- ENH: A one-line build and run diagnostic string; optional native CPU tuning at
  build time; a warning when a serial run is started under an MPI launcher
- BUG: Green-Jacobi preconditioner and single-precision preconditioners fixed

v0.111.0 (03Jul26)
------------------

- ENH: Single-precision floating-point support throughout
- ENH: Q1 (bilinear and trilinear) elements
- ENH: Fused sensitivity, macroscopic right-hand side and average stress on the
  isotropic stiffness operator, so homogenization needs no strain or stress fields
- ENH: The Green-Jacobi (J-FFT) preconditioner; the solver can return the final
  residual
- API: Fields can be created with a dtype; fields can be removed from Python; Python
  object lifetimes are tied to the iterators and exports that use them
- BUG: Input guards across operators and fields; pixel views are zero-copy and
  correctly ordered; several MPI, NetCDF, FFT and GPU memory fixes

v0.110.0 (28Jun26)
------------------

- ENH: Reference-material preconditioner and a pipelined conjugate-gradient solver
  for FFT-accelerated FE homogenization
- ENH: Native multidimensional GPU FFTs, slab decomposition with pencil fallback,
  and an optional staged host transpose for multi-node networks
- ENH: NetCDF I/O for GPU-resident fields; complex GPU linear algebra; fused
  cross-product and Helmholtz projection kernels
- PERF: Fewer device synchronisations, no host bounce on unified-memory devices,
  cached reduction scratch
- API: Field deep copies, host-accessibility queries and empty clones of collections
- DOC: Documentation moved to MkDocs with CPU/GPU scaling benchmarks

v0.109.0 (15Jun26)
------------------

- ENH: Helpers for reductions on GPUs

v0.108.0 (13Jun26)
------------------

- ENH: Non-blocking all-to-all in FFT transposes; ghost reduction without a host buffer
- BUG: NetCDF I/O of local field collections with empty MPI ranks

v0.107.0 (12Jun26)
------------------

- ENH: Multi-GPU halo exchange and FFT transposes scale through contiguous staging
  buffers; solvers run entirely on the device
- ENH: True pencil decomposition for 3D MPI FFTs with per-rank memory scaling as 1/P
- ENH: Operators report and validate the ghost layers they need
- ENH: New preconditioner module (identity, Jacobi, Fourier) for the solver
- API: The conjugate-gradient solver converges on a relative criterion
- BUG: Many MPI, FFT, NetCDF, GPU and binding fixes, including silently dropped
  ghost contributions on devices and garbled multi-component transposes

v0.106.0 (09Jun26)
------------------

- ENH: Preconditioned conjugate gradients; optional coverage instrumentation
- BUG: GPU linear algebra on sub-point fields ran past the buffer; mismatched
  component counts are rejected

v0.105.2 (17Apr26)
------------------

- BUG: Coordinates of FFT engines with ghost cells were garbage in the last dimension

v0.105.1 (11Jan26)
------------------

- BUG: The inverse FFT frequency property was misnamed on GPU engines

v0.105.0 (11Jan26)
------------------

- ENH: 1D FFTs on all backends (serial only)

v0.104.0 (10Jan26)
------------------

- API: FFT engine field methods take a component shape, consistent with field
  collections; scalar fields have no component axis

v0.103.0 (07Jan26)
------------------

- ENH: Operators compute every point for which the stencil has valid input, so wider
  ghost regions get valid results too

v0.102.0 (05Jan26)
------------------

- API: A device class replaces the memory-location enum; devices are given as
  strings such as "cpu", "gpu" or "cuda:1"
- API: The generic stencil operator was renamed; the solver interface and the
  convolution operators were simplified
- ENH: Fused isotropic stiffness operators for linear elasticity in 2D and 3D on CPU
  and GPU, storing only two material parameters per voxel
- ENH: A linear-algebra module for fields that avoids copies in the solver
- ENH: Native rocFFT backend with full stride support; MPI-parallel examples
- BUG: Fixed the 3D stiffness kernel on GPUs and handling of uninitialised MPI

v0.101.2 (29Dec25)
------------------

- PERF: Optimised GPU gradient and divergence kernels without atomic operations

v0.101.1 (28Dec25)
------------------

- BUG: FFT engine field creation returns an existing field of the same name;
  register-only variants raise instead

v0.101.0 (28Dec25)
------------------

- API: Frequency and coordinate arrays are properties of the FFT engine; the
  standalone frequency functions are gone; properties return tuples

v0.100.0 (28Dec25)
------------------

- ENH: Offset, shape and coefficients of all stencil operators are accessible from
  Python

v0.99.0 (28Dec25)
-----------------

- ENH: Fourier-space representation of stencil operators

v0.98.1 (28Dec25)
-----------------

- BUG: Version detection in the publish workflow
- MAINT: Versions come from git tags; build configuration consolidated

v0.98.0 (27Dec25)
-----------------

- ENH: Simplified user-facing API; Windows support; FEM gradient operator with a
  homogenization example; hierarchical timer; ghost reduction
- ENH: 3D Poisson example and an automatic benchmark suite
- MAINT: Unified GPU code for CUDA and HIP
- CI: GPU testing workflow

v0.97.0 (22Dec25)
-----------------

- ENH: GPU support; new parallel FFT with arbitrary ghost buffers; sparse stencils
- BUG: NetCDF output of fields with ghost buffers
- MAINT: muFFT is deprecated

v0.96.0 (15Dec25)
-----------------

- ENH: Looping over strided fields; ghost buffers larger than subdomains
- BUG: Memory leak in ghost communication

v0.95.0 (15Jul25)
-----------------

- ENH: Accessor properties for field data with and without ghosts
- ENH: Parallel conjugate-gradient solver
- API: Flipped axis order of the convolution operator
- MAINT: Dropped Python 3.8

v0.94.0 (18Feb25)
-----------------

- ENH: General convolution operator for fields
- ENH: Domain decomposition with ghost buffer communication

v0.93.3 (11Nov24)
-----------------

- BUG: Do not divide by the smallest stride when it is zero

v0.93.2 (22Oct24)
-----------------

- BUG: Storage-order detection for arrays with a single component

v0.93.1 (20Oct24)
-----------------

- ENH: Numpy wrappers return arrays with exactly the input's shape

v0.93.0 (20Oct24)
-----------------

- API: Always return the full component shape; scalar fields take an empty
  component shape

v0.92.6 (25Sep24)
-----------------

- CI: macOS x86_64 wheels

v0.92.5 (11Jul24)
-----------------

- BUG: Installation without NetCDF

v0.92.4 (30Jun24)
-----------------

- MAINT: Utility for copying a field into a numpy array

v0.92.3 (15Jun24)
-----------------

- BUILD: Do not override dependencies after dl and execinfo are detected

v0.92.2 (14Jun24)
-----------------

- BUILD: Re-added dl and execinfo as requirements

v0.92.1 (14Jun24)
-----------------

- BUILD: Require Eigen 3.4.0 or newer

v0.92.0 (04Jun24)
-----------------

- ENH: NetCDF I/O accepts an mpi4py communicator
- BUG: Import without NetCDF

v0.91.1 (02Jun24)
-----------------

- BUG: Numpy wrapper updated for global field collections without a dimension argument

v0.91.0 (01Jun24)
-----------------

- ENH: Convenience accessors for pixel- and sub-point-shaped arrays; get-or-create
  field accessors; an overwrite open mode for NetCDF
- MAINT: The default communicator is the single-process one when MPI is enabled
- DOC: Documentation and examples of the Python bindings

v0.90.1 (23May24)
-----------------

- BUILD: Fixed wheel and source deployment to PyPI

v0.90.0 (21May24)
-----------------

- MAINT: Split into separate repositories: muGrid and muFFT

v0.27.0 (30Jan24)
-----------------

- ENH: Sensitivity analysis for 3D problems (muSpectre)
- BUILD: Updated Eigen, pybind11 and the meson-python pin

v0.26.4 (04Oct23)
-----------------

- BUILD: Pinned meson-python because of a bug in a newer version

v0.26.3 (16Jul23)
-----------------

- CI: Same as 0.26.2 (debugging CI)

v0.26.2 (10Jul23)
-----------------

- CI: Same as 0.26.1, fixed deployment

v0.26.1 (08Jul23)
-----------------

- CI: macOS wheels

v0.26.0 (31Mar23)
-----------------

- ENH: MPI parallelisation of the sensitivity analysis; wheels for Python 3.11

v0.25.2 (14Jan23)
-----------------

- BUILD: Fixed the macOS build

v0.25.1 (28Dec22)
-----------------

- CI: Same as 0.25.0, automatic deployment to PyPI

v0.25.0 (28Dec22)
-----------------

- ENH: Linux wheels; a PocketFFT engine without external dependencies
- BUILD: Meson build files; MPI autodetected again

v0.24.0 (22Nov22)
-----------------

- BUILD: MPI must be enabled explicitly for the Python install

v0.23.1 (24Mar22)
-----------------

- BUG: Field collection constructor call (muSpectre); pip install on macOS (muFFT)

v0.23.0 (15Oct21)
-----------------

- ENH: Communicating vector operations in solver classes; logical reduction of the
  nonlinearity flag (muSpectre)

v0.22.0 (24Sep21)
-----------------

- BUG: Mean-stress control fixed and usable under MPI, with examples (muSpectre)

v0.21.0 (26Aug21)
-----------------

- CI: Refactored CI with coverage and ccache

v0.20.2 (04Aug21)
-----------------

- MAINT: Removed an unneeded parameter from a finite-element helper (muSpectre)

v0.20.1 (03Aug21)
-----------------

- ENH: Improved 2D and 3D writers; 2D stencil for hexagonal grids (muSpectre)

v0.20.0 (15Jul21)
-----------------

- ENH: Mean-stress control instead of mean strain (muSpectre)

v0.19.2 (28Jun21)
-----------------

- BUG: NetCDF I/O fixes

v0.19.1 (23Jun21)
-----------------

- BUG: Array reshape in the tutorial example (muSpectre)

v0.19.0 (16Jun21)
-----------------

- ENH: Two bilinear elastic, linear strain-softening materials (muSpectre)

v0.18.2 (10Jun21)
-----------------

- ENH: Version reporting; global attributes in NetCDF files
- BUG: Regularisation of slightly non-definite Hessians in the phase-field fracture
  material (muSpectre)

v0.18.1 (02Jun21)
-----------------

- BUG: Trust-region solver resets and position reconstruction (muSpectre); large
  transforms and wrongly shaped input buffers (muFFT)

v0.18.0 (10May21)
-----------------

- ENH: Trust-region FEM solver with preconditioning; reorganised Krylov solver
  hierarchy (muSpectre)

v0.17.0 (24Apr21)
-----------------

- ENH: Trust-region solver, a damage material, named physics, restart strategies and
  gradient integration (muSpectre)

v0.16.0 (31Mar21)
-----------------

- ENH: The gradient projection handles vector and rank-two gradients; the Newton-CG
  solver handles scalar problems (muSpectre)
- BUILD: No warnings with clang

v0.15.1 (30Mar21)
-----------------

- ENH: Phase-field fracture material (muSpectre)
- BUG: Fields with 2^32 elements or more

v0.15.0 (12Mar21)
-----------------

- ENH: Projection operators take a gradient argument and work with discrete
  derivatives; node positions can be reconstructed (muSpectre)
- ENH: Pencil decomposition in the PFFT engine (muFFT)
- BUILD: Fixed CMake installation and cross-platform NetCDF I/O

v0.14.0 (04Feb21)
-----------------

- ENH: Serial wrapper for FFTW half-complex transforms (muFFT)

v0.13.0 (28Jan21)
-----------------

- ENH: Cell data and solver classes for multiphysics; sensitivity analysis (muSpectre)
- BUG: MPI processes without grid points (muFFT)

v0.12.0 (19Nov20)
-----------------

- ENH: Parallel I/O via NetCDF; first and second derivatives in 1D (muFFT)

v0.11.0 (07Sep20)
-----------------

- ENH: Trust-region Newton-CG solver for nonlinear problems with instabilities
- BUILD: Updated the Eigen archive URL that broke pip installs

v0.10.0 (22Jul20)
-----------------

- ENH: Flexible field strides (column-major, row-major, strided pixels); control
  over buffer copies in muFFT; gradient integration for several quadrature points

v0.9.3 (28Jun20)
----------------

- BUG: Source packaging kept a dirty flag that broke the PyPI package

v0.9.2 (28Jun20)
----------------

- BUG: Strided field copies passed the wrong strides, breaking the MPI-parallel
  FFTW forward transform

v0.9.1 (17Jun20)
----------------

- BUG: Source packaging only included the FFT engines present at packaging time

v0.9.0 (17Jun20)
----------------

- ENH: Initial release of µSpectre (FFT-based micromechanical homogenization with
  arbitrary constitutive laws, Krylov and Newton solvers, MPI), µFFT and µGrid

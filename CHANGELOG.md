Change log for µGrid
====================

v0.110.0 (28Jun26)
------------------

- ENH: Added compile-time field allocation profiling with low-overhead runtime control
  and Python bindings
- ENH: Skipped MPI host-staging bounce on unified-memory devices during ghost exchanges
  and FFT transposes
- ENH: Native multidimensional R2C/C2R transforms in cuFFT and rocFFT backends for
  faster serial GPU FFTs
- ENH: Parallel 3D FFT uses native multi-dimensional transforms for slab decompositions
  on CPU and GPU
- ENH: Added slab decomposition option in `FFTEngine` with automatic fallback to pencil
  decomposition
- ENH: Optional staged host FFT transpose utilizing contiguous buffers for multi-node
  networks
- ENH: Added reference-material preconditioner for FFT-accelerated FE homogenization to
  improve CG convergence
- ENH: Added pipelined CG solver option using non-blocking reductions to hide latency
- BUG: GPU kernel launches are now checked and raise exceptions instead of failing
  silently
- MAINT: Cache the linear algebra reduction scratch buffer across calls instead of
  reallocating
- DOC: Converted documentation from Sphinx to MkDocs and added CPU/GPU scaling benchmark
  pages
- ENH: Supported NetCDF I/O for device-resident GPU fields via staging buffers
- ENH: Skipped staging buffer on unified-memory devices for zero-copy NetCDF I/O
- API: Added field deep-copy, host-accessibility queries, and empty-clone overloads for
  field collections
- BUG: Fixed NetCDF variable-field reads with ghosts for structure-of-arrays fields
- ENH: Implemented complex linear algebra operations on CUDA/HIP GPUs, excluding ghost
  regions
- ENH: Fused per-pixel vector operations for three-vector cross product and
  Helmholtz/Leray projection
- ENH: PocketFFT backend performs multi-dimensional transforms in a single call for
  faster serial CPU execution
- API: Renamed the FFT backend header and exposed optional N-dimensional transforms.

0.109.0 (15Jun26)
-----------------

- ENH: Helpers for reduction on GPUs

0.108.0 (13Jun26)
-----------------

- ENH: FFT transpose all-to-all uses non-blocking communication
- ENH: `reduce_ghosts` without intermediate host buffer
- BUG: NetCDF I/O of local field collections now works with empty MPI ranks

0.107.0 (12Jun26)
-----------------

- ENH: Multi-GPU halo exchange and FFT transposes now scale: contiguous staging buffers
  (with a host bounce when MPI is not GPU-aware) instead of strided MPI datatypes; FFT
  scratch is cached across transforms
- ENH: Solvers run entirely on the device
- API: `conjugate_gradients` now converges on a relative criterion
- BUG: Fixed device `reduce_ghosts` silently dropping all contributions received from
  MPI neighbors (the host receive buffer was treated as device memory during
  accumulation); found by a new device-vs-host equivalence test
- BUG: Interior reductions (`norm_sq`, `vecdot`, `axpy_norm_sq`) on host and GPU now sum
  the interior region directly instead of subtracting the ghost contribution from a
  full-buffer reduction (minimizing floating point overflows)
- ENH: Stencil operators now report the ghost layers they need
- ENH: All stencil operators (including the FEM gradients, which previously did not
  check) now validate at apply/transpose time that the field collection provides the
  ghost layers they report
- BUG: Removed the allgather/scatter-only `Transpose` modes, which posted overlapping
  receive buffers (UB) and garbled multi-component 3D transforms
- BUG: FFT transposes now honour field storage order (AoS/SoA); device fields were
  transposed with AoS datatypes, garbling components
- BUG: Fixed a deadlock in `CartesianDecomposition::reduce_ghosts` for uneven
  subdomains; the halo-size check now uses the global minimum interior extent
- BUG: Fixed state-field index rotation using bitwise `&` instead of modulo, which
  aliased `current()`/`old()` for `nb_memory` not of the form 2^k-1
- BUG: Fixed the 3D MPI FFT silently skipping the Y transform for process grids with P2
  == 1 and P1 > 1
- BUG: Fixed NetCDF `compute_tensor_dim_index` never throwing its error and an inverted
  `static_assert` in the NetCDF type mapping
- BUG: Fixed `Unit` streaming emitting an empty string for tagged-but-unitless units
- BUG: Bound `FEMGradientOperator.apply_increment`/`.transpose_increment` and
  `Communicator.max`/`.all`/`.any`; corrected misspelled `uint_state_field`
- BUG: Fixed the `PyGradientOperator.transpose_increment` trampoline calling `transpose`
- BUG: Added Python wrappers for `IsotropicStiffnessOperator2D`/`3D`; `from muGrid
  import *` no longer breaks
- BUG: The DLPack capsule now keeps the owning field (and its collection) alive for the
  lifetime of the exported tensor
- BUG: Fixed host<->device `deep_copy` permuting multi-axis component / multi-sub-point
  data (AoS vs SoA dof ordering)
- BUG: `FieldMap::size()`/`eigen_vec()`/`get_empty_clone()` now handle buffer padding
  and ghost specifications correctly
- BUG: Fixed 64-bit index overflow in index helpers for grids exceeding 2^31 points
- BUG: Free the MPI communicator created by `MPI_Cart_create`; removed collective MPI
  calls from `assert`s (debug/release deadlock)
- BUG: GPU isotropic-stiffness now uploads per-instance G/V matrices; device
  `GenericLinearOperator` caches are invalidated on grid change
- BUG: GPU linalg ghost reductions now handle 1D fields; cuFFT inverse transforms
  synchronize before GPU-aware MPI
- BUG: MPI ghost accumulation now honours device memory and the actual element type
  (Complex/Int) instead of hard-coding host-side `Real`
- BUG: `reduce_ghosts` rejects halos larger than the subdomain instead of silently
  producing a wrong reduction
- ENH: The 3D MPI FFT now uses a true pencil decomposition; per-rank memory scales as
  O(N³/P) and Fourier space distributes X across P2, Y across P1
- ENH: MPI transposes use pure derived-datatype `MPI_Alltoallw` operations (no
  pack/unpack) on host buffers; device buffers use contiguous staging (see above)
- ENH: The FFT engine verifies at construction that its real- and Fourier-space
  collections use the same storage order
- ENH: CMake now probes for C++20 standard-library support at configure time and fails
  with an actionable message instead of a cryptic mid-build error
- ENH: Added `Communicator::min` (C++ and Python), mirroring `max`
- ENH: New `muGrid.Preconditioners` module for the matrix-free CG solver, with
  `Identity`, `Jacobi` and `Fourier` (spectral kernel) preconditioners
- ENH: The Poisson example gained a `-P/--preconditioner` option
  (`fourier`/`fourier-exact`) with per-stage (fft/kernel/ifft) timing
- ENH: The Python `FFTEngine` exposes `communicate_ghosts`/`reduce_ghosts`, so one
  engine can serve as the single decomposition for stencils and FFTs
- MAINT: Removed dead code (`fft_work_buffer.hh`, stray repo-root files, dead
  `[tool.flake8]` section); fixed latent compile-breakers in const accessors
- TST: MPI FFT tests now compare against numpy for unevenly dividing grids,
  multi-component fields and the 3D inverse transform
- TST: Added regression tests for the state-field rotation and the Python API surface
  (star import, FEM-gradient increments, isotropic-stiffness wrapper)
- DOC: Coding convention: use brace initialization (non-narrowing); narrowing
  conversions must be explicit `static_cast`s
- DOC: Corrected numerous C++ doxygen and Sphinx documentation mismatches

0.106.0 (09Jun26)
-----------------

- ENH: Preconditioned conjugate gradients
- ENH: Optional code-coverage instrumentation via `MUGRID_ENABLE_COVERAGE`
- ENH: Bound `LaplaceOperator.apply_increment` and `.transpose` to Python
- BUG: Fixed GPU linalg element count for sub-point fields (`nb_sub_pts > 1`), where
  device kernels ran past the end of the buffer
- BUG: `linalg.axpy`/`axpby`/`copy`/`axpy_norm_sq` now reject fields with mismatched
  component counts
- TST: Added functional tests for the 2D/3D Laplace operator
- TST: Added functional tests for host linear-algebra operations
- MAINT: Removed dead code with no callers

0.105.2 (17Apr26)
-----------------

- BUG: Fixed `coords`/`icoords` returning garbage in the last grid dimension when the
  FFT engine is created with ghost cells
- TST: Added regression test `test_coords_2d_with_ghosts` that exercises `coords` and
  `coordsg` when ghost cells are present

0.105.1 (11Jan26)
-----------------

- BUG: Fixed `ifftfreq` property name on GPU FFT engines (CUDA/ROCm)
  - Was incorrectly named `fftfreqind` instead of `ifftfreq`
  - Now consistent with CPU `FFTEngine` naming

0.105.0 (11Jan26)
-----------------

- ENH: Added 1D FFT support to `FFTEngine`
  - `FFTEngine([N])` creates a 1D FFT engine for grids of size N
  - Forward FFT: `real[N] → complex[N/2+1]` using r2c transform
  - Inverse FFT: `complex[N/2+1] → real[N]` using c2r transform
  - Supports all backends: PocketFFT (CPU), cuFFT (CUDA), rocFFT (ROCm)
  - Supports both AoS (CPU) and SoA (GPU) storage orders
  - Multi-component fields supported (vectors, tensors)
  - Note: 1D FFT is serial-only (no MPI parallelization)
- DOC: Updated FFT documentation to include 1D examples
- TST: Added comprehensive 1D FFT test suite
  - Engine creation and properties tests
  - Roundtrip accuracy tests
  - Comparison with NumPy's `fft.rfft`
  - Multi-component field tests

0.104.0 (10Jan26)
-----------------

- API: FFTEngine field methods now use `components` parameter instead of `nb_components`
  - `real_space_field(name, components=())` - creates scalar field by default
  - `fourier_space_field(name, components=())` - creates scalar field by default
  - `register_real_space_field(name, components=())` - creates scalar field by default
  - `register_fourier_space_field(name, components=())` - creates scalar field by
    default
  - This makes the API consistent with `FieldCollection.real_field()` and
    `CartesianDecomposition.real_field()`
- API: Scalar fields now have different shapes than unit component fields
  - Scalar field (`components=()`): `.p.shape = (nx, ny)`, `.s.shape = (1, nx, ny)`
  - Unit component (`components=(1,)`): `.p.shape = (1, nx, ny)`, `.s.shape = (1, 1, nx,
    ny)`
  - Use `components=(1,)` explicitly if you need the component dimension
- ENH: Added `Shape_t` overloads for all FFTEngine field registration methods
  - Supports arbitrary component shapes like `(3,)` for vectors or `(3, 3)` for tensors
  - Available for CPU, CUDA, and ROCm backends
- DOC: Updated `fourier_derivative.py` example for scalar field indexing

0.103.0 (07Jan26)
-----------------

- ENH: Kernel operators now use stencil-based iteration bounds
  - Operators compute results for all points where the stencil has valid input data
  - If ghost region is larger than stencil requirement, extra ghost points get computed
    results
  - Example: Laplace with 1-wide stencil and 3 ghosts per side → computes 2 extra layers
    beyond interior
- ENH: Updated all operators for stencil-based computation region
  - `GenericLinearOperator`: Dynamic stencil requirements from stencil shape
  - `LaplaceOperator2D/3D`: Requires 1 left, 1 right (centered 5/7-point stencil)
  - `FEMGradientOperator2D/3D`: apply requires 0 left, 1 right; transpose requires 1
    left, 0 right
  - `IsotropicStiffnessOperator2D/3D`: Requires 1 left, 1 right (CPU and GPU kernels)
- DOC: Updated `doc/KERNELS.md` with stencil-based iteration semantics

0.102.0 (05Jan26)
-----------------

- API: Replaced `MemoryLocation` enum with new `Device` class for device selection
  - New `Device` class with factory methods: `Device.cpu()`, `Device.cuda(id)`,
    `Device.rocm(id)`, `Device.gpu(id)`
  - New `DeviceType` enum following DLPack conventions (CPU, CUDA, CUDAHost, ROCm,
    ROCmHost)
  - Supports multi-GPU systems with device IDs (e.g., `Device.cuda(1)` for GPU 1)
- API: Renamed `memory_location` parameter to `device` in Python wrappers
  - Affects `GlobalFieldCollection`, `LocalFieldCollection`, `CartesianDecomposition`
  - Accepts strings (`"cpu"`, `"gpu"`, `"cuda"`, `"cuda:N"`, `"rocm"`, `"rocm:N"`) or
    `Device` objects
- API: Pythonic string-based parameter handling
  - `device`: `"cpu"`, `"gpu"`, `"cuda"`, `"cuda:0"`, `"rocm:1"`, etc.
  - `open_mode` (FileIONetCDF): `"read"`, `"write"`, `"overwrite"`, `"append"`
- API: Renamed `StencilGradientOperator` to `GenericLinearOperator`
  - Clearer naming that reflects the operator's purpose as a general linear convolution
    operator
  - Python bindings updated accordingly
- API: Simplified CG solver interface - removed `hessp_vecdot` parameter
  - CG solver now uses only `hessp` for the Hessian-vector product
  - Fused operations handled internally via `axpy_norm_sq` in linalg module
- API: Removed `apply_vecdot` and `transpose_vecdot` from convolution operators
  - Removed from `ConvolutionOperatorBase`, `ConvolutionOperator`, `LaplaceOperator`,
    `FEMGradientOperator`
  - Performance testing showed negligible benefit; simplifies operator interface
- API: Removed PAPI hardware counter support from Timer class
  - Timer now provides time-based measurements only
  - Removes pypapi dependency and cross-platform compatibility issues
- API: Standardized Python API to use tuples instead of lists for grid dimensions
  - All docstrings now document parameters as "tuple of int" instead of "list of int"
  - Affects `GlobalFieldCollection`, `CartesianDecomposition`, `FFTEngine` parameters
  - Properties like `nb_subdomain_grid_pts` already returned tuples; documentation now
    matches
- ENH: Added `IsotropicStiffnessOperator2D` and `IsotropicStiffnessOperator3D` for solid
  mechanics
  - Fused elliptic operators computing K @ u = B^T C B @ u for isotropic linear elastic
    materials
  - Memory efficient: stores only Lamé parameters (λ, μ) per voxel instead of full
    stiffness matrix
  - Reduces memory from O(N × 24²) for full K storage to O(N × 2) for spatially-varying
    materials
  - GPU support with optimized CUDA and HIP kernels
  - Uses linear tetrahedral FEM with 5-tetrahedra decomposition (3D) or 2-triangle
    decomposition (2D)
- ENH: Added `linalg` module with efficient linear algebra operations for muGrid fields
  - `vecdot(a, b)`: Vector dot product (interior only, excludes ghost regions)
  - `norm_sq(x)`: Squared L2 norm (interior only)
  - `axpy(alpha, x, y)`: y = alpha * x + y (full buffer)
  - `scal(alpha, x)`: x = alpha * x (full buffer)
  - `axpby(alpha, x, beta, y)`: y = alpha * x + beta * y (full buffer, fused operation)
  - `axpy_norm_sq(alpha, x, y)`: y = alpha * x + y, returns ||y||^2 (fused axpy + norm)
  - `copy(src, dst)`: dst = src (full buffer)
  - Avoids GB-scale memory copies from non-contiguous array views in CG solver
  - CPU implementation using Eigen, GPU implementation for CUDA and HIP
- ENH: Updated conjugate gradient solver to use new `linalg` module
  - Uses `axpby` for fused update_p step (2 reads + 1 write instead of 3 reads + 2
    writes)
  - Uses `axpy_norm_sq` for fused residual update (saves 1 memory read per iteration)
- ENH: Native rocFFT backend for AMD GPUs with full stride support
  - Uses `rocfft_plan_description_set_data_layout()` for arbitrary strides
  - Enables 3D MPI-parallel FFTs on AMD GPUs (not possible with cuFFT)
- ENH: Added `Device.gpu()` factory for portable GPU code
  - Automatically selects CUDA or ROCm based on compile-time configuration
  - Falls back to CPU if no GPU backend is available
  - Recommended for code that should work on any GPU platform
- ENH: Added `parprint` utility function for MPI-safe printing
  - MPI-aware print function that only outputs on rank 0
  - Works with NuMPI's MPI stub for compatibility with and without MPI
  - Available as `muGrid.parprint()` in Python API
- ENH: Enabled MPI parallel execution of Poisson and homogenization examples
  - Use `suggest_subdivisions` from NuMPI for automatic domain decomposition
  - Changed from hardcoded serial execution to dynamic MPI-aware subdivision
  - Both examples now scale efficiently across multiple MPI ranks
  - Updated all output to use `parprint` for clean parallel execution
- BUG: Fixed 3D stiffness kernel on GPUs
- BUG: Gracefully handle non-initialized MPI
- BUG: Added guard in cuFFT backend for unsupported strided R2C/C2R transforms
  - cuFFT does not support strides on real data in R2C/C2R transforms
  - 3D MPI-parallel FFTs on NVIDIA GPUs now raise clear `RuntimeError`
  - Workaround: Use CPU FFT backend or 2D grids on NVIDIA hardware
- MAINT: Restructured operators to separate 2D and 3D implementations into distinct
  source files
- MAINT: Updated benchmark scripts for performance testing
- BUILD: Fixed `nodiscard` warnings in HIP linalg implementation
- TST: Added laminate homogenization tests for validating effective material properties
- TST: MPI-parallel laminate homogenization tests
- TST: Added MPI parallel tests for Poisson and homogenization examples in CI
  - Tests run with 2 and 4 MPI ranks for both 2D and 3D cases
  - Validates domain decomposition, ghost communication, and result consistency
  - Automatically runs in GitHub Actions CI when MPI is enabled
- TST: Refactored and unified test infrastructure
- DOC: Added new "Linear Operators" documentation chapter
  - Comprehensive guide to all operator types in µGrid
  - Explains generic, gradient/divergence, and fused operators
  - Details `IsotropicStiffnessOperator` material field requirements
- DOC: Updated GPU and Python API documentation for new device selection interface
- DOC: Added GPU FFT documentation explaining backend limitations
- DOC: Simplified examples to use fused operators for better performance

0.101.2 (29Dec25)
-----------------

- ENH: Optimized GPU divergence kernels (2D and 3D) using gather pattern, eliminating
  atomic operations
- ENH: Hand-unrolled 3D gradient computation exploiting B matrix sparsity
- ENH: Shared memory optimization for 3D gradient kernel (cooperative node loading)
- BUILD: Fixed HIP/CUDA build (missing `d_NODE_OFFSET_3D` constant, `nodiscard`
  warnings)

0.101.1 (28Dec25)
-----------------

- BUG: Fixed `real_space_field` and `fourier_space_field` on `FFTEngine` to return
  existing fields if they already exist (consistent with `real_field` etc. on
  `FieldCollection`)
- API: Added `register_real_space_field` and `register_fourier_space_field` methods to
  `FFTEngine` that throw an error if a field with that name already exists

0.101.0 (28Dec25)
-----------------

- API: Added `fftfreq`, `ifftfreq`, `coords`, `icoords` properties to `FFTEngine`
  - `fftfreq`: Normalized FFT frequencies for local Fourier subdomain
  - `ifftfreq`: Integer FFT frequency indices
  - `coords`: Normalized real-space coordinates for local subdomain
  - `icoords`: Integer real-space coordinate indices
  - `coordsg`/`icoordsg`: Same as above but including ghost cells
  - `spatial_dim`: Returns the spatial dimension (2 or 3)
- API: Removed standalone FFT frequency functions from module level
  - `fft_freq`, `fft_freqind`, `rfft_freq`, `rfft_freqind` are no longer available
  - Use `engine.fftfreq` and `engine.ifftfreq` instead
- API: Properties now return Python tuples instead of C++ objects
  - `nb_fourier_grid_pts`, `nb_fourier_subdomain_grid_pts`,
    `fourier_subdomain_locations`
  - `nb_subdomain_grid_pts`, `subdomain_locations`, `nb_subdivisions`,
    `nb_domain_grid_pts`
- TEST: Comprehensive test suite for FFT frequency and coordinate properties

0.100.0 (28Dec25)
-----------------

- ENH: Added `offset`, `shape`, and `coefficients` properties to all discrete
  convolution operators in Python
  - `ConvolutionOperator`: Access generic stencil metadata and coefficients
  - `LaplaceOperator`: Access hardcoded Laplacian stencil structure
  - `FEMGradientOperator`: Access shape function gradients and node arrangement
- TEST: Comprehensive test suite for stencil property access across all operator types

0.99.0 (28Dec25)
----------------

- ENH: Added `fourier()` method to `ConvolutionOperator` for computing Fourier space
  representations
- ENH: Vectorized Python bindings for `ConvolutionOperator.fourier()` supporting batch
  computation
- TEST: Comprehensive C++ and Python test suites for Fourier method validation

0.98.1 (28Dec25)
----------------

- BUG: Fixed dynamic version detection in publish workflow (was building as 0.0.0)
- MAINT: Configured setuptools_scm for automatic version discovery from git tags
- MAINT: Updated PyPI publish action from deprecated @master to @release/v1
- MAINT: Consolidated setup.cfg and pytest.ini into pyproject.toml
- MAINT: Removed legacy discover_version.py, .gitattributes, and unused requirements.txt

0.98.0 (27Dec25)
----------------

- ENH: **Simplified API**: Streamlined user-facing API for improved usability
- ENH: **Windows support**: Full Windows platform compatibility with exception traceback
  support (closes #48)
- ENH: **3D Poisson solver example**: New example demonstrating 3D Poisson solver usage
- ENH: **Benchmark suite**: Automatic Poisson benchmark suite with fine-grained timing
  and GFLOP/s metrics
- ENH: **FEM gradient operator**: New FEM gradient operator with homogenization example
- ENH: **Hierarchical Timer**: Timer class with hierarchical timing and context manager
  support
- ENH: **Multi-component fields**: Added multi-component field support in
  FEMGradientOperator
- ENH: **reduce_ghosts**: Added reduce_ghosts operation to CartesianDecomposition
- API: Removed standalone FFT field creation functions
- MAINT: **Unified GPU code**: Consolidated GPU backend code for CUDA and HIP
- MAINT: **Laplace kernels**: Hard-coded Laplace kernels are now part of the main
  library
- MAINT: Removed unused pad_size field functionality
- CI: Added GPU testing workflow with Tesla T4 runner
- DOC: Improved API documentation with docstrings and structured references
- BUG: Fixed obtaining raw data pointer in Laplace operator implementation
- BUG: Fixed FEMGradientOperator Python wrapper API mismatch

0.97.0 (22Dec25)
----------------

- ENH: GPU support
- ENH: New parallel FFT with arbitrary ghost buffers
- ENH: Sparse stencils
- BUG: NetCDF output of fields with ghost buffers
- MAINT: Larger code reorganization
- **muFFT** is now deprecated

0.96.0 (15Dec25)
----------------

- ENH: Looping over strides fields
- ENH: Ghost buffers larger than subdomains
- BUG: Memory leak in `communicate_ghosts`

0.95.0 (15Jul25)
----------------

- ENH: Accessor properties for field access without (`s`, `p`) and with ghosts (`sg`,
  `pg`)
- ENH: Parallel conjugate gradient solver
- API: Flipped axes order of convolution operator
- MAINT: Removed support for Python 3.8

0.94.0 (18Feb25)
----------------

- ENH: General convolution operator for fields
- ENH: Domain decomposition with ghost buffer communication

0.93.3 (11Nov24)
----------------

- BUG: Don't divide by smallest stride if it is zero

0.93.2 (22Oct24)
----------------
 
- BUG: Fixed strides in `detect_storage_order` for arrays with single components (but
  non-empty shapes)
- MAINT: Idiot-check strides when constructing a wrapped field

0.93.1 (20Oct2024)
------------------

- ENH: `NumpyProxy` now determine the iter type that is required for returning a numpy
  array with exactly the same shape as the input array

0.93.0 (20Oct2024)
------------------

- API: Always return full component shape, do not cut components with one degree of
  freedom
- API: Scalar fields are now explicitly supporting by passing an empty tuple as the
  component shape

0.92.6 (25Sept2024)
-------------------

- CI: macOS x86_64 wheels

0.92.5 (11Jul2024)
------------------

- BUG: Handle installation without NetCDF

0.92.4 (30June2024)
-------------------

- MAINT: Added utility function for copying (rather than wrapping) field into a numpy
  ndarray

0.92.3 (15June2024)
-------------------

- BUILD: Don't override dependencies after dl and execinfo are detected

0.92.2 (14June2024)
-------------------

- BUILD: (Re)added dl and execinfo as requirements

0.92.1 (14June2024)
-------------------

- BUILD: Require at least eigen3 3.4.0

0.92.0 (04June2024)
-------------------

- ENH: Added wrapper that allows passing an `mpi4py` communicator to `FileIONetCDF`
- BUG: Don't import OpenMode if NetCDF is not available

0.91.1 (02June2024)
-------------------

- BUG: Updated `NumpyProxy` to reflect that a global field collection no longer required
  number of spatial dimensions as first argument

0.91.0 (01June2024)
-------------------

- ENH: Added attributes `p` and `s` for convenience access to pixel-shaped and
  sub-point-shaped numpy arrays
- ENH: Convenience filed accessor function `real_field`, `int_field`, etc. that create
  fields if they don't exist but return them if they do
- ENH: Added `OpenMode::Overwrite` which overwrites an existing file
- MAINT: Default communicator is now MPI_COMM_SELF if MPI is enabled
- DOC: Documentation of Python bindings
- DOC: Python examples

0.90.1 (23May2024)
------------------

- Fixed wheels and source deployment to PyPI

0.90.0 (21May2024)
------------------

- Split code into separate repositories: muGrid and muFFT

0.27.0 (30Jan2024)
------------------

- muSpectre: Sensitivity analysis for 3D problems
- Fixing meson-python to >= 0.15.0
- Updated Eigen3 to v3.4 and pybind11 to v2.11

0.26.4 (04Oct2023)
------------------

- Fixing meson-python to 0.13.2 because of a bug in 0.14.0

0.26.3 (16Jul2023)
------------------

- Same as 0.26.2 (debugging CI)

0.26.2 (10Jul2023)
------------------

- Same as 0.26.1, fixed deployment procedure

0.26.1 (08Jul2023)
------------------

- macOS wheels

0.26.0 (31Mar2023)
------------------

- MPI parallelization of sensitivity analysis
- Wheels for Python 3.11

0.25.2 (14Jan2023)
------------------

- Fixed macOS build

0.25.1 (28Dec2022)
------------------

- Same as 0.25.0, changed CI configuration for automatic deployment to PyPI

0.25.0 (28Dec2022)
------------------

- muSpectre is now distributed with Linux wheels (basic configuration only)
- muFFT: Added PocketFFT engine (that does not require external dependencies)
- Added Meson build files (Python package now exclusively build using Meson)
- Defaulted again to autodetecting MPI

0.24.0 (22Nov2022)
------------------

- muFFT: Changed Python install procedure; default to no MPI and MPI now needs to be
  explicitly enabled

0.23.1 (24Mar2022)
------------------

- muSpectre: A bug fixed in the call of the constructor of FieldCollection
- muFFT: Fixed `pip install muFFT` on macOS

0.23.0 (15Oct2021)
------------------

- muSpectre: making vector operation methods in solver classed with communicating inside
  them
- muSpectre: added logical reduction on was_last_step_nonlinear evaluation in CellData

0.22.0 (24Sep2021)
------------------

- muSpectre: mean stress control bugs resolved
- muSpectre: mean stress control is now usable in MPI
- muSpectre: examples using mean stress control added

0.21.0 (26Aug2021)
------------------

- CI: Refactoring of the CI with addition of coverage and ccache

0.20.2 (04Aug2021)
------------------

- muSpectre: deleted an unnecssary parameter from write_2d_class function in
  iinear_finite_elements.py

0.20.1 (03Aug2021)
------------------

- muSpectre: improve write_2d and write_3d functions, 2D stencil for hexagonal grid

0.20.0 (15Jul2021)
------------------

- muSpectre: capability to apply mean stress (instead of mean strain added)

0.19.2 (28Jun2021)
------------------

- muGrid: correct bugs in the FileIONetCDF

0.19.1 (23Jun2021)
------------------

- muSpectre: Fixed a minor array reshape bug in the tutorial_example_new.py that was
  jeopardizing the output stress plot

0.19.0 (16Jun2021)
------------------

- muSpectre: added material_dunnat_tc (bilinear elastic- linear strain softening with
  tensile-compressive wiegthed norm as strain measure)
- muSpectre: added material_dunnat_t (bilinear elastic- linear strain softening with
  maiximum tensile principal strain as strain measure)

0.18.2 (10Jun2021)
------------------

- muSpectre: Small bugfix and addition of regularization for slightly non-pd Hessians in
  phase field fracture material
- muGrid: Added functions for reporting version
- muGrid: Added global attributes to FileIONetCDF

0.18.1 (02Jun2021)
------------------

- muSpectre: Fix, changed the reset criterion for gradient orthogonality in FEM trust
  region precondtioned Krylov solvers
- muSpectre: Fix, added calling clear_was_last_step_nonlinear in
  fem_newton_trust_region_pc solver
- muSpectre: Fixed get_complemented_positions
- muFFT: Fixed large transforms
- muFFT: Fixed segfault when input buffer had wrong shape

0.18.0 (10May2021)
------------------

- muSpectre: Added trust fem region solver + ability to handle precondtioner
- muSpectre: re-organized the krylov solver hierarchy to circumvent diamond inheritance
  by introducing KrylovSolverXXXTraits classes

0.17.0 (24Apr2021)
------------------

- muSpectre: Added trust region solver class and a simplistic damage material
- muSpectre: physics have their specific name that might be used later for outputs
- muSpectre: trsut region krylov solver has different resetart strategies available
- muSpectre: gradient integration for solver class + cell data is now available

0.16.0 (31Mar2021)
------------------

- µSpectre: The ProjectionGradient works for vectorial and rank-two-tensor gradient
  fields and replaces ProjectionFiniteStrainFast
- µSpectre: The SolverNewtonCG can now handle scalar problems (e.g., diffusion equation,
  heat equation, etc.)
- clang: No more warnings are emitted during compilation

0.15.1 (30Mar2021)
------------------

- muSpectre: Added material for phase field fracture simulations
- muGrid: Fix support for fields with >= 2^32 elements

0.15.0 (12Mar2021)
------------------

- muSpectre: All projection operators now have a gradient argument and can work with
  discrete derivatives
- muSpectre: Projection classes now have an `integrate` method that reconstructs the
  node positions
- muSpectre: Added `linear_finite_elements` to stencil database
- muSpectre: Enabled even number of grid points for discrete stencils
- muFFT: PFFT engine works with pencil decomposition
- all: Fixed installation via CMake and `make install`
- all: Fixed cross platform install of NetCDF I/O

0.14.0 (04Feb2021)
------------------

- muFFT: implement serial wrapper to FFTW hcfft

0.13.0 (28Jan2021)
------------------

- muSpectre: CellData and Solver classes for multiphysics calculations
- muSpectre: Sensitivity analysis
- Bug fix (muFFT): Handle cases where MPI processes have no grid points

0.12.0 (19Nov2020)
------------------

- muGrid: Parallel I/O via NetCDF
- muFFT: Derivatives in 1D
- muFFT: Second derivatives

0.11.0 (07Sep20)
----------------

- Trust-Region Newton-CG solver for nonlinear problems with instabilities
- Updated Eigen archive URL which broke installation via pip

0.10.0 (22Jul2020)
------------------

- Support for more flexible strideste in fields: full column-major, row-major and
  strided pixels portion (#103)
- User control over buffer copies in muFFT with option to avoid them completely
- Gradient integration for multiple quadrature points

0.9.3 (28Jun2020)
-----------------

- Bug fix: Packaging with sdist did not remove dirty flag which lead to a broken PyPI
  package

0.9.2 (28Jun2020)
-----------------

- Bug fix: operator= of TypedFieldBase passed wrong strides during strided copy; this
  broke the MPI-parallel FFTW forward transform (#130)

0.9.1 (17Jun2020)
-----------------

- Bug fix: Packaging with sdist only included FFT engines present during the packaging
  process

0.9.0 (17Jun2020)
-----------------

- Initial release of µSpectre
    * FFT based micromechanical homogeneization
    * Arbitrary constitutive laws in small and finite strain
    * Krylov and Newton solver suite
    * MPI parallelization
- Initial release of µFFT
    * Generic wrapper for MPI-parallel FFT libraries
- Initial release of µGrid
    * Generic library for managing regular grids

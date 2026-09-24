Change log for µGrid
====================

unreleased
----------

- ENH: The reference-material (Green) preconditioner can now *evaluate* its
  symbol per Fourier mode instead of assembling and storing it. A uniform
  operator is a `3^dim` stencil — 243 numbers in 3D — so
  `K(q) = Σ_d S[d] exp(-2πi q·d)` is closed form, and the new
  `AnalyticReferencePreconditioner` keeps the stencil rather than `n²` complex
  values per mode (2.3 GB at 512³ for three components in single precision, and
  it grows with the grid). `linalg/green_symbol.{hh,cc}` is the host kernel and
  `green_symbol_gpu.cc` the CUDA/HIP one, bound as
  `linalg.apply_green_symbol_{2,3}d[_f32]` and `..._gpu_*`
- ENH: `make_reference_stiffness_preconditioner` and
  `make_green_jacobi_preconditioner` take `evaluate_symbol`, which defaults to
  evaluating on a device and storing on the host. On an H200 at 512³ with three
  components, evaluating is 2.7x (float32) and 1.9x (float64) faster per apply
  than the stored symbol, holds 4.9 GB less device memory, and sets up in 1.3 s
  against 168 s. On the host the apply is ~28% slower, so the stored symbol
  stays the default there; pass `evaluate_symbol=True`/`False` to override.
  The measured device win is larger than an arithmetic-intensity estimate
  predicts, because the stored path's per-mode multiply runs component by
  component through CuPy and is far from its own bandwidth floor
- ENH: `reference_stencil` is public and is the single definition of the uniform
  reference operator: it *probes* the real C++ operator on an 8^dim serial grid
  rather than re-deriving `B(q)` from the element tables, so it matches the
  discretisation by construction. `stencil_symbol` generalises the hybrid
  preconditioner's `_z_coupling_blocks` — which already assembled its blocks
  this way — to either phase every axis or leave one explicit, so both
  preconditioners now derive from the same stencil
- ENH: The assembled route uses the stencil too, when the caller names the
  operator (`element` + `grid_spacing` + Lamé), which removes the impulse
  response, the `n` full-grid FFTs and the three engine-sized fields it needed.
  An opaque callable still takes the impulse route, which is the only one that
  works for an operator muGrid cannot name. The two agree to assembly
  round-off: 1.0e-15 in 3D at float64, ~1e-06 at float32

v1.4.0 (23Sep26)
----------------

- FIX: `vecdot`, `norm_sq`, `axpy_norm_sq` and `pipelined_cg_dots` now return
  `linalg::reduction_result_t<T>` — double precision for a `Real32`/`Complex32`
  field — instead of narrowing the double accumulator back to the field's scalar
  type on return. The narrowing turned any single-precision reduction whose true
  value exceeded ~3.4e38 into `inf`, and that bound is reached by ordinary large
  solves: it tightens as `1/sqrt(N)` with the summation and again as `h²` with the
  operator norm, so it is some 500x tighter at 512³ than at 64³. The failure was
  silent rather than loud — an infinite `pAp` makes CG's `alpha = rz/pAp` exactly
  zero, so the iterate stops moving and the residual repeats bit-for-bit until
  maxiter, while the pipelined variant divided by the resulting zero `alpha_prev`
  and raised `ZeroDivisionError` from inside the recurrence. On an `A = 1e25·I`
  system (condition number 1) a float32 solve stalled for 25 iterations where the
  float64 one converged in 1; it now also converges in 1. Python sees no API
  change: both precisions already crossed the binding boundary as a plain `float`
- FIX: Both CG solvers now reject a non-finite residual, curvature term or
  inner product with a `ConvergenceError` naming the cause, instead of checking
  only for NaN and letting infinities through. An infinite curvature term made
  the step length exactly zero, so the solve stalled on an unmoving iterate and
  reported a generic "did not converge"; in the pipelined variant that zero
  became the next iteration's `alpha_prev` and a bare `ZeroDivisionError` escaped
  from inside the recurrence, past the solver's own `ConvergenceError` contract.
  A zero curvature term (a search direction in the operator's null space) and a
  zero step length on an unconverged residual are reported too. NaN and overflow
  get different messages, since one means an indefinite operator and the other
  means the iterate has outgrown the work fields' dynamic range
- FIX: A single-precision solve given an `rtol` below the float32 accuracy floor
  (~1e-6) now warns that the tolerance is unreachable, rather than silently
  running to maxiter. float32 eps is 1.19e-7, so the true residual `b - Ax`
  stagnates around there even while the recursively updated CG residual keeps
  shrinking. Double-precision solves are unaffected
- TST: `tests/python_solvers_test.py` is renamed to `python_solvers_tests.py`.
  pytest collects `python_*_tests.py`, so the file -- and with it the only test
  of `conjugate_gradients_pipelined` in the repo -- had never run
- FIX: The FFT transpose's all-to-all counts and displacements are narrowed to
  MPI's `int` through `checked_mpi_int` instead of a bare `static_cast`, so an
  oversized transform throws rather than wrapping to a negative count and
  corrupting the transform. This is the default exchange path on the GPU, not
  only the env-gated host one. `checked_mpi_int` moved out of
  `cartesian_communicator.cc`'s anonymous namespace into the new
  `mpi/mpi_counts.hh` so both callers narrow the same way
- FIX: GPU kernels compute their global thread index through new
  `global_thread_{x,y,z}()` / `grid_stride_x()` helpers in `memory/gpu_runtime.hh`
  rather than open-coding `blockIdx.x * blockDim.x + threadIdx.x`. All three
  operands are `unsigned int`, so the product was evaluated in 32 bits and
  wrapped above 2^32 threads *before* any widening — assigning the result to an
  `Index_t` did not help. Element counts are `nb_pixels * nb_components *
  nb_sub_pts` and grow with the cube of the resolution, so the bound is reachable.
  48 sites across linalg, laplace, convolution, FEM-gradient and solid-stiffness
  kernels; `ghost_accumulate_gpu.cc` already widened by hand and is unchanged
- FIX: The reduction kernels widen each operand *before* multiplying instead of
  forming the product in the field's precision and widening afterwards. Squaring
  in `float32` discards half the mantissa and overflows at `|x| ~ 1.8e19`, far
  below what the accumulator holds; this affected the scalar (non-vectorised)
  interior paths on the host — so `pipelined_cg_dots`, which has no vectorised
  path, lost ~7e-9 relative accuracy where `vecdot`/`norm_sq` did not — and the
  `dot`/`interior_dot`/`axpy_norm_sq` kernels on the device. The fp32 `sq_norm`
  overload is gone so the mistake cannot be made again
- PERF: `make_reference_stiffness_preconditioner` assembles the symbol at the
  solve precision and inverts it one slab of Fourier modes at a time, in place.
  It previously held `K_hat` in `complex128` regardless of the requested dtype,
  then built four more arrays that size in a row: `np.zeros_like`, the
  `blocks[nonzero]` boolean gather, `np.linalg.inv`'s output, and the
  normalisation multiply. The symbol is n² values per Fourier point — 9.7 GB at
  512³ in `complex128` — so that chain was tens of GB for one preconditioner.
  The `complex128` was also half illusory in a single-precision solve: the data
  reaches it from `column_hat`, a `complex_dtype` field, so it has already been
  through a float32 FFT. Only the per-mode n×n *inversion* gains from double,
  and that still happens in double. The singular q=0 block is made the identity
  before inversion and zeroed after, instead of mask-indexing around it.
  Measured peak RSS at 96³ with 3 components: 508 → 376 MB single, 575 → 444 MB
  double. `SYMBOL_INVERSION_SLAB_BYTES` bounds the working set and is
  module-level so tests can shrink it; the slab boundaries provably do not
  change the result (`test_reference_stiffness_symbol_is_slab_invariant`)
- PERF: `BlockFourierPreconditioner`'s Hermitian detection tests one component
  pair at a time. `np.abs(blocks)` and `blocks - conj(swapaxes(blocks, 0, 1))`
  each allocated an array the size of the whole symbol, the second complex
- ENH: New `GridTransfer{2,3}D`: multilinear prolongation `P` between two
  nested nodal grids differing by a factor of two in every direction, and its
  exact adjoint `R = Pᵀ`, the tensor product of the `[1/2, 1, 1/2]` stencil.
  These are the grid-transfer half of a geometric multigrid hierarchy. Both act
  component-wise and never mix the components of a vector field, which is what
  makes them right for elasticity: multilinear interpolation reproduces linear
  displacement fields exactly, so `range(P)` contains every rigid-body mode and
  every constant strain — the near-nullspace property an algebraic multigrid
  must be told about explicitly. `R = Pᵀ` rather than the `1/2^dim`-normalised
  full weighting, because the restricted residual is a force (a functional) and
  carries no measure factor; that pairing is also what keeps a V-cycle built
  from these operators symmetric, as plain CG requires of a preconditioner
- ENH: The transfers require the two decompositions to be *nested* — the fine
  subdomain twice the coarse one in extent and starting at twice its global
  location — so no transfer crosses a rank boundary and both directions are
  pure local work plus the caller's existing halo exchange. A violation is
  rejected with an error naming the offending axis and both extents rather than
  silently corrupting the subdomain seams, which is the failure mode that would
  otherwise only show up as a slightly wrong convergence rate under MPI
- ENH: New `MultigridReferencePreconditioner`: a V-cycle approximation of the
  reference-stiffness inverse `Kʳᵉᶠ⁻¹`, with the existing FFT block symbol
  solving the coarsest level exactly. A drop-in for
  `make_reference_stiffness_preconditioner`, and usable as the `green` argument
  of `GreenJacobiPreconditioner` exactly as the FFT version is. The motivation
  is parallel scaling: an FFT apply costs four all-to-all transposes, each a
  full barrier across all ranks, and forces the solver onto the FFT engine's
  pencil decomposition in which the x axis is never distributed; a V-cycle needs
  only nearest-neighbour halo exchange per level. The cycle runs on the
  *uniform* reference operator, so no material field is restricted and every
  level is a rediscretisation rather than a Galerkin product — heterogeneity
  stays where it already is, in the `J^{1/2} · G · J^{1/2}` scaling. Serial for
  now; the MPI path raises rather than returning a wrong answer
- ENH: The Jacobi damping is derived, not configured: `ω = 1.7 / λ_max(D⁻¹K)`
  with `λ_max` measured by power iteration at setup. The stability limit is
  `2/λ_max` and `λ_max` moves with dimension and element kind, so a hardcoded
  value that is optimal in 2D (0.7) diverges outright in 3D. `D⁻¹K` is
  invariant under uniform refinement, so one estimate on the coarsest level
  serves the whole hierarchy. The nodal block is likewise probed rather than
  assumed: it is `c·I` for Q1 in either dimension and for P1 in 3D, so the
  smoother is a single scaled `axpy`, while 2D P1 — whose two-triangle Kuhn
  split leaves `K01 = λ + μ` — is refused with an explanatory error instead of
  being smoothed with the wrong diagonal
- BUILD: Clang's `--gcc-install-dir` is now derived from the directory holding
  `crtbegin.o` rather than the one holding `libstdc++.so`. The two coincide on
  many installations but not all: where `libstdc++.so` sits in `<prefix>/lib64`
  — as it does on an EasyBuild GCCcore toolchain — the hint named a directory
  that is not a GCC installation at all, and configuring a HIP build died with
  `does not contain a GCC installation` before compiling a line. The reported
  path is also normalised, since GCC returns it relative to its own driver
  location with `..` components left in
- BUG: `MultigridReferencePreconditioner` now builds its coarse levels on the
  same device as the fine one. They were created without a device argument and
  so always landed on the host, which under a device fine grid is a mismatch
  rather than a slow path: the cycle moves fields straight between levels. The
  V-cycle still cannot run on a device end to end, because `GridTransfer` has
  host-space overloads only, but every other part of it now can — which is
  what lets the cycle be priced on a GPU a piece at a time
- BUG: Two failures that used to surface far from their cause now say what is
  wrong. A grid too coarse to halve even once produced an `AttributeError`
  about a missing `real_space_collection`, and is now rejected on the grounds
  that a V-cycle needs at least two levels. Applying the cycle to device fields
  produced a pybind overload mismatch from three frames down, and now names the
  missing device grid-transfer kernels
- ENH: `examples/homogenization.py` gained `-P multigrid`, which applies the
  same reference operator as `-P reference` by a V-cycle instead of a fine-grid
  FFT, and `--sync-timers`, which brackets every timed region with a device
  synchronisation. The latter is needed for any GPU cost attribution from this
  example: kernel launches are asynchronous, so an unsynchronised host-side
  timer around a region that only launches work measures the launch, and the
  work is charged to whichever region is open at the next implicit
  synchronisation -- usually a CG dot product pulling a scalar back. Totals are
  unaffected; the breakdown is not, and the symptom is a sub-timer that stops
  growing with the grid or shrinks
- ENH: New `examples/vcycle_vs_fft.py`, which measures the one number that
  decides whether the V-cycle is worth having: R, the cost of a V-cycle apply
  over the cost of an FFT apply, on a single device. The two scale in opposite
  directions -- a cycle is halo-only, an FFT's all-to-all is not -- so R at one
  rank predicts the crossover instead of waiting to observe it. R > 1 is
  expected and is not a failure. It is reported three ways: modelled from the
  matvec alone, priced from the cycle's parts as timed on a GPU, and measured
  end to end wherever the cycle runs. The script also carries the iteration
  penalty, since a cheaper apply that needs more CG iterations is not cheaper
- ENH: `examples/homogenization.py --decomposition` unties the domain split
  from the preconditioner. `muGrid.FFTEngine` does not merely add transforms to
  a `CartesianDecomposition`, it also dictates how the domain is divided, so a
  measurement that swaps the preconditioner alone moves both at once and cannot
  say which it moved. The JSON output now also records the split that was
  actually used -- rank count, subdivisions and subdomain extents -- since the
  FFT engine picks its own and ignores `suggest_subdivisions`
- ENH: New `examples/decomposition_baseline.py`, which measures what the FFT
  engine's domain split costs on its own, with the preconditioner held at
  `-P none` so that no transform runs in either arm. The split turns out to be
  a *slab*, `[1, 1, P]` -- only the last axis is ever distributed -- which caps
  a run at `P <= N` ranks and grows its halo twice as fast as a 3D split. On a
  single shared-memory node it nonetheless costs nothing measurable, and its
  halo exchange is the faster of the two, because it has 2 MPI neighbours where
  a 3D split has 6 and per-message latency beats volume there
- ENH: `examples/homogenization.py --mg-nu` and `--mg-cycles` expose the
  V-cycle's smoothing and cycle counts, so the cost/convergence trade can be
  measured rather than guessed. Measured, `nu = 1` beats the default of 2 at
  every grid tried: smoothing costs `2*nu+1` per level while iterations fall
  far more slowly, so the whole-solve penalty against the FFT preconditioner
  drops from 4.5x to 3.8x at 256 cubed. Raising `nb_cycles` instead is strictly
  worse -- `nu=1, cycles=2` buys exactly the iteration count of `nu=2` for about
  27% more work
- TST: `test_preconditioner_is_symmetric` and
  `test_vcycle_converges_on_the_reference_operator` now sweep `nu` over
  {1, 2, 3} in both dimensions. Symmetry is what allows plain CG to be used at
  all, and a guarantee that held only at whichever `nu` happened to be the
  default would be worth little now that `nu` is a tuning parameter
- ENH: New `HybridFourierTridiagonalPreconditioner`, reachable as
  `examples/homogenization.py -P hybrid`. It applies the same reference
  operator as `-P reference` but transforms only the rank-local axes, solving
  block-tridiagonally along the distributed one -- so it never performs the
  all-to-all a full FFT forces, and unlike the V-cycle it is *exact*. Measured
  at 1, 2, 4 and 8 ranks it reproduces `-P reference`'s CG count to the
  iteration, where the V-cycle costs about 1.5x of it. Two properties make it
  work, and neither is separability: the reference operator is uniform, so
  transforming the local axes decouples every mode exactly, and the stencil
  reaches one node along the distributed axis for Q1 and P1 alike. The
  distributed solve is the Spike/partitioned-Thomas scheme, whose small reduced
  interface system also absorbs the periodic wrap-around. Slab decomposition
  only
- ENH: `HybridFourierTridiagonalPreconditioner` runs on the GPU. Host and device
  share one implementation -- the array module follows the decomposition -- so
  the two cannot drift apart, and a test asserts the device reproduces the host
  *result* rather than merely a small residual. Validated on 2x MI300A at 1, 2
  and 4 ranks: the solve is exact to 8e-16 and its checksum matches the host's
  to twelve decimals. The interface exchange stages device buffers through the
  host, so this does not require a GPU-aware MPI build
- PERF: The hybrid's tridiagonal sweep is a fused HIP/CUDA kernel -- one thread
  per Fourier mode, marching the distributed axis with the coupling blocks in
  registers -- replacing a Python loop that issued four kernels per plane. It is
  52x faster at 256 cubed and 119x at 64 cubed, reaching 1755 GB/s, and agrees
  with the loop to round-off. The internal layout is now z-major so a wavefront
  reads contiguous bytes, which cost nothing to adopt because a permutation was
  already being materialised. Two consequences of making the sweep fast: the
  real-space mean projection is gone, since projecting off the rigid
  translations is just zeroing the all-zero mode's z-mean, and the spike arrays
  are no longer stored at all, since correcting the right-hand side at its two
  end planes and solving again is the same thing for one extra sweep instead of
  a pass over 2.4 GB
- PERF: The hybrid's Thomas factors are stored compressed. Their recurrence has
  constant coefficients and is therefore a fixed-point iteration whose
  convergence distribution turns out to be independent of the grid -- median 10
  steps, 90th percentile 16 at every size measured -- so the first 32 factors
  are kept densely, everything beyond reuses the last, and the ~2% of modes that
  have not converged by then keep a full line. Exact rather than approximate: a
  mode is exceptional when reusing the last factor would be wrong anywhere along
  the remaining axis. 6.9x less storage at 256 cubed, 1217 MB down to 177 MB,
  and the sweep gains 21% because it streams those factors
- BUG: The hybrid preconditioner deflates the nullspace of its singular `q = 0`
  z-line instead of pseudo-inverting it. The kernel there is known exactly --
  the rigid translation, one per component, constant along the distributed axis
  -- so shifting it out of the way and inverting is both exact and better
  conditioned. `pinv` had to separate it by magnitude instead, and the gap it
  was given is not one it can resolve: the computed zeros sit within a digit of
  its default `1e-15 * sigma_max` cutoff, so which side they land on is a
  property of the LAPACK build. numpy 2.5 lands one of them on the wrong side
  at 32 planes, which keeps a kernel direction scaled by `1e13` and costs the
  2D preconditioner four digits of exactness -- enough to break
  `test_hybrid_is_the_exact_reference_inverse`, and, unnoticed, to slow every
  solve that used it
- BUG: `_batched_inverse` is told which mode may be singular instead of
  discovering it. It used to mark a block for pseudo-inversion when
  `|A A⁻¹ - I|` exceeded an absolute `1e-8` — a threshold below the rounding
  noise of a *healthy* complex64 block, so at `dtype=np.float32` it fired on
  hundreds of sound modes (174 in 2D at 32, 1747 in 3D at 16) and replaced each
  one, through a Python loop over modes, with a pseudo-inverse computed at that
  same precision. The one mode that is genuinely singular is the all-zero one,
  it is singular by construction, and `apply` discards its solve wholesale, so
  it is now named by its caller and zeroed — the same thing
  `make_reference_stiffness_preconditioner` does at `q = 0`. The tolerance that
  remains is a verification, not a switch, and scales with the working
  precision; a mode that is singular without being declared so now raises
  rather than riding along pseudo-inverted
- BUG: `_compress_factors` compares its deviation against a scaled tolerance
  instead of dividing by a guard floor of `1e-300`, which is not a small number
  in single precision but zero
- TST: `test_hybrid_is_exact_in_single_precision` runs the hybrid end to end at
  `dtype=np.float32`, which nothing else covered — every tolerance on that path
  was unmeasured, and two of them were wrong. It is also the sharper form of
  `test_hybrid_is_the_exact_reference_inverse`: in double the computed zeros of
  the singular mode straddle `pinv`'s cutoff, so a retained nullspace showed up
  in 2D but not 3D and only on some numpy versions, while at complex64 they sit
  ~1e-7 of the largest singular value and are always on the wrong side of it.
  The same defect therefore registers at every size, in both dimensions, on any
  numpy — 1.4e-1 in 2D and 1.1e-2 in 3D against a floor of ~1e-6


v1.3.0 (16Sep26)
----------------

- ENH: New `NodalMomentOperator{2,3}D`: the cell moments `∫_e rho^k dx`
  (k = 2, 3, 4) of a nodal scalar field's FE interpolant and their nodal
  gradients, in one fused host/GPU pass. A polynomial phase-field energy is a
  fixed combination of these — the double well `rho^2 (1-rho)^2` has cell
  integral `M2 - 2 M3 + M4` — so the energy stays with the caller and µGrid
  stays material-model-agnostic, as `compute_sensitivity` already is. The
  array-at-a-time form of this computation materialises the interpolant at
  every quadrature point of every cell at once (a 27-fold copy of the grid in
  3D, plus a temporary per polynomial term); here every quadrature point is
  consumed in registers and the pass is O(1) in scratch memory
- ENH: `fem_element.hh` gains shape-function *values* and a moment quadrature
  per element, alongside the existing gradient tables: 3-point-per-axis tensor
  Gauss for Q1, and the same rule placed on each sub-simplex for P1. The
  simplex rules are Gauss-Jacobi, not Gauss-Legendre: the collapsed map's
  Jacobian is folded into the weight function, which keeps 3 points per axis
  exact for the quartic integrand (leaving it in the integrand needs 4, and a
  3-point Gauss-Legendre rule is exact for M2 and M3 but wrong for M4) and
  makes every weight positive, so a cell's double-well energy is never
  negative
- ENH: The P1 sub-simplex decompositions are now data in `fem_element.hh`
  (`Nodes`/`Frac`) rather than only comments, so downstream code need not keep
  its own copy
- TST: `fem_element.hh` now encodes each simplex decomposition twice — as the
  gradient tables (`B`/`Wfrac`) and as the moment tables (`Nodes`/`Frac`) — so
  `static_assert`s tie the two together: the volume fractions must equal the
  gradient rule's weights, and a node the moment tables omit from a
  sub-simplex must have a vanishing gradient at that quadrature point. All
  four moment rules are also asserted to be partitions of the cell

v1.2.0 (14Sep26)
----------------

- ENH: Device-to-device copies use a kernel instead of `Memcpy`/`Memcpy2D`,
  which the HIP runtime services on the host CPU for managed allocations
  (30x faster there, no slower on plain device memory); the serial ghost
  exchange and the pencil transpose gain the most (30-38% per CG iteration)
- ENH: N-D transforms only block the host when a decomposed engine hands
  their output to GPU-aware MPI; a serial engine keeps the stream ordering
  it already had (4-13% per CG iteration)
- BUG: Device builds disable pybind11's LTO extras, which made the device
  compiler silently drop every binding at link time (`import muGrid` then
  failed with a missing module export function)
- BUG: Throw on first GPU use when muGrid is linked against a CUDA/HIP runtime
  older than the headers it was compiled with. The mismatch does not fail at
  link time and reads device properties as garbage, which surfaced as a
  segfault inside MPI
- BUG: `sendrecv_staged` verifies that a device pointer really has a host
  mapping before handing it to an MPI that does not report GPU support,
  instead of faulting inside MPI's memcpy

v1.1.1 (12Sep26)
----------------

- ENH: The fused stiffness kernels keep their G/V geometry matrices in
  `__constant__` memory at the kernel's working precision; a float32 apply no
  longer converts a `double` per inner-loop term on the fp64 pipe (~15x faster
  in 3D single precision, bit-identical results)
- ENH: The 3D stiffness kernel bounds its registers so two blocks fit per SM,
  lifting it from latency-bound at 17% occupancy to 93% of DRAM throughput
- ENH: `BlockFourierPreconditioner` applies its per-mode block product in one
  fused GPU kernel instead of a kernel per term (10-22% per CG iteration)
- ENH: GPU interior reductions (`vecdot`, `norm_sq`, `axpy_norm_sq`) cap their
  grid at 1024 blocks and use the grid-stride loop the kernels already had, so
  the shared-memory tree is amortised over many elements
- ENH: The 3D stiffness kernel stages its displacement tile, halo included,
  in shared memory instead of re-reading each node from L2 for all 27
  stencils that touch it (global load traffic down 29x; 8-9% per CG
  iteration at 96-128^3, growing with grid size)
- MAINT: `GreenJacobiPreconditioner` scales into its output field and
  applies the inner Green preconditioner in place, dropping a resident
  vector-sized work buffer per solve
- MAINT: `mpi4py` is imported only by MPI-enabled builds; a serial build no
  longer runs `MPI_Init` at `import muGrid`

v1.1.0 (04Sep26)
----------------

- ENH: `reduce_ghosts` supports halos wider than a rank's interior (and ranks
  without grid points) by relaying through intermediate ranks; it is now the
  exact adjoint of `communicate_ghosts` for any decomposition
- ENH: The FFT engine's field helpers (`real_space_field`,
  `fourier_space_field` and their `register_*` variants) accept `sub_pt`
- BUG: FFT of fields with multiple sub-points returned garbage (issue #192);
  `fft`/`ifft` now reject pairs with differing component or sub-point counts
- BUG: `communicate_ghosts`/`reduce_ghosts` reject fields from a foreign field
  collection, e.g. Fourier-space fields (issue #191)
- BUG: `Communicator::sum` on a dynamically sized `Eigen::Matrix` crashed
  under MPI (unsized result)
- TST: `MUGRID_FFT_NO_ND=1` hides pocketfft's N-D transforms so the
  axis-by-axis and general pencil FFT paths run in the test suite

v1.0.0 (16Jul26)
----------------

- ENH: `FileIONetCDF.sync()` flushes buffered frames to disk for incremental
  checkpointing of long runs
- BUG: Autodetect GPU architecture; guard against GPU API failures
- BUG: Guard NetCDF reads against dimension/data-type mismatch (were garbled)
- BUG: `float32` field support in NetCDF I/O
- BUG: Reject creating a field collection on a GPU device whose backend was not
  compiled in (e.g. `Device.cuda()` in a CPU-only build)
- BUG: `version_string()` reports the GPU backend as `ON`/`OFF` from the build;
  a passed device appends its id (e.g. `CUDA: ON, device cuda:2`)

v0.112.0 (07Jul26)
------------------

- ENH: Double-precision accumulators even in single-precision runs
- ENH: `FileIONetCDF.register_frame_variable` stores per-frame, grid-less
  quantities (small replicated scalars/vectors/tensors) alongside fields
- ENH: `version_string()` returns a one-line build/run diagnostic (version, MPI
  rank, GPU device, NetCDF backend); added `netcdf_backend`/`netcdf_version`
- ENH: `MUGRID_NATIVE_ARCH` CMake option to tune code for the host CPU
  (`-march=native`; non-portable, off by default)
- ENH: Warn when a serial `Communicator` is built under an MPI launcher
- BUG: Fixed the Green-Jacobi preconditioner
- BUG: Single-precision support in the preconditioner
- BUG: Match field precision in Fourier space

v0.111.0 (03Jul26)
------------------

- ENH: Single-precision floating-point support
- ENH: Added fused `compute_sensitivity` to the isotropic stiffness operator
- ENH: Added the Green-Jacobi (J-FFT) preconditioner
- ENH: Added Q1 (bilinear quad / trilinear hex) elements
- ENH: Added fused `apply_macro_rhs` and `average_stress` to the isotropic stiffness
  operator, so FE homogenization needs no resident strain/stress fields
- ENH: `conjugate_gradients` can return the final residual `b - Ax` (for
  adjoint-corrected objectives) via an optional out-field
- ENH: Input guards across operators and fields — dtype-matched `fft`/`ifft`,
  scalar-only Laplace, storage-order/contiguity checks, size-checked Eigen
  field operators, rejection of duplicate local pixels and device state fields
- API: Python lifetime fixes — `keep_alive` on `Pixels`/`FileFrame` iterators,
  and `pop_field` kept safe against live DLPack exports
- API: `dtype=` field-creation API
- API: Bound `FieldCollection.pop_field` to free a field's memory from Python
- MAINT: Unified FE shape-function tables into element traits
- MAINT: Aliased the CG preconditioned-residual onto the residual when unpreconditioned
- MAINT: Reference preconditioner frees its impulse-response scratch after assembly and
  stores Hermitian symbols as a triangle
- MAINT: Reference preconditioner apply is now einsum-free
- BUG: `average_stress` integrates the owned region, not the padded one
- BUG: Fixed np=8 deadlock in the block-Fourier preconditioner (Hermitian
  detection diverged on ranks with empty Fourier subdomains)
- BUG: `Field.p`/`.pg` pixel views are zero-copy and correctly ordered for
  multi-component, multi-sub-point fields (were silently copied and permuted);
  fixed the underlying `get_strides(Pixel)` for structure-of-arrays fields
- BUG: `Array::resize` no longer double-frees when reallocation throws (GPU OOM)
- BUG: Guard against zero-size GPU linalg kernel launches on empty MPI ranks
- BUG: NetCDF attribute read no longer overflows on mismatched lengths, and the
  registered-vs-file consistency check is now effective
- BUG: `reduce_ghosts` supports single-precision (`Real32`/`Complex32`) fields
- BUG: MPI ghost/gather paths use C complex datatypes and range-checked counts
  (correct on Fortran-less MPI and halos above 2³¹ elements)
- BUG: Fixed FFT-engine subcommunicator leak, `CartesianCommunicator`
  copy-assignment, and unsafe `FieldCollection` moves of populated collections
- BUG: Fixed non-contiguous `Pixels` iteration and cuFFT fp32 N-D alignment checks

v0.110.0 (28Jun26)
------------------

- ENH: Removed redundant device-wide synchronizations after GPU compute kernels so the
  command queue pipelines; remaining MPI/FFT barriers scoped to the default stream
- ENH: Added a uniform-coefficient apply to the fused isotropic stiffness operator so the
  reference-material preconditioner runs from the Lamé means, dropping the full
  stiffness-tensor field and its GPU memory cost
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

# FFT Engine Design

Design notes for the muGrid FFT engine (`src/libmugrid/fft/`). The
user-facing documentation — transform definitions, normalisation, frequency
arrays and the choice between slab and pencil decomposition — is
`docs/fft.md`. This document describes how the engine lays out data, which
transform path it takes, and how the MPI transposes are implemented.

Notation: grid $N_0 \times N_1 [\times N_2]$, half-complex extent
$F_0 = \lfloor N_0/2 \rfloor + 1$, $n_c$ components per pixel, process grid
$P = P_1 \times P_2$ (3D: axis 1 split over $P_2$, axis 2 over $P_1$; 2D:
axis 1 split over $P_2 = P$).

## Storage order

A field with $n_c$ components on $M$ pixels (ghosts included) is stored in one
of two layouts. With pixel index $p$ and component index $c$, the linear
offset of entry $(c, p)$ is

| Layout | Offset | Used for |
|---|---|---|
| Array of structures (AoS) | $c + n_c\,p$ | `HostSpace` (CPU) |
| Structure of arrays (SoA) | $c\,M + p$ | `CUDASpace`, `ROCmSpace` (GPU) |

- **SoA on the GPU.** Threads that process consecutive pixels of one component
  access consecutive addresses, which the hardware coalesces into few memory
  transactions. SoA also gives each component unit stride, which cuFFT
  requires for real-to-complex (R2C) and complex-to-real (C2R) transforms:
  cuFFT does not support strided real data for R2C/C2R.
- **AoS on the CPU.** All components of a pixel share a cache line, which
  suits kernels that operate on whole per-pixel vectors or tensors.

The engine's FFT work buffers and the MPI transposes use the storage order of
the memory space they live in, so no transform ever converts between layouts.
The engine verifies once, at construction, that its real-space and
Fourier-space collections share a storage order; `fft()`/`ifft()` only accept
fields from these collections, so every transform sees a single layout.

## Strides

For a buffer of extent $L_0 \times L_1 [\times L_2]$ (ghosts included) with
$M = \prod_\alpha L_\alpha$ pixels, the element strides are

| | component | axis 0 | axis 1 | axis 2 |
|---|---|---|---|---|
| AoS | $1$ | $n_c$ | $n_c L_0$ | $n_c L_0 L_1$ |
| SoA | $M$ | $1$ | $L_0$ | $L_0 L_1$ |

The first interior pixel sits at pixel offset
$g_0 + g_1 L_0 + g_2 L_0 L_1$ for left ghost counts $g_\alpha$. The backends
receive these strides directly, so ghosted real-space fields are transformed
without repacking.

### Ghost padding along axis 0

GPU backends require the real array of an R2C/C2R transform to be aligned like
a complex array (16 bytes for double precision). The base pointer of each
transformed row is offset from the aligned allocation by
$g_0 + i\,L_0$ (plus multiples of $L_0 L_1$) real values, which stays aligned
iff both $g_0$ and $L_0$ are even. The engine therefore increases the left
ghost count along axis 0 to the next even number and adds one right ghost
layer if $L_0$ would otherwise be odd. Host engines are padded identically so
that host and device engines with equal parameters produce identical buffer
layouts. Axis 0 is never distributed, so the padding is the same on every
rank. The extra layers contain valid communicated ghost data.

## Backends

`FFTBackend` (`fft_backend.hh`) declares batched one-dimensional primitives

- `r2c(n, batch, in, in_stride, in_dist, out, out_stride, out_dist)`,
- `c2r(...)`, `c2c_forward(...)`, `c2c_backward(...)`,

in double and single precision, and optional planned $d$-dimensional
transforms `r2c_nd`/`c2r_nd` over an arbitrary strided shape with a list of
transformed axes (the half-complex axis last), reported by `supports_nd()`.
pocketfft, cuFFT and rocFFT all implement the $d$-dimensional entry points.

A backend that cannot plan a particular layout (observed with rocFFT for some
batched, strided, ghost-padded real transforms) throws `FFTPlanUnsupported`
before writing any output. The engine then falls back to the axis-by-axis path
for that grid and remembers the outcome, so later transforms skip the attempt.
Any other exception (e.g. out of memory) propagates.

rocFFT's native API (`rocfft_plan_description_set_data_layout`) accepts
arbitrary strides on both sides; the engine nevertheless uses the same SoA
layout on AMD and NVIDIA GPUs so that one device code path serves both.

## Transform paths

The forward transform takes the first applicable path below; the inverse runs
the same steps in reverse order with C2R in place of R2C.

1. **Serial, $d$-dimensional.** If the whole grid is local and the backend
   supports $d$-dimensional transforms, one planned R2C transform over all
   $d$ axes is issued, with the component axis as a non-transformed batch axis
   and input/output strides folded from the ghost and AoS/SoA layout. No work
   buffer is used.

2. **2D, MPI.** R2C along axis 0 into a work buffer of shape
   $F_0 \times N_1/P$, an all-to-all on all $P$ ranks that scatters axis 0
   and gathers axis 1 (result $F_0/P \times N_1$), then C2C along axis 1.

3. **3D slab** ($P_2 = 1$). One planned 2D R2C transform of the local
   $(x, y)$ planes per component, batched over the $N_2/P$ local planes and
   written directly into the $F_0 \times N_1 \times N_2/P$ layout; an
   all-to-all on all $P$ ranks that scatters axis 1 and gathers axis 2
   (result $F_0 \times N_1/P \times N_2$); C2C along axis 2.

4. **3D pencil** (general):

   | stage | local shape | operation |
   |---|---|---|
   | Z-pencil | $F_0 \times N_1/P_2 \times N_2/P_1$ | R2C along axis 0 |
   | Y-pencil | $F_0/P_2 \times N_1 \times N_2/P_1$ | all-to-all in the row communicator ($P_2$ ranks), then C2C along axis 1 |
   | X-pencil | $F_0/P_2 \times N_1/P_1 \times N_2$ | all-to-all in the column communicator ($P_1$ ranks), then C2C along axis 2 |

   When $P_2 = 1$ (resp. $P_1 = 1$) the corresponding all-to-all is a local
   copy. Each rank holds $O(N/P)$ values at every stage; no axis is replicated.
   The X-pencil layout is the Fourier-space layout of the engine.

The 1D engine is serial; it issues one batched 1D R2C transform per component.

The row communicator contains the ranks with equal $P_1$ coordinate and the
column communicator the ranks with equal $P_2$ coordinate; ranks are numbered
$r = p_2 P_1 + p_1$.

### Component batching

On the axis-by-axis and slab paths the engine issues one batched transform per
component (batched over rows or planes), rather than one transform batched over
components. With SoA storage each component is then a unit-stride array, which
satisfies cuFFT's R2C/C2R constraint; with AoS storage the component offset is
simply the starting address. The cost is $n_c$ backend calls instead of one;
the serial $d$-dimensional path batches over components in a single call.

## MPI transposes

`Transpose` (`transpose.hh/.cc`) redistributes a block that is distributed
along `axis_in` and complete along `axis_out` into one that is distributed
along `axis_out` and complete along `axis_in`. Both axes are split into
contiguous blocks whose sizes differ by at most one between ranks. Every rank
sends a disjoint block to each peer and receives into a disjoint block.

- **Host memory:** one `MPI_Alltoallw` per transpose. Each peer's block is
  described by a derived datatype — `MPI_Type_create_subarray` for the spatial
  block combined with `MPI_Type_contiguous` (AoS) or `MPI_Type_create_hvector`
  (SoA) for the components — so there is no explicit pack/unpack buffer and
  the MPI library performs the strided access. Setting
  `MUGRID_STAGED_TRANSPOSE=1` selects the contiguous-staging exchange below on
  the host as well.
- **Device memory:** each peer's block is gathered into a cached contiguous
  device staging buffer, exchanged with non-blocking `MPI_Isend`/`MPI_Irecv`
  to every peer and a single `MPI_Waitall`, and scattered on the receiver. MPI
  implementations pack strided datatypes on device memory block by block,
  which is much slower than a device-side gather kernel.

Both paths serialise blocks identically. The element datatype is
`MPI_C_DOUBLE_COMPLEX` or `MPI_C_FLOAT_COMPLEX` (the C types are used because
the Fortran `MPI_DOUBLE_COMPLEX` may be `MPI_DATATYPE_NULL` on MPI builds
without Fortran support). Whether device buffers are handed to MPI directly or
staged through host memory depends on GPU-aware-MPI detection; see the
"GPU data and MPI" section of `docs/gpu.md`.

## Testing the axis-by-axis paths

pocketfft, cuFFT and rocFFT all support $d$-dimensional transforms, so a
normal run takes path 1 in serial and path 3 for slab decompositions; the
per-axis serial path and the slab fallback are otherwise reached only when a
backend rejects a plan. Setting `MUGRID_FFT_NO_ND=1` before constructing a
host `FFTEngine` substitutes `PocketFFTAxisBackend`, which hides the
$d$-dimensional entry points of `PocketFFTBackend`. Results are bit-for-bit
those of `PocketFFTBackend` (the 1D primitives are the same), the engine runs
the per-axis paths in double and single precision, and `engine.backend_name`
reports `PocketFFT (axis-by-axis)`. CTest registers the FFT tests a second
time with this variable set (the `*_fft_no_nd_*` tests).

## Normalisation

Like FFTW, cuFFT and rocFFT, the transforms are unnormalised: `ifft(fft(u))`
equals $\prod_\alpha N_\alpha\,u$. `normalisation` returns
$1/\prod_\alpha N_\alpha$.

## File structure

- `fft_engine.hh`: `FFTEngine<MemorySpace>` with all transform paths
- `fft_engine_base.hh/.cc`: process-grid setup, sub-communicators, field
  collections, work buffers and transpose management
- `fft_utils.hh/.cc`: Hermitian grid size, frequency helpers, process-grid
  selection (`select_process_grid`), block distribution
- `fft_backend.hh`: abstract backend interface (1D primitives, optional
  $d$-dimensional transforms, `FFTPlanUnsupported`)
- `fft_backend_traits.hh`: backend selection per memory space
- `pocketfft_backend.hh/.cc`, `pocketfft/pocketfft_hdronly.h`: host backend
  and the vendored pocketfft header
- `gpu_fft_backend.hh`, `cufft_backend.hh/.cc`, `rocfft_backend.hh/.cc`: GPU
  backends
- `transpose.hh/.cc`: MPI transposes

## Limitations

- Device-side transforms on the axis-by-axis and slab paths are not batched
  across components (see [Component batching](#component-batching)).
- GPU-aware MPI implementations differ in how they move device memory. If the
  transposes dominate on a given system, a vendor multi-GPU FFT library
  (cuFFTMp, heFFTe) is an alternative that the engine does not support.

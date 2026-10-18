# Architecture

This page describes the internal design of two µGrid components: the
distributed FFT engine and the stencil-operator kernels. What the operators
compute and how to call them is on the [Operators](operators.md) and
[FFT](fft.md) pages; this page explains how the implementation is laid out and
why.

Notation: $d$ is the spatial dimension, $N_\alpha$ the number of grid points
along axis $\alpha$, $N = \prod_\alpha N_\alpha$, $h_\alpha$ the grid spacing,
$P$ the number of MPI ranks, and $C$ the number of degrees of freedom per pixel
(components × sub-points).

## Storage order

A field with $C$ degrees of freedom per pixel on $M$ pixels (ghosts included)
is stored in one of two orders. Writing $c \in \{0, \dots, C-1\}$ for the
degree of freedom and $p \in \{0, \dots, M-1\}$ for the linear pixel index
(column-major, axis 0 fastest), the memory offset of entry $(c, p)$ is

$$
\text{AoS (array of structures):}\quad c + C\,p, \qquad\qquad
\text{SoA (structure of arrays):}\quad c\,M + p .
$$

AoS is the default in host memory (`HostSpace`), SoA in device memory
(`CUDASpace`, `ROCmSpace`). For a pixel with grid index
$(i_0, i_1, i_2)$ on a subdomain of $(n_0, n_1, n_2)$ points,
$p = i_0 + n_0 i_1 + n_0 n_1 i_2$, so the stride between neighbours along axis
0 is $C$ in AoS and $1$ in SoA, and the offset between consecutive components
is $1$ in AoS and $M$ in SoA. Kernels are written against these strides and
therefore run on either layout.

The two defaults follow from the memory systems:

- **GPU (SoA).** Consecutive threads process consecutive pixels. With SoA they
  read consecutive addresses of one component, which the hardware coalesces
  into a single memory transaction. SoA also gives every component unit stride
  along axis 0, which cuFFT requires for the real side of real-to-complex and
  complex-to-real transforms (see [Backends](#backends)).
- **CPU (AoS).** Pointwise operations (constitutive updates, tensor
  contractions, the fused stiffness kernel) touch all $C$ values of a pixel at
  once; with AoS these share a cache line.

FFT work buffers use the storage order of the memory space they live in, so no
layout conversion takes place inside the transform; the engine reads the order
from each field and computes the strides per buffer.

## FFT engine design

### Transform

The engine computes the unnormalised discrete Fourier transform of a real field
$u$,

$$
\hat u(\mathbf{q}) = \sum_{\mathbf{x}} u(\mathbf{x})\,
e^{-2\pi i \sum_\alpha q_\alpha x_\alpha},
\qquad q_\alpha \in \{0, \tfrac{1}{N_\alpha}, \dots\} \text{ cycles per grid point},
$$

and its inverse without the factor $1/N$ (the engine exposes $1/N$ as
`normalisation`). Because $u$ is real, $\hat u(-\mathbf{q}) =
\overline{\hat u(\mathbf{q})}$, so only the half-complex range
$q_0 \in [0, \tfrac12]$ of axis 0 is stored: the Fourier grid has
$F_0 = \lfloor N_0/2 \rfloor + 1$ points along axis 0. The transform is
separable: a real-to-complex (r2c) transform along axis 0 followed by
complex-to-complex (c2c) transforms along the remaining axes. Each
one-dimensional pass costs $\mathcal{O}(N \log N_\alpha)$, the full transform
$\mathcal{O}(N \log N)$.

### Domain decomposition

The $P$ ranks form a process grid $P = P_1 P_2$:

| Space | Local block (3D) | Local block (2D) | Ghosts |
|-------|------------------|------------------|--------|
| Real | $N_0 \times N_1/P_2 \times N_2/P_1$ | $N_0 \times N_1/P_2$ | user-defined |
| Fourier | $F_0/P_2 \times N_1/P_1 \times N_2$ | $F_0/P_2 \times N_1$ | none |

Axis 0 is always local in real space, so the r2c pass needs no communication.
The real-space collection carries the user's ghost layers and doubles as the
domain decomposition for stencil operators. The forward transform is

1. r2c along axis 0;
2. transpose axis 0 ↔ axis 1 within each row communicator (ranks with equal
   $P_1$ coordinate; skipped if it has one rank), c2c along axis 1;
3. (3D) transpose axis 1 ↔ axis 2 within each column communicator, c2c along
   axis 2.

The inverse runs the adjoint sequence with c2r last. Each transpose moves
$\mathcal{O}(N/P)$ data per rank.

- **Slab** ($P_1 = P$, $P_2 = 1$; requires $P \le N_2$). Axes 0 and 1 are both
  local, so they are transformed together by one planned two-dimensional
  transform and only the axis 1 ↔ axis 2 transpose is needed.
- **Pencil** (general $P_1 \times P_2$). Each axis is transformed separately
  between two transposes. The factorisation maximises a score that favours
  $P_1 \approx P_2$ and $P_1 \mid N_2$, $P_2 \mid N_1$. Because two axes are
  distributed, pencil can use more than $N_2$ ranks.
- **auto** (default) uses slab when $P \le N_2$, otherwise pencil: slab
  performs one transpose instead of two and so moves less data, and pencil is
  needed only when there are more ranks than planes.

In 2D the decomposition is always a slab along axis 1 ($P_1 = 1$, $P_2 = P$).
One-dimensional transforms are serial only.

### Execution paths

- **Serial N-D path.** On one rank the whole grid is local, and backends that
  support it (`supports_nd()`) execute the complete r2c/c2r transform as one
  planned multidimensional call with strides that absorb the ghost padding and
  the component layout. This avoids work buffers, intermediate copies and
  per-plane launches. If a backend cannot plan a particular strided layout
  (rocFFT fails for some ghost-padded grids), the failure is detected before
  data is written, remembered for that precision, and the per-axis path is
  used. Setting `MUGRID_FFT_NO_ND=1` before constructing a host engine hides
  PocketFFT's N-D transforms, so the test suite can cover the per-axis path in
  serial runs.
- **Per-axis path.** Used under MPI and as the fallback. Each pass loops over
  the $C$ degrees of freedom and issues one batched one-dimensional transform
  per degree of freedom, batched over the pixel rows of the local block. On SoA
  storage each such batch has unit element stride, which satisfies cuFFT's
  real-side requirement; on AoS storage the element stride is $C$, which
  PocketFFT accepts. Batching over degrees of freedom instead would need
  non-unit strides on the GPU. The cost is one backend call per degree of
  freedom and pass.

### Transposes

The `Transpose` class redistributes data between the two layouts of a
transpose. It takes the storage order of the fields, so both AoS and SoA are
handled without conversion.

- **Host memory.** Blocks are described by MPI derived datatypes
  (`MPI_Type_create_subarray`, replicated over components with
  `MPI_Type_create_hvector`) and exchanged with `MPI_Alltoallw`, so no explicit
  pack/unpack buffers are needed and MPI handles the non-contiguous access.
  `MUGRID_STAGED_TRANSPOSE=1` selects an alternative that packs into
  contiguous buffers and uses `MPI_Alltoallv`.
- **Device memory.** MPI implementations pack derived datatypes on device
  memory block by block, which is orders of magnitude slower than a device-side
  gather. Device transposes therefore gather each peer's block into a
  contiguous staging buffer on the device, exchange flat messages with
  non-blocking point-to-point MPI, and scatter on the receiver (see
  [GPU data and MPI](gpu.md#gpu-data-and-mpi)).

FFT plans and work buffers are cached per engine and released with it. Device
reduction and ghost-accumulation scratch buffers are cached per device for the
lifetime of the process and only grow; they are sized by the largest reduction
or halo seen, a small fraction of a field.

### Backends

The engine is `FFTEngine<MemorySpace>` over an abstract `FFT1DBackend`:

- **PocketFFT** on the host. It is header-only and vendored in the source tree
  (`fft/pocketfft/`), so the CPU build has no external FFT dependency, and it is
  BSD-3-licensed, which is compatible with µGrid's LGPL. It supports arbitrary
  strides and both double and single precision.
- **cuFFT** on NVIDIA GPUs. cuFFT does not support non-unit strides on the real
  side of r2c/c2r transforms, which the SoA device layout satisfies.
- **rocFFT** on AMD GPUs, called directly. rocFFT supports arbitrary strides;
  µGrid uses the same SoA layout as on NVIDIA devices so both share one code
  path.

### File structure

```text
src/libmugrid/fft/
├── fft_engine_base.hh/cc     # decomposition, collections, transposes
├── fft_engine.hh/cc          # FFTEngine<MemorySpace>: transform execution
├── fft_backend.hh            # abstract FFT1DBackend (1D primitives, N-D entry points)
├── fft_backend_traits.hh     # backend selection per memory space
├── gpu_fft_backend.hh        # shared GPU backend code
├── pocketfft_backend.hh/cc   # host backend
├── cufft_backend.hh/cc       # NVIDIA backend
├── rocfft_backend.hh/cc      # AMD backend
├── transpose.hh/cc           # MPI transposes
├── fft_utils.hh/cc           # frequencies, normalisation, process grid
└── pocketfft/                # vendored PocketFFT header
```

## Operator kernel design

| Operator | Stencil | Input → output |
|----------|---------|----------------|
| `GenericLinearOperator` | arbitrary | nodal → quadrature (or nodal → nodal) |
| `FEMGradientOperator` | $2^d$ nodes per element | nodal → quadrature |
| `LaplaceOperator` | $2d + 1$ points | nodal → nodal |
| `IsotropicStiffnessOperator` | $3^d$ nodes (gather from $2^d$ elements) | nodal → nodal |

### Ghost layers and boundary conditions

Operators make no assumption about boundary conditions. Periodicity and
parallelism come entirely from ghost layers filled by
`communicate_ghosts()` before the operator runs: on a periodic domain the ghost
layers hold copies from the opposite side (or from the neighbouring rank). The
same kernel therefore serves serial, MPI-parallel, periodic and non-periodic
runs on CPU and GPU.

Each operator declares the ghost layers it reads. For a stencil with offset
$o_\alpha$ and extent $s_\alpha$ along axis $\alpha$, `apply` reads input at
$p_\alpha + o_\alpha, \dots, p_\alpha + o_\alpha + s_\alpha - 1$ and requires

$$
g^\mathrm{left}_\alpha = \max(0, -o_\alpha), \qquad
g^\mathrm{right}_\alpha = \max(0, o_\alpha + s_\alpha - 1)
$$

ghost layers. The transpose is computed as a *gather* that reads at mirrored
offsets, so its requirement is the mirror image (left and right swapped). The
operator reports both (`get_apply_ghost_requirement()`,
`get_transpose_ghost_requirement()`) and their elementwise maximum
(`get_ghost_requirement()`), which the Python `ghosts=` argument uses to size
collections. Before a kernel runs, the field collection is checked once against
the requirement and an invalid configuration raises; the inner loops contain no
bounds checks. The computed region is every point whose stencil data lies
inside the ghost-padded subdomain.

Because transposes gather into owned points rather than scatter into
neighbours, they write no ghost layers and their results need no
`reduce_ghosts()`.

| Operator | apply (left, right) | transpose (left, right) |
|----------|---------------------|-------------------------|
| `GenericLinearOperator` | $(\max(0,-o), \max(0, o+s-1))$ | swapped |
| `LaplaceOperator` | $(1, 1)$ | $(1, 1)$ |
| `FEMGradientOperator` | $(0, 1)$ | $(1, 0)$ |
| `IsotropicStiffnessOperator` | $(1, 1)$ | — |

For example, a stencil with offset $(0,0,0)$ and extent $(2,2,2)$ reads ahead
in `apply` (one right ghost layer) and gathers from behind in `transpose` (one
left ghost layer).

### Finite-element tables

The finite-element gradient and the fused stiffness operator reduce to the
same per-pixel contraction of nodal values with a reference table of
shape-function gradients $B_{q,\alpha,n}$ (quadrature point $q$, direction
$\alpha$, node $n$ of the $2^d$ pixel corners, numbered by binary offsets with
axis 0 fastest), scaled by $1/h_\alpha$ and summed over quadrature points with
weights $w_q = W^\mathrm{frac}_q \prod_\alpha h_\alpha$,
$\sum_q W^\mathrm{frac}_q = 1$. The element type only changes the numbers in
the table, not the kernel, so the tables live in `fem_element.hh` and the
kernels are templated on the element. The supported elements (Q1 and P1 in 2D
and 3D), their quadrature rules and weights are listed on the
[Operators](operators.md#femgradientoperator) page.

The physical gradient at quadrature point $q$ is

$$
(\nabla u)_{q,\alpha} = \frac{1}{h_\alpha} \sum_n B_{q,\alpha,n}\, u_n .
$$

`FEMGradientOperator.apply` evaluates this for every element whose $2^d$
corner nodes are available (hence one right ghost layer). `transpose` computes
$\mathbf{B}^\top W$ by letting every owned node gather the contributions of the
$2^d$ elements that contain it, which reach one layer to the left.

### LaplaceOperator

The $(2d+1)$-point finite-difference Laplacian (defined on the
[Operators](operators.md#laplaceoperator) page) reads one neighbour on each side
of every axis, so it needs one ghost layer on each side. It is self-adjoint,
so `transpose` executes the same kernel as `apply`.

### IsotropicStiffnessOperator

The operator and the split $K_e = 2\mu_e G + \lambda_e V$ are defined on the
[Operators](operators.md#isotropicstiffnessoperator) page. In the
implementation $G$ and $V$ are assembled once at construction in Voigt
notation with engineering shear strains,

$$
G = \sum_q w_q\, B_q^{\varepsilon\top} I' B_q^{\varepsilon}, \qquad
V = \sum_q w_q\, (B_q^{\varepsilon\top} m)(m^\top B_q^\varepsilon),
$$

where $B_q^\varepsilon$ maps the element's nodal displacements to the Voigt
strain at $q$, $I' = \operatorname{diag}(1, 1, 1, \tfrac12, \tfrac12,
\tfrac12)$ (in 2D $\operatorname{diag}(1, 1, \tfrac12)$) so that
$\varepsilon^\top I' \varepsilon = \varepsilon : \varepsilon$, and
$m = (1, 1, 1, 0, 0, 0)^\top$ (in 2D $(1,1,0)^\top$) extracts the trace. Only
this one-time precomputation depends on the element; the per-call kernels are
element-agnostic, so the element is a constructor argument rather than a
template parameter.

**Gather.** The force at node $I$ is

$$
f_I = \sum_{e \ni I} \sum_{J \in e} \bigl(2\mu_e G + \lambda_e V\bigr)_{IJ}\, u_J ,
$$

the sum over the $2^d$ elements incident to $I$. The kernel assigns one thread
(or loop iteration) per node, which gathers these contributions and writes only
its own node. No two threads write the same address, so no atomic operations
are needed on the GPU, and the result is deterministic. The element $e$ with
origin at pixel $\mathbf{i}$ holds $\lambda_e, \mu_e$ at pixel $\mathbf{i}$;
node $\mathbf{j}$ is corner $\mathbf{j} - \mathbf{i} \in \{0,1\}^d$ of the
elements with origins $\mathbf{j} - \boldsymbol{\delta}$,
$\boldsymbol{\delta} \in \{0,1\}^d$. Hence the displacement and the Lamé fields
need one ghost layer on each side. `apply_macro_rhs`, `assemble_diagonal`,
`average_stress` and `compute_sensitivity` use the same gather (or a
per-element reduction) and need no resident strain or stress field.

The table below gives the local corner index of a node in each incident
element, by element offset $\boldsymbol{\delta}$ relative to the node:

| 2D offset | corner | 3D offset | corner | 3D offset | corner |
|-----------|--------|-----------|--------|-----------|--------|
| $(-1,-1)$ | 3 | $(-1,-1,-1)$ | 7 | $(0,-1,-1)$ | 6 |
| $(0,-1)$ | 2 | $(-1,0,-1)$ | 5 | $(0,0,-1)$ | 4 |
| $(-1,0)$ | 1 | $(-1,-1,0)$ | 3 | $(0,-1,0)$ | 2 |
| $(0,0)$ | 0 | $(-1,0,0)$ | 1 | $(0,0,0)$ | 0 |

### Memory layout in the kernels

For a displacement field with $d$ components on a subdomain of
$n_0 \times n_1 \times n_2$ points (ghosts included), the strides of the
component index $c$ and the grid indices are

| | $c$ | $i_0$ | $i_1$ | $i_2$ |
|---|---|---|---|---|
| CPU (AoS) | $1$ | $d$ | $d\,n_0$ | $d\,n_0 n_1$ |
| GPU (SoA) | $n_0 n_1 n_2$ | $1$ | $n_0$ | $n_0 n_1$ |

Data pointers are offset by the left ghost layers, so index 0 is the first
owned point and index $-1$ the left ghost. The CPU and GPU kernels are
separate source files (`*_gpu.cc`, compiled as CUDA or HIP) dispatched through
`kernel_dispatcher.hh`, so each can be written for its architecture.

### File structure

```text
src/libmugrid/operators/
├── linear.hh                 # LinearOperator base, ghost requirements
├── kernel_dispatcher.hh      # host/device dispatch
├── convolution_kernels_cpu.hh / _gpu.hh
├── generic.hh/.cc, generic_gpu.cc          # GenericLinearOperator
├── laplace.hh/.cc, laplace_gpu.cc          # LaplaceOperator<Dim>
├── fem_element.hh                           # element tables (P1, Q1)
├── fem_gradient.hh, fem_gradient_gpu.cc    # FEMGradientOperator<Element, T>
├── nodal_moments.hh/.cc, nodal_moments_gpu.cc
├── transfer.hh/.cc                          # multigrid prolongation/restriction
└── solids/
    ├── isotropic_stiffness.hh               # IsotropicStiffnessOperator<Dim>
    ├── isotropic_stiffness_2d.cc
    ├── isotropic_stiffness_3d.cc
    └── isotropic_stiffness_gpu.cc
```

### Testing

The fused stiffness operator is tested in
`tests/python_isotropic_stiffness_operator_tests.py` against an explicit
$\mathbf{B}^\top \mathbf{C} \mathbf{B}$ assembled from `GenericLinearOperator`,
for symmetry of $\mathbf{K}$, for unit-impulse responses (periodic and
non-periodic), for the patch test on both elements, for `apply_macro_rhs`,
`average_stress`, `assemble_diagonal` and `compute_sensitivity`, for
single-versus-double precision, for CPU/GPU agreement, and for the validation
errors raised on missing ghost layers or mismatched material fields. The
element tables and the assembled P1 stencil are pinned in
`tests/python_fem_element_tests.py`.

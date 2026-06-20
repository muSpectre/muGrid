# Linear algebra

µGrid provides a set of linear algebra operations optimized for field data. These operations work directly on muGrid fields, avoiding the overhead of creating non-contiguous array views. The `linalg` module largely captures Level 1 BLAS functionality, providing the fundamental building blocks for iterative solvers and other numerical algorithms.

## Overview

The linear algebra operations are designed with two key principles:

1. **MPI correctness**: Operations that compute scalar results (like dot products and norms) iterate only over interior pixels, excluding ghost regions. This prevents double-counting of values that are duplicated across MPI ranks.
2. **Performance**: Operations that modify field data (like `axpy` and `scal`) operate on the full buffer including ghost regions. This is efficient because the memory is contiguous, and ghost values are typically overwritten by subsequent ghost communication anyway.

## Available operations

The following operations are available in the `muGrid.linalg` namespace (C++) or as functions in the Python bindings:

| Operation | Description | BLAS equivalent |
|-----------|-------------|-----------------|
| `vecdot(a, b)` | Vector dot product: \(\sum_i a_i b_i\) | SDOT / DDOT |
| `axpy(alpha, x, y)` | \(y \leftarrow \alpha x + y\) | SAXPY / DAXPY |
| `scal(alpha, x)` | \(x \leftarrow \alpha x\) | SSCAL / DSCAL |
| `scal(alpha_field, x)` | \(x_{c,i} \leftarrow \alpha_{c,i}\, x_{c,i}\) (per-pixel multiplier; single-component \(\alpha\) broadcasts over the components of \(x\)) | (extended BLAS) |
| `axpby(alpha, x, beta, y)` | \(y \leftarrow \alpha x + \beta y\) | (extended BLAS) |
| `copy(src, dst)` | \(\text{dst} \leftarrow \text{src}\) | SCOPY / DCOPY |
| `norm_sq(x)` | Squared norm: \(\sum_i x_i^2\) | SNRM2² / DNRM2² |
| `axpy_norm_sq(alpha, x, y)` | Fused: \(y \leftarrow \alpha x + y\), returns \(\|y\|^2\) | (fused operation) |
| `cross(a, b, out)` | Per-pixel three-vector cross product: \(\text{out} = a \times b\) | (extended BLAS) |
| `leray_project(k, invk, N, out)` | Fused Helmholtz/Leray projection update: \(\text{out}_c \mathrel{-}= k_c \sum_d \text{invk}_d\, N_d\) | (extended BLAS) |

The first group (`vecdot` through `axpy_norm_sq`) are the Level-1 BLAS-style primitives. `cross` and `leray_project` are *fused per-pixel vector kernels* that operate on three-component fields; they are described in [Per-pixel vector kernels](#per-pixel-vector-kernels) below.

## Ghost region handling

Understanding how ghost regions are handled is important for MPI-parallel codes:

**Scalar-producing operations** (`vecdot`, `norm_sq`, `axpy_norm_sq`): These iterate only over interior pixels. Ghost values are excluded because they are duplicates of values owned by neighboring MPI ranks. The returned scalar is a local result that must be MPI-reduced if a global result is needed.

**Field-modifying operations** (`axpy`, `scal`, `axpby`, `copy`, `cross`, `leray_project`): These operate on the full buffer including ghost regions. This is more efficient than iterating only over interior pixels, and ghost values will be overwritten by subsequent `communicate_ghosts()` calls anyway.

## Usage examples

### C++ usage

```cpp
#include "linalg/linalg.hh"

using namespace muGrid;
using namespace muGrid::linalg;

// Assuming fields a, b are TypedField<Real, HostSpace>

// Dot product (interior only)
Real dot = vecdot(a, b);

// AXPY: y = 0.5 * x + y
axpy(0.5, x, y);

// Scale: x = 2.0 * x
scal(2.0, x);

// Combined scale and add: y = alpha * x + beta * y
axpby(1.0, x, -1.0, y);  // y = x - y

// Copy: dst = src
copy(src, dst);

// Squared norm (interior only)
Real norm2 = norm_sq(x);

// Fused AXPY + norm: y = alpha * x + y, returns ||y||²
Real new_norm2 = axpy_norm_sq(alpha, x, y);
```

### Python usage

The linear algebra operations are available through the `muGrid.linalg` module:

```python
import muGrid
import muGrid.linalg as la

# Create fields
fc = muGrid.GlobalFieldCollection([64, 64])
x = fc.real_field("x")
y = fc.real_field("y")

# Initialize fields
x.p[...] = 1.0
y.p[...] = 2.0

# Dot product (returns local scalar, not MPI-reduced)
dot = la.vecdot(x, y)

# AXPY: y = 0.5 * x + y
la.axpy(0.5, x, y)

# Scale: x = 2.0 * x
la.scal(2.0, x)

# Scale by a per-pixel multiplier (e.g. an inverse operator diagonal
# or a spectral kernel); alpha is a real field on the same collection.
# A single-component alpha broadcasts over the components of x; an
# alpha with x's number of components is applied elementwise. x may
# be real or complex.
alpha = fc.real_field("alpha")
alpha.p[...] = 0.5
la.scal(alpha, x)

# Squared norm
norm2 = la.norm_sq(x)
```

### MPI-parallel example

For MPI-parallel computations, scalar results must be reduced across ranks:

```python
from mpi4py import MPI
import muGrid
import muGrid.linalg as la

# Create parallel decomposition
comm = muGrid.Communicator(MPI.COMM_WORLD)
decomp = muGrid.CartesianDecomposition(
    comm,
    nb_domain_grid_pts=[128, 128],
    nb_ghosts_left=[1, 1],
    nb_ghosts_right=[1, 1],
)

x = decomp.real_field("x")
y = decomp.real_field("y")

# ... initialize fields ...

# Local dot product
local_dot = la.vecdot(x, y)

# MPI reduce to get global dot product
global_dot = MPI.COMM_WORLD.allreduce(local_dot, op=MPI.SUM)
```

## Fused operations

The `axpy_norm_sq` function provides a fused operation that computes both an AXPY update and the squared norm of the result in a single pass through memory. This is more efficient than separate `axpy` + `norm_sq` calls:

```python
# Separate operations (less efficient):
# - 2 reads of x, 2 reads of y, 1 write of y
la.axpy(alpha, x, y)
norm2 = la.norm_sq(y)

# Fused operation (more efficient):
# - 1 read of x, 1 read of y, 1 write of y
norm2 = la.axpy_norm_sq(alpha, x, y)
```

This optimization is particularly valuable in iterative solvers where memory bandwidth is often the limiting factor.

## Per-pixel vector kernels

Beyond the BLAS-style primitives, `linalg` provides two fused kernels that act on **three-component** fields a single pixel at a time. They exist for the same reason as `axpy_norm_sq`: expressing them as arithmetic on field views would allocate temporaries and traverse memory several times, whereas the fused kernels compute the whole result in one pass and run unchanged on host and device. Both operate on the full buffer (ghosts included). They were introduced for pseudo-spectral fluid solvers but are not specific to them.

**`cross(a, b, out)`**

Per-pixel three-vector cross product \(\text{out} = a \times b\), i.e.

$$
\text{out}_0 = a_1 b_2 - a_2 b_1, \quad
\text{out}_1 = a_2 b_0 - a_0 b_2, \quad
\text{out}_2 = a_0 b_1 - a_1 b_0 .
$$

All three fields must have exactly three components and share a collection. `out` must be **distinct** from both `a` and `b`: a cross product cannot be formed in place, because the first component written would be read back while forming the others. Available for both real and complex fields (e.g. the Fourier-space vorticity \(i\mathbf{k}\times\hat{\mathbf{u}}\) with complex fields, and the real-space Lamb vector \(\mathbf{u}\times\boldsymbol{\omega}\)).

**`leray_project(k, invk, N, out)`**

Fused Helmholtz/Leray projection *update*:

$$
\text{out}_c \mathrel{-}= k_c \sum_d \text{invk}_d\, N_d .
$$

With the wavevector \(\mathbf{k}\) in `k` and \(\mathbf{k}/|\mathbf{k}|^2\) in `invk` (the \(\mathbf{k}=0\) mode regularised to zero by the caller), this subtracts \(\mathbf{k}\,(\mathbf{k}\cdot\mathbf{N})/|\mathbf{k}|^2\) from `out`, removing its longitudinal (compressible) part. `k` and `invk` are **real** three-component coefficient fields; `N` and `out` are the **complex** vector fields. Because the coefficients are real, the real and imaginary parts are updated independently. `out` may alias `N`; all four fields share a collection and have three components.

```python
import muGrid
import muGrid.linalg as la

fc = muGrid.GlobalFieldCollection([64, 64, 64])
a = fc.real_field("a", (3,))
b = fc.real_field("b", (3,))
omega = fc.real_field("omega", (3,))

# omega = a x b, in a single fused pass (out distinct from inputs)
la.cross(a, b, omega)

# Leray projection of a complex Fourier-space field onto div-free modes
k = fc.real_field("k", (3,))         # wavevector
invk = fc.real_field("invk", (3,))   # k / |k|^2, k=0 mode zeroed
N = fc.complex_field("N", (3,))
out = fc.complex_field("out", (3,))
la.copy(N, out)                       # start from out = N
la.leray_project(k, invk, N, out)     # out <- out - k (k.N)/|k|^2
```

## GPU support

All linear algebra operations support GPU fields (CUDA and ROCm). The operations automatically use GPU-optimized kernels when the input fields reside in device memory:

```python
import muGrid
import muGrid.linalg as la

if muGrid.has_gpu:
    # Create GPU field collection
    fc = muGrid.GlobalFieldCollection([64, 64], device="gpu")

    x = fc.real_field("x")
    y = fc.real_field("y")

    # Operations execute on GPU
    la.axpy(0.5, x, y)  # GPU kernel
    dot = la.vecdot(x, y)  # GPU reduction
```

The GPU kernels are optimized for coalesced memory access and use efficient parallel reduction algorithms for scalar-producing operations.

All operations — including the complex specialisations (`vecdot`, `norm_sq`, `axpy`, `scal`, `axpby`, `copy`, `axpy_norm_sq`) and the per-pixel vector kernels (`cross` for real and complex fields, `leray_project`) — run on the device. Complex buffers are processed through their underlying real/imaginary components, so the kernels need no device complex type; the sesquilinear `vecdot` accumulates the real and imaginary parts of \(\overline{a}\,b\) separately.

## Integration with solvers

The linear algebra module provides the core operations used by iterative solvers in `muGrid.Solvers`. For example, the conjugate gradient solver uses these operations internally:

```python
from muGrid.Solvers import conjugate_gradients

# The CG solver uses vecdot, axpy, scal, etc. internally
x = conjugate_gradients(comm, fc, b, x0, hessp=apply_operator)
```

If you are implementing custom solvers, using the `muGrid.linalg` operations ensures efficient execution on both CPU and GPU, with proper handling of ghost regions for MPI parallelism.

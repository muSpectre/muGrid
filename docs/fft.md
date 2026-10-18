# FFT

µGrid provides a distributed real-to-complex Fast Fourier Transform engine,
`FFTEngine`, that transforms µGrid fields in place in their memory space (host
or GPU) and under MPI. This page defines the transforms, describes the
parallel decomposition and documents the Python API. Implementation details
(strides, work buffers, the MPI transpose) are in the
[FFT engine design notes](https://github.com/muSpectre/muGrid/blob/main/doc/FFT.md).

## Definitions

Consider a periodic grid of dimension $d \in \{1, 2, 3\}$ with $N_\alpha$ grid
points along axis $\alpha$ and grid spacing $h_\alpha$. A real field $u$ is
given at the integer positions $\mathbf{n} = (n_0, \ldots, n_{d-1})$,
$0 \le n_\alpha < N_\alpha$. Its Fourier coefficients are indexed by integer
frequencies $\mathbf{m}$, or equivalently by the frequency
$q_\alpha = m_\alpha / N_\alpha$ in cycles per grid point.

The forward transform (`fft`) and the inverse transform (`ifft`) are

$$
\hat{u}(\mathbf{m}) = \sum_{\mathbf{n}} u(\mathbf{n})\,
  \exp\!\Big(-2\pi i \sum_\alpha \frac{m_\alpha n_\alpha}{N_\alpha}\Big),
\qquad
u(\mathbf{n}) \leftarrow \sum_{\mathbf{m}} \hat{u}(\mathbf{m})\,
  \exp\!\Big(+2\pi i \sum_\alpha \frac{m_\alpha n_\alpha}{N_\alpha}\Big).
$$

Both are **unnormalised**: a forward-inverse round trip multiplies the data by
$\prod_\alpha N_\alpha$. The engine property `normalisation` is
$1/\prod_\alpha N_\alpha$; multiply by it after `ifft` to recover the input.
Leaving the normalisation to the caller means it can be folded into a kernel
that is applied anyway (e.g. a spectral preconditioner) at no extra cost, and
any of the common conventions (normalise forward, inverse, or both
symmetrically) can be implemented on top.

Because $u$ is real, $\hat{u}(-\mathbf{m}) = \overline{\hat{u}(\mathbf{m})}$
(Hermitian symmetry). The engine therefore stores only
$0 \le m_0 \le \lfloor N_0/2 \rfloor$ along axis 0 (the *half-complex* axis):
the Fourier grid has $F_0 = \lfloor N_0/2 \rfloor + 1$ points along axis 0 and
$N_\alpha$ along the other axes. This halves memory and work compared with a
complex transform. On the other axes, $m_\alpha$ follows NumPy's `fftfreq`
convention, $m_\alpha \in [-\lfloor N_\alpha/2 \rfloor, \lceil N_\alpha/2 \rceil)$.

The physical wavevector is $\mathbf{k} = 2\pi\mathbf{q}/\mathbf{h}$, i.e.
$k_\alpha = 2\pi q_\alpha / h_\alpha = 2\pi m_\alpha / (N_\alpha h_\alpha)$. A
derivative $\partial_\alpha$ is the multiplication of $\hat{u}$ by
$i k_\alpha$.

`ifft` is a complex-to-real transform. It assumes that the Fourier field is
the transform of real data, in particular that the coefficients at
$m_0 = 0$ and (for even $N_0$) at $m_0 = N_0/2$ are consistent with Hermitian
symmetry. For input that violates this, the result depends on the backend
(multi-axis cuFFT/rocFFT C2R transforms may differ from pocketfft), so project
such data first, e.g. by taking the real part of a full complex transform.

### Cost

A transform of a field with $n_c$ components costs
$O(n_c N \log N)$ operations with $N = \prod_\alpha N_\alpha$, distributed over
$P$ MPI ranks as $O(n_c (N/P) \log N)$ per rank, plus the all-to-all
communication described below.

## Backends

| Memory space | Backend | Notes |
|---|---|---|
| Host | [pocketfft](https://github.com/mreineck/pocketfft) | Header-only C++ library, vendored in `src/libmugrid/fft/pocketfft`, BSD-3-Clause licensed, so the host engine has no external FFT dependency. |
| CUDA | cuFFT | Selected with `device="cuda"`/`"gpu"` on a CUDA build. |
| ROCm | rocFFT | Selected with `device="rocm"`/`"gpu"` on a HIP build. |

`engine.backend_name` reports the backend in use. Every backend offers
one-dimensional batched transforms and planned $d$-dimensional transforms.
Whenever the local data layout allows it, the engine issues a single planned
multi-dimensional transform (batched over the field components) instead of one
transform per axis, which avoids intermediate work buffers and, on GPUs, one
kernel launch and synchronisation per axis and component. Building and using
the engine on the GPU is described on the [GPU page](gpu.md#fft-on-gpu).

## The FFTEngine class

```python
import muGrid

engine = muGrid.FFTEngine([nx])          # 1D (serial only)
engine = muGrid.FFTEngine([nx, ny])      # 2D
engine = muGrid.FFTEngine([nx, ny, nz])  # 3D
```

The full constructor signature is

```python
muGrid.FFTEngine(nb_domain_grid_pts, communicator=None,
                 nb_ghosts_left=None, nb_ghosts_right=None, nb_sub_pts=None,
                 device=None, ghosts=None, decomposition="auto")
```

- `communicator` — a `muGrid.Communicator` (or an mpi4py communicator); serial
  if omitted.
- `nb_ghosts_left`, `nb_ghosts_right`, `ghosts` — ghost layers of the
  real-space fields (see [FFT with ghost regions](#fft-with-ghost-regions)).
- `nb_sub_pts` — sub-points per pixel, e.g. `{"quad": 4}`.
- `device` — `"cpu"` (default), `"gpu"`, `"cuda"`, `"cuda:N"`, `"rocm"`, or a
  `muGrid.Device`.
- `decomposition` — `"auto"`, `"slab"` or `"pencil"` (see
  [MPI decomposition](#mpi-decomposition)).

The engine reports the grid geometry:

```python
engine = muGrid.FFTEngine([16, 20])

engine.nb_domain_grid_pts             # (16, 20)  global real-space grid
engine.nb_fourier_grid_pts            # (9, 20)   global Fourier grid
engine.nb_subdomain_grid_pts          # local real-space grid (no ghosts)
engine.nb_fourier_subdomain_grid_pts  # local Fourier grid
engine.normalisation                  # 1/320
engine.backend_name                   # "PocketFFT"
```

1D transforms are serial only; constructing a 1D engine on more than one MPI
rank raises an error. 2D and 3D engines run under MPI.

## Creating FFT fields

The engine owns two field collections, one in real space and one in Fourier
space. Fields passed to `fft`/`ifft` must belong to these collections.

```python
engine = muGrid.FFTEngine([16, 20])

real_field = engine.real_space_field("real")              # float64
fourier_field = engine.fourier_space_field("fourier")     # complex128
print(real_field.s.shape)     # (1, 16, 20)
print(fourier_field.s.shape)  # (1, 9, 20)
```

`real_space_field(name, components=(), dtype=np.float64, sub_pt="pixel")` and
`fourier_space_field(name, components=(), dtype=np.complex128, sub_pt="pixel")`
return the field if it already exists and create it otherwise;
`register_real_space_field`/`register_fourier_space_field` take the same
arguments but raise if the name is taken.

Fields may have any number of components; all of them are transformed:

```python
velocity = engine.real_space_field("velocity", components=(3,))
velocity_hat = engine.fourier_space_field("velocity_hat", components=(3,))
print(velocity.s.shape)       # (3, 1, 16, 20)
print(velocity_hat.s.shape)   # (3, 1, 9, 20)
```

### Precision

The transform runs in the precision of the fields: a `float64` real field
pairs with a `complex128` Fourier field, a `float32` field with `complex64`.
Mismatched pairs raise an error.

```python
r = engine.real_space_field("r32", dtype=np.float32)
f = engine.fourier_space_field("f32", dtype=np.complex64)
engine.fft(r, f)
```

## Forward and inverse transforms

`fft(real_field, fourier_field)` computes the forward transform,
`ifft(fourier_field, real_field)` the inverse:

```python
import numpy as np
import muGrid

engine = muGrid.FFTEngine([16, 20])
real_field = engine.real_space_field("real")
fourier_field = engine.fourier_space_field("fourier")

real_field.p[...] = np.random.randn(16, 20)
original = real_field.p.copy()

engine.fft(real_field, fourier_field)
engine.ifft(fourier_field, real_field)
real_field.p[...] *= engine.normalisation

np.testing.assert_allclose(real_field.p, original, atol=1e-14)
```

The 1D transform matches NumPy's `rfft`:

```python
engine = muGrid.FFTEngine([64])
real_field = engine.real_space_field("signal")
fourier_field = engine.fourier_space_field("spectrum")

data = np.random.randn(64)
real_field.p[...] = data
engine.fft(real_field, fourier_field)
np.testing.assert_allclose(fourier_field.p.ravel(), np.fft.rfft(data))
```

## Utility functions

```python
muGrid.get_hermitian_grid_pts([64, 64, 64])   # (33, 64, 64)
muGrid.get_hermitian_grid_pts([64, 64], r2c_axis=1)
muGrid.fft_normalization([16, 20])            # 1/320
```

`get_hermitian_grid_pts(nb_grid_pts, r2c_axis=0)` returns the Fourier grid
for a half-complex transform along `r2c_axis`; `fft_normalization(nb_grid_pts)`
returns $1/\prod_\alpha N_\alpha$.

## Frequency and coordinate arrays

The engine returns frequency and coordinate arrays for the **rank-local**
subdomain, so code that uses them is identical in serial and MPI runs.

| Property | Shape | Value |
|---|---|---|
| `fftfreq` | `(d, *nb_fourier_subdomain_grid_pts)` | $q_\alpha = m_\alpha / N_\alpha$ (cycles per grid point) |
| `ifftfreq` | same | $m_\alpha$ (integer) |
| `coords` | `(d, *nb_subdomain_grid_pts)` | $n_\alpha / N_\alpha \in [0, 1)$ |
| `icoords` | same | $n_\alpha$ (integer) |
| `coordsg`, `icoordsg` | `(d, *local grid with ghosts)` | as above, including ghost cells |

Along the half-complex axis 0, `fftfreq` follows NumPy's `rfftfreq`
($q_0 \in [0, 1/2]$, with an even Nyquist mode at $+1/2$); along the other
axes it follows `fftfreq` ($q_\alpha \in [-1/2, 1/2)$).

```python
import numpy as np
import muGrid

engine = muGrid.FFTEngine([7, 4])
qx, qy = engine.fftfreq           # shape (2, 4, 4)
mx, my = engine.ifftfreq          # integer frequencies

ref = np.array(np.meshgrid(np.fft.rfftfreq(7), np.fft.fftfreq(4),
                           indexing="ij"))
np.testing.assert_allclose(engine.fftfreq, ref)

x, y = engine.coords              # shape (2, 7, 4), values in [0, 1)
ix, iy = engine.icoords
np.testing.assert_allclose(ix, x * 7)
```

A spectral operator is built from these arrays. For example, the Fourier
symbol of the continuous Laplacian on a domain of size $L_\alpha = N_\alpha
h_\alpha$ is $-|\mathbf{k}|^2$:

```python
nx, ny = 64, 64
Lx, Ly = 1.0, 1.0
engine = muGrid.FFTEngine([nx, ny])
qx, qy = engine.fftfreq
kx = 2 * np.pi * qx * nx / Lx     # k = 2 pi q / h with h = L / N
ky = 2 * np.pi * qy * ny / Ly

u = engine.real_space_field("u")
u_hat = engine.fourier_space_field("u_hat")
x, y = engine.coords
u.p[...] = np.cos(2 * np.pi * x)
engine.fft(u, u_hat)
laplacian_hat = -(kx**2 + ky**2) * u_hat.p
```

`examples/fourier_derivative.py` computes a spectral gradient this way (see
[Examples](examples.md#fourier-derivative)).

## MPI decomposition

In real space, axis 0 is never distributed. The Fourier-space layout is the
transpose of the real-space layout, so every transform involves an all-to-all
exchange. With $F_0 = \lfloor N_0/2 \rfloor + 1$ and a process grid
$P = P_1 \times P_2$:

| | Real space (local) | Fourier space (local) |
|---|---|---|
| 2D | $N_0 \times N_1/P$ | $F_0/P \times N_1$ |
| 3D | $N_0 \times N_1/P_2 \times N_2/P_1$ | $F_0/P_2 \times N_1/P_1 \times N_2$ |

(Extents that do not divide evenly differ by at most one grid point between
ranks.)
`engine.process_grid` returns $(P_1, P_2)$ and `engine.process_coords` the
rank's coordinates in it.

**2D.** Axis 1 is split over all $P$ ranks ($P_1 = 1$, $P_2 = P$). The forward
transform is a real-to-complex transform along axis 0, one all-to-all that
redistributes the half-complex axis 0 and gathers axis 1, and a
complex-to-complex transform along axis 1. The `decomposition` argument has no
effect in 2D.

**3D slab** ($P_1 = P$, $P_2 = 1$, requires $P \le N_2$). Each rank holds
complete $(x, y)$ planes. The forward transform is one planned 2D
real-to-complex transform of the local planes, one all-to-all over all $P$
ranks that scatters axis 1 and gathers axis 2, and a complex-to-complex
transform along axis 2.

**3D pencil** ($P_1 P_2 = P$, $P_2 > 1$). The forward transform is a
real-to-complex transform along axis 0; an all-to-all within each row
communicator (the $P_2$ ranks sharing a $P_1$ coordinate) that scatters axis 0
and gathers axis 1; a complex-to-complex transform along axis 1; an all-to-all
within each column communicator ($P_1$ ranks) that scatters axis 1 and gathers
axis 2; and a complex-to-complex transform along axis 2. The inverse
transform runs the same steps in reverse order.

### Communication cost

Let $V = n_c F_0 N_1 N_2$ be the number of complex values of the
Fourier-space field. In the usual latency-bandwidth model with latency
$\alpha$ per message and $\beta$ per transferred value, one all-to-all over a
communicator of $p$ ranks costs each rank about
$(p-1)\,\alpha + \frac{p-1}{p}\frac{V}{P}\,\beta$. Hence, per transform,

$$
T_\text{slab} \approx (P-1)\,\alpha + \frac{V}{P}\,\beta,
\qquad
T_\text{pencil} \approx (P_1 + P_2 - 2)\,\alpha + 2\,\frac{V}{P}\,\beta .
$$

The slab moves every value once and the pencil twice; the pencil sends fewer,
larger messages per rank ($\sim 2\sqrt{P}$ instead of $P$) and can use up to
$\min(N_1, N_2)\cdot\min(N_1, F_0)$ ranks instead of $N_2$.

### Choosing the decomposition

```python
engine = muGrid.FFTEngine(nb_grid_pts, comm, decomposition="auto")
```

- `"slab"` — $P_1 = P$, $P_2 = 1$. Raises if $P > N_2$.
- `"pencil"` — the factorisation $P = P_1 P_2$ chosen by the heuristic below.
- `"auto"` (default) — slab whenever $P \le N_2$, pencil otherwise. The slab
  moves half the data of the pencil (one transpose instead of two) and
  transforms the two local axes with a single planned 2D transform, so it is
  preferred whenever it can use all ranks.

The pencil factorisation maximises
$s = 1 - \frac{|P_1 - P_2|}{P_1 + P_2}
 + \tfrac{1}{2}\big([N_2 \bmod P_1 = 0] + [N_1 \bmod P_2 = 0]\big)
 + 10\,[\text{every rank holds data}]$,
i.e. it prefers factorisations in which every rank owns data in both spaces
($P_1 \le \min(N_1, N_2)$, $P_2 \le \min(N_1, F_0)$), then balanced process
grids (which minimise the latency term above), then even divisibility of the
grid.

### Basic parallel usage

```python
from mpi4py import MPI
import muGrid

comm = muGrid.Communicator(MPI.COMM_WORLD)
engine = muGrid.FFTEngine([128, 128, 128], comm)

print(f"Rank {comm.rank}: local {engine.nb_subdomain_grid_pts}, "
      f"process grid {engine.process_grid}, "
      f"coords {engine.process_coords}")
```

```bash
mpirun -np 4 python parallel_fft.py
```

## FFT with ghost regions

`FFTEngine` is a `CartesianDecomposition`: its real-space fields can carry
ghost layers, and `engine.communicate_ghosts(field)` /
`engine.reduce_ghosts(field)` fill and reduce them. Stencil operators and
transforms can therefore act on the same fields, which is how spectral
preconditioners are combined with finite-difference or finite-element
operators (see [Linear algebra](linalg.md#preconditioners)). Fourier-space
fields have no ghosts.

```python
from mpi4py import MPI
import muGrid

comm = muGrid.Communicator(MPI.COMM_WORLD)
laplace = muGrid.LaplaceOperator(3)

engine = muGrid.FFTEngine([64, 64, 64], comm, ghosts=laplace)
# equivalently: nb_ghosts_left=[1, 1, 1], nb_ghosts_right=[1, 1, 1]

u = engine.real_space_field("u", components=(3,))      # has ghosts
u_hat = engine.fourier_space_field("u_hat", components=(3,))
engine.communicate_ghosts(u)
```

Passing an operator (or a list of operators) as `ghosts` sizes the ghost
layers from the operators' reported `ghost_requirement`.

!!! note
    The engine pads the ghost layers along axis 0 so that the left ghost count
    and the stored row width (interior plus ghosts) are even. GPU
    real-to-complex transforms need the real rows aligned like complex ones
    (16 bytes); host engines are padded identically so that host and device
    fields of the same engine parameters share a layout. With one requested
    ghost layer on each side, axis 0 gets 2 + 2. The extra layers hold valid,
    communicated ghost data, so stencils see at least the ghosts they asked
    for, but `.sg`/`.pg` and `coordsg` are wider than requested along axis 0.
    Read the actual counts from
    `engine.real_space_collection.nb_ghosts_left`/`nb_ghosts_right`.

```python
engine = muGrid.FFTEngine([64, 64], nb_ghosts_left=[1, 1],
                          nb_ghosts_right=[1, 1])
print(engine.coords.shape)    # (2, 64, 64)
print(engine.coordsg.shape)   # (2, 68, 66)
```

## GPU support

On a CUDA or HIP build, `FFTEngine(..., device="gpu")` allocates its fields in
device memory and dispatches to cuFFT or rocFFT; the engine classes
`FFTEngineCUDA`/`FFTEngineROCm` are the underlying bindings. Device storage is
structure-of-arrays, which gives the unit-stride real data that cuFFT requires
for real-to-complex transforms. See the [GPU page](gpu.md#fft-on-gpu) for
building with GPU support and working with device fields.

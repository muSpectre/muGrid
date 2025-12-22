Fast Fourier Transform
######################

*µ*\Grid includes a built-in Fast Fourier Transform (FFT) engine that operates on
*µ*\Grid fields. The FFT engine uses `pocketfft <https://github.com/mreineck/pocketfft>`_
for efficient FFT computation.

The FFTEngine class
*******************

Instantiating an FFT engine is straightforward:

.. code-block:: python

    import muGrid

    # Create a 2D FFT engine
    engine = muGrid.FFTEngine([nx, ny])

    # Create a 3D FFT engine
    engine = muGrid.FFTEngine([nx, ny, nz])

The FFT engine provides properties for querying the grid dimensions:

.. code-block:: python

    engine = muGrid.FFTEngine([16, 20])

    print(f"Real-space grid: {engine.nb_domain_grid_pts}")      # (16, 20)
    print(f"Fourier-space grid: {engine.nb_fourier_grid_pts}")  # (9, 20)
    print(f"Backend: {engine.backend_name}")                     # "pocketfft"

The Fourier-space grid has reduced dimensions due to the real-to-complex (r2c) transform
exploiting Hermitian symmetry.

Creating FFT fields
*******************

The FFT engine manages two field collections: one for real-space fields and one for
Fourier-space fields. Use the convenience functions to create fields:

.. code-block:: python

    import muGrid
    from muGrid import fft_real_space_field, fft_fourier_space_field

    engine = muGrid.FFTEngine([16, 20])

    # Create real-space field (real-valued)
    real_field = fft_real_space_field(engine, "real")
    print(f"Real field shape: {real_field.s.shape}")  # (1, 1, 16, 20)

    # Create Fourier-space field (complex-valued)
    fourier_field = fft_fourier_space_field(engine, "fourier")
    print(f"Fourier field shape: {fourier_field.s.shape}")  # (1, 1, 9, 20)

Forward and inverse transforms
******************************

The ``fft`` method performs the forward transform (real to Fourier space) and
``ifft`` performs the inverse transform (Fourier to real space):

.. code-block:: python

    import numpy as np
    import muGrid
    from muGrid import fft_real_space_field, fft_fourier_space_field

    engine = muGrid.FFTEngine([16, 20])

    real_field = fft_real_space_field(engine, "real")
    fourier_field = fft_fourier_space_field(engine, "fourier")

    # Initialize real-space field
    real_field.s[:] = np.random.randn(1, 1, 16, 20)
    original = real_field.s.copy()

    # Forward FFT: real -> Fourier
    engine.fft(real_field._cpp, fourier_field._cpp)

    # Inverse FFT: Fourier -> real
    engine.ifft(fourier_field._cpp, real_field._cpp)

    # Apply normalization for roundtrip
    real_field.s[:] *= engine.normalisation

    # Verify roundtrip
    np.testing.assert_allclose(real_field.s, original, atol=1e-14)

Normalization
*************

*µ*\Grid FFT transforms are **not normalized** by default. A forward-inverse roundtrip
picks up a global factor equal to the total number of grid points. This factor is
available as the ``normalisation`` property:

.. code-block:: python

    engine = muGrid.FFTEngine([8, 10])
    print(f"Normalization factor: {engine.normalisation}")  # 1/80 = 0.0125

To get the original values after a roundtrip, multiply by the normalization factor
after the inverse transform.

The reason for leaving normalization to the user is that there are multiple valid
conventions (normalize forward, normalize inverse, or split between both), and
different applications may prefer different choices.

Wavevector utilities
********************

*µ*\Grid provides utility functions for working with FFT frequencies:

.. code-block:: python

    import muGrid

    n = 8

    # Integer frequency indices (same as numpy.fft.fftfreq(n) * n)
    freq_idx = muGrid.fft_freqind(n)
    print(freq_idx)  # [0, 1, 2, 3, -4, -3, -2, -1]

    # Frequency values with spacing dx
    dx = 0.1
    freqs = muGrid.fft_freq(n, dx)  # Same as numpy.fft.fftfreq(n, dx)

    # For real FFT (half-complex output)
    rfft_idx = muGrid.rfft_freqind(n)
    print(rfft_idx)  # [0, 1, 2, 3, 4]

    rfft_freqs = muGrid.rfft_freq(n, dx)  # Same as numpy.fft.rfftfreq(n, dx)

Hermitian grid dimensions
*************************

For real-to-complex transforms, the output has reduced dimensions due to Hermitian
symmetry. Use ``get_hermitian_grid_pts`` to compute the Fourier-space grid size:

.. code-block:: python

    import muGrid

    real_grid = [64, 64, 64]
    fourier_grid = muGrid.get_hermitian_grid_pts(real_grid)
    print(f"Fourier grid: {fourier_grid}")  # [33, 64, 64]

The first dimension is reduced to ``n//2 + 1``.

FFT normalization factor
************************

The ``fft_normalization`` function returns the normalization factor for a given
grid size:

.. code-block:: python

    import muGrid

    norm = muGrid.fft_normalization([16, 20])
    print(f"Normalization: {norm}")  # 1/320

Example: Fourier derivative
***************************

A common use of FFTs is computing derivatives in Fourier space. Here's an example
computing the gradient of a 2D field:

.. code-block:: python

    import numpy as np
    import muGrid
    from muGrid import fft_real_space_field, fft_fourier_space_field

    # Grid parameters
    nx, ny = 64, 64
    Lx, Ly = 2 * np.pi, 2 * np.pi
    dx, dy = Lx / nx, Ly / ny

    engine = muGrid.FFTEngine([nx, ny])

    # Create fields
    f = fft_real_space_field(engine, "f")
    f_hat = fft_fourier_space_field(engine, "f_hat")
    grad_x = fft_real_space_field(engine, "grad_x")
    grad_y = fft_real_space_field(engine, "grad_y")
    grad_hat = fft_fourier_space_field(engine, "grad_hat")

    # Initialize with a smooth function
    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    f.s[0, 0, :, :] = np.sin(X) * np.cos(Y)

    # Forward FFT
    engine.fft(f._cpp, f_hat._cpp)

    # Compute wavevectors
    kx = 2 * np.pi * np.array(muGrid.rfft_freqind(nx)) / Lx
    ky = 2 * np.pi * np.array(muGrid.fft_freqind(ny)) / Ly
    KX, KY = np.meshgrid(kx, ky, indexing='ij')

    # Derivative in x: multiply by i*kx
    grad_hat.s[0, 0, :, :] = 1j * KX * f_hat.s[0, 0, :, :]
    engine.ifft(grad_hat._cpp, grad_x._cpp)
    grad_x.s[:] *= engine.normalisation

    # Derivative in y: multiply by i*ky
    grad_hat.s[0, 0, :, :] = 1j * KY * f_hat.s[0, 0, :, :]
    engine.ifft(grad_hat._cpp, grad_y._cpp)
    grad_y.s[:] *= engine.normalisation

    # Verify: d/dx[sin(x)cos(y)] = cos(x)cos(y)
    expected_grad_x = np.cos(X) * np.cos(Y)
    np.testing.assert_allclose(grad_x.s[0, 0], expected_grad_x, atol=1e-10)

Multi-component fields
**********************

FFT fields can have multiple components:

.. code-block:: python

    import muGrid
    from muGrid import fft_real_space_field, fft_fourier_space_field

    engine = muGrid.FFTEngine([16, 20])

    # Create field with 3 components (e.g., velocity vector)
    velocity = fft_real_space_field(engine, "velocity", nb_components=3)
    velocity_hat = fft_fourier_space_field(engine, "velocity_hat", nb_components=3)

    print(f"Velocity shape: {velocity.s.shape}")      # (3, 1, 16, 20)
    print(f"Velocity_hat shape: {velocity_hat.s.shape}")  # (3, 1, 9, 20)

    # FFT transforms all components
    engine.fft(velocity._cpp, velocity_hat._cpp)
    engine.ifft(velocity_hat._cpp, velocity._cpp)

Derivative operators (µFFT)
***************************

The separate `µFFT <https://github.com/muSpectre/muFFT>`_ library provides two types
of derivative operators for computing derivatives in Fourier space: spectral
derivatives (``FourierDerivative``) and finite-difference stencil derivatives
(``DiscreteDerivative``). These can be used together with *µ*\Grid's FFT engine.

FourierDerivative
=================

The ``FourierDerivative`` class computes spectral derivatives by multiplying with
the wavevector in Fourier space. This gives the highest accuracy possible on a
discrete grid:

.. code-block:: python

    import numpy as np
    from muFFT import FFT, FourierDerivative

    # Create FFT engine
    nb_grid_pts = (64, 64)
    physical_sizes = (2 * np.pi, 2 * np.pi)
    fft = FFT(nb_grid_pts)

    # Create derivative operators
    # FourierDerivative(nb_dims, direction)
    fourier_x = FourierDerivative(2, 0)  # Derivative in x direction
    fourier_y = FourierDerivative(2, 1)  # Derivative in y direction

    # Create fields
    rfield = fft.real_space_field('scalar')
    ffield = fft.fourier_space_field('scalar')

    # Initialize with sin(x)
    x, y = fft.coords
    rfield.p = np.sin(2 * np.pi * x)

    # Forward FFT
    fft.fft(rfield, ffield)

    # Compute derivative in Fourier space
    dx = physical_sizes[0] / nb_grid_pts[0]
    fgrad = fft.fourier_space_field('gradient')
    fgrad.p = fourier_x.fourier(fft.fftfreq) * ffield.p / dx

    # Inverse FFT
    rgrad = fft.real_space_field('gradient')
    fft.ifft(fgrad, rgrad)
    rgrad.p *= fft.normalisation

    # Result is cos(x)
    np.testing.assert_allclose(rgrad.p, 2 * np.pi * np.cos(2 * np.pi * x), atol=1e-12)

The ``fourier`` method returns the Fourier-space multiplier for the derivative,
which is ``2πi * k`` where ``k`` is the wavevector.

DiscreteDerivative
==================

The ``DiscreteDerivative`` class computes derivatives using finite-difference stencils.
While less accurate than spectral derivatives, they correspond exactly to discrete
finite-difference operations in real space:

.. code-block:: python

    import numpy as np
    from muFFT import FFT, DiscreteDerivative

    # Create FFT engine
    nb_grid_pts = (64, 64)
    fft = FFT(nb_grid_pts)

    # Create first-order upwind stencil: f'(x) ≈ [f(x+h) - f(x)] / h
    # DiscreteDerivative(offsets, coefficients)
    upwind_x = DiscreteDerivative([0, 0], [[-1], [1]])
    upwind_y = DiscreteDerivative([0, 0], [[-1, 1]])

    # Create second-order central stencil: f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
    central_x = DiscreteDerivative([-1, 0], [[-0.5], [0], [0.5]])
    central_y = DiscreteDerivative([0, -1], [[-0.5, 0, 0.5]])

The first argument is the offset (where the stencil starts relative to the current
point), and the second argument is a 2D array of coefficients.

Stencil libraries (µFFT)
========================

*µ*\FFT provides pre-built stencil libraries for common derivative operators:

.. code-block:: python

    from muFFT.Stencils1D import upwind, central, central6, central_2nd
    from muFFT.Stencils2D import upwind_x, upwind_y, central_x, central_y
    from muFFT.Stencils2D import linear_finite_elements
    from muFFT.Stencils3D import upwind_x, upwind_y, upwind_z

**1D Stencils** (``muFFT.Stencils1D``):

- ``upwind``: First-order upwind difference
- ``central``: Second-order central difference
- ``central6``: Sixth-order central difference
- ``central_2nd``: Fourth-order second derivative

**2D Stencils** (``muFFT.Stencils2D``):

- ``upwind_x``, ``upwind_y``: First-order upwind differences
- ``central_x``, ``central_y``: Second-order central differences
- ``central_2nd_x``, ``central_2nd_y``: Fourth-order second derivatives
- ``averaged_upwind_x``, ``averaged_upwind_y``: Averaged upwind differences
- ``linear_finite_elements``: Tuple of 4 stencils for FEM on triangular mesh

**3D Stencils** (``muFFT.Stencils3D``):

- ``upwind_x``, ``upwind_y``, ``upwind_z``: First-order upwind differences
- ``central_x``, ``central_y``, ``central_z``: Second-order central differences
- ``linear_finite_elements``: Tuple of 18 stencils for FEM on tetrahedral mesh

Example using stencils:

.. code-block:: python

    import numpy as np
    from muFFT import FFT
    from muFFT.Stencils2D import upwind_x, upwind_y

    nb_grid_pts = (54, 17)
    physical_sizes = (1.4, 2.3)
    dx, dy = np.array(physical_sizes) / np.array(nb_grid_pts)

    fft = FFT(nb_grid_pts)

    # Create and initialize field
    rfield = fft.real_space_field('scalar')
    x, y = fft.coords
    rfield.p = np.sin(2 * np.pi * x)

    # Forward FFT
    ffield = fft.fourier_space_field('scalar')
    fft.fft(rfield, ffield)

    # Compute gradient using stencils
    fgrad = fft.fourier_space_field('gradient', (2,))
    fgrad.p[0] = upwind_x.fourier(fft.fftfreq) * ffield.p / dx
    fgrad.p[1] = upwind_y.fourier(fft.fftfreq) * ffield.p / dy

    # Inverse FFT
    rgrad = fft.real_space_field('gradient', (2,))
    fft.ifft(fgrad, rgrad)
    rgrad.p *= fft.normalisation

MPI-parallel FFT
****************

The ``FFTEngine`` class supports MPI parallelization using pencil (2D) decomposition,
which allows efficient scaling to large numbers of ranks.

Basic parallel usage
====================

.. code-block:: python

    import numpy as np
    from mpi4py import MPI
    import muGrid
    from muGrid import Communicator

    # Create parallel FFT engine
    nb_grid_pts = [128, 128, 128]
    comm = Communicator(MPI.COMM_WORLD)
    engine = muGrid.FFTEngine(nb_grid_pts, comm)

    # Each rank has a subdomain
    print(f"Rank {MPI.COMM_WORLD.rank}:")
    print(f"  Global grid: {engine.nb_domain_grid_pts}")
    print(f"  Local subdomain: {engine.nb_subdomain_grid_pts}")
    print(f"  Process grid: {engine.process_grid}")
    print(f"  Process coords: {engine.process_coords}")

The FFT engine uses pencil decomposition, distributing the grid across a 2D process
grid for efficient scaling.

FFT with ghost regions
======================

For computations requiring ghost cells (e.g., stencil operations), specify the
ghost buffer sizes:

.. code-block:: python

    import muGrid
    from muGrid import Communicator
    from mpi4py import MPI

    comm = Communicator(MPI.COMM_WORLD)

    # Create FFT engine with ghost regions
    engine = muGrid.FFTEngine(
        nb_domain_grid_pts=[64, 64, 64],
        comm=comm,
        nb_ghosts_left=[1, 1, 1],
        nb_ghosts_right=[1, 1, 1]
    )

    # Real-space fields include ghost regions
    real_field = engine.real_space_field("displacement", nb_components=3)

    # Fourier-space fields have no ghosts (hard assumption)
    fourier_field = engine.fourier_space_field("displacement_k", nb_components=3)

Run with MPI:

.. code-block:: sh

    $ mpirun -np 4 python parallel_fft.py

GPU support
***********

When *µ*\Grid is compiled with CUDA or HIP support, FFT operations can run on GPU.
See the :doc:`GPU` documentation for details on building with GPU support and
working with GPU fields.

Currently, the FFT engine operates on host (CPU) memory. For GPU-accelerated FFT,
you can transfer data between host and device fields as needed.

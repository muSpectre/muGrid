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

    # Create a 1D FFT engine
    engine = muGrid.FFTEngine([nx])

    # Create a 2D FFT engine
    engine = muGrid.FFTEngine([nx, ny])

    # Create a 3D FFT engine
    engine = muGrid.FFTEngine([nx, ny, nz])

The FFT engine supports 1D, 2D, and 3D grids. Note that 1D FFT is serial-only
(no MPI parallelization), while 2D and 3D support MPI-parallel execution.

The FFT engine provides properties for querying the grid dimensions:

.. code-block:: python

    engine = muGrid.FFTEngine([16, 20])

    print(f"Real-space grid: {engine.nb_domain_grid_pts}")      # (16, 20)
    print(f"Fourier-space grid: {engine.nb_fourier_grid_pts}")  # (9, 20)
    print(f"Backend: {engine.backend_name}")                    # "pocketfft"

The Fourier-space grid has reduced dimensions due to the real-to-complex (r2c) transform
exploiting Hermitian symmetry.

Creating FFT fields
*******************

The FFT engine manages two field collections: one for real-space fields and one for
Fourier-space fields. Use the engine's methods to create fields:

.. code-block:: python

    import muGrid

    engine = muGrid.FFTEngine([16, 20])

    # Create real-space field (real-valued)
    real_field = engine.real_space_field("real")
    print(f"Real field shape: {real_field.s.shape}")  # (1, 1, 16, 20)

    # Create Fourier-space field (complex-valued)
    fourier_field = engine.fourier_space_field("fourier")
    print(f"Fourier field shape: {fourier_field.s.shape}")  # (1, 1, 9, 20)

Forward and inverse transforms
******************************

The ``fft`` method performs the forward transform (real to Fourier space) and
``ifft`` performs the inverse transform (Fourier to real space):

.. literalinclude:: ../../examples/fft_roundtrip.py
    :language: python

1D FFT
******

For 1D problems (e.g., time series analysis, 1D signal processing), create
a 1D FFT engine by passing a single-element list:

.. code-block:: python

    import numpy as np
    import muGrid

    # Create 1D FFT engine
    N = 64
    engine = muGrid.FFTEngine([N])

    # Create fields
    real_field = engine.real_space_field("signal")
    fourier_field = engine.fourier_space_field("spectrum")

    # Initialize with a test signal (e.g., sine wave)
    x = np.linspace(0, 1, N, endpoint=False)
    real_field.p[:] = np.sin(2 * np.pi * 3 * x)  # 3 Hz sine wave

    # Forward FFT
    engine.fft(real_field, fourier_field)

    # The Fourier field has N/2+1 complex values due to Hermitian symmetry
    print(f"Fourier shape: {fourier_field.p.shape}")  # (33,) for N=64

    # Inverse FFT and normalize
    engine.ifft(fourier_field, real_field)
    real_field.p[:] *= engine.normalisation

The 1D FFT output matches NumPy's ``fft.rfft``:

.. code-block:: python

    # Compare with NumPy
    data = np.random.randn(64)
    real_field.p[:] = data
    engine.fft(real_field, fourier_field)

    numpy_result = np.fft.rfft(data)
    np.testing.assert_allclose(fourier_field.p.flatten(), numpy_result)

Note that 1D FFT is **serial-only** and will raise an error if used with
multiple MPI ranks.

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

Frequency and coordinate arrays
*******************************

The FFT engine provides properties for accessing frequency and coordinate arrays
that are correctly shaped for the local subdomain. This is particularly useful
for MPI-parallel computations where each rank only handles a portion of the grid.

Frequency arrays
================

The ``fftfreq`` and ``ifftfreq`` properties provide FFT frequency arrays:

.. code-block:: python

    import muGrid
    import numpy as np

    engine = muGrid.FFTEngine([7, 4])

    # Normalized frequencies (range [-0.5, 0.5))
    qx, qy = engine.fftfreq
    print(f"fftfreq shape: {engine.fftfreq.shape}")  # (2, 4, 4) for r2c transform

    # Integer frequency indices
    iqx, iqy = engine.ifftfreq
    print(f"ifftfreq dtype: {iqx.dtype}")  # integer type

    # These match numpy's frequency arrays (sliced for r2c transform)
    freq_ref = np.array(np.meshgrid(
        *(np.fft.fftfreq(n) for n in [7, 4]), indexing="ij"
    ))
    freq_ref = freq_ref[:, :4, :]  # Slice for half-complex
    np.testing.assert_allclose(engine.fftfreq, freq_ref)

The frequency arrays have shape ``[dim, local_fx, local_fy, ...]`` where the first
axis indexes the spatial dimension and the remaining axes match the local Fourier
subdomain dimensions.

Coordinate arrays
=================

The ``coords`` and ``icoords`` properties provide real-space coordinate arrays:

.. code-block:: python

    import muGrid

    engine = muGrid.FFTEngine([7, 4])

    # Normalized coordinates (range [0, 1))
    x, y = engine.coords
    print(f"coords shape: {engine.coords.shape}")  # (2, 7, 4)

    # Integer coordinate indices
    ix, iy = engine.icoords
    print(f"icoords dtype: {ix.dtype}")  # integer type

    # Verify: integer coords = fractional * n
    np.testing.assert_allclose(ix, x * 7)
    np.testing.assert_allclose(iy, y * 4)

For fields with ghost regions, use ``coordsg`` and ``icoordsg`` to get coordinates
including the ghost cells:

.. code-block:: python

    engine = muGrid.FFTEngine(
        [64, 64],
        nb_ghosts_left=[1, 1],
        nb_ghosts_right=[1, 1]
    )

    # Coordinates without ghosts
    x, y = engine.coords

    # Coordinates with ghosts (larger array)
    xg, yg = engine.coordsg
    print(f"Without ghosts: {engine.coords.shape}")   # (2, local_nx, local_ny)
    print(f"With ghosts: {engine.coordsg.shape}")     # (2, local_nx+2, local_ny+2)

MPI-parallel frequency arrays
=============================

In MPI-parallel runs, frequency and coordinate arrays return only the data for the
local subdomain owned by each rank:

.. code-block:: python

    from mpi4py import MPI
    import muGrid

    comm = muGrid.Communicator(MPI.COMM_WORLD)
    engine = muGrid.FFTEngine([128, 128], comm)

    # Each rank gets frequencies for its local Fourier subdomain
    qx, qy = engine.fftfreq
    print(f"Rank {MPI.COMM_WORLD.rank}: fftfreq shape = {engine.fftfreq.shape}")

    # Coordinates for local real-space subdomain
    x, y = engine.coords
    print(f"Rank {MPI.COMM_WORLD.rank}: coords shape = {engine.coords.shape}")

Example: Fourier-space operations
=================================

A common pattern is using frequency arrays to construct Fourier-space operators:

.. code-block:: python

    import numpy as np
    import muGrid

    engine = muGrid.FFTEngine([64, 64])
    qx, qy = engine.fftfreq

    real_field = engine.real_space_field("u")
    fourier_field = engine.fourier_space_field("u_hat")

    # Initialize with a cosine wave
    x, y = engine.coords
    real_field.p[0] = np.cos(2 * np.pi * x)

    # Forward FFT
    engine.fft(real_field, fourier_field)

    # Compute Laplacian in Fourier space: -4π²(qx² + qy²) * u_hat
    laplacian_hat = -4 * np.pi**2 * (qx**2 + qy**2) * fourier_field.p[0]

    # Or use integer frequencies for identifying specific modes
    iqx, iqy = engine.ifftfreq
    # Find the (1, 0) mode
    mode_mask = (np.abs(iqx) == 1) & (iqy == 0)

Hermitian grid dimensions
*************************

For real-to-complex transforms, the output has reduced dimensions due to Hermitian
symmetry. Use ``get_hermitian_grid_pts`` to compute the Fourier-space grid size:

.. code-block:: python

    import muGrid

    # 1D example
    fourier_1d = muGrid.get_hermitian_grid_pts([64])
    print(f"1D Fourier grid: {fourier_1d}")  # [33]

    # 2D example
    fourier_2d = muGrid.get_hermitian_grid_pts([64, 64])
    print(f"2D Fourier grid: {fourier_2d}")  # [33, 64]

    # 3D example
    fourier_3d = muGrid.get_hermitian_grid_pts([64, 64, 64])
    print(f"3D Fourier grid: {fourier_3d}")  # [33, 64, 64]

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

.. literalinclude:: ../../examples/fourier_derivative.py
    :language: python

Multi-component fields
**********************

FFT fields can have multiple components:

.. code-block:: python

    import muGrid

    engine = muGrid.FFTEngine([16, 20])

    # Create field with 3 components (e.g., velocity vector)
    velocity = engine.real_space_field("velocity", components=(3,))
    velocity_hat = engine.fourier_space_field("velocity_hat", components=(3,))

    print(f"Velocity shape: {velocity.s.shape}")      # (3, 1, 16, 20)
    print(f"Velocity_hat shape: {velocity_hat.s.shape}")  # (3, 1, 9, 20)

    # FFT transforms all components - fields are passed directly
    engine.fft(velocity, velocity_hat)
    engine.ifft(velocity_hat, velocity)

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
    real_field = engine.real_space_field("displacement", components=(3,))

    # Fourier-space fields have no ghosts (hard assumption)
    fourier_field = engine.fourier_space_field("displacement_k", components=(3,))

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

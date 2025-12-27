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

.. literalinclude:: ../../examples/fourier_derivative.py
    :language: python

Multi-component fields
**********************

FFT fields can have multiple components:

.. code-block:: python

    import muGrid

    engine = muGrid.FFTEngine([16, 20])

    # Create field with 3 components (e.g., velocity vector)
    velocity = engine.real_space_field("velocity", nb_components=3)
    velocity_hat = engine.fourier_space_field("velocity_hat", nb_components=3)

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

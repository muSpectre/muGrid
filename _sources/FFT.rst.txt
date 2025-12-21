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

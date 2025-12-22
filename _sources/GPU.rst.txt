GPU Computing
#############

*µ*\Grid supports GPU acceleration through CUDA (NVIDIA) and HIP (AMD ROCm) backends.
Fields can be allocated on either host (CPU) or device (GPU) memory, and operations
on GPU fields are executed on the GPU.

Checking GPU availability
*************************

Before using GPU features, you can check if *µ*\Grid was compiled with GPU support:

.. code-block:: python

    import muGrid

    print(f"CUDA available: {muGrid.has_cuda}")
    print(f"ROCm available: {muGrid.has_rocm}")
    print(f"Any GPU backend: {muGrid.has_gpu}")

The ``has_gpu`` flag is ``True`` if either CUDA or ROCm support is available.

Memory locations
****************

Field collections can be created with a specific memory location using a string:

* ``"host"``: Allocate fields in CPU memory (default)
* ``"device"``: Allocate fields in GPU memory

Here is an example of creating a field collection on the GPU:

.. code-block:: python

    import muGrid

    # Create a GPU field collection
    fc = muGrid.GlobalFieldCollection([64, 64], memory_location="device")

    # Create a field on the GPU
    field = fc.real_field("my_field")
    print(f"Field is on GPU: {field.is_on_gpu}")
    print(f"Device: {field.device}")

Working with GPU arrays
***********************

When accessing GPU field data, the array views (``s``, ``p``, ``sg``, ``pg``) return
`CuPy <https://cupy.dev/>`_ arrays instead of numpy arrays. CuPy provides a numpy-compatible
API for GPU arrays:

.. code-block:: python

    import muGrid

    # Check GPU is available
    if not muGrid.has_gpu:
        raise RuntimeError("GPU support not available")

    import cupy as cp

    # Create GPU field collection
    fc = muGrid.GlobalFieldCollection([64, 64], memory_location="device")

    # Create fields
    field_a = fc.real_field("a")
    field_b = fc.real_field("b")

    # Initialize with CuPy (GPU) operations
    field_a.p[...] = cp.random.randn(*field_a.p.shape)
    field_b.p[...] = cp.sin(field_a.p) + cp.cos(field_a.p)

    # Compute on GPU
    result = cp.sum(field_b.p ** 2)
    print(f"Sum of squares: {result}")

For writing device-agnostic code, avoid importing `numpy` or `cupy` directly.
In the above example, execute the sum with:

.. code-block:: python

    result = (field_b.p ** 2).sum()

This may not always be possible. In this case, it may be useful to either import
`numpy` or `scipy` under the same module alias:

.. code-block:: python

    if memory == "host":
        import numpy as xp
    else:
        import cupy as xp

Zero-copy data exchange
***********************

*µ*\Grid uses the `DLPack <https://github.com/dmlc/dlpack>`_ protocol for zero-copy
data exchange between the C++ library and Python. This means:

* No data is copied when accessing field arrays
* Modifications to the array directly modify the underlying field data
* GPU data stays on the GPU

This is particularly important for performance when working with large arrays on GPUs.

CartesianDecomposition with GPU
*******************************

When using domain decomposition with ``CartesianDecomposition``, you can also
specify the memory location:

.. code-block:: python

    import muGrid

    # Create communicator (serial or MPI)
    comm = muGrid.Communicator()

    # Create decomposition on GPU
    decomp = muGrid.CartesianDecomposition(
        comm,
        nb_domain_grid_pts=[128, 128],
        nb_ghosts_left=[1, 1],
        nb_ghosts_right=[1, 1],
        memory_location="device"
    )

    # Create GPU field using real_field helper
    field = muGrid.real_field(decomp, "gpu_field")

    # Access coordinates (returned as CuPy arrays on GPU)
    x, y = decomp.coords

Installing CuPy
***************

To work with GPU fields, you need to install `CuPy <https://cupy.dev/>`_. The installation
depends on your CUDA version:

.. code-block:: sh

    # For CUDA 11.x
    pip install cupy-cuda11x

    # For CUDA 12.x
    pip install cupy-cuda12x

    # For ROCm
    pip install cupy-rocm-5-0  # or appropriate ROCm version

Convolution operators on GPU
****************************

Convolution operators can also operate on GPU fields. The convolution is performed
on the GPU when both input and output fields are on the GPU:

.. code-block:: python

    import numpy as np
    import muGrid

    if not muGrid.has_gpu:
        raise RuntimeError("GPU support not available")

    import cupy as cp

    # Create GPU field collection
    fc = muGrid.GlobalFieldCollection([64, 64], memory_location="device")

    # Create Laplacian stencil (defined on CPU as numpy array)
    stencil = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    laplace = muGrid.ConvolutionOperator([-1, -1], stencil)

    # Create fields
    input_field = fc.real_field("input")
    output_field = fc.real_field("output")

    # Initialize input (using CuPy)
    input_field.p[...] = cp.random.randn(*input_field.p.shape)

    # Apply convolution (executed on GPU) - fields are passed directly
    laplace.apply(input_field, output_field)

Performance considerations
**************************

GPU acceleration is most beneficial when:

* Working with large grids (the GPU parallelism outweighs data transfer overhead)
* Performing many operations on the same data (data stays on GPU)
* Using operations that parallelize well (element-wise operations, FFTs, convolutions)

For small grids or infrequent operations, the overhead of CPU-GPU data transfer may
outweigh the benefits of GPU computation.

Building with GPU support
*************************

To build *µ*\Grid with GPU support, enable the appropriate CMake option:

.. code-block:: sh

    # For CUDA
    cmake -DMUGRID_ENABLE_CUDA=ON ..

    # For ROCm/HIP
    cmake -DMUGRID_ENABLE_HIP=ON ..

You can also specify the GPU architectures to target:

.. code-block:: sh

    # CUDA architectures (e.g., 70=V100, 80=A100, 90=H100)
    cmake -DMUGRID_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="70;80;90" ..

    # HIP architectures (e.g., gfx906=MI50, gfx90a=MI200)
    cmake -DMUGRID_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="gfx906;gfx90a" ..

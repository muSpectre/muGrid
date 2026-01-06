Linear Algebra
##############

*µ*\Grid provides a set of linear algebra operations optimized for field data.
These operations work directly on muGrid fields, avoiding the overhead of
creating non-contiguous array views. The ``linalg`` module largely captures
Level 1 BLAS functionality, providing the fundamental building blocks for
iterative solvers and other numerical algorithms.

Overview
********

The linear algebra operations are designed with two key principles:

1. **MPI correctness**: Operations that compute scalar results (like dot products
   and norms) iterate only over interior pixels, excluding ghost regions. This
   prevents double-counting of values that are duplicated across MPI ranks.

2. **Performance**: Operations that modify field data (like ``axpy`` and ``scal``)
   operate on the full buffer including ghost regions. This is efficient because
   the memory is contiguous, and ghost values are typically overwritten by
   subsequent ghost communication anyway.

Available Operations
********************

The following operations are available in the ``muGrid.linalg`` namespace
(C++) or as functions in the Python bindings:

.. list-table:: Linear Algebra Operations
   :header-rows: 1
   :widths: 25 35 40

   * - Operation
     - Description
     - BLAS Equivalent
   * - ``vecdot(a, b)``
     - Vector dot product: :math:`\sum_i a_i b_i`
     - SDOT / DDOT
   * - ``axpy(alpha, x, y)``
     - :math:`y \leftarrow \alpha x + y`
     - SAXPY / DAXPY
   * - ``scal(alpha, x)``
     - :math:`x \leftarrow \alpha x`
     - SSCAL / DSCAL
   * - ``axpby(alpha, x, beta, y)``
     - :math:`y \leftarrow \alpha x + \beta y`
     - (extended BLAS)
   * - ``copy(src, dst)``
     - :math:`\text{dst} \leftarrow \text{src}`
     - SCOPY / DCOPY
   * - ``norm_sq(x)``
     - Squared norm: :math:`\sum_i x_i^2`
     - SNRM2² / DNRM2²
   * - ``axpy_norm_sq(alpha, x, y)``
     - Fused: :math:`y \leftarrow \alpha x + y`, returns :math:`\|y\|^2`
     - (fused operation)

Ghost Region Handling
*********************

Understanding how ghost regions are handled is important for MPI-parallel codes:

**Scalar-producing operations** (``vecdot``, ``norm_sq``, ``axpy_norm_sq``):
  These iterate only over interior pixels. Ghost values are excluded because
  they are duplicates of values owned by neighboring MPI ranks. The returned
  scalar is a local result that must be MPI-reduced if a global result is needed.

**Field-modifying operations** (``axpy``, ``scal``, ``axpby``, ``copy``):
  These operate on the full buffer including ghost regions. This is more
  efficient than iterating only over interior pixels, and ghost values will
  be overwritten by subsequent ``communicate_ghosts()`` calls anyway.

Usage Examples
**************

C++ Usage
---------

.. code-block:: cpp

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

Python Usage
------------

The linear algebra operations are available through the ``muGrid.linalg``
module:

.. code-block:: python

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

    # Squared norm
    norm2 = la.norm_sq(x)

MPI-Parallel Example
--------------------

For MPI-parallel computations, scalar results must be reduced across ranks:

.. code-block:: python

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

Fused Operations
****************

The ``axpy_norm_sq`` function provides a fused operation that computes both
an AXPY update and the squared norm of the result in a single pass through
memory. This is more efficient than separate ``axpy`` + ``norm_sq`` calls:

.. code-block:: python

    # Separate operations (less efficient):
    # - 2 reads of x, 2 reads of y, 1 write of y
    la.axpy(alpha, x, y)
    norm2 = la.norm_sq(y)

    # Fused operation (more efficient):
    # - 1 read of x, 1 read of y, 1 write of y
    norm2 = la.axpy_norm_sq(alpha, x, y)

This optimization is particularly valuable in iterative solvers where
memory bandwidth is often the limiting factor.

GPU Support
***********

All linear algebra operations support GPU fields (CUDA and ROCm). The
operations automatically use GPU-optimized kernels when the input fields
reside in device memory:

.. code-block:: python

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

The GPU kernels are optimized for coalesced memory access and use
efficient parallel reduction algorithms for scalar-producing operations.

Integration with Solvers
************************

The linear algebra module provides the core operations used by iterative
solvers in ``muGrid.Solvers``. For example, the conjugate gradient solver
uses these operations internally:

.. code-block:: python

    from muGrid.Solvers import conjugate_gradients

    # The CG solver uses vecdot, axpy, scal, etc. internally
    x = conjugate_gradients(comm, fc, b, x0, hessp=apply_operator)

If you are implementing custom solvers, using the ``muGrid.linalg``
operations ensures efficient execution on both CPU and GPU, with proper
handling of ghost regions for MPI parallelism.

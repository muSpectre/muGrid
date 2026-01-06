Code Architecture
#################

This section describes the internal design and implementation of key *µ*\Grid
components, including the FFT engine and discrete differential operators.

.. contents:: Table of Contents
   :local:
   :depth: 2

FFT Engine Design
*****************

The FFT engine provides distributed Fast Fourier Transform operations on
structured grids with MPI parallelization. It uses pencil decomposition for
efficient scaling to large numbers of MPI ranks.

Storage Order
=============

Array of Structures (AoS) vs Structure of Arrays (SoA)
------------------------------------------------------

*µ*\Grid supports two storage orders for multi-component fields:

**AoS (Array of Structures)**: Components are interleaved per pixel

- Memory layout: ``[c0p0, c1p0, c2p0, c0p1, c1p1, c2p1, ...]``
- Used on **CPU** (default for ``HostSpace``)
- Stride between consecutive X elements = ``nb_components``
- Component offset = ``comp`` (component index)

**SoA (Structure of Arrays)**: Components are stored in separate contiguous blocks

- Memory layout: ``[c0p0, c0p1, c0p2, ..., c1p0, c1p1, c1p2, ...]``
- Used on **GPU** (default for ``CUDASpace`` and ``ROCmSpace``)
- Stride between consecutive X elements = 1
- Component offset = ``comp * nb_pixels``

Why Different Storage Orders?
-----------------------------

**GPU (SoA)**: Coalesced memory access is critical for GPU performance. When
threads access consecutive memory locations, the hardware can combine multiple
accesses into a single transaction. SoA ensures that threads processing different
pixels of the same component access contiguous memory.

**CPU (AoS)**: Cache locality for multi-component operations. When processing a
single pixel, all its components are in the same cache line.

**Design Decision**: FFT work buffers use the same storage order as the memory
space they reside in (SoA on GPU, AoS on CPU). This avoids expensive storage
order conversions during FFT operations. The FFT engine detects the storage
order of input/output fields using ``field.get_storage_order()`` and computes
the appropriate strides for each buffer.

Stride Calculations
===================

2D Case
-------

For a 2D field with dimensions ``[Nx, Ny]`` and ``nb_components`` components:

**SoA strides:**

.. code-block:: text

    comp_offset_factor = nb_pixels        // Component i starts at i * nb_pixels
    x_stride = 1                          // Consecutive X elements are contiguous
    row_dist = row_width                  // Distance between rows (Y direction)

**AoS strides:**

.. code-block:: text

    comp_offset_factor = 1                // Component i starts at offset i
    x_stride = nb_components              // Skip over all components between X elements
    row_dist = row_width * nb_components  // Distance between rows includes all components

3D Case
-------

For a 3D field with dimensions ``[Nx, Ny, Nz]``:

**SoA strides:**

.. code-block:: text

    comp_offset_factor = nb_pixels
    x_stride = 1
    y_dist = row_x
    z_dist = row_x * rows_y

**AoS strides:**

.. code-block:: text

    comp_offset_factor = 1
    x_stride = nb_components
    y_dist = row_x * nb_components
    z_dist = row_x * rows_y * nb_components

GPU Backend Limitations
=======================

cuFFT Strided R2C/C2R Limitation
--------------------------------

cuFFT has a documented limitation: **strided real-to-complex (R2C) and
complex-to-real (C2R) transforms are not supported**. The stride on the real
data side must be 1.

This affects multi-component FFTs because with AoS storage, the stride between
consecutive real values would be ``nb_components`` (not 1).

**Solution**: With SoA storage order on GPU, each component's data is contiguous
(stride = 1), which satisfies cuFFT's requirement. The FFT engine loops over
components, executing one batched FFT per component.

rocFFT Native API
-----------------

Unlike cuFFT, rocFFT's native API (``rocfft_plan_description_set_data_layout()``)
supports arbitrary strides for both input and output. However, for consistency
and to avoid storage order conversions, we use the same SoA approach on AMD GPUs.

MPI Parallel FFT Design
=======================

Pencil Decomposition
--------------------

For 3D distributed FFT, the engine uses pencil decomposition:

1. **Z-pencil**: Data distributed in Y and Z, FFT along X (r2c)
2. **Y-pencil**: Transpose Y<->Z, FFT along Y (c2c)
3. **X-pencil**: Transpose X<->Z, FFT along Z (c2c)

Transpose Operations
--------------------

The ``Transpose`` class handles MPI all-to-all communication for redistributing
data between pencil orientations.

**Current limitation**: Transpose operations assume AoS layout. When using GPU
memory, this may require storage order conversion or updating the Transpose
class to handle SoA.

Communication Efficiency
------------------------

**Design Decision**: The FFT module is designed so that MPI communication does
not require explicit packing and unpacking. Data is laid out so that
``MPI_Alltoall`` can operate directly on contiguous memory regions.

Forward 2D FFT Algorithm
========================

.. code-block:: text

    1. r2c FFT along X for each component
       - Input: real field (with ghosts)
       - Output: work buffer (half-complex, no ghosts)
       - Loop over components, batched over Y rows

    2. [MPI only] Transpose X<->Y
       - Redistributes data across ranks
       - Changes from Y-local to X-local

    3. c2c FFT along Y for each component
       - In-place on work buffer (serial) or output (MPI)
       - Batched over X values

    4. [Serial only] Copy work to output
       - Same storage order, direct copy

Forward 3D FFT Algorithm
========================

.. code-block:: text

    1. r2c FFT along X for each component
       - Input: real field
       - Output: Z-pencil work buffer (half-complex in X)
       - Batched over Y rows for each Z plane

    2a. [MPI] Transpose Y<->Z
        - Z-pencil to Y-pencil

    2b. c2c FFT along Y for each component
        - On Y-pencil buffer
        - Individual transforms per (X, Z) position

    2c. [MPI] Transpose Z<->Y
        - Y-pencil back to Z-pencil

    3. [MPI] Transpose X<->Z (or copy)
       - Z-pencil to X-pencil (output)

    4. c2c FFT along Z for each component
       - On output buffer
       - Individual transforms per (X, Y) position

    5. [Serial only] Copy work to output

Per-Component Looping
=====================

Instead of attempting to batch across components (which would require non-unit
strides on GPU), the FFT engine loops over components:

.. code-block:: cpp

    for (Index_t comp = 0; comp < nb_components; ++comp) {
        Index_t in_comp_offset = comp * in_comp_factor;
        Index_t out_comp_offset = comp * work_comp_factor;
        backend->r2c(Nx, batch_size,
                     input_ptr + in_base + in_comp_offset,
                     in_x_stride, in_row_dist,
                     work_ptr + out_comp_offset,
                     work_x_stride, work_row_dist);
    }

This approach:

- Satisfies cuFFT's unit-stride requirement for R2C/C2R
- Allows efficient batching within each component
- Works correctly for both SoA and AoS storage orders
- Has minimal overhead (one kernel launch per component)

Normalization
=============

Like FFTW and cuFFT, the transforms are **unnormalized**. A forward FFT followed
by an inverse FFT multiplies the result by N (the transform size). Users must
explicitly normalize if needed.

FFT File Structure
==================

.. code-block:: text

    src/libmugrid/fft/
    ├── fft_engine.hh         # Template class FFTEngine<MemorySpace>
    ├── fft_engine_base.hh/cc # Base class with field collections and transpose management
    ├── fft_1d_backend.hh     # Abstract interface for 1D FFT backends
    ├── pocketfft_backend.cc  # CPU backend using pocketfft
    ├── cufft_backend.cc      # NVIDIA GPU backend using cuFFT
    ├── rocfft_backend.cc     # AMD GPU backend using rocFFT
    └── transpose.hh/cc       # MPI transpose operations

Operator Kernel Design
**********************

This section describes the design of the discrete differential operators in
*µ*\Grid, including their common design principles for ghost region handling
and periodicity.

Operator Overview
=================

*µ*\Grid provides several discrete differential operators optimized for structured grids:

.. list-table::
   :header-rows: 1
   :widths: 25 30 20 25

   * - Operator
     - Description
     - Stencil Size
     - Input -> Output
   * - ``GenericLinearOperator``
     - Flexible stencil-based convolution
     - Configurable
     - nodal -> quad or nodal -> nodal
   * - ``FEMGradientOperator``
     - FEM gradient (nodal -> quadrature)
     - 8 nodes (3D), 4 nodes (2D)
     - nodal -> quad
   * - ``LaplaceOperator``
     - 7-point (3D) or 5-point (2D) Laplacian
     - 7 (3D), 5 (2D)
     - nodal -> nodal
   * - ``IsotropicStiffnessOperator``
     - Fused elasticity kernel
     - 8 nodes (3D), 4 nodes (2D)
     - nodal -> nodal

Common Design Principles
========================

All operators follow these design principles for consistency and correctness:

Ghost Region Handling
---------------------

All operators make no assumption over boundary conditions and operate on an
interior region only. Distributed-memory parallelization and periodic boundary
conditions are implemented through ghost cell communication:

- **Ghost regions** are filled by ``CartesianDecomposition::communicate_ghosts()``
  before the operator is called
- For **periodic BC**: ghost regions contain copies from the opposite boundary
- For **non-periodic BC** with Dirichlet conditions: boundary node values are
  constrained, and their forces are not computed by the fused kernels (they
  would be overwritten anyway)
- **Interior node computation only**: For all BCs, operators compute forces only
  for interior nodes (e.g. indices 1 to n-2, assuming a ghost buffer of width 1),
  not boundary nodes (indices 0 and n-1)

Stencil Offset Convention
-------------------------

All operators use a stencil offset to define where the operator "centers":

.. code-block:: python

    # Example: FEMGradientOperator with offset [0, 0, 0]
    # Output element (0,0,0) uses input nodes at (0,0,0), (1,0,0), (0,1,0), etc.
    grad_op = muGrid.FEMGradientOperator(3, grid_spacing)

    # Example: GenericLinearOperator with custom offset
    conv_op = muGrid.GenericLinearOperator([0, 0, 0], coefficients)

The offset determines which element's quadrature points correspond to which
nodal region.

Upfront Validation
------------------

Ghost region requirements are validated **once** before kernel execution, not
during iteration:

- Operators check that sufficient ghost cells exist in the field collection
- Invalid configurations throw clear error messages
- No bounds checking inside the hot loop

Memory Layout
-------------

CPU operators use AoS, and GPU operators use SoA memory layouts:

- **CPU**: DOF components contiguous ``[ux0, uy0, uz0, ux1, uy1, uz1, ...]``
- **GPU**: Spatial indices contiguous ``[ux0, ux1, ..., uy0, uy1, ..., uz0, uz1, ...]``

The kernel implementations may differ between CPU and GPU as they may be
optimized for the respective target architecture.

GenericLinearOperator Design
============================

Purpose
-------

Flexible stencil-based convolution operator that can represent any linear
differential operator as a sparse stencil.

Design
------

- Takes a list of coefficient arrays, one per output component per stencil point
- Stencil defined by relative offsets from the current position
- Supports both forward (apply) and adjoint (transpose) operations

Ghost Requirements
------------------

The forward and transpose operations have different ghost requirements because
the stencil direction is reversed:

**Forward (apply)**: reads at position ``p + s`` (stencil offset s)

- Left ghosts: ``max(-offset, 0)`` in each dimension
- Right ghosts: ``max(stencil_shape - 1 + offset, 0)`` in each dimension

**Transpose**: reads at position ``p - s`` (reversed stencil direction)

- Left ghosts: ``max(stencil_shape - 1 + offset, 0)`` in each dimension (swapped from apply's right)
- Right ghosts: ``max(-offset, 0)`` in each dimension (swapped from apply's left)

**Example**: For a stencil with offset ``[0, 0, 0]`` and shape ``[2, 2, 2]``:

- Apply: left=0, right=1 (reads ahead)
- Transpose: left=1, right=0 (gathers from behind)

Usage
-----

.. code-block:: python

    # Create from FEMGradientOperator coefficients
    grad_op = muGrid.FEMGradientOperator(3, grid_spacing)
    conv_op = muGrid.GenericLinearOperator([0, 0, 0], grad_op.coefficients)

    # Apply gradient
    conv_op.apply(nodal_field, quadrature_field)

    # Apply divergence (transpose)
    conv_op.transpose(quadrature_field, nodal_field, quad_weights)

FEMGradientOperator Design
==========================

Purpose
-------

Computes gradients at quadrature points from nodal displacements using linear
finite element (P1) shape functions.

Mathematical Formulation
------------------------

For linear FEM on structured grids:

- **2D**: Each pixel decomposed into 2 triangles, 2 quadrature points
- **3D**: Each voxel decomposed into 5 tetrahedra (Kuhn triangulation), 5 quadrature points

The gradient at quadrature point q is:

.. math::

    \frac{\partial u}{\partial x_i} = \sum_I B_q[i,I] \times u_I / h_i

where :math:`B_q` contains shape function gradients for the element containing q.

Shape Function Gradients
------------------------

**2D (2 triangles per pixel):**

.. code-block:: text

    B_2D[triangle][dim][node]
    Triangle 0: nodes 0,1,2 (lower-left)
    Triangle 1: nodes 1,2,3 (upper-right)

**3D (5 tetrahedra per voxel):**

.. code-block:: text

    B_3D[tet][dim][node]
    Tet 0: central tetrahedron (weight 1/3)
    Tets 1-4: corner tetrahedra (weight 1/6 each)

Ghost Requirements
------------------

**Forward (apply): Gradient - nodal -> quadrature**

- Input (nodal field): 1 ghost cell on **right** in each dimension
- Output (quadrature field): no ghosts needed
- Iteration: over elements [0, n-2], reads nodal values at element corners
  [ix, ix+1] x [iy, iy+1] x [iz, iz+1]

**Transpose (transpose): Divergence - quadrature -> nodal**

- Input (quadrature/stress field): 1 ghost cell on **left** in each dimension (for periodic BC)
- Output (nodal field): no specific ghost requirement
- Iteration: over elements [0, n-2], accumulates contributions to nodal values
  at element corners

The ghost requirement swap occurs because:

- Forward reads "ahead" (node ix+1 when at element ix) -> needs right ghosts
- Transpose gathers from elements "behind" (element ix-1 contributes to node ix)
  -> needs left ghosts for periodic wraparound

LaplaceOperator Design
======================

Purpose
-------

Optimized discrete Laplacian for Poisson-type problems.

Mathematical Formulation
------------------------

Standard finite difference Laplacian:

.. math::

    \nabla^2 u \approx \frac{u[i+1] + u[i-1] - 2u[i]}{h^2} + \ldots

**2D (5-point stencil):**

.. code-block:: text

         1
       1 -4 1
         1

**3D (7-point stencil):**

.. code-block:: text

    Center: -6, Neighbors: +1 (x6)

Ghost Requirements
------------------

- 1 ghost cell in each direction (for accessing neighbors)

Properties
----------

- Self-adjoint: transpose equals forward apply
- Negative semi-definite (positive semi-definite with ``-scale``)
- Configurable scale factor for grid spacing and sign conventions

Usage
-----

.. code-block:: python

    # Create Laplacian with scale = -1/h^2 for positive-definite form
    laplace = muGrid.LaplaceOperator3D(scale=-1.0 / h**2)
    laplace.apply(u, laplacian_u)

IsotropicStiffnessOperator Design
=================================

Purpose
-------

Fused kernel computing ``force = K @ displacement`` for isotropic linear elastic
materials, avoiding explicit assembly of the stiffness matrix K.

Mathematical Formulation
------------------------

For isotropic materials, the element stiffness matrix decomposes as:

.. math::

    K_e = 2\mu G + \lambda V

where:

- :math:`G = \sum_q w_q B_q^T I' B_q` - geometry matrix (shear stiffness)
- :math:`V = \sum_q w_q (B_q^T m)(m^T B_q)` - volumetric coupling matrix
- :math:`I' = \text{diag}(1, 1, 1, 0.5, 0.5, 0.5)` - Voigt scaling for strain energy
- :math:`m = [1, 1, 1, 0, 0, 0]^T` - trace selector vector

This reduces memory from O(N x DOF^2) for full K to O(N x 2) for spatially-varying
materials.

Element Decomposition
---------------------

**2D:** 2 triangles per pixel

- Quadrature weights: ``[0.5, 0.5]``

**3D:** 5 tetrahedra per voxel (Kuhn triangulation)

- Quadrature weights: ``[1/3, 1/6, 1/6, 1/6, 1/6]``

Field Requirements
------------------

All fields use **node-based indexing** with the same grid dimensions:

.. list-table::
   :header-rows: 1
   :widths: 30 25 22 23

   * - Field
     - Dimensions
     - Left Ghosts
     - Right Ghosts
   * - Displacement
     - (nx, ny, nz)
     - 1
     - 1
   * - Force
     - (nx, ny, nz)
     - 1
     - 1
   * - Material (lam, mu)
     - (nx, ny, nz)
     - 1
     - 1

**Key design principle**: The kernel does not distinguish between periodic and
non-periodic boundary conditions. Ghost cell communication (via
``CartesianDecomposition::communicate_ghosts()``) handles periodicity:

- For **periodic BC**: ghost cells contain copies from the opposite boundary
- For **non-periodic BC**: ghost cells are filled with appropriate boundary values

This unified approach simplifies the kernel implementation and enables consistent
behavior across CPU and GPU.

Gather Pattern
--------------

The kernel iterates over **interior nodes** and gathers contributions from all
neighboring elements:

**2D (4 elements per node):**

.. code-block:: text

    Element offsets and local node indices:
      (-1,-1) -> local node 3    (0,-1) -> local node 2
      (-1, 0) -> local node 1    (0, 0) -> local node 0

**3D (8 elements per node):**

.. code-block:: text

    Element offsets and local node indices:
      (-1,-1,-1) -> local node 7    (0,-1,-1) -> local node 6
      (-1, 0,-1) -> local node 5    (0, 0,-1) -> local node 4
      (-1,-1, 0) -> local node 3    (0,-1, 0) -> local node 2
      (-1, 0, 0) -> local node 1    (0, 0, 0) -> local node 0

Iteration Bounds
----------------

The kernel iterates over all interior nodes (indices 0 to n-1):

.. code-block:: cpp

    // Iterate over all nodes - ghost communication handles periodicity
    for (ix = 0; ix < nnx; ix++):
        for (iy = 0; iy < nny; iy++):
            // Compute force at node (ix, iy)

The kernel reads from ghost cells at positions -1 (left) and n (right), which
must be populated before calling the operator. For periodic BC, these contain
copies from the opposite boundary. For non-periodic BC, these contain boundary
values (typically zero for homogeneous Dirichlet conditions).

Kernel Pseudocode
-----------------

.. code-block:: cpp

    for each node (ix, iy, iz) in [0, nnx) x [0, nny) x [0, nnz):
        f = [0, 0, 0]  // force accumulator

        for each neighboring element (8 in 3D):
            // Element indices can be -1 (left ghost) or nx (right ghost)
            ex, ey, ez = element indices relative to node
            local_node = which corner of element is this node

            // Material fields are node-indexed, read from element position
            // (element at ex,ey,ez has material stored at node ex,ey,ez)
            lam = material_lambda[ex, ey, ez]
            mu = material_mu[ex, ey, ez]

            // Gather displacement from all element nodes (may read ghosts)
            u = [displacement at each element corner node]

            // Compute stiffness contribution
            for each DOF d:
                row = local_node * NB_DOFS + d
                f[d] += sum_j (2*mu * G[row,j] + lam * V[row,j]) * u[j]

        force[ix, iy, iz] = f

Memory Layout Details
=====================

CPU Layout (Array of Structures)
--------------------------------

DOF components are contiguous in memory:

.. code-block:: text

    Memory: [ux0, uy0, uz0, ux1, uy1, uz1, ux2, uy2, uz2, ...]

Strides:

.. code-block:: cpp

    disp_stride_d = 1;                    // DOF components contiguous
    disp_stride_x = NB_DOFS;              // = 3 for 3D
    disp_stride_y = NB_DOFS * nx;
    disp_stride_z = NB_DOFS * nx * ny;

GPU Layout (Structure of Arrays)
--------------------------------

Spatial indices are contiguous for coalesced memory access:

.. code-block:: text

    Memory: [ux0, ux1, ux2, ..., uy0, uy1, uy2, ..., uz0, uz1, uz2, ...]

Strides:

.. code-block:: cpp

    disp_stride_x = 1;                    // Spatial x contiguous
    disp_stride_y = nx;
    disp_stride_z = nx * ny;
    disp_stride_d = nx * ny * nz;         // DOF components separated

Data Pointer Offset
-------------------

Pointers are offset to account for ghost cells:

.. code-block:: cpp

    const Real* disp_data = displacement.data() + ghost_offset;
    // Now index 0 = first interior node, index -1 = left ghost

Operator File Structure
=======================

.. code-block:: text

    src/libmugrid/operators/
    ├── linear.hh              # Base LinearOperator class
    ├── generic.hh/.cc         # GenericLinearOperator
    ├── laplace_2d.hh/.cc      # LaplaceOperator2D
    ├── laplace_3d.hh/.cc      # LaplaceOperator3D
    ├── fem_gradient_2d.hh/.cc # FEMGradientOperator2D
    ├── fem_gradient_3d.hh/.cc # FEMGradientOperator3D
    └── solids/
        ├── isotropic_stiffness_2d.hh/.cc  # IsotropicStiffnessOperator2D
        ├── isotropic_stiffness_3d.hh/.cc  # IsotropicStiffnessOperator3D
        └── isotropic_stiffness_gpu.cc     # GPU kernels

Testing
=======

Tests are in ``tests/python_isotropic_stiffness_operator_tests.py``:

- ``test_compare_with_generic_operator``: Compares fused kernel against explicit B^T C B
- ``test_unit_impulse_*``: Verifies response to unit displacement at specific nodes
- ``test_symmetry``: Verifies K is symmetric
- ``ValidationGuardTest*``: Verifies proper error handling for invalid configurations
- ``GPUUnitImpulseTest``: Verifies GPU kernel matches CPU kernel output

With node-based indexing, all tests compare the full output (all nodes), not
just interior nodes.

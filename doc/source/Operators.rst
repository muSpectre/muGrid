Linear Operators
################

*µ*\Grid provides a hierarchy of linear operators for discretizing partial
differential equations on structured grids. This chapter explains the different
operator types and when to use each one.

Overview
********

Linear operators in *µ*\Grid fall into three categories:

1. **Generic operators**: Flexible, user-defined stencils for arbitrary convolutions
2. **Gradient/divergence operators**: FEM-based operators for computing derivatives
3. **Fused operators**: Highly optimized operators for specific PDEs (Laplace, elasticity)

The general recommendation is to use fused operators when available, as they
provide the best performance. Generic operators are useful for prototyping or
implementing custom stencils.

.. list-table:: Operator Summary
   :header-rows: 1
   :widths: 25 35 40

   * - Operator
     - Use Case
     - Performance
   * - ``GenericLinearOperator``
     - Custom stencils, prototyping
     - Baseline
   * - ``FEMGradientOperator``
     - Gradient/divergence with FEM
     - 2-3× faster than generic
   * - ``LaplaceOperator``
     - Scalar Poisson problems
     - 2-4× faster than generic
   * - ``IsotropicStiffnessOperator``
     - Linear elasticity (isotropic)
     - 5-10× faster than unfused

Generic Linear Operators
************************

The ``GenericLinearOperator`` class (formerly ``StencilGradientOperator``)
implements arbitrary convolution stencils. It is the most flexible operator
type but also the slowest due to indirect memory access patterns.

Creating a generic operator
---------------------------

A generic operator requires:

1. **Stencil coefficients**: A numpy array of any shape containing the weights
2. **Offset**: The position of the stencil origin relative to array indices

Example: 5-point Laplacian in 2D:

.. code-block:: python

    import numpy as np
    import muGrid

    # 5-point Laplacian stencil
    h = 0.1  # Grid spacing
    stencil = np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ]) / h**2

    # Offset: stencil[0,0] corresponds to neighbor at (-1, -1)
    offset = [-1, -1]

    laplace = muGrid.GenericLinearOperator(offset, stencil)

Multi-component stencils
------------------------

For operators with multiple input/output components (like gradients), use
higher-dimensional coefficient arrays:

.. code-block:: python

    # 2D gradient stencil: 2 output components, 1 input, 2×2 stencil
    # Shape: [nb_output, nb_quad, nb_input, stencil_x, stencil_y]
    gradient_coeffs = np.zeros((2, 1, 1, 2, 2))

    # ∂/∂x: forward difference
    gradient_coeffs[0, 0, 0, 0, 0] = -1/h
    gradient_coeffs[0, 0, 0, 1, 0] = +1/h

    # ∂/∂y: forward difference
    gradient_coeffs[1, 0, 0, 0, 0] = -1/h
    gradient_coeffs[1, 0, 0, 0, 1] = +1/h

    gradient_op = muGrid.GenericLinearOperator([0, 0], gradient_coeffs)

Using generic operators
-----------------------

.. code-block:: python

    # Create field collection with ghost regions
    decomposition = muGrid.CartesianDecomposition(
        comm, (64, 64),
        nb_ghosts_left=(1, 1),
        nb_ghosts_right=(1, 1),
    )

    input_field = decomposition.real_field("input")
    output_field = decomposition.real_field("output")

    # Fill ghost regions before applying operator
    decomposition.communicate_ghosts(input_field)

    # Apply the operator
    laplace.apply(input_field, output_field)

FEM Gradient Operator
*********************

The ``FEMGradientOperator`` computes gradients using linear finite element
shape functions. Each pixel/voxel is subdivided into simplicial elements
(triangles in 2D, tetrahedra in 3D) with multiple quadrature points.

Properties
----------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Dimension
     - Elements
     - Quadrature Points
   * - 2D
     - 2 triangles per pixel
     - 2 (one per triangle)
   * - 3D
     - 5 tetrahedra per voxel
     - 5 (one per tetrahedron)

Creating the operator
---------------------

.. code-block:: python

    # Grid spacing (can be non-uniform)
    grid_spacing = (0.1, 0.1)  # 2D
    # grid_spacing = (0.1, 0.1, 0.1)  # 3D

    gradient_op = muGrid.FEMGradientOperator(dim=2, grid_spacing=grid_spacing)

    # Query operator properties
    print(f"Quadrature points: {gradient_op.nb_quad_pts}")
    print(f"Quadrature weights: {gradient_op.quadrature_weights}")

Gradient operation
------------------

The forward operation computes gradients at quadrature points:

.. code-block:: python

    # Input: scalar or vector field at nodes
    # Output: gradient tensor at quadrature points

    # For a displacement field u with shape [dim, nx, ny]:
    # gradient_op.apply(u, grad_u) produces grad_u with shape [dim, dim, quad, nx, ny]

    decomposition.communicate_ghosts(u_field)
    gradient_op.apply(u_field, gradient_field)

Divergence (transpose) operation
--------------------------------

The transpose operation computes the (negative) divergence, weighted by
quadrature weights:

.. code-block:: python

    # Input: tensor field at quadrature points
    # Output: vector field at nodes

    gradient_op.transpose(stress_field, force_field, quad_weights)

The quadrature weights are typically the element volumes divided by the number
of quadrature points per element.

Laplace Operator
****************

The ``LaplaceOperator`` provides an optimized implementation of the discrete
Laplacian using the standard 5-point (2D) or 7-point (3D) stencil.

.. code-block:: python

    # Create Laplacian with scaling factor
    # Negative scale makes the operator positive-definite (for CG solver)
    scale = -1.0 / h**2
    laplace = muGrid.LaplaceOperator(dim=2, scale=scale)

    # Apply to fields
    decomposition.communicate_ghosts(u_field)
    laplace.apply(u_field, result_field)

The hard-coded implementation is significantly faster than an equivalent
``GenericLinearOperator`` because:

- Memory access patterns are predictable
- The compiler can vectorize the inner loops (SIMD)
- GPU kernels are highly optimized

Isotropic Stiffness Operator
****************************

The ``IsotropicStiffnessOperator2D`` and ``IsotropicStiffnessOperator3D``
classes implement fused stiffness operators for isotropic linear elastic
materials. They compute:

.. math::

    \mathbf{f} = \mathbf{K} \mathbf{u} = \mathbf{B}^T \mathbf{C} \mathbf{B} \mathbf{u}

where:

- :math:`\mathbf{u}` is the displacement field
- :math:`\mathbf{B}` is the strain-displacement matrix (gradient operator)
- :math:`\mathbf{C}` is the material stiffness tensor
- :math:`\mathbf{f}` is the internal force vector

Mathematical formulation
------------------------

For isotropic materials, the stiffness tensor :math:`\mathbf{C}` depends only
on two Lamé parameters:

- :math:`\lambda` (first Lamé parameter, related to bulk modulus)
- :math:`\mu` (shear modulus)

The element stiffness matrix decomposes as:

.. math::

    \mathbf{K}_e = 2\mu \mathbf{G} + \lambda \mathbf{V}

where :math:`\mathbf{G}` and :math:`\mathbf{V}` are geometry-only matrices
computed once at construction time. This decomposition enables:

1. **Memory efficiency**: Store only 2 scalars per element instead of a full stiffness matrix
2. **Computational efficiency**: Avoid explicit matrix assembly and storage

Creating the operator
---------------------

.. code-block:: python

    # 2D operator
    grid_spacing = (0.1, 0.1)
    stiffness_op_2d = muGrid.IsotropicStiffnessOperator2D(grid_spacing)

    # 3D operator
    grid_spacing = (0.1, 0.1, 0.1)
    stiffness_op_3d = muGrid.IsotropicStiffnessOperator3D(grid_spacing)

Material fields
---------------

The operator requires two material fields containing the Lamé parameters at
each element (pixel/voxel):

.. code-block:: python

    # Create decomposition for element-based fields (one fewer grid point in each direction)
    # because elements are defined between nodes
    element_grid_pts = tuple(n - 1 for n in nb_grid_pts)

    element_decomposition = muGrid.CartesianDecomposition(
        comm, element_grid_pts,
        nb_subdivisions=subdivisions,
        nb_ghosts_left=(1,) * dim,
        nb_ghosts_right=(1,) * dim,
    )

    # Create material fields with shape [nx-1, ny-1] (2D) or [nx-1, ny-1, nz-1] (3D)
    lambda_field = element_decomposition.real_field("lambda")
    mu_field = element_decomposition.real_field("mu")

Computing Lamé parameters from engineering constants
----------------------------------------------------

.. code-block:: python

    def lame_parameters(E, nu):
        """
        Compute Lamé parameters from Young's modulus E and Poisson's ratio nu.

        Parameters
        ----------
        E : float
            Young's modulus
        nu : float
            Poisson's ratio

        Returns
        -------
        lam : float
            First Lamé parameter (λ)
        mu : float
            Shear modulus (μ)
        """
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        return lam, mu

    # Example: heterogeneous material with matrix and inclusions
    E_matrix, E_inclusion = 1.0, 10.0
    nu = 0.3

    lam_matrix, mu_matrix = lame_parameters(E_matrix, nu)
    lam_inclusion, mu_inclusion = lame_parameters(E_inclusion, nu)

    # Set material properties based on phase field (0 = matrix, 1 = inclusion)
    lambda_field.p[...] = lam_matrix * (1 - phase) + lam_inclusion * phase
    mu_field.p[...] = mu_matrix * (1 - phase) + mu_inclusion * phase

    # Fill ghost regions (only needs to be done once)
    element_decomposition.communicate_ghosts(lambda_field)
    element_decomposition.communicate_ghosts(mu_field)

Applying the operator
---------------------

.. code-block:: python

    # Displacement and force fields on node-based grid
    # Shape: [dim, nx, ny] for 2D, [dim, nx, ny, nz] for 3D
    u_field = decomposition.real_field("displacement", (dim,))
    f_field = decomposition.real_field("force", (dim,))

    # Apply stiffness operator: f = K @ u
    decomposition.communicate_ghosts(u_field)
    stiffness_op.apply(u_field, lambda_field, mu_field, f_field)

    # Increment form: f += alpha * K @ u
    stiffness_op.apply_increment(u_field, lambda_field, mu_field, alpha, f_field)

Complete linear elasticity example
----------------------------------

.. code-block:: python

    import numpy as np
    import muGrid
    from muGrid.Solvers import conjugate_gradients

    # Parameters
    nb_grid_pts = (32, 32)
    dim = 2
    E_matrix, E_inclusion = 1.0, 10.0
    nu = 0.3

    # Compute Lamé parameters
    lam_matrix = E_matrix * nu / ((1 + nu) * (1 - 2 * nu))
    mu_matrix = E_matrix / (2 * (1 + nu))
    lam_inclusion = E_inclusion * nu / ((1 + nu) * (1 - 2 * nu))
    mu_inclusion = E_inclusion / (2 * (1 + nu))

    # Grid spacing
    grid_spacing = tuple(1.0 / n for n in nb_grid_pts)

    # Create operator
    stiffness_op = muGrid.IsotropicStiffnessOperator2D(grid_spacing)

    # Domain decomposition for nodal fields
    comm = muGrid.Communicator()
    decomposition = muGrid.CartesianDecomposition(
        comm, nb_grid_pts,
        nb_ghosts_left=(1, 1),
        nb_ghosts_right=(1, 1),
    )

    # Domain decomposition for element fields
    element_decomposition = muGrid.CartesianDecomposition(
        comm, tuple(n - 1 for n in nb_grid_pts),
        nb_ghosts_left=(1, 1),
        nb_ghosts_right=(1, 1),
    )

    # Create fields
    u_field = decomposition.real_field("displacement", (dim,))
    f_field = decomposition.real_field("force", (dim,))
    lambda_field = element_decomposition.real_field("lambda")
    mu_field = element_decomposition.real_field("mu")

    # Set up material (circular inclusion)
    coords = element_decomposition.coords
    X, Y = coords[0], coords[1]
    phase = ((X - 0.5)**2 + (Y - 0.5)**2 < 0.25**2).astype(float)

    lambda_field.p[...] = lam_matrix * (1 - phase) + lam_inclusion * phase
    mu_field.p[...] = mu_matrix * (1 - phase) + mu_inclusion * phase
    element_decomposition.communicate_ghosts(lambda_field)
    element_decomposition.communicate_ghosts(mu_field)

    # Stiffness operator wrapper for CG solver
    def apply_stiffness(u, f):
        decomposition.communicate_ghosts(u)
        stiffness_op.apply(u, lambda_field, mu_field, f)

    # Solve (with appropriate RHS setup for homogenization...)
    # conjugate_gradients(comm, decomposition, rhs, u_field, hessp=apply_stiffness, ...)

Performance Comparison
**********************

The fused ``IsotropicStiffnessOperator`` provides significant performance
advantages over manually computing :math:`\mathbf{B}^T \mathbf{C} \mathbf{B}`:

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Approach
     - Relative Speed
     - Memory per Element
   * - Full stiffness matrix
     - 1× (baseline)
     - 576 floats (3D)
   * - Separate B, C, B^T ops
     - 2-3×
     - Full C tensor + intermediates
   * - Fused isotropic operator
     - 5-10×
     - 2 floats (λ, μ)

The speedup comes from:

1. **Reduced memory traffic**: Only read 2 material values per element instead of full tensors
2. **No intermediate storage**: Strain and stress computed on-the-fly
3. **Optimized kernels**: Hand-tuned CPU loops and GPU kernels
4. **Better cache utilization**: Predictable access patterns

GPU Performance
---------------

On GPUs, the fused operators show even greater advantages:

- Atomic-free implementation using gather patterns
- Shared memory optimization for cooperative node loading
- High occupancy due to low register pressure

Typical GPU speedups are 5-10× over unfused approaches on modern NVIDIA and AMD GPUs.

When to Use Each Operator
*************************

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Operator
     - Recommended Use
   * - ``GenericLinearOperator``
     - Prototyping, custom stencils, uncommon PDEs
   * - ``FEMGradientOperator``
     - Computing gradients/divergence, anisotropic materials
   * - ``LaplaceOperator``
     - Scalar Poisson problems, diffusion equations
   * - ``IsotropicStiffnessOperator``
     - Linear elasticity with isotropic materials

For production code solving standard PDEs, always prefer the fused operators
when available. They provide the best performance while maintaining numerical
accuracy identical to the unfused approach.

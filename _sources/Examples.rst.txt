Examples
########

This chapter provides step-by-step examples for building numerical solvers with
*µ*\Grid. We present two complete examples: a Poisson solver and a linear
elasticity solver for micromechanical homogenization.

.. contents:: Table of Contents
   :local:
   :depth: 2

Poisson Solver
**************

The Poisson equation is a fundamental PDE that appears in many physical contexts
(heat conduction, electrostatics, etc.). We solve:

.. math::

    -\nabla^2 u = f

with periodic boundary conditions on a unit domain.

Setting up the grid
-------------------

First, we import the necessary modules and set up a 2D grid with ghost regions
for the stencil operations:

.. code-block:: python

    import numpy as np
    import muGrid
    from muGrid.Solvers import conjugate_gradients

    # Create a communicator (serial execution)
    comm = muGrid.Communicator()

    # Grid parameters
    nb_grid_pts = [64, 64]
    dim = len(nb_grid_pts)

    # Ghost layers: 1 cell on each side for the 5-point stencil
    nb_ghosts_left = [1, 1]
    nb_ghosts_right = [1, 1]

    # Create the domain decomposition
    # (works for both serial and MPI-parallel execution)
    decomposition = muGrid.CartesianDecomposition(
        comm,
        nb_domain_grid_pts=nb_grid_pts,
        nb_ghosts_left=nb_ghosts_left,
        nb_ghosts_right=nb_ghosts_right,
    )

Creating the Laplacian operator
-------------------------------

*µ*\Grid provides two ways to create discrete operators:

1. **Generic convolution operators**: Flexible, user-defined stencils
2. **Hard-coded operators**: Optimized implementations for common operators

**Generic convolution operator:**

The 5-point Laplacian stencil in 2D is:

.. math::

    \nabla^2 u \approx \frac{u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1} - 4u_{i,j}}{h^2}

We create this with a ``ConvolutionOperator``:

.. code-block:: python

    # Grid spacing (assuming unit domain)
    h = 1.0 / nb_grid_pts[0]

    # Scale factor: negative because -∇² must be positive definite for CG
    scale = -1.0 / h**2

    # 5-point Laplacian stencil
    stencil = scale * np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ])
    stencil_offset = [-1, -1]  # Stencil origin relative to center

    laplace_generic = muGrid.ConvolutionOperator(stencil_offset, stencil)

**Hard-coded Laplacian operator:**

For better performance, use the optimized ``LaplaceOperator``:

.. code-block:: python

    laplace_hardcoded = muGrid.LaplaceOperator(dim, scale)

Both operators have the same interface (``apply`` method) and produce identical
results, but the hard-coded version is significantly faster (see
:ref:`performance-comparison` below).

Creating fields and setting up the RHS
--------------------------------------

.. code-block:: python

    # Create fields using the decomposition
    rhs = decomposition.real_field("rhs")
    solution = decomposition.real_field("solution")

    # Set up a smooth right-hand side
    # Get coordinates for each pixel in the local domain
    x = np.linspace(0, 1, nb_grid_pts[0], endpoint=False)
    y = np.linspace(0, 1, nb_grid_pts[1], endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    rhs.p[...] = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)

    # Remove mean (necessary for periodic Poisson with Neumann-like conditions)
    rhs.p[...] -= np.mean(rhs.p)

Solving with conjugate gradients
--------------------------------

The conjugate gradient solver requires a function that applies the linear operator:

.. code-block:: python

    # Choose which operator to use
    laplace = laplace_hardcoded  # or laplace_generic

    def apply_laplacian(x, Ax):
        """Apply the Laplacian operator: Ax = L @ x"""
        # Fill ghost regions with periodic boundary values
        decomposition.communicate_ghosts(x)
        # Apply the stencil
        laplace.apply(x, Ax)

    # Solve the system
    conjugate_gradients(
        comm,
        decomposition,
        apply_laplacian,
        rhs,
        solution,
        tol=1e-6,
        maxiter=1000,
    )

    print(f"Solution range: [{solution.p.min():.6f}, {solution.p.max():.6f}]")

Complete Poisson solver
-----------------------

Here is the complete, minimal Poisson solver:

.. code-block:: python

    import numpy as np
    import muGrid
    from muGrid.Solvers import conjugate_gradients

    # Setup
    comm = muGrid.Communicator()
    nb_grid_pts = [64, 64]
    h = 1.0 / nb_grid_pts[0]

    decomposition = muGrid.CartesianDecomposition(
        comm,
        nb_domain_grid_pts=nb_grid_pts,
        nb_ghosts_left=[1, 1],
        nb_ghosts_right=[1, 1],
    )

    # Laplacian operator (negative for positive-definiteness)
    laplace = muGrid.LaplaceOperator(2, -1.0 / h**2)

    # Fields
    rhs = decomposition.real_field("rhs")
    solution = decomposition.real_field("solution")

    # RHS: smooth function with zero mean
    x = np.linspace(0, 1, nb_grid_pts[0], endpoint=False)
    y = np.linspace(0, 1, nb_grid_pts[1], endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    rhs.p[...] = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    rhs.p[...] -= np.mean(rhs.p)

    # Linear operator for CG
    def apply_laplacian(x, Ax):
        decomposition.communicate_ghosts(x)
        laplace.apply(x, Ax)

    # Solve
    conjugate_gradients(comm, decomposition, apply_laplacian, rhs, solution,
                        tol=1e-6, maxiter=1000)

    print(f"Solved! Solution range: [{solution.p.min():.4f}, {solution.p.max():.4f}]")

Linear Elasticity Solver
************************

This example computes the effective elastic properties of a heterogeneous
material using FEM-based homogenization. The governing equation is:

.. math::

    \nabla \cdot \boldsymbol{\sigma} = 0

where the stress is related to strain by Hooke's law:

.. math::

    \boldsymbol{\sigma} = \mathbf{C} : \boldsymbol{\varepsilon}

and strain is the symmetric gradient of displacement:

.. math::

    \boldsymbol{\varepsilon} = \frac{1}{2}(\nabla \mathbf{u} + \nabla \mathbf{u}^T)

Material properties
-------------------

We define isotropic elastic materials using Young's modulus *E* and Poisson's
ratio *ν*:

.. code-block:: python

    import numpy as np
    import muGrid
    from muGrid.Solvers import conjugate_gradients

    def isotropic_stiffness_2d(E, nu):
        """
        Create 2D plane strain stiffness tensor in Voigt notation.
        Voigt ordering: [xx, yy, xy]
        """
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        C = np.zeros((3, 3))
        C[0, 0] = lam + 2 * mu  # C_xxxx
        C[1, 1] = lam + 2 * mu  # C_yyyy
        C[2, 2] = mu            # C_xyxy
        C[0, 1] = C[1, 0] = lam # C_xxyy
        return C

    # Material parameters
    E_matrix = 1.0      # Young's modulus of matrix
    E_inclusion = 10.0  # Young's modulus of inclusion (10x stiffer)
    nu = 0.3            # Poisson's ratio (same for both)

    C_matrix = isotropic_stiffness_2d(E_matrix, nu)
    C_inclusion = isotropic_stiffness_2d(E_inclusion, nu)

Setting up the microstructure
-----------------------------

We create a simple circular inclusion in the center:

.. code-block:: python

    nb_grid_pts = [32, 32]
    dim = 2

    # Grid coordinates (cell centers)
    x = np.linspace(0, 1, nb_grid_pts[0], endpoint=False) + 0.5 / nb_grid_pts[0]
    y = np.linspace(0, 1, nb_grid_pts[1], endpoint=False) + 0.5 / nb_grid_pts[1]
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Circular inclusion at center with radius 0.25
    radius = 0.25
    distance = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
    phase = (distance < radius).astype(float)  # 1 = inclusion, 0 = matrix

    print(f"Inclusion volume fraction: {np.mean(phase):.4f}")

Using the FEM gradient operator
-------------------------------

The ``FEMGradientOperator`` computes gradients using linear finite element
shape functions. It subdivides each pixel into triangular elements with
quadrature points:

.. code-block:: python

    # Grid spacing
    grid_spacing = [1.0 / n for n in nb_grid_pts]

    # Create the FEM gradient operator
    gradient_op = muGrid.FEMGradientOperator(dim, grid_spacing)

    # Get quadrature information
    nb_quad = gradient_op.nb_quad_pts      # Number of quadrature points per pixel
    nb_nodes = gradient_op.nb_nodal_pts    # Number of nodal points per pixel
    quad_weights = gradient_op.get_quadrature_weights()

    print(f"Quadrature points per pixel: {nb_quad}")
    print(f"Quadrature weights: {quad_weights}")

Setting up the domain and fields
--------------------------------

.. code-block:: python

    comm = muGrid.Communicator()

    # Domain decomposition with ghost regions for the gradient stencil
    decomposition = muGrid.CartesianDecomposition(
        comm,
        nb_domain_grid_pts=nb_grid_pts,
        nb_ghosts_left=[1, 1],
        nb_ghosts_right=[1, 1],
        nb_sub_pts={"quad": nb_quad},  # Register quadrature sub-points
    )

    # Displacement field: vector with dim components
    u_field = decomposition.real_field("displacement", (dim,))

    # Force field (RHS): vector with dim components
    f_field = decomposition.real_field("force", (dim,))

    # Strain and stress at quadrature points: tensors with (dim, dim) components
    strain_field = decomposition.real_field("strain", (dim, dim), "quad")
    stress_field = decomposition.real_field("stress", (dim, dim), "quad")

    # Material stiffness at each quadrature point
    # Shape: (3, 3, nb_quad, nx, ny) for Voigt notation
    C_field = np.zeros((3, 3, nb_quad) + tuple(nb_grid_pts))
    for q in range(nb_quad):
        for i in range(3):
            for j in range(3):
                C_field[i, j, q] = (
                    C_matrix[i, j] * (1 - phase) +
                    C_inclusion[i, j] * phase
                )

Computing strain from displacement
----------------------------------

.. code-block:: python

    def compute_strain(u, strain_out):
        """
        Compute strain from displacement: ε = sym(∇u)
        """
        # Fill ghost values for periodic boundaries
        decomposition.communicate_ghosts(u)

        # Apply gradient operator: computes ∂u_i/∂x_j
        # Input shape: (dim, 1, nx, ny) - vector field at nodes
        # Output shape: (dim, dim, quad, nx, ny) - tensor at quad points
        gradient_op.apply(u, strain_out)

        # Symmetrize: ε_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
        grad = strain_out.s
        strain_out.s[...] = 0.5 * (grad + np.swapaxes(grad, 0, 1))

Computing stress from strain
----------------------------

Using Voigt notation for efficient tensor contraction:

.. code-block:: python

    def compute_stress(strain, stress_out, C):
        """
        Compute stress from strain: σ = C : ε
        Uses Voigt notation for the contraction.
        """
        eps = strain.s

        # Convert strain tensor to Voigt vector
        eps_voigt = np.zeros((3, nb_quad) + tuple(nb_grid_pts))
        eps_voigt[0] = eps[0, 0]          # εxx
        eps_voigt[1] = eps[1, 1]          # εyy
        eps_voigt[2] = 2 * eps[0, 1]      # 2εxy (engineering shear)

        # Stress in Voigt: σ = C @ ε
        sig_voigt = np.einsum('ijq...,jq...->iq...', C, eps_voigt)

        # Convert back to tensor
        stress_out.s[0, 0] = sig_voigt[0]  # σxx
        stress_out.s[1, 1] = sig_voigt[1]  # σyy
        stress_out.s[0, 1] = sig_voigt[2]  # σxy
        stress_out.s[1, 0] = sig_voigt[2]  # σyx (symmetric)

Computing divergence (equilibrium)
----------------------------------

The transpose of the gradient operator computes the divergence:

.. code-block:: python

    def compute_divergence(stress, f_out):
        """
        Compute divergence of stress: f = ∇·σ
        Uses the transpose of the gradient operator.
        """
        # Fill ghost values
        decomposition.communicate_ghosts(stress)

        # Apply transpose with quadrature weights
        f_out.pg[...] = 0.0
        gradient_op.transpose(stress, f_out, list(quad_weights))

The stiffness operator
----------------------

Combining the above, the stiffness operator computes :math:`\mathbf{K}\mathbf{u} = \mathbf{B}^T \mathbf{C} \mathbf{B} \mathbf{u}`:

.. code-block:: python

    def apply_stiffness(u_in, f_out):
        """
        Apply stiffness operator: f = B^T C B u
        """
        # Strain: ε = B u
        compute_strain(u_in, strain_field)

        # Stress: σ = C : ε
        compute_stress(strain_field, stress_field, C_field)

        # Force: f = B^T σ
        compute_divergence(stress_field, f_out)

Solving for effective properties
--------------------------------

To compute effective properties, we apply unit macroscopic strains and
measure the resulting average stress:

.. code-block:: python

    # Apply unit strain in xx-direction
    E_macro = np.zeros((dim, dim))
    E_macro[0, 0] = 1.0  # Unit strain εxx

    # Compute RHS: f = -B^T C E_macro
    # (uniform macroscopic strain throughout the domain)
    strain_field.s[...] = 0.0
    strain_field.s[0, 0, ...] = E_macro[0, 0]
    compute_stress(strain_field, stress_field, C_field)
    compute_divergence(stress_field, f_field)
    f_field.s[...] *= -1.0  # RHS is negative

    # Initialize displacement to zero
    u_field.s[...] = 0.0

    # Solve equilibrium
    conjugate_gradients(
        comm,
        decomposition,
        apply_stiffness,
        f_field,
        u_field,
        tol=1e-6,
        maxiter=500,
    )

    # Compute total strain (macroscopic + fluctuation)
    compute_strain(u_field, strain_field)
    strain_field.s[0, 0, ...] += E_macro[0, 0]

    # Compute stress from total strain
    compute_stress(strain_field, stress_field, C_field)

    # Average stress = effective stress for unit applied strain
    sig_xx_avg = np.mean(stress_field.s[0, 0])
    sig_yy_avg = np.mean(stress_field.s[1, 1])

    print(f"Average stress for εxx=1: σxx={sig_xx_avg:.4f}, σyy={sig_yy_avg:.4f}")
    print(f"These are entries C_eff[0,0] and C_eff[1,0] of the effective stiffness")

.. _performance-comparison:

Convolution Operator Performance
********************************

*µ*\Grid provides two types of convolution operators with different performance
characteristics.

Generic ConvolutionOperator
---------------------------

The ``ConvolutionOperator`` class accepts arbitrary stencils:

.. code-block:: python

    # Any stencil shape and values
    stencil = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    op = muGrid.ConvolutionOperator([-1, -1], stencil)

**Advantages:**

- Fully flexible: any stencil shape and coefficients
- Easy to implement custom operators
- Supports multiple operators and quadrature points

**Disadvantages:**

- Sparse memory access pattern (indirect indexing)
- Cannot be fully optimized by the compiler

Hard-coded operators
--------------------

Hard-coded operators like ``LaplaceOperator`` and ``FEMGradientOperator`` have
optimized implementations:

.. code-block:: python

    laplace = muGrid.LaplaceOperator(dim=2, scale=1.0)
    gradient = muGrid.FEMGradientOperator(dim=2, grid_spacing=[0.1, 0.1])

**Advantages:**

- Compiler can optimize memory access patterns
- Better cache utilization
- SIMD vectorization possible
- GPU kernels can be highly optimized

**Disadvantages:**

- Fixed stencil structure
- Less flexible

Performance comparison
----------------------

Typical performance differences on CPU:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Operator
     - Relative Speed
     - Notes
   * - Generic ConvolutionOperator
     - 1× (baseline)
     - Flexible but slower
   * - LaplaceOperator
     - 2-4× faster
     - Optimized 5/7-point stencil
   * - FEMGradientOperator
     - 2-3× faster
     - Optimized FEM kernels

On GPU, the differences can be even more pronounced (5-10× or more) because:

- Hard-coded operators have predictable memory access patterns
- Better occupancy and fewer register spills
- Specialized GPU kernels avoid sparse indexing overhead

**Recommendation:**

- Use hard-coded operators (``LaplaceOperator``, ``FEMGradientOperator``) for
  production code where performance matters
- Use ``ConvolutionOperator`` for prototyping, testing custom stencils, or when
  no hard-coded alternative exists

Example: comparing both approaches
----------------------------------

.. code-block:: python

    import numpy as np
    import muGrid
    import time

    # Setup
    nb_grid_pts = [512, 512]
    h = 1.0 / 512
    scale = -1.0 / h**2

    fc = muGrid.GlobalFieldCollection(
        nb_grid_pts,
        nb_ghosts_left=[1, 1],
        nb_ghosts_right=[1, 1],
    )

    input_field = fc.real_field("input")
    output_field = fc.real_field("output")
    input_field.p[...] = np.random.randn(*nb_grid_pts)

    # Pad ghost regions
    input_field.pg[...] = np.pad(input_field.p, ((1, 1), (1, 1)), mode='wrap')

    # Generic operator
    stencil = scale * np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    generic_op = muGrid.ConvolutionOperator([-1, -1], stencil)

    # Hard-coded operator
    hardcoded_op = muGrid.LaplaceOperator(2, scale)

    # Benchmark
    n_iter = 100

    start = time.perf_counter()
    for _ in range(n_iter):
        generic_op.apply(input_field, output_field)
    generic_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(n_iter):
        hardcoded_op.apply(input_field, output_field)
    hardcoded_time = time.perf_counter() - start

    print(f"Generic operator: {generic_time:.3f}s ({n_iter} iterations)")
    print(f"Hard-coded operator: {hardcoded_time:.3f}s ({n_iter} iterations)")
    print(f"Speedup: {generic_time / hardcoded_time:.1f}×")

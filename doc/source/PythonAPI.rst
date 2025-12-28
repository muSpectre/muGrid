Python API Reference
####################

This section provides the API reference for the µGrid Python module.
The Python bindings provide access to the core µGrid functionality
with a Pythonic interface.

.. contents:: Table of Contents
   :local:
   :depth: 2

Field Collections
*****************

Field collections manage groups of fields on structured grids.

GlobalFieldCollection
---------------------

.. py:class:: muGrid.GlobalFieldCollection(nb_grid_pts, nb_sub_pts=None, nb_ghosts_left=None, nb_ghosts_right=None, memory_location=None)

   A GlobalFieldCollection manages a set of fields that share the same
   global grid structure. It can allocate fields in either host (CPU)
   or device (GPU) memory.

   :param nb_grid_pts: Grid dimensions, e.g., ``[64, 64]`` for 2D or ``[32, 32, 32]`` for 3D.
   :type nb_grid_pts: sequence of int
   :param nb_sub_pts: Number of sub-points per pixel for each sub-point type. Default is ``{}``.
   :type nb_sub_pts: dict, optional
   :param nb_ghosts_left: Ghost cells on low-index side. Default is no ghosts.
   :type nb_ghosts_left: sequence of int, optional
   :param nb_ghosts_right: Ghost cells on high-index side. Default is no ghosts.
   :type nb_ghosts_right: sequence of int, optional
   :param memory_location: Memory location for field data: ``"host"`` (CPU) or ``"device"`` (GPU). Default is ``"host"``.
   :type memory_location: str, optional

   **Example**::

       >>> fc = muGrid.GlobalFieldCollection([64, 64])
       >>> field = fc.real_field("temperature")
       >>> field.p[:] = 300.0  # Set temperature to 300K

       >>> # GPU field collection
       >>> fc_gpu = muGrid.GlobalFieldCollection([64, 64], memory_location="device")

   .. py:method:: real_field(name, components=(), sub_pt="pixel")

      Create a real-valued field.

      :param name: Unique name for the field.
      :type name: str
      :param components: Shape of field components. Default is ``()`` for scalar.
      :type components: tuple of int, optional
      :param sub_pt: Sub-point type. Default is ``"pixel"``.
      :type sub_pt: str, optional
      :returns: Wrapped field with ``.s``, ``.p``, ``.sg``, ``.pg`` accessors.
      :rtype: Field

   .. py:method:: complex_field(name, components=(), sub_pt="pixel")

      Create a complex-valued field.

      :param name: Unique name for the field.
      :type name: str
      :param components: Shape of field components. Default is ``()`` for scalar.
      :type components: tuple of int, optional
      :param sub_pt: Sub-point type. Default is ``"pixel"``.
      :type sub_pt: str, optional
      :returns: Wrapped field with ``.s``, ``.p``, ``.sg``, ``.pg`` accessors.
      :rtype: Field

   .. py:method:: int_field(name, components=(), sub_pt="pixel")

      Create an integer field.

      :param name: Unique name for the field.
      :type name: str
      :param components: Shape of field components. Default is ``()`` for scalar.
      :type components: tuple of int, optional
      :param sub_pt: Sub-point type. Default is ``"pixel"``.
      :type sub_pt: str, optional
      :returns: Wrapped field with ``.s``, ``.p``, ``.sg``, ``.pg`` accessors.
      :rtype: Field

   .. py:method:: uint_field(name, components=(), sub_pt="pixel")

      Create an unsigned integer field.

      :param name: Unique name for the field.
      :type name: str
      :param components: Shape of field components. Default is ``()`` for scalar.
      :type components: tuple of int, optional
      :param sub_pt: Sub-point type. Default is ``"pixel"``.
      :type sub_pt: str, optional
      :returns: Wrapped field with ``.s``, ``.p``, ``.sg``, ``.pg`` accessors.
      :rtype: Field

LocalFieldCollection
--------------------

.. py:class:: muGrid.LocalFieldCollection(spatial_dim, name="", nb_sub_pts=None, memory_location=None)

   A LocalFieldCollection manages fields on a subset of pixels, typically
   used for material-specific data in heterogeneous simulations.

   :param spatial_dim: Spatial dimension (2 or 3).
   :type spatial_dim: int
   :param name: Name for the collection. Default is ``""``.
   :type name: str, optional
   :param nb_sub_pts: Number of sub-points per pixel for each sub-point type.
   :type nb_sub_pts: dict, optional
   :param memory_location: Memory location: ``"host"`` or ``"device"``. Default is ``"host"``.
   :type memory_location: str, optional

   The field creation methods (``real_field``, ``complex_field``, ``int_field``, ``uint_field``)
   are the same as for ``GlobalFieldCollection``.

Fields
******

The :class:`Field` class wraps C++ fields with convenient numpy array access.

Field
-----

.. py:class:: muGrid.Field(cpp_field)

   Python wrapper for muGrid fields providing numpy/cupy array views.

   This class wraps a C++ muGrid field and provides the following properties
   for accessing the underlying data as arrays:

   - ``s``: SubPt layout, excluding ghost regions
   - ``sg``: SubPt layout, including ghost regions
   - ``p``: Pixel layout, excluding ghost regions
   - ``pg``: Pixel layout, including ghost regions

   For CPU fields, the arrays are numpy arrays. For GPU fields (CUDA/ROCm),
   the arrays are CuPy arrays. Both are views into the underlying C++ data,
   so modifications to the arrays will modify the field data directly (zero-copy).

   :param cpp_field: The underlying C++ field object.

   .. py:property:: s

      SubPt layout array excluding ghost regions.

      Shape: ``(*components_shape, nb_sub_pts, *spatial_dims_without_ghosts)``

   .. py:property:: sg

      SubPt layout array including ghost regions.

      Shape: ``(*components_shape, nb_sub_pts, *spatial_dims_with_ghosts)``

   .. py:property:: p

      Pixel layout array excluding ghost regions.

      Shape: ``(nb_components * nb_sub_pts, *spatial_dims_without_ghosts)``

   .. py:property:: pg

      Pixel layout array including ghost regions.

      Shape: ``(nb_components * nb_sub_pts, *spatial_dims_with_ghosts)``

   .. py:property:: is_on_gpu

      Check if this field resides on GPU memory.

      :rtype: bool

   .. py:property:: device

      Get the device where this field resides (``'cpu'`` or ``'cuda:N'``).

      :rtype: str

wrap_field
----------

.. py:function:: muGrid.wrap_field(field)

   Wrap a C++ field in a Python Field object.

   :param field: The underlying C++ field.
   :returns: Wrapped field with numpy array access.
   :rtype: Field

Domain Decomposition
********************

Classes for parallel domain decomposition with MPI.

CartesianDecomposition
----------------------

.. py:class:: muGrid.CartesianDecomposition(communicator, nb_domain_grid_pts, nb_subdivisions=None, nb_ghosts_left=None, nb_ghosts_right=None, nb_sub_pts=None, memory_location=None)

   CartesianDecomposition manages domain decomposition for MPI-parallel
   computations on structured grids, including ghost buffer regions for
   stencil operations.

   :param communicator: MPI communicator for parallel execution.
   :type communicator: Communicator
   :param nb_domain_grid_pts: Global domain grid dimensions.
   :type nb_domain_grid_pts: sequence of int
   :param nb_subdivisions: Number of subdivisions in each dimension. Default is automatic.
   :type nb_subdivisions: sequence of int, optional
   :param nb_ghosts_left: Ghost cells on low-index side. Default is no ghosts.
   :type nb_ghosts_left: sequence of int, optional
   :param nb_ghosts_right: Ghost cells on high-index side. Default is no ghosts.
   :type nb_ghosts_right: sequence of int, optional
   :param nb_sub_pts: Number of sub-points per pixel for each sub-point type.
   :type nb_sub_pts: dict, optional
   :param memory_location: Memory location: ``"host"`` or ``"device"``. Default is ``"host"``.
   :type memory_location: str, optional

   **Example**::

       >>> from muGrid import Communicator, CartesianDecomposition
       >>> comm = Communicator()
       >>> decomp = CartesianDecomposition(
       ...     comm,
       ...     nb_domain_grid_pts=[128, 128],
       ...     nb_ghosts_left=[1, 1],
       ...     nb_ghosts_right=[1, 1]
       ... )
       >>> field = decomp.real_field("displacement", components=(3,))

   .. py:property:: nb_grid_pts

      Local subdomain grid dimensions (alias for ``nb_subdomain_grid_pts``).

      :rtype: list of int

   .. py:method:: set_nb_sub_pts(sub_pt_type, nb_sub_pts)

      Set the number of sub-points for a given sub-point type.

      :param sub_pt_type: Name of the sub-point type (e.g., ``"quad"``).
      :type sub_pt_type: str
      :param nb_sub_pts: Number of sub-points per pixel for this type.
      :type nb_sub_pts: int

   .. py:method:: communicate_ghosts(field)

      Exchange ghost buffer data for a field.

      :param field: The field whose ghost buffers should be filled from neighbors.
      :type field: Field

   .. py:method:: reduce_ghosts(field)

      Accumulate ghost buffer contributions back to the interior domain.

      This is the adjoint operation of ``communicate_ghosts`` and is needed
      for transpose operations (e.g., divergence) with periodic BCs.
      After the operation, ghost buffers are zeroed.

      :param field: The field whose ghost buffers should be reduced to interior.
      :type field: Field

Communicator
------------

.. py:function:: muGrid.Communicator(communicator=None)

   Factory function for the communicator class.

   :param communicator: The bare MPI communicator (mpi4py or muGrid communicator object).
      Default is a serial communicator containing just the present process.
   :returns: A muGrid communicator object.

   **Example**::

       >>> from muGrid import Communicator
       >>> comm = Communicator()  # Serial communicator
       >>> # Or with MPI:
       >>> from mpi4py import MPI
       >>> comm = Communicator(MPI.COMM_WORLD)

Operators
*********

Discrete operators for stencil-based computations.

ConvolutionOperator
-------------------

.. py:class:: muGrid.ConvolutionOperator(offset, stencil)

   Applies convolution (stencil) operations to fields. Useful for computing
   gradients, Laplacians, and other discrete differential operators.

   :param offset: Offset of the stencil origin relative to the current pixel.
   :type offset: sequence of int
   :param stencil: Stencil coefficients. Shape determines the stencil size.
   :type stencil: array_like

   **Example**::

       >>> # Create a 2D Laplacian stencil
       >>> import numpy as np
       >>> stencil = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
       >>> laplace = muGrid.ConvolutionOperator([-1, -1], stencil)
       >>> laplace.apply(input_field, output_field)

   .. py:method:: apply(nodal_field, quadrature_point_field)

      Apply convolution to fields.

      :param nodal_field: Input field.
      :type nodal_field: Field
      :param quadrature_point_field: Output field.
      :type quadrature_point_field: Field

   .. py:method:: transpose(quadrature_point_field, nodal_field, weights=None)

      Apply transpose convolution to fields.

      :param quadrature_point_field: Input field.
      :type quadrature_point_field: Field
      :param nodal_field: Output field.
      :type nodal_field: Field
      :param weights: Weights for the transpose operation.
      :type weights: sequence of float, optional

LaplaceOperator
---------------

.. py:class:: muGrid.LaplaceOperator(spatial_dim, scale=1.0)

   A hard-coded, optimized Laplacian stencil operator using the standard
   5-point (2D) or 7-point (3D) finite difference stencil.

   :param spatial_dim: Spatial dimension (2 or 3).
   :type spatial_dim: int
   :param scale: Scaling factor for the Laplacian. Default is ``1.0``.
   :type scale: float, optional

   **Example**::

       >>> laplace = muGrid.LaplaceOperator(2, scale=-1.0)
       >>> laplace.apply(input_field, output_field)

   .. py:method:: apply(input_field, output_field)

      Apply the Laplacian operator.

      :param input_field: Input field.
      :type input_field: Field
      :param output_field: Output field.
      :type output_field: Field

   .. py:method:: apply_increment(input_field, alpha, output_field)

      Apply Laplacian and add scaled result to output: ``output += alpha * L(input)``.

      :param input_field: Input field.
      :type input_field: Field
      :param alpha: Scaling factor.
      :type alpha: float
      :param output_field: Output field (updated in-place).
      :type output_field: Field

   .. py:method:: transpose(input_field, output_field, weights=None)

      Apply transpose operator. For Laplacian, this is the same as apply.

      :param input_field: Input field.
      :type input_field: Field
      :param output_field: Output field.
      :type output_field: Field
      :param weights: Weights (unused for Laplacian, included for API compatibility).
      :type weights: sequence of float, optional

FEMGradientOperator
-------------------

.. py:class:: muGrid.FEMGradientOperator(spatial_dim, grid_spacing=None)

   A hard-coded, optimized gradient operator using linear finite element
   shape functions on triangles (2D) or tetrahedra (3D).

   :param spatial_dim: Spatial dimension (2 or 3).
   :type spatial_dim: int
   :param grid_spacing: Grid spacing in each direction. Default is ``[1.0, ...]`` for each dimension.
   :type grid_spacing: sequence of float, optional

   **Example**::

       >>> grad = muGrid.FEMGradientOperator(2)
       >>> grad.apply(nodal_field, quadrature_point_gradient_field)

   .. py:method:: apply(nodal_field, quadrature_point_field)

      Apply the gradient operator (nodal values → quadrature point gradients).

      :param nodal_field: Input field at nodal points.
      :type nodal_field: Field
      :param quadrature_point_field: Output field at quadrature points.
      :type quadrature_point_field: Field

   .. py:method:: apply_increment(nodal_field, alpha, quadrature_point_field)

      Apply gradient and add scaled result to output: ``output += alpha * grad(input)``.

      :param nodal_field: Input field at nodal points.
      :type nodal_field: Field
      :param alpha: Scaling factor.
      :type alpha: float
      :param quadrature_point_field: Output field at quadrature points (updated in-place).
      :type quadrature_point_field: Field

   .. py:method:: transpose(quadrature_point_field, nodal_field, weights=None)

      Apply transpose (divergence) operator (quadrature points → nodal values).

      :param quadrature_point_field: Input field at quadrature points.
      :type quadrature_point_field: Field
      :param nodal_field: Output field at nodal points.
      :type nodal_field: Field
      :param weights: Weights for the transpose operation.
      :type weights: sequence of float, optional

   .. py:method:: transpose_increment(quadrature_point_field, alpha, nodal_field, weights=None)

      Apply transpose and add scaled result: ``output += alpha * div(input)``.

      :param quadrature_point_field: Input field at quadrature points.
      :type quadrature_point_field: Field
      :param alpha: Scaling factor.
      :type alpha: float
      :param nodal_field: Output field at nodal points (updated in-place).
      :type nodal_field: Field
      :param weights: Weights for the transpose operation.
      :type weights: sequence of float, optional

FFT Engine
**********

Distributed FFT operations using pencil decomposition.

FFTEngine
---------

.. py:class:: muGrid.FFTEngine(nb_domain_grid_pts, communicator=None, nb_ghosts_left=None, nb_ghosts_right=None, nb_sub_pts=None)

   The FFTEngine provides distributed FFT operations on structured grids
   with MPI parallelization using pencil (2D) decomposition.

   :param nb_domain_grid_pts: Global grid dimensions ``[Nx, Ny]`` or ``[Nx, Ny, Nz]``.
   :type nb_domain_grid_pts: sequence of int
   :param communicator: MPI communicator. Default is serial execution.
   :type communicator: Communicator, optional
   :param nb_ghosts_left: Ghost cells on low-index side of each dimension.
   :type nb_ghosts_left: sequence of int, optional
   :param nb_ghosts_right: Ghost cells on high-index side of each dimension.
   :type nb_ghosts_right: sequence of int, optional
   :param nb_sub_pts: Number of sub-points per pixel.
   :type nb_sub_pts: dict, optional

   **Example**::

       >>> engine = muGrid.FFTEngine([64, 64])
       >>> real_field = engine.real_space_field("displacement", nb_components=3)
       >>> fourier_field = engine.fourier_space_field("displacement_k", nb_components=3)
       >>> engine.fft(real_field, fourier_field)
       >>> engine.ifft(fourier_field, real_field)
       >>> real_field.s[:] *= engine.normalisation

   .. py:method:: fft(input_field, output_field)

      Forward FFT: real space → Fourier space.

      The transform is unnormalized. To recover original data after
      ``ifft(fft(x))``, multiply by ``normalisation``.

      :param input_field: Real-space field (must be in this engine's real collection).
      :type input_field: Field
      :param output_field: Fourier-space field (must be in this engine's Fourier collection).
      :type output_field: Field

   .. py:method:: ifft(input_field, output_field)

      Inverse FFT: Fourier space → real space.

      The transform is unnormalized. To recover original data after
      ``ifft(fft(x))``, multiply by ``normalisation``.

      :param input_field: Fourier-space field (must be in this engine's Fourier collection).
      :type input_field: Field
      :param output_field: Real-space field (must be in this engine's real collection).
      :type output_field: Field

   .. py:method:: real_space_field(name, nb_components=1)

      Create a real-space field for FFT operations.

      :param name: Unique field name.
      :type name: str
      :param nb_components: Number of components. Default is 1.
      :type nb_components: int, optional
      :returns: Wrapped real-valued field with array accessors.
      :rtype: Field

   .. py:method:: fourier_space_field(name, nb_components=1)

      Create a Fourier-space field for FFT operations.

      :param name: Unique field name.
      :type name: str
      :param nb_components: Number of components. Default is 1.
      :type nb_components: int, optional
      :returns: Wrapped complex-valued field with array accessors.
      :rtype: Field

   .. py:attribute:: normalisation

      Normalization factor for FFT. Multiply by this after ``ifft(fft(x))``
      to recover the original data.

      :type: float

FFT Utilities
-------------

.. py:function:: muGrid.fft_freq(nb_grid_pts, lengths)

   Return FFT sample frequencies for a given grid.

   :param nb_grid_pts: Number of grid points in each dimension.
   :type nb_grid_pts: sequence of int
   :param lengths: Physical lengths in each dimension.
   :type lengths: sequence of float
   :returns: Array of frequency values.
   :rtype: numpy.ndarray

.. py:function:: muGrid.fft_freqind(nb_grid_pts)

   Return FFT frequency indices for a given grid.

   :param nb_grid_pts: Number of grid points in each dimension.
   :type nb_grid_pts: sequence of int
   :returns: Array of frequency indices.
   :rtype: numpy.ndarray

.. py:function:: muGrid.rfft_freq(nb_grid_pts, lengths)

   Return real FFT sample frequencies (Hermitian-symmetric) for a given grid.

   :param nb_grid_pts: Number of grid points in each dimension.
   :type nb_grid_pts: sequence of int
   :param lengths: Physical lengths in each dimension.
   :type lengths: sequence of float
   :returns: Array of frequency values.
   :rtype: numpy.ndarray

.. py:function:: muGrid.rfft_freqind(nb_grid_pts)

   Return real FFT frequency indices for a given grid.

   :param nb_grid_pts: Number of grid points in each dimension.
   :type nb_grid_pts: sequence of int
   :returns: Array of frequency indices.
   :rtype: numpy.ndarray

.. py:function:: muGrid.fft_normalization(nb_grid_pts)

   Return FFT normalization factor.

   :param nb_grid_pts: Number of grid points in each dimension.
   :type nb_grid_pts: sequence of int
   :returns: Normalization factor.
   :rtype: float

.. py:function:: muGrid.get_hermitian_grid_pts(nb_grid_pts)

   Return the number of grid points in the Hermitian (rfft) representation.

   :param nb_grid_pts: Number of grid points in each dimension.
   :type nb_grid_pts: sequence of int
   :returns: Number of Hermitian grid points.
   :rtype: tuple of int

File I/O
********

Classes for reading and writing fields to NetCDF files.

FileIONetCDF
------------

.. py:class:: muGrid.FileIONetCDF(file_name, open_mode="read", communicator=None)

   Provides NetCDF file I/O for muGrid fields with optional MPI support.

   :param file_name: Path to the NetCDF file.
   :type file_name: str
   :param open_mode: File open mode: ``"read"``, ``"write"``, ``"overwrite"``, or ``"append"``. Default is ``"read"``.
   :type open_mode: str or OpenMode, optional
   :param communicator: MPI communicator for parallel I/O. Default is serial.
   :type communicator: Communicator, optional

   **Example**::

       >>> file = muGrid.FileIONetCDF("output.nc", open_mode="overwrite")
       >>> file.register_field_collection(field_collection)
       >>> file.append_frame().write()

   .. py:method:: register_field_collection(collection)

      Register a field collection for I/O.

      :param collection: The field collection to register. If a ``CartesianDecomposition`` is
         passed, its underlying field collection is used.
      :type collection: GlobalFieldCollection, LocalFieldCollection, or CartesianDecomposition

   .. py:method:: append_frame()

      Append a new frame to the file.

      :returns: Frame object for writing.

   .. py:attribute:: OpenMode

      Enum for file open modes: ``Read``, ``Write``, ``Overwrite``, ``Append``.

Utilities
*********

Timer
-----

.. py:class:: muGrid.Timer(name="")

   Timer utility for performance measurement.

   :param name: Name for the timer. Default is ``""``.
   :type name: str, optional

   .. py:method:: start()

      Start the timer.

   .. py:method:: stop()

      Stop the timer.

   .. py:method:: elapsed()

      Return elapsed time in seconds.

      :rtype: float

Solvers
-------

.. py:module:: muGrid.Solvers

The Solvers module provides simple parallel iterative solvers.

.. py:function:: muGrid.Solvers.conjugate_gradients(comm, fc, hessp, b, x, tol=1e-6, maxiter=1000, callback=None)

   Conjugate gradient method for matrix-free solution of the linear problem
   ``Ax = b``, where ``A`` is represented by the function ``hessp`` (which computes the
   product of ``A`` with a vector). The method iteratively refines the solution ``x``
   until the residual ``||Ax - b||`` is less than ``tol`` or until ``maxiter`` iterations
   are reached.

   :param comm: Communicator for parallel processing.
   :type comm: muGrid.Communicator
   :param fc: Collection for creating temporary fields used by the CG algorithm.
   :type fc: muGrid.GlobalFieldCollection, muGrid.LocalFieldCollection, or muGrid.CartesianDecomposition
   :param hessp: Function that computes the product of the Hessian matrix ``A`` with a vector.
      Signature: ``hessp(input_field, output_field)`` where both are ``muGrid.Field``.
   :type hessp: callable
   :param b: Right-hand side vector.
   :type b: muGrid.Field
   :param x: Initial guess for the solution (modified in place).
   :type x: muGrid.Field
   :param tol: Tolerance for convergence. Default is ``1e-6``.
   :type tol: float, optional
   :param maxiter: Maximum number of iterations. Default is ``1000``.
   :type maxiter: int, optional
   :param callback: Function to call after each iteration with signature:
      ``callback(iteration, x_array, residual_array, search_direction_array)``.
   :type callback: callable, optional
   :returns: Solution to the system ``Ax = b`` (same as input field ``x``).
   :rtype: muGrid.Field
   :raises RuntimeError: If the algorithm does not converge within ``maxiter`` iterations,
      or if the Hessian is not positive definite.

Enumerations
************

.. py:class:: muGrid.IterUnit

   Enumeration for iteration unit types.

   .. py:attribute:: Pixel

      Iterate over pixels.

   .. py:attribute:: SubPt

      Iterate over sub-points.

.. py:class:: muGrid.StorageOrder

   Enumeration for array storage order.

   .. py:attribute:: ColMajor

      Column-major (Fortran) storage order.

   .. py:attribute:: RowMajor

      Row-major (C) storage order.

Module Constants
****************

The following constants indicate compile-time configuration:

.. py:data:: muGrid.has_mpi

   ``True`` if MPI support is enabled.

.. py:data:: muGrid.has_cuda

   ``True`` if CUDA GPU support is compiled in.

.. py:data:: muGrid.has_rocm

   ``True`` if ROCm/HIP GPU support is compiled in.

.. py:data:: muGrid.has_gpu

   ``True`` if any GPU support is available.

.. py:data:: muGrid.has_netcdf

   ``True`` if NetCDF I/O support is available.

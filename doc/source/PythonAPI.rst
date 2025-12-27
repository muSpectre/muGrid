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

.. autoclass:: muGrid.GlobalFieldCollection
   :members:
   :undoc-members:
   :show-inheritance:

LocalFieldCollection
--------------------

.. autoclass:: muGrid.LocalFieldCollection
   :members:
   :undoc-members:
   :show-inheritance:

Fields
******

The :class:`Field` class wraps C++ fields with convenient numpy array access.

.. autoclass:: muGrid.Field
   :members:
   :undoc-members:

.. autofunction:: muGrid.wrap_field

Domain Decomposition
********************

Classes for parallel domain decomposition with MPI.

CartesianDecomposition
----------------------

.. autoclass:: muGrid.CartesianDecomposition
   :members:
   :undoc-members:
   :show-inheritance:

Communicator
------------

.. autoclass:: muGrid.Communicator
   :members:
   :undoc-members:

Operators
*********

Discrete operators for stencil-based computations.

ConvolutionOperator
-------------------

.. autoclass:: muGrid.ConvolutionOperator
   :members:
   :undoc-members:
   :show-inheritance:

LaplaceOperator
---------------

.. autoclass:: muGrid.LaplaceOperator
   :members:
   :undoc-members:
   :show-inheritance:

FEMGradientOperator
-------------------

.. autoclass:: muGrid.FEMGradientOperator
   :members:
   :undoc-members:
   :show-inheritance:

FFT Engine
**********

Distributed FFT operations using pencil decomposition.

.. autoclass:: muGrid.FFTEngine
   :members:
   :undoc-members:
   :show-inheritance:

FFT Utilities
-------------

.. autofunction:: muGrid.fft_freq

.. autofunction:: muGrid.fft_freqind

.. autofunction:: muGrid.rfft_freq

.. autofunction:: muGrid.rfft_freqind

.. autofunction:: muGrid.fft_normalization

.. autofunction:: muGrid.get_hermitian_grid_pts

File I/O
********

Classes for reading and writing fields to NetCDF files.

.. autoclass:: muGrid.FileIONetCDF
   :members:
   :undoc-members:

Utilities
*********

Timer
-----

.. autoclass:: muGrid.Timer
   :members:
   :undoc-members:

Solvers
-------

.. automodule:: muGrid.Solvers
   :members:
   :undoc-members:

Enumerations
************

.. autoclass:: muGrid.IterUnit
   :members:
   :undoc-members:

.. autoclass:: muGrid.StorageOrder
   :members:
   :undoc-members:

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

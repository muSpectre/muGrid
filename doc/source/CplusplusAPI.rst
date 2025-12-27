C++ API Reference
#################

This section provides the API reference for the µGrid C++ library.
The library is organized into several modules for field management,
grid operations, MPI communication, and I/O.

Field Collections
*****************

Field collections manage groups of fields on structured grids. All fields
in a collection share the same spatial discretization.

.. doxygenclass:: muGrid::FieldCollection
   :members:
   :undoc-members:

.. doxygenclass:: muGrid::GlobalFieldCollection
   :members:
   :undoc-members:

.. doxygenclass:: muGrid::LocalFieldCollection
   :members:
   :undoc-members:

Fields
******

Fields represent scalar, vector, or tensor quantities that vary over
the grid. Each field belongs to a field collection.

.. doxygenclass:: muGrid::Field
   :members:
   :undoc-members:

.. doxygenclass:: muGrid::TypedFieldBase
   :members:
   :undoc-members:

.. doxygenclass:: muGrid::StateField
   :members:
   :undoc-members:

Domain Decomposition
********************

Classes for parallel domain decomposition and ghost communication.

.. doxygenclass:: muGrid::Decomposition
   :members:
   :undoc-members:

.. doxygenclass:: muGrid::CartesianDecomposition
   :members:
   :undoc-members:

.. doxygenclass:: muGrid::Communicator
   :members:
   :undoc-members:

Operators
*********

Discrete operators for stencil-based computations like gradients and
the Laplacian.

.. doxygenclass:: muGrid::ConvolutionOperatorBase
   :members:
   :undoc-members:

.. doxygenclass:: muGrid::ConvolutionOperator
   :members:
   :undoc-members:

.. doxygenclass:: muGrid::LaplaceOperator
   :members:
   :undoc-members:

.. doxygenclass:: muGrid::FEMGradientOperator
   :members:
   :undoc-members:

FFT Engine
**********

Distributed FFT operations using pencil decomposition.

.. doxygenclass:: muGrid::FFTEngineBase
   :members:
   :undoc-members:

.. doxygenclass:: muGrid::FFTEngine
   :members:
   :undoc-members:

File I/O
********

Classes for reading and writing fields to disk in NetCDF format.

.. doxygenclass:: muGrid::FileIOBase
   :members:
   :undoc-members:

.. doxygenclass:: muGrid::FileIONetCDF
   :members:
   :undoc-members:

Core Types
**********

Fundamental types and enumerations used throughout the library.

.. doxygentypedef:: muGrid::Real

.. doxygentypedef:: muGrid::Complex

.. doxygentypedef:: muGrid::Int

.. doxygentypedef:: muGrid::Uint

.. doxygentypedef:: muGrid::Index_t

.. doxygenenum:: muGrid::IterUnit

.. doxygenenum:: muGrid::StorageOrder

Grid Utilities
**************

Utilities for working with grid coordinates and indices.

.. doxygenfunction:: muGrid::get_domain_ccoord

.. doxygenfunction:: muGrid::get_domain_index

.. doxygenfunction:: muGrid::fft_freq

.. doxygenfunction:: muGrid::fft_freqind

.. doxygenfunction:: muGrid::rfft_freq

.. doxygenfunction:: muGrid::rfft_freqind

Full Index
**********

For a complete listing of all classes, functions, and types:

.. doxygenindex::

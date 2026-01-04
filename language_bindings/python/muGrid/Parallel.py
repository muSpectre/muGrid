#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   Parallel.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   21 Mar 2018

@brief  muGrid Communicator object

Copyright © 2018 Till Junge

µGrid is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µGrid is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with µGrid; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.

Additional permission under GNU GPL version 3 section 7

If you modify this Program, or any covered work, by linking or combining it
with proprietary FFT implementations or numerical libraries, containing parts
covered by the terms of those libraries' licenses, the licensors of this
Program grant you additional permission to convey the resulting work.
"""

import numpy as np

# Import the C++ extension module
# Try relative import first (for installed wheels),
# fall back to absolute (for development)
try:
    from . import _muGrid
except ImportError:
    import _muGrid

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


# CuPy is optional - only imported when GPU arrays are used
_cupy = None


def _get_cupy():
    """Lazy import of CuPy for GPU array support."""
    global _cupy
    if _cupy is None:
        try:
            import cupy

            _cupy = cupy
        except ImportError:
            raise ImportError(
                "CuPy is required for GPU array operations. "
                "Install it with: pip install cupy-cuda12x "
                "(or appropriate CUDA or ROCm version)"
            )
    return _cupy


def _is_cupy_array(obj):
    """Check if an object is a CuPy array without importing CuPy."""
    return type(obj).__module__.startswith("cupy")


def _get_mpi_dtype(arr):
    """Get the MPI datatype corresponding to a numpy/cupy array dtype."""
    dtype = arr.dtype
    if dtype == np.float64:
        return MPI.DOUBLE
    elif dtype == np.float32:
        return MPI.FLOAT
    elif dtype == np.int32:
        return MPI.INT
    elif dtype == np.int64:
        return MPI.LONG
    elif dtype == np.complex128:
        return MPI.DOUBLE_COMPLEX
    elif dtype == np.complex64:
        return MPI.COMPLEX
    else:
        raise TypeError(f"Unsupported dtype for MPI reduction: {dtype}")


class CommunicatorWrapper:
    """
    Python wrapper for muGrid Communicator with GPU-aware reduction operations.

    This wrapper provides reduction operations (sum, max, etc.) that work
    efficiently with both CPU (numpy) and GPU (CuPy) arrays:

    - For serial execution (size=1): GPU arrays are returned directly without
      forcing synchronization or device-to-host copies.
    - For MPI parallel execution: Uses GPU-aware MPI when available to perform
      reductions directly on GPU memory.

    The wrapper delegates all other operations to the underlying C++ Communicator.
    """

    def __init__(self, cpp_comm, mpi_comm=None):
        """
        Initialize the Communicator wrapper.

        Parameters
        ----------
        cpp_comm : _muGrid.Communicator
            The underlying C++ communicator object
        mpi_comm : mpi4py.MPI.Comm, optional
            The mpi4py communicator for GPU-aware operations
        """
        self._cpp = cpp_comm
        self._mpi_comm = mpi_comm

    @property
    def rank(self):
        """Get rank of present process."""
        return self._cpp.rank

    @property
    def size(self):
        """Get total number of processes."""
        return self._cpp.size

    def barrier(self):
        """Barrier synchronization."""
        self._cpp.barrier()

    def sum(self, arg):
        """
        Sum reduction across all processes.

        For GPU arrays (CuPy), this method avoids unnecessary synchronization:
        - Serial (size=1): Returns the input array directly (no sync)
        - MPI parallel: Uses GPU-aware MPI for efficient reduction

        Parameters
        ----------
        arg : scalar, numpy array, or CuPy array
            Value to reduce

        Returns
        -------
        Same type as input
            Global sum across all processes
        """
        # Check if it's a GPU array
        if hasattr(arg, "__cuda_array_interface__") or _is_cupy_array(arg):
            return self._sum_gpu(arg)

        # For CPU scalars and arrays, use the C++ implementation
        return self._cpp.sum(arg)

    def _sum_gpu(self, arg):
        """GPU-aware sum reduction."""
        cp = _get_cupy()

        # Serial case: no reduction needed, return as-is
        if self.size == 1:
            return arg

        # MPI case: use GPU-aware MPI
        if self._mpi_comm is None:
            raise RuntimeError(
                "MPI communicator not available for GPU reduction. "
                "This should not happen if the wrapper was created correctly."
            )

        # Handle 0-D arrays (scalars on GPU)
        is_scalar = arg.ndim == 0
        if is_scalar:
            # Convert to 1-D for MPI, then back to 0-D
            arg = arg.reshape(1)

        # Allocate output buffer on GPU
        result = cp.empty_like(arg)

        # Use GPU-aware MPI Allreduce
        self._mpi_comm.Allreduce(arg, result, op=MPI.SUM)

        # Convert back to 0-D if input was scalar
        if is_scalar:
            result = result.reshape(())

        return result

    def max(self, arg):
        """
        Max reduction across all processes.

        For GPU arrays (CuPy), this method avoids unnecessary synchronization:
        - Serial (size=1): Returns the input array directly (no sync)
        - MPI parallel: Uses GPU-aware MPI for efficient reduction

        Parameters
        ----------
        arg : scalar, numpy array, or CuPy array
            Value to reduce

        Returns
        -------
        Same type as input
            Global maximum across all processes
        """
        # Check if it's a GPU array
        if hasattr(arg, "__cuda_array_interface__") or _is_cupy_array(arg):
            return self._max_gpu(arg)

        # For CPU scalars and arrays, use the C++ implementation
        return self._cpp.max(arg)

    def _max_gpu(self, arg):
        """GPU-aware max reduction."""
        cp = _get_cupy()

        # Serial case: no reduction needed, return as-is
        if self.size == 1:
            return arg

        # MPI case: use GPU-aware MPI
        if self._mpi_comm is None:
            raise RuntimeError("MPI communicator not available for GPU reduction.")

        # Handle 0-D arrays (scalars on GPU)
        is_scalar = arg.ndim == 0
        if is_scalar:
            arg = arg.reshape(1)

        # Allocate output buffer on GPU
        result = cp.empty_like(arg)

        # Use GPU-aware MPI Allreduce
        self._mpi_comm.Allreduce(arg, result, op=MPI.MAX)

        if is_scalar:
            result = result.reshape(())

        return result

    def all(self, arg):
        """
        Logical AND reduction across all processes.

        Parameters
        ----------
        arg : bool
            Local boolean value

        Returns
        -------
        bool
            True if all processes have True, False otherwise
        """
        # For GPU scalars, need to sync to get the boolean value
        if hasattr(arg, "__cuda_array_interface__") or _is_cupy_array(arg):
            if self.size == 1:
                # Still need to convert to Python bool for the return type
                return bool(arg)
            # For MPI, we need the boolean value on CPU
            arg = bool(arg)

        return self._cpp.all(arg)

    def any(self, arg):
        """
        Logical OR reduction across all processes.

        Parameters
        ----------
        arg : bool
            Local boolean value

        Returns
        -------
        bool
            True if any process has True, False otherwise
        """
        # For GPU scalars, need to sync to get the boolean value
        if hasattr(arg, "__cuda_array_interface__") or _is_cupy_array(arg):
            if self.size == 1:
                return bool(arg)
            arg = bool(arg)

        return self._cpp.any(arg)

    def cumulative_sum(self, arg):
        """
        Ordered partial cumulative sum across processes.

        Parameters
        ----------
        arg : scalar
            Local value to include in cumulative sum

        Returns
        -------
        scalar
            Cumulative sum up to and including this process
        """
        # For GPU scalars, convert to CPU (MPI_Scan doesn't have GPU-aware
        # variant in most implementations)
        if hasattr(arg, "__cuda_array_interface__") or _is_cupy_array(arg):
            arg = float(arg)

        return self._cpp.cumulative_sum(arg)

    def gather(self, arg):
        """
        Gather arrays from all processes.

        Parameters
        ----------
        arg : array
            Local array to gather

        Returns
        -------
        array
            Gathered array containing data from all processes
        """
        # For now, delegate to C++ implementation (requires CPU array)
        if hasattr(arg, "__cuda_array_interface__") or _is_cupy_array(arg):
            import warnings

            warnings.warn(
                "GPU gather not yet implemented, copying to CPU", RuntimeWarning
            )
            arg = arg.get()

        return self._cpp.gather(arg)

    def bcast(self, scalar_arg, root):
        """
        Broadcast value from root to all processes.

        Parameters
        ----------
        scalar_arg : scalar
            Value to broadcast (only used on root)
        root : int
            Rank of the broadcasting process

        Returns
        -------
        scalar
            Broadcasted value
        """
        # For GPU scalars, convert to CPU
        if hasattr(scalar_arg, "__cuda_array_interface__") or _is_cupy_array(
            scalar_arg
        ):
            scalar_arg = float(scalar_arg)

        return self._cpp.bcast(scalar_arg, root)

    # Delegate other attributes to the C++ communicator
    def __getattr__(self, name):
        """Delegate attribute access to the underlying C++ communicator."""
        return getattr(self._cpp, name)


def Communicator(communicator=None):
    """
    Factory function for the communicator class.

    Parameters
    ----------
    communicator: mpi4py or muGrid communicator object
        The bare MPI communicator. (Default: _muGrid.Communicator())

    Returns
    -------
    CommunicatorWrapper
        A wrapped communicator with GPU-aware reduction operations
    """
    # If the communicator is None, we return a communicator that contains just
    # the present process.
    if communicator is None:
        cpp_comm = _muGrid.Communicator()
        return CommunicatorWrapper(cpp_comm, mpi_comm=None)

    # If the communicator is already a CommunicatorWrapper, return it
    if isinstance(communicator, CommunicatorWrapper):
        return communicator

    # If the communicator is already an instance of _muGrid.Communicator,
    # wrap it
    if isinstance(communicator, _muGrid.Communicator):
        return CommunicatorWrapper(communicator, mpi_comm=None)

    # Now we need to do some magic. See if the communicator that was passed
    # conforms with the mpi4py interface, i.e. it has a method 'Get_size'.
    # The present magic enables using either mpi4py or stub implementations
    # of the same interface.
    if hasattr(communicator, "Get_size"):
        # If the size of the communicator group is 1, just return a
        # communicator that contains just the present process.
        if communicator.Get_size() == 1:
            cpp_comm = _muGrid.Communicator()
            return CommunicatorWrapper(cpp_comm, mpi_comm=None)
        # Otherwise, check if muGrid does actually have MPI support. If yes
        # we assume that the communicator is an mpi4py communicator.
        elif _muGrid.Communicator.has_mpi:
            if not MPI:
                raise RuntimeError(
                    "muGrid was compiled with MPI support but "
                    "mpi4py could not be loaded."
                )
            cpp_comm = _muGrid.Communicator(MPI._handleof(communicator))
            # Store the mpi4py communicator for GPU-aware operations
            return CommunicatorWrapper(cpp_comm, mpi_comm=communicator)
        else:
            raise RuntimeError("muGrid was compiled without MPI support.")
    else:
        raise RuntimeError(
            "The communicator does not have a 'Get_size' "
            "method. muGrid only supports communicators that "
            "conform to the mpi4py interface."
        )


def parprint(*args, comm=None, **kwargs):
    """
    MPI-safe print function that only prints on rank 0.

    This function behaves like the built-in print() function but only executes
    on the master process (rank 0) in an MPI environment. This prevents
    duplicate output when running in parallel.

    Parameters
    ----------
    *args
        Positional arguments passed to print()
    comm : Communicator, optional
        The communicator to use. If None, prints unconditionally (useful for
        serial execution or when used outside of an MPI context).
    **kwargs
        Keyword arguments passed to print()

    Examples
    --------
    >>> from muGrid import Communicator
    >>> from muGrid.Parallel import parprint
    >>> comm = Communicator()
    >>> parprint("This only prints on rank 0", comm=comm)
    >>> parprint("This prints unconditionally")  # No comm specified
    """
    if comm is None or comm.rank == 0:
        print(*args, **kwargs)

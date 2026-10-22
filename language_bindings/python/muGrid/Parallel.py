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

import os
import pickle
import sys
import warnings

import numpy as np

# Import the C++ extension module
# Try relative import first (for installed wheels),
# fall back to absolute (for development)
try:
    from . import _muGrid
except ImportError:
    import _muGrid


def _loaded_mpi4py():
    """The ``mpi4py.MPI`` module if the process has imported it, else None.

    muGrid never imports mpi4py itself: importing it initialises MPI, which
    muGrid does only when asked (:meth:`Communicator.world`), and loads a
    second MPI library if mpi4py was built against another one than muGrid.
    An mpi4py communicator can only have come from an already imported
    mpi4py, so looking it up is enough.
    """
    return sys.modules.get("mpi4py.MPI")


def _check_same_mpi_library(MPI):
    """Raise unless mpi4py and muGrid are linked against one MPI library.

    Handles are only meaningful within one library, so wrapping a
    communicator from another would at best fail obscurely. Two libraries
    never agree on the value of ``MPI_COMM_WORLD`` (a pointer in Open MPI, a
    fixed integer in MPICH), which makes it a cheap and exact test; version
    strings are only for the message."""
    if MPI._handleof(MPI.COMM_WORLD) != _muGrid.mpi_comm_world_handle():
        raise RuntimeError(
            "mpi4py and muGrid are linked against different MPI libraries, "
            "so their communicators cannot be exchanged. mpi4py: "
            f"{MPI.Get_library_version().splitlines()[0]!r}; muGrid: "
            f"{_muGrid.mpi_library_version().splitlines()[0]!r}. Use "
            "muGrid.Communicator.world() instead of mpi4py, or install an "
            "mpi4py built against muGrid's MPI."
        )


#: DLPack device types of host memory (plain, CUDA-pinned, ROCm-pinned);
#: every other device type is GPU memory
_DLPACK_HOST_DEVICES = (1, 3, 11)


def _is_device_array(obj):
    """Whether `obj` is an array in GPU memory, of any library: its DLPack
    device (or, for objects without DLPack, CUDA's array interface) says so.
    muGrid never imports an array library to find out."""
    device = getattr(obj, "__dlpack_device__", None)
    if device is not None:
        try:
            return device()[0] not in _DLPACK_HOST_DEVICES
        except Exception:  # e.g. a 0-d view some libraries refuse to export
            pass
    return hasattr(obj, "__cuda_array_interface__")


class Communicator:
    """
    Python conveniences on top of the C++ communicator (`_muGrid.Communicator`).

    ``Communicator()`` is a serial communicator (one rank; it never starts
    MPI), also under an MPI launcher, where it warns; ``Communicator.world()``
    is ``MPI_COMM_WORLD``, and the one thing that starts MPI. ``Communicator(x)``
    wraps a bare ``_muGrid.Communicator`` or an mpi4py communicator, and
    returns a ``Communicator`` unchanged.

    Array exchanges on NumPy and CuPy arrays (``allreduce``, ``allgather``,
    ``allgatherv``, ``sendrecv``) are the C++ communicator's own, which hands
    device buffers to MPI directly when it is GPU-aware and stages them
    through the host otherwise. On top of those this wrapper adds:

    - reductions that dispatch on the argument -- host scalars by dtype,
      CuPy arrays on the device, and serial runs without a copy;
    - full-array reductions (``reduce_*``) and a NuMPI-compatible
      ``reduction`` object;
    - ``isum``/``isum_wait``, a non-blocking sum on the C++ communicator's
      ``iallreduce``;
    - ``bcast``, ``gather`` and ``allgather`` of Python objects, with
      mpi4py's lower-case semantics (objects pickled, arrays as buffers);
    - ``split``/``dup`` returning wrapped communicators, and the mpi4py
      communicator (``mpi4py_comm``) when there is one.

    Everything else is delegated to the C++ communicator.
    """

    def __new__(cls, communicator=None):
        """
        Parameters
        ----------
        communicator : None, Communicator, _muGrid.Communicator or mpi4py.MPI.Comm
            ``None`` (default) gives a serial communicator. A
            ``Communicator`` is returned unchanged; a bare
            ``_muGrid.Communicator`` is wrapped. An mpi4py communicator of
            size 1 gives a serial communicator, and of larger size is wrapped
            together with its mpi4py handle, after a check that mpi4py and
            muGrid use one MPI library.

        Raises
        ------
        RuntimeError
            If a multi-rank mpi4py communicator is passed to a build without
            MPI support or one linked against another MPI library, or if the
            object is none of the above.

        Warns
        -----
        RuntimeWarning
            When a serial communicator is constructed (``communicator=None``)
            while the environment indicates the process was started by a
            multi-rank MPI launcher -- usually an accident that makes every
            rank solve the full problem independently. Intentional serial use
            under a launcher keeps working; suppress the warning with
            :func:`warnings.catch_warnings` if it is deliberate.
        """
        if communicator is None:
            _warn_if_serial_under_launcher()
            return cls._wrap(_muGrid.Communicator())
        if isinstance(communicator, Communicator):
            return communicator
        if isinstance(communicator, _muGrid.Communicator):
            return cls._wrap(communicator)
        # Duck-type mpi4py's interface, so that a stub of it (NuMPI's, say)
        # of size 1 works without mpi4py
        if not hasattr(communicator, "Get_size"):
            raise RuntimeError(
                f"cannot make a muGrid Communicator from {communicator!r}: "
                "pass a Communicator, a _muGrid.Communicator or an mpi4py "
                "communicator, or call Communicator.world()"
            )
        if communicator.Get_size() == 1:
            # Serial, but keep the object: tools asking for mpi4py_comm get
            # what the caller had
            return cls._wrap(_muGrid.Communicator(), mpi_comm=communicator)
        if not _muGrid.Communicator.has_mpi:
            raise RuntimeError("muGrid was compiled without MPI support.")
        MPI = _loaded_mpi4py()
        if MPI is None or not isinstance(communicator, MPI.Comm):
            raise RuntimeError(
                f"cannot make a muGrid Communicator from {communicator!r}: "
                "only mpi4py communicators can be wrapped, or call "
                "Communicator.world()"
            )
        _check_same_mpi_library(MPI)
        cpp_comm = _muGrid.Communicator(MPI._handleof(communicator))
        cpp_comm.discover_node()
        return cls._wrap(cpp_comm, mpi_comm=communicator)

    def __init__(self, communicator=None):
        # Everything happens in __new__, which may return an existing
        # instance; nothing to redo here.
        pass

    @classmethod
    def _wrap(cls, cpp_comm, mpi_comm=None):
        """A wrapper around `cpp_comm`, with `mpi_comm` the mpi4py
        communicator it was made from, if any."""
        self = object.__new__(cls)
        self._cpp = cpp_comm
        self._mpi_comm = mpi_comm
        return self

    @classmethod
    def world(cls):
        """``MPI_COMM_WORLD``, starting MPI if nobody has.

        The only thing in muGrid that initialises MPI (with
        ``MPI_THREAD_SERIALIZED``); importing muGrid does not. Collective:
        every rank calls it. In a build without MPI it is the serial
        communicator, unless the process runs under an MPI launcher, where
        that would silently make every rank solve the whole problem -- then
        it raises.
        """
        if not _muGrid.Communicator.has_mpi:
            nb_ranks = _mpi_launcher_world_size()
            if nb_ranks > 1:
                raise RuntimeError(
                    f"Communicator.world() under an MPI launcher with "
                    f"{nb_ranks} ranks, but this build of muGrid has no MPI "
                    "support; install or build muGrid with MPI"
                )
        return cls._wrap(_muGrid.Communicator.world())

    @property
    def mpi4py_comm(self):
        """The mpi4py communicator this was made from, or, if mpi4py is
        imported, one made from the same handle; else None.

        Tools like muTimer use this to gather Python objects across ranks.
        Note that `mpi_comm` (delegated to the C++ communicator) returns the
        raw MPI handle instead.
        """
        if self._mpi_comm is None and self.size > 1:
            MPI = _loaded_mpi4py()
            if MPI is not None and hasattr(MPI.Comm, "fromhandle"):
                _check_same_mpi_library(MPI)
                self._mpi_comm = MPI.Comm.fromhandle(self._cpp.mpi_comm)
        return self._mpi_comm

    def split(self, color, key=0):
        """A communicator over the ranks passing the same `color`, ordered
        by `key` (``MPI_Comm_split``). Collective."""
        return self._wrap(self._cpp.split(color, key))

    def dup(self):
        """A new handle to the same group (``MPI_Comm_dup``). Collective."""
        return self._wrap(self._cpp.dup())

    def compare(self, other):
        """How this communicator relates to `other`: ``"identical"``,
        ``"congruent"`` (same ranks, same order), ``"similar"`` (same ranks)
        or ``"unequal"``."""
        return self._cpp.compare(Communicator(other)._cpp)

    @property
    def node_rank(self):
        """Rank among the ranks on this node, for one GPU per rank."""
        return self._cpp.node_rank

    @property
    def node_size(self):
        """Number of ranks on this node."""
        return self._cpp.node_size

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
        Sum reduction across all processes, elementwise for arrays.

        Arrays -- NumPy or CuPy, of any shape and of dtype float32/64,
        complex64/128, int32/64 or uint32 -- go through the C++
        communicator's ``allreduce`` and come back with their shape, dtype
        and kind; device arrays are handed to MPI directly or staged through
        the host, as the communicator decides. In a serial run a CuPy array
        is returned as it is, without waiting for the device. Host scalars
        are reduced with the MPI datatype of their dtype (see
        `_reduce_host_scalar`).

        Parameters
        ----------
        arg : scalar, array-like, or CuPy array
            Value to reduce

        Returns
        -------
        Same type as input
            Global sum across all processes
        """
        if not self._is_gpu(arg) and np.ndim(arg) == 0:
            return self._reduce_host_scalar(arg, "sum")
        return self._allreduce_array(arg, "sum")

    def isum(self, arg):
        """Begin a non-blocking sum reduction across all processes.

        Returns a handle to pass to :meth:`isum_wait`. Between the two calls
        the caller can perform work that does not depend on the reduced
        value, so the global synchronisation latency is hidden behind that
        work (the key to pipelined CG; Ghysels & Vanroose 2014). The handle
        is the C++ communicator's ``iallreduce`` request; its ``wait()`` does
        what :meth:`isum_wait` does.

        Meant for small arrays (the use case is fusing a handful of inner
        products into one reduction); device arrays are staged through the
        host. In a serial run the sum is ready at once.

        Parameters
        ----------
        arg : numpy or CuPy array, or scalar
            Local contribution to reduce. Must not be modified until the
            reduction completes.

        Returns
        -------
        handle
            Pass to :meth:`isum_wait` to obtain the global sum.
        """
        return self._cpp.iallreduce(arg, "sum")

    def isum_wait(self, handle):
        """Complete a reduction started with :meth:`isum` and return its sum,
        an array of the argument's shape and kind."""
        return handle.wait()

    def _allreduce_array(self, arg, op):
        """Elementwise `op` over all ranks of a host or device array.

        The C++ communicator's ``allreduce``: the result has the shape, dtype
        and kind (NumPy or CuPy) of `arg`, which can be anything array-like.
        A CuPy array in a serial run is returned as it is, so that a serial
        solve never waits for the device."""
        if self.size == 1 and self._is_gpu(arg):
            return arg
        return self._cpp.allreduce(arg, op)

    def max(self, arg):
        """Max reduction across all processes, elementwise for arrays; see
        :meth:`sum` (complex values have no maximum)."""
        if not self._is_gpu(arg) and np.ndim(arg) == 0:
            return self._reduce_host_scalar(arg, "max")
        return self._allreduce_array(arg, "max")

    def min(self, arg):
        """Min reduction across all processes; see :meth:`sum`."""
        if not self._is_gpu(arg) and np.ndim(arg) == 0:
            return self._reduce_host_scalar(arg, "min")
        return self._allreduce_array(arg, "min")

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
        if _is_device_array(arg):
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
        if _is_device_array(arg):
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
        if _is_device_array(arg):
            arg = float(arg)

        return self._cpp.cumulative_sum(arg)

    # -- Python objects, as mpi4py's lower-case collectives ---------------- #
    #
    # The C++ communicator moves bytes and numbers; what an object is made of
    # is Python's business, so objects travel pickled, as uint8 arrays. Arrays
    # of numbers are the exception: they go as buffers, so that a CuPy array
    # stays on its device and a large one is not copied through pickle.

    @staticmethod
    def _is_array(obj):
        """Whether `obj` goes as a buffer: anything with an array interface
        or DLPack, host or device, including NumPy scalars."""
        return (
            hasattr(obj, "__array_interface__")
            or hasattr(obj, "__dlpack__")
            or hasattr(obj, "__cuda_array_interface__")
        )

    def _bcast_bytes(self, data, root):
        """Broadcast a bytes-like from `root` (ignored elsewhere)."""
        length = np.zeros(1, dtype=np.int64)
        if self.rank == root:
            length[0] = len(data)
        self._cpp.bcast(length, root)
        buffer = np.empty(int(length[0]), dtype=np.uint8)
        if self.rank == root:
            buffer[...] = np.frombuffer(data, dtype=np.uint8)
        self._cpp.bcast(buffer, root)
        return buffer.tobytes()

    def bcast(self, obj, root=0):
        """Rank `root`'s `obj` on every rank, as mpi4py's ``bcast``.

        An array (of any library speaking DLPack: NumPy, CuPy, PyTorch, ...)
        arrives as an array of the same library on every rank -- a device
        array on the receiving rank's current device -- through a buffer
        broadcast; any other object -- a number
        keeping its type, a dict of settings, None -- is pickled. Only the
        root's argument is read.

        Parameters
        ----------
        obj : object
            What to broadcast; whatever the other ranks pass is ignored.
        root : int, optional
            Rank whose `obj` is broadcast (default 0).
        """
        if self.size == 1:
            return obj
        is_root = self.rank == root
        if is_root and isinstance(obj, np.ndarray) and obj.ndim > 0:
            # NumPy describes any of its dtypes, also those DLPack cannot
            array = np.ascontiguousarray(obj)
            header = ("numpy", array.shape,
                      np.lib.format.dtype_to_descr(array.dtype))
        elif is_root and self._is_array(obj) and np.ndim(obj) > 0:
            array = _muGrid._exchange_contiguous(obj)
            header = ("array", _muGrid._exchange_describe(array))
        elif is_root:
            header = ("object", obj)
        else:
            header = None
        header = pickle.loads(self._bcast_bytes(
            pickle.dumps(header, protocol=pickle.HIGHEST_PROTOCOL)
            if is_root else b"", root))
        if header[0] == "object":
            return header[1]
        if not is_root and header[0] == "numpy":
            array = np.empty(header[1],
                             dtype=np.lib.format.descr_to_dtype(header[2]))
        elif not is_root:
            # Same shape, dtype, kind of memory and library as the root's,
            # on this rank's current device
            array = _muGrid._exchange_empty(*header[1])
        self._cpp.bcast(array, root)
        return array

    def gather(self, obj, root=0):
        """Every rank's `obj`, as a list in rank order on rank `root` and
        None on the other ranks, as mpi4py's ``gather``.

        Objects are pickled (a CuPy array is thereby copied to the host);
        for arrays of one shape on every rank that should stay on the device
        and arrive stacked, see :meth:`allgather`.
        """
        payload = np.frombuffer(
            pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL), dtype=np.uint8
        )
        if self.size == 1:
            return [pickle.loads(payload.tobytes())]
        lengths = self._cpp.allgather(np.asarray([payload.size]))[:, 0]
        data = self._cpp.gatherv(payload, root)
        if data is None:
            return None
        data = data.tobytes()
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        return [pickle.loads(data[offsets[p]:offsets[p + 1]])
                for p in range(self.size)]

    def allgather(self, obj):
        """Every rank's `obj`, on every rank.

        An array (NumPy or CuPy, of one shape and dtype on every rank) or a
        NumPy scalar comes back stacked along a new leading axis, shape
        ``(size, *obj.shape)``, of the same kind; anything else -- a Python
        number, a string, a dict -- comes back as a list in rank order, as
        from mpi4py's ``allgather``.
        """
        if self._is_array(obj):
            return self._cpp.allgather(obj)
        payload = np.frombuffer(
            pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL), dtype=np.uint8
        )
        if self.size == 1:
            return [pickle.loads(payload.tobytes())]
        lengths = self._cpp.allgather(np.asarray([payload.size]))[:, 0]
        data = self._cpp.allgatherv(payload).tobytes()
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        return [pickle.loads(data[offsets[p]:offsets[p + 1]])
                for p in range(self.size)]

    # ------------------------------------------------------------------ #
    # Full-array global reductions (input array -> single global scalar)  #
    # ------------------------------------------------------------------ #
    #
    # Note the distinction from `sum`/`max` above: those reduce *element-
    # wise* across ranks and return an array of the same shape. The
    # `reduce_*` methods below reduce over *all elements on all ranks* and
    # return a single scalar (Allreduce semantics: every rank gets it).

    @staticmethod
    def _is_gpu(a):
        """True if `a` is an array in GPU memory (see `_is_device_array`)."""
        return _is_device_array(a)

    @staticmethod
    def _as_array(a):
        """Coerce a Python scalar/sequence to a numpy array; pass arrays
        (numpy or CuPy) through unchanged so device placement is preserved."""
        if hasattr(a, "size"):
            return a
        return np.asarray(a)

    def _local_reduce(self, a, op):
        """Reduce the local array `a` to a 0-d scalar on its own device.

        Empty subdomains contribute the identity for `op` (0 for sum,
        +inf for min, -inf for max) as a 0-d array on the same device, so
        that every rank takes the same code path in the subsequent
        Allreduce (mixing host/device buffers would mismatch the
        collective)."""
        if a.size:
            return getattr(a, op)()
        dtype = np.dtype(a.dtype if hasattr(a, "dtype") else np.float64)
        if np.issubdtype(dtype, np.integer):
            # +-inf does not exist in an integer type: converting it would
            # raise OverflowError on the empty rank only, while the other
            # ranks wait forever in the Allreduce.
            info = np.iinfo(dtype)
            identity = {"sum": 0, "min": info.max, "max": info.min}[op]
        else:
            identity = {"sum": 0.0, "min": float("inf"),
                        "max": float("-inf")}[op]
        if self._is_gpu(a):
            return _muGrid._exchange_full_like(a, identity)
        return np.asarray(identity, dtype=dtype)

    def _allreduce_scalar(self, local, op):
        """Allreduce a per-rank scalar to a global scalar.

        `op` is one of "sum", "min", "max". `local` is a Python number or
        a 0-d numpy/CuPy array. Returns the same kind as the input (0-d
        array on device for GPU inputs, Python/numpy scalar on CPU)."""
        # Serial fast path: no MPI call at all (works for CuPy without
        # CUDA-aware MPI).
        if self.size == 1:
            return local

        if self._is_gpu(local):
            return self._cpp.allreduce(local, op)

        # CPU scalar: route through the C++ communicator (robust when
        # `_mpi_comm` is None but the C++ communicator is MPI-backed).
        return self._reduce_host_scalar(local, op)

    def _reduce_host_scalar(self, value, op):
        """Reduce one host scalar with the MPI datatype taken from its dtype.

        Converting through ``float()`` would drop the imaginary part of
        complex values and round integers beyond 2^53; picking the C++
        overload by value could make two ranks use different datatypes in one
        collective. The dtype kind is the same on every rank, so dispatching
        on it keeps the collective consistent."""
        value = np.asarray(value)
        kind = value.dtype.kind
        if kind == "c":
            if op != "sum":
                raise TypeError(f"no {op} for complex values")
            return getattr(self._cpp, "_sum_complex128")(complex(value))
        if kind in "iub":
            return getattr(self._cpp, f"_{op}_int64")(int(value))
        return getattr(self._cpp, f"_{op}_float64")(float(value))

    def reduce_sum(self, a):
        """Global sum over all elements of `a` on all ranks (scalar)."""
        a = self._as_array(a)
        return self._allreduce_scalar(self._local_reduce(a, "sum"), "sum")

    def reduce_min(self, a):
        """Global minimum element of `a` over all ranks (scalar).

        All-empty input returns +inf."""
        a = self._as_array(a)
        return self._allreduce_scalar(self._local_reduce(a, "min"), "min")

    def reduce_max(self, a):
        """Global maximum element of `a` over all ranks (scalar).

        All-empty input returns -inf."""
        a = self._as_array(a)
        return self._allreduce_scalar(self._local_reduce(a, "max"), "max")

    def reduce_mean(self, a):
        """Count-weighted global mean over all elements on all ranks.

        Equals (Σ_ranks Σ_elements a) / (Σ_ranks a.size), i.e. the mean of
        the global data, not the mean of per-rank means. Returns nan if the
        global element count is zero."""
        a = self._as_array(a)
        gsum = self.reduce_sum(a)
        # The element count is a host integer on every rank, so reduce it on
        # the CPU regardless of where `a` lives.
        gcount = float(self._allreduce_scalar(int(a.size), "sum"))
        if gcount == 0:
            return float("nan")
        return gsum / gcount

    @property
    def reduction(self):
        """A NuMPI-compatible reduction object with `.sum/.min/.max/.mean`.

        A drop-in for ``NuMPI.Tools.Reduction(comm)``: the methods have
        full-array global-reduction semantics (see `reduce_sum` etc.)."""
        return _Reduction(self)

    # Delegate other attributes to the C++ communicator
    def __getattr__(self, name):
        """Delegate attribute access to the underlying C++ communicator."""
        if name.startswith("_"):
            # Not yet wrapped, or a private name: never the C++ object's
            raise AttributeError(name)
        return getattr(self._cpp, name)


class _Reduction:
    """NuMPI-compatible adapter exposing full-array global reductions as
    ``.sum``/``.min``/``.max``/``.mean`` (see `CommunicatorWrapper.reduce_*`)."""

    def __init__(self, comm):
        self._comm = comm

    def sum(self, a):
        return self._comm.reduce_sum(a)

    def min(self, a):
        return self._comm.reduce_min(a)

    def max(self, a):
        return self._comm.reduce_max(a)

    def mean(self, a):
        return self._comm.reduce_mean(a)


#: Environment variables MPI launchers use to announce the world size.
_MPI_LAUNCHER_SIZE_VARS = (
    "OMPI_COMM_WORLD_SIZE",  # Open MPI
    "PMI_SIZE",  # MPICH / Hydra
    "MV2_COMM_WORLD_SIZE",  # MVAPICH2
    "SLURM_NTASKS",  # srun
)


def _mpi_launcher_world_size():
    """World size announced by an MPI launcher's environment, or 1 when not
    running under a (detectable) launcher."""
    for var in _MPI_LAUNCHER_SIZE_VARS:
        value = os.environ.get(var)
        if value:
            try:
                return int(value)
            except ValueError:
                pass
    return 1


def _warn_if_serial_under_launcher():
    nb_ranks = _mpi_launcher_world_size()
    if nb_ranks > 1:
        warnings.warn(
            f"This process runs under an MPI launcher with {nb_ranks} ranks, "
            "but a serial muGrid Communicator is being constructed: every "
            "rank will redundantly work on the full problem by itself. To "
            "enable domain decomposition, use muGrid.Communicator.world() "
            "(and pass it to the FFT engine).",
            RuntimeWarning,
            stacklevel=3,
        )


#: Former name of :class:`Communicator`
CommunicatorWrapper = Communicator


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

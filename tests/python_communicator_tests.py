#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_communicator_tests.py

@author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date   21 May 2019

@brief  Test muGrid's wrapper of the MPI communicator

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

import warnings

import numpy as np
import pytest

import muGrid


def test_sum_default():
    # The default communicator is COMM_SELF, i.e. each process by itself
    comm = muGrid.Communicator()
    assert comm.sum(comm.rank + 3) == 3


def test_serial_communicator_under_mpi_launcher_warns(monkeypatch):
    """Constructing a serial (default) communicator inside a multi-rank MPI
    launcher environment is almost always an accident (every rank solves the
    full problem) and must warn; a single-rank or launcher-free environment
    must stay silent."""
    from muGrid.Parallel import _MPI_LAUNCHER_SIZE_VARS

    for var in _MPI_LAUNCHER_SIZE_VARS:
        monkeypatch.delenv(var, raising=False)

    # No launcher detected: no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        muGrid.Communicator()

    # Single-rank launcher: no warning.
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "1")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        muGrid.Communicator()

    # Multi-rank launcher: warn.
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    with pytest.warns(RuntimeWarning, match="MPI launcher with 4 ranks"):
        muGrid.Communicator()

    # Explicitly passing a communicator never warns.
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    try:
        from mpi4py import MPI
    except ImportError:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        muGrid.Communicator(MPI.COMM_WORLD)


@pytest.mark.skipif(
    not muGrid.has_mpi, reason="muGrid was compiled without MPI support"
)
def test_sum_comm_world():
    comm = _make_comm()
    # 1 + 2 + 3 + ... + n = n*(n+1)/2
    assert comm.sum(comm.rank + 3) == comm.size * (comm.size + 1) / 2 + 2 * comm.size


def test_cum_sum_comm_world():
    comm = _make_comm()
    # 1 + 2 + 3 + ... + n = n*(n+1)/2
    assert (
        comm.cumulative_sum(comm.rank + 1)
        == comm.rank * (comm.rank + 1) / 2 + comm.rank + 1
    )


def test_bcast_1():
    # The default communicator is COMM_SELF, i.e. each process by itself
    comm = muGrid.Communicator()
    scalar_arg = comm.rank + 3
    res = comm.bcast(scalar_arg, 0)
    assert res == 3

    scalar_arg = comm.rank + 1
    res = comm.bcast(scalar_arg, root=comm.size - 1)
    assert res == comm.size


@pytest.mark.skipif(not muGrid.has_mpi, reason="muFFT was compiled without MPI support")
def test_bcast_2():
    comm = _make_comm()
    scalar_arg = comm.rank + 3
    res = comm.bcast(scalar_arg, 0)
    assert res == 3

    scalar_arg = comm.rank + 1
    res = comm.bcast(scalar_arg, root=comm.size - 1)
    assert res == comm.size


def _make_comm():
    return muGrid.Communicator.world() if muGrid.has_mpi else muGrid.Communicator()


def test_reduce_correctness_cpu():
    # Acceptance test 1 (CPU): reduce_* match numpy on the local array for the
    # serial communicator.
    comm = muGrid.Communicator()
    rng = np.random.default_rng(42)
    a = rng.standard_normal((4, 5, 6)).astype(np.float64)

    assert np.isclose(float(comm.reduce_sum(a)), a.sum(), rtol=1e-12)
    assert np.isclose(float(comm.reduce_min(a)), a.min(), rtol=1e-12)
    assert np.isclose(float(comm.reduce_max(a)), a.max(), rtol=1e-12)
    assert np.isclose(float(comm.reduce_mean(a)), a.mean(), rtol=1e-12)

    # 0-d input is treated as a single element.
    s = np.asarray(3.5)
    assert float(comm.reduce_sum(s)) == 3.5
    assert float(comm.reduce_mean(s)) == 3.5


def test_reduce_decomposition_invariance():
    # Acceptance test 2: split a global array across ranks with unequal
    # subdomains (including at least one empty rank) and check the four
    # reductions equal the single-process result.
    comm = _make_comm()
    rank, size = comm.rank, comm.size

    # Build the same global array on every rank from a fixed seed.
    rng = np.random.default_rng(1234)
    global_a = rng.standard_normal(101).astype(np.float64)

    # Uneven, contiguous partition. With >=2 ranks the last rank gets an
    # empty slice so the empty-subdomain path is exercised.
    bounds = [0] * (size + 1)
    if size == 1:
        bounds = [0, len(global_a)]
    else:
        # Put everything in the first (size-1) ranks, leave the last empty.
        per = len(global_a) // (size - 1)
        for i in range(size - 1):
            bounds[i + 1] = bounds[i] + per
        bounds[size - 1] = len(global_a)  # absorb remainder
        bounds[size] = len(global_a)  # last rank: empty
    local = global_a[bounds[rank]:bounds[rank + 1]]

    assert np.isclose(float(comm.reduce_sum(local)), global_a.sum(), rtol=1e-12)
    assert np.isclose(float(comm.reduce_min(local)), global_a.min(), rtol=1e-12)
    assert np.isclose(float(comm.reduce_max(local)), global_a.max(), rtol=1e-12)
    assert np.isclose(float(comm.reduce_mean(local)), global_a.mean(), rtol=1e-12)


def test_reduce_serial_no_allreduce():
    # Acceptance test 3: with size == 1, no Allreduce is invoked.
    comm = muGrid.Communicator()
    assert comm.size == 1

    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    # A 0-d local result is returned directly by the serial fast path.
    local = comm._local_reduce(a, "sum")
    assert comm._allreduce_scalar(local, "sum") is local


def test_reduce_collective_result():
    # Acceptance test 4: every rank returns the same value.
    comm = _make_comm()
    rng = np.random.default_rng(comm.rank + 7)
    a = rng.standard_normal(comm.rank * 3 + 5).astype(np.float64)

    for val in (comm.reduce_sum(a), comm.reduce_min(a),
                comm.reduce_max(a), comm.reduce_mean(a)):
        val = float(val)
        assert val == comm.bcast(val, 0)


def test_reduce_empty_and_mean_count():
    # Empty local input returns the identity rather than raising.
    comm = muGrid.Communicator()
    empty = np.zeros(0, dtype=np.float64)
    assert float(comm.reduce_sum(empty)) == 0.0
    assert float(comm.reduce_min(empty)) == float("inf")
    assert float(comm.reduce_max(empty)) == float("-inf")
    # Mean with zero global count is nan, not a division error.
    assert np.isnan(float(comm.reduce_mean(empty)))


def test_reduction_object_nuMPI_compatible():
    # The .reduction adapter mirrors NuMPI.Tools.Reduction's interface.
    comm = muGrid.Communicator()
    red = comm.reduction
    a = np.arange(10, dtype=np.float64)
    assert float(red.sum(a)) == a.sum()
    assert float(red.min(a)) == a.min()
    assert float(red.max(a)) == a.max()
    assert float(red.mean(a)) == a.mean()


def test_reduce_gpu_matches_numpy():
    # Acceptance tests 1/3/5 (GPU): same results as numpy on a CuPy input,
    # handled without CUDA-aware MPI in the serial case. Skips cleanly when
    # CuPy / a GPU is unavailable.
    cp = pytest.importorskip("cupy")
    try:
        rng = np.random.default_rng(99)
        a_np = rng.standard_normal((4, 5)).astype(np.float64)
        a_cp = cp.asarray(a_np)
    except Exception as exc:  # no usable GPU
        pytest.skip(f"CuPy present but no usable GPU: {exc}")

    comm = muGrid.Communicator()
    for name, ref in (
        ("reduce_sum", a_np.sum()),
        ("reduce_min", a_np.min()),
        ("reduce_max", a_np.max()),
        ("reduce_mean", a_np.mean()),
    ):
        res = getattr(comm, name)(a_cp)
        # Result is convertible to a Python float and matches numpy.
        assert np.isclose(float(res), ref, rtol=1e-12)


def test_allgatherv():
    comm = _make_comm()

    # gather arrays "a" with different lengths on the ranks
    a = np.arange(comm.rank * 2 + 4).reshape((-1, 2)).T

    a_gathered = comm.allgatherv(a, axis=1)

    # construct reference
    for i in range(comm.size):
        if i == 0:
            a_ref = np.arange(i * 2 + 4).reshape((-1, 2))
        elif i >= 1:
            a_tmp = np.arange(i * 2 + 4).reshape((-1, 2))
            a_ref = np.concatenate((a_ref, a_tmp), axis=0)

    assert (a_gathered == a_ref.T).all()


def _rank_array(xp, rank, dtype=np.complex128):
    """A small non-contiguous array that differs between ranks."""
    a = xp.arange(12, dtype=dtype).reshape(3, 4) * (1 + 1j) + rank
    return a[:, ::2]


def _array_modules():
    """numpy, and CuPy when there is a usable GPU."""
    try:
        import cupy

        cupy.zeros(1)
    except Exception:  # no CuPy, or no usable GPU
        return [np]
    return [np, cupy]


def _on_host(a):
    return a.get() if hasattr(a, "get") else a


@pytest.mark.parametrize("dtype", [np.float32, np.complex128])
def test_allgather(dtype):
    # Host and device arrays alike: device ones go to MPI directly or through
    # the host, whichever the MPI takes, and come back as device arrays.
    comm = _make_comm()
    for xp in _array_modules():
        local = _rank_array(xp, comm.rank, dtype)
        gathered = comm.allgather(local)
        assert type(gathered) is type(local)
        assert gathered.shape == (comm.size,) + local.shape
        for p in range(comm.size):
            np.testing.assert_array_equal(_on_host(gathered[p]),
                                          _rank_array(np, p, dtype))


def test_sendrecv_around_a_ring():
    comm = _make_comm()
    up, down = (comm.rank + 1) % comm.size, (comm.rank - 1) % comm.size
    for xp in _array_modules():
        local = _rank_array(xp, comm.rank)
        from_down = comm.sendrecv(local, dest=up, source=down, tag=3)
        assert type(from_down) is type(local)
        np.testing.assert_array_equal(_on_host(from_down),
                                      _rank_array(np, down))


def test_exchanges_on_the_bare_cpp_communicator():
    # The array exchanges are the C++ communicator's own: no wrapper, and so
    # no mpi4py communicator, needed.
    from muGrid import _muGrid

    comm = _make_comm()
    cpp = (_muGrid.Communicator(comm.mpi_comm) if muGrid.has_mpi
           else _muGrid.Communicator())
    for xp in _array_modules():
        x = xp.asarray([1.0, -2.0]) * (comm.rank + 1)
        total = cpp.allreduce(x, "sum")
        assert type(total) is type(x)
        n = comm.size * (comm.size + 1) / 2
        np.testing.assert_array_equal(_on_host(total), [n, -2 * n])
        np.testing.assert_array_equal(
            _on_host(cpp.allreduce(x, "max")), [comm.size, -2.0])
        np.testing.assert_array_equal(
            _on_host(cpp.allreduce(x, "min")), [1.0, -2.0 * comm.size])
        scalar = cpp.allreduce(xp.asarray(3.0), "sum")
        assert scalar.shape == ()
        assert float(scalar) == 3.0 * comm.size
        assert cpp.allgather(x).shape == (comm.size, 2)


def test_exchanges_reject_unsupported_dtypes():
    comm = _make_comm()
    with pytest.raises(TypeError, match="dtype"):
        comm.allreduce(np.zeros(2, dtype=np.int16))
    with pytest.raises(ValueError, match="op"):
        comm.allreduce(np.zeros(2), "prod")


def test_isum():
    comm = _make_comm()
    n = comm.size * (comm.size + 1) / 2
    for xp in _array_modules():
        local = xp.asarray([1.0, 2.0, -1.0]) * (comm.rank + 1)
        handle = comm.isum(local)
        total = comm.isum_wait(handle)
        assert type(total) is type(local)
        np.testing.assert_array_equal(_on_host(total), [n, 2 * n, -n])
        # waiting again returns the same result
        np.testing.assert_array_equal(_on_host(handle.wait()), _on_host(total))
    # a request dropped without waiting completes when it is collected
    comm.isum(np.ones(2))
    assert float(comm.isum_wait(comm.isum(2.0))) == 2.0 * comm.size


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.complex128,
                                   np.complex64, np.int32, np.int64,
                                   np.uint32])
@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 2, 2)])
def test_array_reductions_keep_shape_and_dtype(dtype, shape):
    comm = _make_comm()
    for xp in _array_modules():
        local = (xp.arange(np.prod(shape)).reshape(shape) + comm.rank
                 ).astype(dtype)
        reference = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
        total = comm.sum(local)
        assert type(total) is type(local)
        assert total.shape == shape and total.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(
            _on_host(total),
            reference * comm.size + sum(range(comm.size)))
        if np.issubdtype(dtype, np.complexfloating):
            continue
        np.testing.assert_array_equal(_on_host(comm.max(local)),
                                      reference + comm.size - 1)
        np.testing.assert_array_equal(_on_host(comm.min(local)), reference)


def test_allgatherv_concatenates_like_numpy():
    comm = _make_comm()
    for xp in _array_modules():
        parts = [xp.arange(3 * (p + 1), dtype=np.float32).reshape(p + 1, 3) + p
                 for p in range(comm.size)]
        result = comm.allgatherv(parts[comm.rank])
        assert type(result) is type(parts[0])
        np.testing.assert_array_equal(
            _on_host(result), np.concatenate([_on_host(p) for p in parts]))
        # along another axis
        result = comm.allgatherv(parts[comm.rank].T, axis=-1)
        np.testing.assert_array_equal(
            _on_host(result),
            np.concatenate([_on_host(p).T for p in parts], axis=-1))


@pytest.mark.parametrize("dtype", [np.uint8, np.bool_, np.int16, "S5",
                                   [("a", np.int8), ("b", np.float64)]])
def test_exchanges_move_any_fixed_size_dtype(dtype):
    # Bytes of varying length per rank -- what muTimer needs to gather its
    # serialised timers -- and other dtypes MPI has no arithmetic for
    comm = _make_comm()
    parts = [np.zeros(p + 2, dtype=dtype) for p in range(comm.size)]
    for p, part in enumerate(parts):
        part.view(np.uint8)[:] = np.arange(part.nbytes) % 2 + p
    gathered = comm.allgatherv(parts[comm.rank])
    assert gathered.dtype == np.dtype(dtype)
    assert gathered.tobytes() == b"".join(p.tobytes() for p in parts)
    same = np.zeros(3, dtype=dtype)
    same.view(np.uint8)[:] = comm.rank
    assert comm.allgather(same).shape == (comm.size, 3)
    up, down = (comm.rank + 1) % comm.size, (comm.rank - 1) % comm.size
    received = comm.sendrecv(same, dest=up, source=down)
    assert received.view(np.uint8).tolist() == [down] * same.nbytes


def test_allgatherv_round_trips_serialised_objects():
    import json

    comm = _make_comm()
    local = {"rank": comm.rank, "timers": ["solve"] * (comm.rank + 1)}
    payload = np.frombuffer(json.dumps(local).encode(), dtype=np.uint8)
    lengths = comm.allgather(np.asarray([payload.size]))[:, 0]
    data = comm.allgatherv(payload).tobytes()
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    decoded = [json.loads(data[offsets[p]:offsets[p + 1]])
               for p in range(comm.size)]
    assert [d["rank"] for d in decoded] == list(range(comm.size))


def test_allgatherv_rejects_mismatched_dtypes_and_objects():
    comm = _make_comm()
    with pytest.raises(TypeError, match="Python objects"):
        comm.allgatherv(np.array([object()]))
    if comm.size > 1:
        dtype = np.float32 if comm.rank == 0 else np.float64
        with pytest.raises(TypeError, match="one dtype"):
            comm.allgatherv(np.ones(2, dtype=dtype))


# -- Python objects, as mpi4py's lower-case collectives --------------------- #


def test_bcast_objects_and_arrays():
    comm = _make_comm()
    root = comm.size - 1
    # objects, pickled; only the root's argument counts
    assert comm.bcast({"n": 3, "name": "x"} if comm.rank == root else None,
                      root=root) == {"n": 3, "name": "x"}
    assert comm.bcast(None, root=root) is None
    assert comm.bcast("abc" if comm.rank == root else 0, root=root) == "abc"
    # numbers keep their type
    for value in (3, 2.5, 1 + 2j, True, np.float32(1.5)):
        got = comm.bcast(value if comm.rank == root else None, root=root)
        assert got == value and type(got) is type(value)
    # arrays go as buffers and arrive as arrays of the same kind
    for xp in _array_modules():
        sent = xp.arange(6, dtype=np.float32).reshape(2, 3)
        got = comm.bcast(sent if comm.rank == root else None, root=root)
        assert type(got) is type(sent)
        np.testing.assert_array_equal(_on_host(got), _on_host(sent))
    # default root is 0
    assert comm.bcast([comm.rank]) == [0]


def test_gather_is_a_list_on_the_root():
    comm = _make_comm()
    root = comm.size - 1
    local = {"rank": comm.rank, "timers": ["solve"] * (comm.rank + 1)}
    gathered = comm.gather(local, root=root)
    if comm.rank == root:
        assert [d["rank"] for d in gathered] == list(range(comm.size))
        assert gathered[-1]["timers"] == ["solve"] * comm.size
    else:
        assert gathered is None
    # what muTimer sends: a (dict, list) tuple per rank, to rank 0
    timers = comm.gather(({"a": {"total": 1.0 * comm.rank}}, ["a"]), root=0)
    if comm.rank == 0:
        assert [t[0]["a"]["total"] for t in timers] == list(range(comm.size))
    else:
        assert timers is None


def test_allgather_of_objects_is_a_list():
    comm = _make_comm()
    assert comm.allgather(f"rank {comm.rank}") == [
        f"rank {p}" for p in range(comm.size)]
    assert comm.allgather({"r": comm.rank}) == [{"r": p} for p in range(comm.size)]
    # arrays and NumPy scalars still come back stacked
    assert comm.allgather(np.float64(comm.rank)).shape == (comm.size,)
    assert comm.allgather(np.ones(2)).shape == (comm.size, 2)
    # a Python number is an object: a list, as from mpi4py
    assert comm.allgather(comm.rank) == list(range(comm.size))


def test_allreduce_scalars_and_mpi4py_ops():
    comm = _make_comm()
    n = comm.size
    total = comm.allreduce(2.0)
    assert total == 2.0 * n and type(total) is float
    assert type(comm.allreduce(3, "max")) is int
    assert comm.allreduce(np.float64(1.0)).shape == ()  # NumPy scalar: array
    with pytest.raises(ValueError, match="op must be"):
        comm.allreduce(1.0, "prod")
    MPI = pytest.importorskip("mpi4py.MPI", exc_type=ImportError)
    assert comm.allreduce(1.0, MPI.SUM) == n
    assert comm.allreduce(comm.rank, op=MPI.MAX) == n - 1
    np.testing.assert_array_equal(
        comm.allreduce(np.arange(2.0), MPI.MIN), [0.0, 1.0])
    with pytest.raises(ValueError, match="op must be"):
        comm.allreduce(1.0, MPI.PROD)


# -- Arrays of any library: read through DLPack, results handed back --------- #
#
# The exchanges know no array library: they read arrays through DLPack (or,
# without it, CUDA's array interface) and give results back through the
# input's own library's from_dlpack -- or as muGrid's ExchangeBuffer when it
# has none, as for the stand-in producers below.

_DLPACK_CPU = 1


class _DLPackOnly:
    """An array that speaks DLPack and nothing else, recording the `stream`
    each export was asked to order its work before."""

    def __init__(self, array):
        self._array = array
        self.streams = []

    def __dlpack__(self, stream=None, max_version=None, dl_device=None,
                   copy=None):
        self.streams.append(stream)
        if max_version is None:
            return self._array.__dlpack__(stream=stream)
        try:
            return self._array.__dlpack__(stream=stream,
                                          max_version=max_version)
        except TypeError:  # a producer predating DLPack 1.0
            return self._array.__dlpack__(stream=stream)

    def __dlpack_device__(self):
        return self._array.__dlpack_device__()


def test_dlpack_only_producer_on_the_host():
    comm = _make_comm()
    a = _DLPackOnly(np.arange(4.0) + comm.rank)
    s = comm._cpp.allreduce(a, "sum")
    # The producer's module has no from_dlpack: the result stays muGrid's
    assert type(s).__name__ == "ExchangeBuffer"
    assert s.__dlpack_device__() == (_DLPACK_CPU, 0)
    np.testing.assert_array_equal(
        np.from_dlpack(s), comm.size * np.arange(4.0) + sum(range(comm.size)))
    assert a.streams == [None]  # host memory: no stream


def test_exchanges_pack_any_strides():
    comm = _make_comm()
    base = np.arange(24.0).reshape(4, 6) + comm.rank
    total = comm.size * np.arange(24.0).reshape(4, 6) + sum(range(comm.size))
    for view, expected in [
        (base[:, ::2], total[:, ::2]),       # strided
        (base[::-1], total[::-1]),           # negative strides
        (base.T, total.T),                   # transposed
        (np.broadcast_to(base[0], (3, 6)),   # zero strides
         np.broadcast_to(total[0], (3, 6))),
    ]:
        np.testing.assert_array_equal(comm._cpp.allreduce(view, "sum"),
                                      expected)
    # The gathered axis of a strided view, and the result's axis order
    gathered = comm.allgatherv(base[:, ::2], axis=1)
    np.testing.assert_array_equal(
        gathered,
        np.concatenate([(np.arange(24.0).reshape(4, 6) + p)[:, ::2]
                        for p in range(comm.size)], axis=1))


def test_bcast_needs_a_writeable_array():
    comm = _make_comm()
    a = np.zeros(3)
    a.flags.writeable = False
    with pytest.raises(ValueError, match="writeable"):
        comm._cpp.bcast(a, 0)


def _cupy_or_skip():
    try:
        import cupy

        cupy.zeros(1)
    except Exception:
        pytest.skip("no CuPy with a usable GPU")
    return cupy


def test_dlpack_only_producer_on_the_device():
    """A device producer is asked to order its work before the legacy
    default stream: stream=1 on CUDA, None on ROCm."""
    cupy = _cupy_or_skip()
    comm = _make_comm()
    a = _DLPackOnly(cupy.arange(4.0) + comm.rank)
    s = comm._cpp.allreduce(a, "sum")
    assert s.__dlpack_device__() == a.__dlpack_device__()
    cupy.testing.assert_array_equal(
        cupy.from_dlpack(s),
        comm.size * cupy.arange(4.0) + sum(range(comm.size)))
    assert a.streams == [1 if muGrid.has_cuda else None]


class _CudaArrayInterfaceOnly:
    """A device array known only through CUDA's array interface (v3), with
    or without its `stream` key."""

    def __init__(self, array, stream="keep"):
        self._array = array
        interface = dict(array.__cuda_array_interface__)
        if stream is None:
            interface.pop("stream", None)
        elif stream != "keep":
            interface["stream"] = stream
        self.__cuda_array_interface__ = interface


@pytest.mark.parametrize("stream", ["keep", None, "other"])
def test_cuda_array_interface_only_producer(stream):
    """Without DLPack, the interface's stream is synchronised (an unknown
    one by draining the device)."""
    cupy = _cupy_or_skip()
    comm = _make_comm()
    other = cupy.cuda.Stream(non_blocking=True)
    with other:
        x = cupy.arange(1000.0) + comm.rank  # written on another stream
        # CuPy reports the *current* stream in its interface, so "keep"
        # must read it while `other` is current, as a producer would
        a = _CudaArrayInterfaceOnly(
            x, other.ptr if stream == "other" else stream)
    s = comm._cpp.allreduce(a, "sum")
    cupy.testing.assert_array_equal(
        cupy.from_dlpack(s),
        comm.size * cupy.arange(1000.0) + sum(range(comm.size)))


def test_device_results_come_back_in_the_input_library():
    cupy = _cupy_or_skip()
    comm = _make_comm()
    x = cupy.arange(6.0).reshape(2, 3) + comm.rank
    s = comm._cpp.allreduce(x, "sum")
    assert isinstance(s, cupy.ndarray)
    assert s.device.id == x.device.id
    g = comm.allgather(x)
    assert isinstance(g, cupy.ndarray) and g.shape == (comm.size, 2, 3)

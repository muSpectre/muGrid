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
    try:
        from mpi4py import MPI

        comm = muGrid.Communicator(MPI.COMM_WORLD)
    except ImportError:
        comm = muGrid.Communicator()
    # 1 + 2 + 3 + ... + n = n*(n+1)/2
    assert comm.sum(comm.rank + 3) == comm.size * (comm.size + 1) / 2 + 2 * comm.size


def test_cum_sum_comm_world():
    try:
        from mpi4py import MPI

        comm = muGrid.Communicator(MPI.COMM_WORLD)
    except ImportError:
        comm = muGrid.Communicator()
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
    res = comm.bcast(scalar_arg=scalar_arg, root=comm.size - 1)
    assert res == comm.size


@pytest.mark.skipif(not muGrid.has_mpi, reason="muFFT was compiled without MPI support")
def test_bcast_2():
    try:
        from mpi4py import MPI

        comm = muGrid.Communicator(MPI.COMM_WORLD)
    except ImportError:
        comm = muGrid.Communicator()
    scalar_arg = comm.rank + 3
    res = comm.bcast(scalar_arg, 0)
    assert res == 3

    scalar_arg = comm.rank + 1
    res = comm.bcast(scalar_arg=scalar_arg, root=comm.size - 1)
    assert res == comm.size


def _make_comm():
    try:
        from mpi4py import MPI

        return muGrid.Communicator(MPI.COMM_WORLD)
    except ImportError:
        return muGrid.Communicator()


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


def test_gather():
    try:
        from mpi4py import MPI

        comm = muGrid.Communicator(MPI.COMM_WORLD)
    except ImportError:
        comm = muGrid.Communicator()

    # gather arrays "a" with different lengths on the ranks
    a = np.arange(comm.rank * 2 + 4).reshape((-1, 2)).T

    a_gathered = comm.gather(a)

    # construct reference
    for i in range(comm.size):
        if i == 0:
            a_ref = np.arange(i * 2 + 4).reshape((-1, 2))
        elif i >= 1:
            a_tmp = np.arange(i * 2 + 4).reshape((-1, 2))
            a_ref = np.concatenate((a_ref, a_tmp), axis=0)

    assert (a_gathered == a_ref.T).all()

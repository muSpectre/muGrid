"""
Starting and stopping MPI, and the communicators that come from it.

muGrid never starts MPI on import; ``Communicator.world()`` does, and muGrid
finalises MPI at exit only if it started it. The lifecycle itself can only be
observed in a fresh interpreter, so those tests run subprocesses -- and only
when this process is not itself under an MPI launcher, where a child that
calls MPI_Init would try to join the launcher's job. The communicator tests
run in-process, under mpirun or not.
"""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import muGrid
from muGrid.Parallel import _MPI_LAUNCHER_SIZE_VARS, _mpi_launcher_world_size

needs_mpi = pytest.mark.skipif(not muGrid.has_mpi, reason="needs an MPI build")

#: What compare() says of two communicators over the same ranks that are not
#: one handle; the serial stub of a build without MPI has one handle only
SAME_GROUP = "congruent" if muGrid.has_mpi else "identical"


def _run(code, env=None):
    """Run `code` in a fresh interpreter; return its stdout lines."""
    full_env = dict(os.environ, **(env or {}))
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=full_env,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.split()


# Under a launcher -- even with one rank, as ctest runs these -- a child
# that calls MPI_Init would try to join the launcher's job and hang
subprocess_ok = pytest.mark.skipif(
    any(var in os.environ for var in _MPI_LAUNCHER_SIZE_VARS),
    reason="a child process would join the MPI launcher's job",
)


@subprocess_ok
def test_import_does_not_start_mpi():
    out = _run(
        """
        import sys, muGrid
        print(muGrid.mpi_initialized(), "mpi4py.MPI" in sys.modules)
        muGrid.Communicator()
        print(muGrid.mpi_initialized())
        """
    )
    assert out == ["False", "False", "False"]


@needs_mpi
@subprocess_ok
def test_world_starts_mpi_and_exit_stops_it():
    # An atexit hook registered before muGrid is imported runs after
    # muGrid's (LIFO), so it sees whether muGrid finalised MPI.
    out = _run(
        """
        import atexit
        # Registered before muGrid's own exit hook, so it runs after it (LIFO)
        atexit.register(lambda: print(
            "finalized", __import__("muGrid")._muGrid.mpi_finalized()))
        import muGrid
        w = muGrid.Communicator.world()
        print(muGrid.mpi_initialized(),
              muGrid._muGrid.mpi_initialized_by_mugrid(),
              muGrid.mpi_thread_level(), w.size)
        """
    )
    assert out == ["True", "True", "serialized", "1", "finalized", "True"]


@needs_mpi
@subprocess_ok
def test_finalize_can_be_left_to_the_os():
    out = _run(
        """
        import atexit
        # Registered before muGrid's own exit hook, so it runs after it (LIFO)
        atexit.register(lambda: print(
            "finalized", __import__("muGrid")._muGrid.mpi_finalized()))
        import muGrid
        muGrid.Communicator.world()
        """,
        env={"MUGRID_MPI_FINALIZE": "0"},
    )
    assert out == ["finalized", "False"]


@needs_mpi
@subprocess_ok
def test_mpi_started_by_someone_else_is_left_alone():
    pytest.importorskip("mpi4py.MPI", exc_type=ImportError)
    out = _run(
        """
        import atexit
        # Registered before muGrid's own exit hook, so it runs after it (LIFO)
        atexit.register(lambda: print(
            "finalized", __import__("muGrid")._muGrid.mpi_finalized()))
        from mpi4py import MPI
        import muGrid
        w = muGrid.Communicator.world()
        print(muGrid._muGrid.mpi_initialized_by_mugrid(), w.size)
        muGrid.finalize_mpi()  # not ours to finalise
        print(muGrid._muGrid.mpi_finalized())
        """
    )
    # mpi4py finalises at exit in its own hook; which hook runs last depends
    # on registration order, so only muGrid's answers are asserted
    assert out[:3] == ["False", "1", "False"]


@needs_mpi
@subprocess_ok
def test_world_after_finalize_raises():
    out = _run(
        """
        import muGrid
        muGrid.Communicator.world()
        muGrid.finalize_mpi()
        try:
            muGrid.Communicator.world()
        except RuntimeError as error:
            print("raised", "finalised" in str(error))
        """
    )
    assert out == ["raised", "True"]


@subprocess_ok
def test_world_without_mpi_under_a_launcher_raises():
    if muGrid.has_mpi:
        pytest.skip("needs a build without MPI")
    out = _run(
        """
        import muGrid
        try:
            muGrid.Communicator.world()
        except RuntimeError as error:
            print("raised", "no MPI support" in str(error))
        """,
        env={"OMPI_COMM_WORLD_SIZE": "4"},
    )
    assert out == ["raised", "True"]


# -- in-process: the communicators world() gives --------------------------- #


def test_communicator_is_a_class(comm):
    assert isinstance(comm, muGrid.Communicator)
    assert muGrid.Communicator(comm) is comm
    assert muGrid.Parallel.CommunicatorWrapper is muGrid.Communicator
    assert isinstance(muGrid.Communicator(comm._cpp), muGrid.Communicator)


def test_world_is_the_launcher_size(comm):
    assert comm.size == max(_mpi_launcher_world_size(), 1) or not muGrid.has_mpi
    assert comm.compare(comm) == "identical"
    if muGrid.has_mpi:
        assert comm.compare(muGrid.Communicator.world()) == "identical"


def test_split_dup_compare(comm):
    dup = comm.dup()
    assert dup.size == comm.size and dup.rank == comm.rank
    assert comm.compare(dup) == SAME_GROUP
    half = comm.split(comm.rank % 2, key=-comm.rank)
    assert half.size == (comm.size + 1 - comm.rank % 2) // 2
    # key=-rank orders the ranks the other way round
    assert half.rank == half.size - 1 - comm.rank // 2
    assert comm.compare(half) == (SAME_GROUP if comm.size == 1 else "unequal")
    # the split carries the exchanges
    assert half.allgather(np.asarray(comm.rank)).tolist() == sorted(
        range(comm.rank % 2, comm.size, 2), reverse=True
    )
    del half, dup  # owned handles are freed with their last reference


def test_serial_communicators(comm):
    serial = muGrid.Communicator()
    assert serial.compare(muGrid.Communicator()) in ("identical", "congruent")
    assert serial.compare(comm) == (SAME_GROUP if comm.size == 1 else "unequal")
    assert serial.split(0).size == 1 and serial.dup().size == 1
    assert (serial.node_rank, serial.node_size) == (0, 1)


def test_node_rank(comm):
    # Single-node test runs: every rank is on this node
    assert comm.node_size == comm.size
    assert sorted(comm.allgather(np.asarray(comm.node_rank)).tolist()) == list(
        range(comm.size)
    )
    assert comm.split(0).node_size == comm.size


@needs_mpi
def test_node_rank_of_a_wrapped_handle_needs_discovery(comm):
    from muGrid import _muGrid

    if comm.size == 1:
        pytest.skip("a single rank knows it is rank 0 of 1 on its node")
    bare = _muGrid.Communicator(comm.mpi_comm)
    with pytest.raises(RuntimeError, match="discover_node"):
        bare.node_rank
    bare.discover_node()
    assert bare.node_rank == comm.node_rank


# -- mpi4py interop --------------------------------------------------------- #


@needs_mpi
def test_mpi4py_communicator_is_wrapped(comm):
    MPI = pytest.importorskip("mpi4py.MPI", exc_type=ImportError)
    from_mpi4py = muGrid.Communicator(MPI.COMM_WORLD)
    assert from_mpi4py.size == comm.size
    # on one rank mpi4py's COMM_WORLD becomes the serial communicator
    assert from_mpi4py.compare(comm) == ("identical" if comm.size > 1
                                         else "congruent")
    assert from_mpi4py.node_rank == comm.node_rank
    assert from_mpi4py.mpi4py_comm is MPI.COMM_WORLD
    # and one made by muGrid can be handed back to mpi4py
    if comm.size > 1:
        assert isinstance(comm.mpi4py_comm, MPI.Comm)
        assert comm.mpi4py_comm.Get_size() == comm.size
    assert "rank" in muGrid.version_string(MPI.COMM_WORLD)


@needs_mpi
def test_another_mpi_library_is_rejected():
    """A communicator from an mpi4py linked against another MPI has a
    different MPI_COMM_WORLD; simulated with a stand-in module."""
    from muGrid.Parallel import _check_same_mpi_library

    class Other:
        COMM_WORLD = object()

        @staticmethod
        def _handleof(_):
            return 0xBAD

        @staticmethod
        def Get_library_version():
            return "Some other MPI 1.0\nmore"

    with pytest.raises(RuntimeError, match="different MPI libraries"):
        _check_same_mpi_library(Other)


def test_unknown_objects_are_rejected():
    with pytest.raises(RuntimeError, match="cannot make a muGrid Communicator"):
        muGrid.Communicator("COMM_WORLD")

    class Stub:  # mpi4py's interface, size 1: fine without mpi4py
        def Get_size(self):
            return 1

    assert muGrid.Communicator(Stub()).size == 1

    class BigStub:
        def Get_size(self):
            return 4

    with pytest.raises(RuntimeError, match="only mpi4py|without MPI"):
        muGrid.Communicator(BigStub())

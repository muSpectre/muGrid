import pytest

import muGrid

@pytest.fixture
def comm():
    try:
        from mpi4py import MPI
        return muGrid.Communicator(MPI.COMM_WORLD)
    except ImportError:
        return muGrid.Communicator()

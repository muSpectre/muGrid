"""
MPI tests of the heterogeneous multigrid (muGrid.Multigrid): on any number of
ranks the multigrid must reproduce the serial one. Each rank also solves the
serial problem on its own and compares.

The serial tests are in python_heterogeneous_multigrid_tests.py, whose
helpers this file reuses.
"""

import numpy as np
import pytest
from python_heterogeneous_multigrid_tests import (
    FusedProblem,
    cg_iterations,
    devices,
    global_rhs,
    host,
    phase_field,
    put,
)


@pytest.mark.parametrize("device", devices(),
                         ids=lambda d: "gpu" if d is not None else "cpu")
@pytest.mark.parametrize("nb_grid_pts, precision", [
    ((32, 32), "double"), ((16, 16, 16), "double"), ((32, 32), "single")])
def test_distributed_multigrid_matches_serial(comm, nb_grid_pts, precision,
                                              device):
    """On any number of ranks: the same V-cycle output, smoother bounds and
    CG iteration count as the serial multigrid of the same problem."""
    scale = phase_field(nb_grid_pts, 1e4)
    serial = FusedProblem(nb_grid_pts, precision=precision, device=device)
    distributed = FusedProblem(nb_grid_pts, precision=precision,
                               device=device, comm=comm)
    for problem in (serial, distributed):
        problem.set_scale(scale)
        problem.mg.setup()
    dim = len(nb_grid_pts)
    outputs = []
    for problem in (serial, distributed):
        d = problem.decomposition
        r = d.real_field("vc-r", (dim,))
        z = d.real_field("vc-z", (dim,))
        put(r.s, global_rhs(problem, dim, 11))
        problem.mg(r, z)
        outputs.append(host(z.s))
    expected = outputs[0][(slice(None), slice(None)) + distributed.window]
    tol = 1e-9 if precision == "double" else 1e-4
    np.testing.assert_allclose(outputs[1], expected, rtol=tol,
                               atol=tol * np.abs(outputs[0]).max())
    serial_bounds = [b for _, b, _ in serial.mg.level_summary()]
    distributed_bounds = [b for _, b, _ in distributed.mg.level_summary()]
    if comm.rank == 0:            # the tail's bounds live on rank 0
        np.testing.assert_allclose(distributed_bounds, serial_bounds,
                                   rtol=1e-12)
    assert cg_iterations(serial) == cg_iterations(distributed)


def test_distributed_multigrid_on_a_small_grid(comm):
    """A grid already below n_agglomerate still gets one distributed
    coarsening (the fine operator is a black box, so the level gathered on rank
    0 must be a stencil) and matches the serial iteration count."""
    n = (16, 16)
    scale = phase_field(n, 1e3)
    serial = FusedProblem(n)
    distributed = FusedProblem(n, comm=comm)
    for problem in (serial, distributed):
        problem.set_scale(scale)
        problem.mg.setup()
    if comm.size > 1:
        assert len(distributed.mg.levels) == 2
    assert cg_iterations(serial) == cg_iterations(distributed)

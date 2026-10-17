"""
The FEM gradient's transpose under domain decomposition (audit finding O1).

The protocol is the one every muGrid operator follows: communicate the
*input's* ghosts, then apply or transpose; the output's ghosts are not written
and are never reduced. These tests hold the transpose to it on any rank count,
through the adjoint identity ``<W apply(u), g> = <u, transpose(g)>`` summed
over owned points only.
"""

import numpy as np
import pytest
from NuMPI.Testing.Subdivision import suggest_subdivisions

import muGrid

# Odd and unequal sizes, so subdomains are uneven under MPI
GRIDS = {2: (11, 7), 3: (7, 5, 6)}
SPACING = (0.3, 0.7, 1.1)
ELEMENTS = [muGrid.FEMElement.q1, muGrid.FEMElement.p1]


def _setup(comm, dim, element, ghosts=None):
    op = muGrid.FEMGradientOperator(dim, list(SPACING[:dim]), element)
    decomposition = muGrid.CartesianDecomposition(
        comm, list(GRIDS[dim]), suggest_subdivisions(dim, comm.size),
        ghosts=op if ghosts is None else ghosts,
        nb_sub_pts={"quad": op.nb_quad_pts})
    return op, decomposition


def _fill(decomposition, field, seed):
    """Owned values of `field` from one global random array, so the global
    state is the same on every rank count."""
    dim = len(decomposition.nb_subdomain_grid_pts)
    lead = field.s.shape[:-dim]
    values = np.random.default_rng(seed).standard_normal(
        lead + tuple(decomposition.nb_domain_grid_pts))
    owned = tuple(slice(start, start + n) for start, n in zip(
        decomposition.subdomain_locations,
        decomposition.nb_subdomain_grid_pts))
    field.s[...] = values[(Ellipsis,) + owned]


def _owned_dot(comm, a, b):
    return comm.sum(float(np.vdot(np.asarray(a), np.asarray(b)).real))


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("element", ELEMENTS)
def test_transpose_is_the_adjoint_on_owned_nodes(comm, dim, element):
    op, decomposition = _setup(comm, dim, element)
    weights = np.asarray(op.quadrature_weights)
    tag = f"{dim}-{element}"

    g = decomposition.real_field(f"g-{tag}", (dim,), "quad")
    _fill(decomposition, g, seed=1)
    decomposition.communicate_ghosts(g)
    tg = decomposition.real_field(f"tg-{tag}")
    # Stale data in the output's ghosts must neither leak into the owned
    # nodes nor be overwritten
    tg.sg[...] = 1e30
    op.transpose(g, tg)
    assert np.all(np.isfinite(tg.s))
    ghosts = np.ones(tg.sg.shape, dtype=bool)
    ghosts[(Ellipsis,) + tuple(
        slice(left, left + n) for left, n in zip(
            decomposition.collection.nb_ghosts_left,
            decomposition.nb_subdomain_grid_pts))] = False
    assert np.all(tg.sg[ghosts] == 1e30)

    # Several directions, so a transpose that is wrong only on the
    # subdomain faces (missing or double-counted elements) cannot pass
    u = decomposition.real_field(f"u-{tag}")
    gu = decomposition.real_field(f"gu-{tag}", (dim,), "quad")
    for seed in range(2, 5):
        _fill(decomposition, u, seed=seed)
        decomposition.communicate_ghosts(u)
        op.apply(u, gu)
        # W apply(u), contracted with g over owned quadrature points
        lhs = 0.0
        for q, w in enumerate(weights):
            lhs += w * float(np.sum(gu.s[:, q] * g.s[:, q]))
        lhs = comm.sum(lhs)
        rhs = _owned_dot(comm, u.s, tg.s)
        assert np.isclose(lhs, rhs, rtol=1e-12, atol=0), (lhs, rhs)


@pytest.mark.parametrize("dim", [2, 3])
def test_transpose_matches_on_every_rank_count(comm, dim):
    """The owned result equals the serial one on the same global data."""
    element = muGrid.FEMElement.q1
    op, decomposition = _setup(comm, dim, element)
    g = decomposition.real_field(f"g-ref-{dim}", (dim,), "quad")
    _fill(decomposition, g, seed=7)
    decomposition.communicate_ghosts(g)
    tg = decomposition.real_field(f"tg-ref-{dim}")
    op.transpose(g, tg)

    serial_op, serial = _setup(muGrid.Communicator(), dim, element)
    gs = serial.real_field(f"g-ref-{dim}", (dim,), "quad")
    _fill(serial, gs, seed=7)
    serial.communicate_ghosts(gs)
    ts = serial.real_field(f"tg-ref-{dim}")
    serial_op.transpose(gs, ts)
    owned = tuple(slice(start, start + n) for start, n in zip(
        decomposition.subdomain_locations,
        decomposition.nb_subdomain_grid_pts))
    np.testing.assert_allclose(tg.s, ts.s[(Ellipsis,) + owned], rtol=1e-13,
                               atol=1e-13 * np.abs(ts.s).max())


@pytest.mark.parametrize("dim", [2, 3])
def test_transpose_increment_adds_on_owned_nodes(comm, dim):
    op, decomposition = _setup(comm, dim, muGrid.FEMElement.q1)
    g = decomposition.real_field(f"g-inc-{dim}", (dim,), "quad")
    _fill(decomposition, g, seed=3)
    decomposition.communicate_ghosts(g)
    ref = decomposition.real_field(f"ref-inc-{dim}")
    op.transpose(g, ref)
    out = decomposition.real_field(f"out-inc-{dim}")
    out.sg[...] = 2.0
    op.transpose_increment(g, 0.5, out)
    np.testing.assert_allclose(out.s, 2.0 + 0.5 * ref.s, rtol=1e-14,
                               atol=1e-14 * np.abs(ref.s).max())


def test_transpose_requires_a_left_ghost_layer():
    """apply() alone needs only right ghosts; the transpose must refuse a
    collection without left ones instead of silently dropping the element
    below each first owned node."""
    op = muGrid.FEMGradientOperator(2, [0.1, 0.1])
    assert op.transpose_ghost_requirement == ((1, 1), (0, 0))
    fc = muGrid.GlobalFieldCollection(
        [8, 8], ghosts=op.apply_ghost_requirement,
        nb_sub_pts={"quad": op.nb_quad_pts})
    nodal = fc.real_field("nodal")
    quad = fc.real_field("quad", (2,), "quad")
    op.apply(nodal, quad)  # apply is fine
    with pytest.raises(RuntimeError, match="ghost"):
        op.transpose(quad, nodal)

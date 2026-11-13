"""
@file     python_ghost_overlap_tests.py

@author  Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date    13 Nov 2026

@brief   Split ghost exchange (begin/end) and plane-range operator applies

These are the two halves of overlapping a slab's halo exchange with the
stencil: CartesianDecomposition.begin_communicate_ghosts posts the exchange
along the last axis and returns, IsotropicStiffnessOperator.apply_planes
computes the planes that do not read those ghosts, and
end_communicate_ghosts completes the exchange for the rest. Run under MPI at
any rank count; with one rank the exchange is local and begin does it all.

Copyright © 2026 Lars Pastewka

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
import pytest

from conftest import create_device, get_test_devices, skip_if_gpu_unavailable

import muGrid


def _host(array):
    """A NumPy copy of a host or device array."""
    return array.get() if hasattr(array, "get") else np.array(array)


def _assign(target, values):
    """target[...] = values for a host or device target."""
    if hasattr(target, "get"):
        import cupy as cp

        target[...] = cp.asarray(values)
    else:
        target[...] = values


def _slab(comm, device, dim, n, ghosts=1):
    nb_ranks = comm.size
    if n % nb_ranks or n // nb_ranks < 3:
        pytest.skip(f"{n} planes do not split into slabs of >= 3 for "
                    f"{nb_ranks} ranks")
    kwargs = {}
    device_obj = create_device(device)
    if device_obj is not None:
        kwargs["device"] = device_obj
    return muGrid.CartesianDecomposition(
        comm, [n] * dim, nb_subdivisions=[1] * (dim - 1) + [nb_ranks],
        nb_ghosts_left=(ghosts,) * dim, nb_ghosts_right=(ghosts,) * dim,
        **kwargs)


def _random_interior(comm, field, seed):
    """Fill the interior with random values and the ghosts with garbage."""
    rng = np.random.default_rng(seed + comm.rank)
    _assign(field.pg, np.full(field.pg.shape, -7.0))
    _assign(field.p, rng.standard_normal(field.p.shape))


@pytest.mark.parametrize("device", get_test_devices())
@pytest.mark.parametrize("dim,n,ghosts", [(2, 24, 1), (3, 12, 1), (2, 24, 2)])
@pytest.mark.parametrize("nb_components", [1, 3])
def test_begin_end_equals_communicate_ghosts(comm, device, dim, n, ghosts,
                                             nb_components):
    skip_if_gpu_unavailable(device)
    decomp = _slab(comm, device, dim, n, ghosts)
    a = decomp.real_field(f"ov-a-{dim}-{ghosts}-{nb_components}",
                          (nb_components,))
    b = decomp.real_field(f"ov-b-{dim}-{ghosts}-{nb_components}",
                          (nb_components,))
    _random_interior(comm, a, 3)
    _assign(b.pg, _host(a.pg))

    decomp.communicate_ghosts(a)
    decomp.begin_communicate_ghosts(b)
    decomp.end_communicate_ghosts(b)

    np.testing.assert_array_equal(_host(b.pg), _host(a.pg))


def test_overlap_flag(comm):
    decomp = muGrid.CartesianDecomposition(
        comm, [8, 4 * comm.size], nb_subdivisions=[1, comm.size],
        nb_ghosts_left=(1, 1), nb_ghosts_right=(1, 1))
    assert decomp.overlaps_ghost_communication == (
        muGrid.has_mpi and comm.size > 1)


def test_one_exchange_in_flight(comm):
    if comm.size < 2:
        pytest.skip("with one rank begin completes the exchange itself")
    decomp = muGrid.CartesianDecomposition(
        comm, [8, 4 * comm.size], nb_subdivisions=[1, comm.size],
        nb_ghosts_left=(1, 1), nb_ghosts_right=(1, 1))
    a = decomp.real_field("ov-flight-a")
    b = decomp.real_field("ov-flight-b")
    decomp.begin_communicate_ghosts(a)
    with pytest.raises(RuntimeError, match="in flight"):
        decomp.begin_communicate_ghosts(b)
    with pytest.raises(RuntimeError, match="in flight"):
        decomp.communicate_ghosts(b)
    with pytest.raises(RuntimeError, match="another field"):
        decomp.end_communicate_ghosts(b)
    # None of the refusals touched the exchange in flight
    decomp.end_communicate_ghosts(a)
    decomp.communicate_ghosts(b)


def _stiffness_setup(comm, device, dim, n, dtype=np.float64):
    decomp = _slab(comm, device, dim, n)
    spacing = (1.0 / n,) * dim
    op = (muGrid.IsotropicStiffnessOperator2D if dim == 2
          else muGrid.IsotropicStiffnessOperator3D)(spacing,
                                                    muGrid.FEMElement.q1)
    tag = f"{dim}-{n}-{np.dtype(dtype).name}"
    u = decomp.real_field(f"st-u-{tag}", (dim,), dtype=dtype)
    lam = decomp.real_field(f"st-lam-{tag}", (1,), dtype=dtype)
    mu = decomp.real_field(f"st-mu-{tag}", (1,), dtype=dtype)
    rng = np.random.default_rng(5 + comm.rank)
    _assign(u.p, rng.standard_normal(u.p.shape).astype(dtype))
    _assign(lam.p, (1.0 + rng.random(lam.p.shape)).astype(dtype))
    _assign(mu.p, (0.5 + rng.random(mu.p.shape)).astype(dtype))
    decomp.communicate_ghosts(lam)
    decomp.communicate_ghosts(mu)
    return decomp, op, u, lam, mu, tag


@pytest.mark.parametrize("device", get_test_devices())
@pytest.mark.parametrize("dim,n", [(2, 24), (3, 12)])
def test_apply_planes_pieces_make_the_whole(comm, device, dim, n):
    """Interior and boundary planes together are apply(), node by node."""
    skip_if_gpu_unavailable(device)
    decomp, op, u, lam, mu, tag = _stiffness_setup(comm, device, dim, n)
    full = decomp.real_field(f"st-full-{tag}", (dim,))
    parts = decomp.real_field(f"st-parts-{tag}", (dim,))
    decomp.communicate_ghosts(u)
    op.apply(u, lam, mu, full)

    nz = int(decomp.nb_subdomain_grid_pts[-1])
    _assign(parts.pg, np.zeros(parts.pg.shape))
    op.apply_planes(u, lam, mu, parts, 1, nz - 1)
    op.apply_planes(u, lam, mu, parts, 0, 1)
    op.apply_planes(u, lam, mu, parts, nz - 1)  # to the end

    np.testing.assert_allclose(_host(parts.p), _host(full.p), rtol=1e-13,
                               atol=1e-13 * float(np.abs(_host(full.p)).max()))


@pytest.mark.parametrize("device", get_test_devices())
def test_apply_planes_leaves_other_planes_alone(comm, device):
    skip_if_gpu_unavailable(device)
    decomp, op, u, lam, mu, tag = _stiffness_setup(comm, device, 3, 12)
    f = decomp.real_field(f"st-alone-{tag}", (3,))
    decomp.communicate_ghosts(u)
    _assign(f.p, np.full(f.p.shape, 42.0))
    op.apply_planes(u, lam, mu, f, 1, 2)
    got = _host(f.p)
    assert np.all(got[..., 0] == 42.0)
    assert np.all(got[..., 2:] == 42.0)
    assert not np.all(got[..., 1] == 42.0)


def test_apply_planes_rejects_bad_ranges(comm):
    decomp, op, u, lam, mu, tag = _stiffness_setup(comm, "cpu", 2, 24)
    f = decomp.real_field(f"st-bad-{tag}", (2,))
    nz = int(decomp.nb_subdomain_grid_pts[-1])
    for begin, end in [(-1, 2), (3, 2), (0, nz + 1)]:
        with pytest.raises(RuntimeError, match="plane range"):
            op.apply_planes(u, lam, mu, f, begin, end)


@pytest.mark.parametrize("device", get_test_devices())
@pytest.mark.parametrize("dim,n", [(2, 24), (3, 12)])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_overlapped_apply_equals_blocking_apply(comm, device, dim, n, dtype):
    """The pattern this exists for: begin, interior, end, boundary."""
    skip_if_gpu_unavailable(device)
    decomp, op, u, lam, mu, tag = _stiffness_setup(comm, device, dim, n,
                                                   dtype)
    want = decomp.real_field(f"st-want-{tag}", (dim,), dtype=dtype)
    got = decomp.real_field(f"st-got-{tag}", (dim,), dtype=dtype)
    v = decomp.real_field(f"st-v-{tag}", (dim,), dtype=dtype)
    _assign(v.pg, _host(u.pg))  # same data, ghosts still stale

    decomp.communicate_ghosts(u)
    op.apply(u, lam, mu, want)

    nz = int(decomp.nb_subdomain_grid_pts[-1])
    decomp.begin_communicate_ghosts(v)
    op.apply_planes(v, lam, mu, got, 1, nz - 1)
    decomp.end_communicate_ghosts(v)
    op.apply_planes(v, lam, mu, got, 0, 1)
    op.apply_planes(v, lam, mu, got, nz - 1, nz)

    scale = float(np.abs(_host(want.p)).max())
    tol = 1e-12 if dtype == np.float64 else 1e-5
    np.testing.assert_allclose(_host(got.p), _host(want.p), rtol=tol,
                               atol=tol * scale)


@pytest.mark.parametrize("device", get_test_devices())
@pytest.mark.parametrize("dim", [2, 3])
def test_fft_engine_begin_end_equals_communicate_ghosts(comm, device, dim):
    """The FFT engine is the decomposition muTopOpt overlaps with."""
    skip_if_gpu_unavailable(device)
    n = 4 * comm.size if dim == 3 else 8 * comm.size
    kwargs = {}
    device_obj = create_device(device)
    if device_obj is not None:
        kwargs["device"] = device_obj
    engine = muGrid.FFTEngine([n] * dim, comm, nb_ghosts_left=(1,) * dim,
                              nb_ghosts_right=(1,) * dim, **kwargs)
    subdivisions = tuple(engine.nb_subdivisions)
    assert engine.overlaps_ghost_communication == (
        muGrid.has_mpi and subdivisions[-1] > 1
        and int(engine.nb_subdomain_grid_pts[-1]) >= 1)

    a = engine.real_space_field(f"fe-a-{dim}", (dim,))
    b = engine.real_space_field(f"fe-b-{dim}", (dim,))
    _random_interior(comm, a, 11)
    _assign(b.pg, _host(a.pg))
    engine.communicate_ghosts(a)
    engine.begin_communicate_ghosts(b)
    engine.end_communicate_ghosts(b)
    np.testing.assert_array_equal(_host(b.pg), _host(a.pg))


@pytest.mark.parametrize("device", get_test_devices())
def test_wide_ghosts_fall_back_to_the_blocking_exchange(comm, device):
    """Ghosts wider than a rank's slab are relayed in several steps, which
    begin cannot leave in flight: it must do the whole exchange itself."""
    skip_if_gpu_unavailable(device)
    kwargs = {}
    device_obj = create_device(device)
    if device_obj is not None:
        kwargs["device"] = device_obj
    # One plane per rank along the last axis, two ghost planes either side
    decomp = muGrid.CartesianDecomposition(
        comm, [6, comm.size], nb_subdivisions=[1, comm.size],
        nb_ghosts_left=(1, 2), nb_ghosts_right=(1, 2), **kwargs)
    assert not decomp.overlaps_ghost_communication
    a = decomp.real_field("wide-a", (2,))
    b = decomp.real_field("wide-b", (2,))
    _random_interior(comm, a, 13)
    _assign(b.pg, _host(a.pg))
    decomp.communicate_ghosts(a)
    decomp.begin_communicate_ghosts(b)
    # Nothing is in flight: another exchange may start right away
    decomp.communicate_ghosts(a)
    decomp.end_communicate_ghosts(b)
    np.testing.assert_array_equal(_host(b.pg), _host(a.pg))


@pytest.mark.parametrize("device", get_test_devices())
@pytest.mark.parametrize("ghosts_left,ghosts_right",
                         [((1, 1), (1, 0)), ((1, 0), (1, 2))])
def test_one_sided_ghosts(comm, device, ghosts_left, ghosts_right):
    """No ghosts on one side of the last axis: that direction sends and
    receives nothing, and the other must still arrive."""
    skip_if_gpu_unavailable(device)
    kwargs = {}
    device_obj = create_device(device)
    if device_obj is not None:
        kwargs["device"] = device_obj
    decomp = muGrid.CartesianDecomposition(
        comm, [6, 4 * comm.size], nb_subdivisions=[1, comm.size],
        nb_ghosts_left=ghosts_left, nb_ghosts_right=ghosts_right, **kwargs)
    a = decomp.real_field(f"side-a-{ghosts_left}-{ghosts_right}", (3,))
    b = decomp.real_field(f"side-b-{ghosts_left}-{ghosts_right}", (3,))
    _random_interior(comm, a, 17)
    _assign(b.pg, _host(a.pg))
    decomp.communicate_ghosts(a)
    decomp.begin_communicate_ghosts(b)
    decomp.end_communicate_ghosts(b)
    np.testing.assert_array_equal(_host(b.pg), _host(a.pg))


@pytest.mark.parametrize("device", get_test_devices())
@pytest.mark.parametrize("dim,n", [(2, 24), (3, 12)])
def test_apply_planes_empty_range(comm, device, dim, n):
    """An empty range computes nothing and leaves the force alone."""
    skip_if_gpu_unavailable(device)
    decomp, op, u, lam, mu, tag = _stiffness_setup(comm, device, dim, n)
    f = decomp.real_field(f"st-empty-{tag}", (dim,))
    decomp.communicate_ghosts(u)
    _assign(f.pg, np.full(f.pg.shape, 3.5))
    op.apply_planes(u, lam, mu, f, 2, 2)
    assert np.all(_host(f.pg) == 3.5)

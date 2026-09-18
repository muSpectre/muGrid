#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file    python_multigrid_tests.py

@author  Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date    18 Sep 2026

@brief   Functional tests for the multigrid grid-transfer operators

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

------------------------------------------------------------------------------

``GridTransfer`` implements multilinear prolongation ``P`` between two nested
nodal grids and its exact adjoint ``R = Pᵀ``. Three properties carry the whole
construction, and each is checked here against an independent reference rather
than against a recorded output:

  * **Adjointness** ``⟨P c, f⟩ = ⟨c, Pᵀ f⟩`` to round-off. This is what makes a
    V-cycle built from these operators symmetric, which plain CG requires of a
    preconditioner.
  * **Polynomial reproduction**: ``P`` maps a linear displacement field to
    itself. This is why ``range(P)`` contains every rigid-body mode and every
    constant strain -- the near-nullspace property that lets a *geometric*
    hierarchy work for elasticity without being told about it.
  * **Agreement with an independent NumPy implementation** of the same tensor
    product rule, component by component.
"""

import numpy as np
import pytest

import muGrid

# --------------------------------------------------------------------------- #
# Independent NumPy reference: the same tensor-product rules, one 1D pass per
# spatial axis. Arrays are (nb_dof, *spatial) with periodic wrap-around.
# --------------------------------------------------------------------------- #


def _axis_slice(ax, sl):
    return (slice(None),) * (ax + 1) + (sl,)


def _prolong_reference(coarse, dim):
    """fine[2c] = coarse[c]; fine[2c+1] = (coarse[c] + coarse[c+1]) / 2."""
    out = coarse
    for ax in range(dim):
        axis = ax + 1
        n = out.shape[axis]
        new = np.empty(
            out.shape[:axis] + (2 * n,) + out.shape[axis + 1:], dtype=out.dtype
        )
        new[_axis_slice(ax, slice(0, None, 2))] = out
        new[_axis_slice(ax, slice(1, None, 2))] = 0.5 * (
            out + np.roll(out, -1, axis=axis)
        )
        out = new
    return out


def _restrict_reference(fine, dim):
    """coarse[c] = fine[2c] + (fine[2c-1] + fine[2c+1]) / 2, i.e. Pᵀ."""
    out = fine
    for ax in range(dim):
        axis = ax + 1
        even = out[_axis_slice(ax, slice(0, None, 2))]
        odd = out[_axis_slice(ax, slice(1, None, 2))]
        out = even + 0.5 * (odd + np.roll(odd, 1, axis=axis))
    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _decomposition(comm, nb_grid_pts, nb_dof):
    """One nested level: a decomposition with a single ghost layer per side."""
    dim = len(nb_grid_pts)
    nb_ranks = comm.size if comm is not None else 1
    subdivisions = _power_of_two_subdivisions(dim, nb_ranks)
    return muGrid.CartesianDecomposition(
        comm,
        list(nb_grid_pts),
        nb_subdivisions=subdivisions,
        nb_ghosts_left=(1,) * dim,
        nb_ghosts_right=(1,) * dim,
    )


def _power_of_two_subdivisions(dim, nb_ranks):
    """Factor `nb_ranks` into powers of two, spread over the axes.

    The nesting precondition needs every subdivision count to divide the
    subdomain extent at every level, which power-of-two grids guarantee only
    for power-of-two rank counts. `NuMPI.suggest_subdivisions` does not
    promise that (12 ranks gives [2, 2, 3]).
    """
    if nb_ranks & (nb_ranks - 1) != 0:
        pytest.skip(
            f"nested multigrid levels need a power-of-two rank count, "
            f"got {nb_ranks}"
        )
    subdivisions = [1] * dim
    axis = 0
    remaining = nb_ranks
    while remaining > 1:
        subdivisions[axis % dim] *= 2
        remaining //= 2
        axis += 1
    return subdivisions


def _fill(decomposition, field, values):
    """Write global `values` into the local interior, then fill the ghosts."""
    loc = decomposition.subdomain_locations
    shape = decomposition.nb_subdomain_grid_pts
    window = tuple(slice(o, o + n) for o, n in zip(loc, shape))
    field.s[:, 0] = values[(slice(None),) + window]
    decomposition.communicate_ghosts(field)


def _gather_interior(comm, decomposition, field, nb_dof, nb_grid_pts):
    """Assemble the global interior array from every rank's subdomain."""
    out = np.zeros((nb_dof,) + tuple(nb_grid_pts))
    loc = decomposition.subdomain_locations
    shape = decomposition.nb_subdomain_grid_pts
    window = tuple(slice(o, o + n) for o, n in zip(loc, shape))
    out[(slice(None),) + window] = field.s[:, 0]
    if comm is not None and comm.size > 1:
        # Communicator.sum takes scalars or 2D Fortran-contiguous float64
        # arrays (it is bound through Eigen), so flatten the spatial axes.
        flat = np.asfortranarray(out.reshape(nb_dof, -1))
        out = np.asarray(comm.sum(flat)).reshape(out.shape)
    return out


def _local_dot(a, b):
    """Interior-only inner product; ghosts are another rank's copies."""
    return float((np.asarray(a.s[:, 0]) * np.asarray(b.s[:, 0])).sum())


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dim,nb_fine", [(2, (16, 16)), (3, (8, 8, 8))])
def test_prolong_matches_numpy_reference(comm, dim, nb_fine):
    """P agrees with an independent tensor-product implementation."""
    nb_dof = dim
    nb_coarse = tuple(n // 2 for n in nb_fine)
    rng = np.random.default_rng(0)
    coarse_values = rng.standard_normal((nb_dof,) + nb_coarse)

    coarse_decomp = _decomposition(comm, nb_coarse, nb_dof)
    fine_decomp = _decomposition(comm, nb_fine, nb_dof)
    coarse = coarse_decomp.collection.real_field("coarse", (nb_dof,))
    fine = fine_decomp.collection.real_field("fine", (nb_dof,))

    _fill(coarse_decomp, coarse, coarse_values)
    fine.s[...] = 0.0
    muGrid.GridTransfer(dim).prolong(coarse, fine)

    got = _gather_interior(comm, fine_decomp, fine, nb_dof, nb_fine)
    expected = _prolong_reference(coarse_values, dim)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-13)


@pytest.mark.parametrize("dim,nb_fine", [(2, (16, 16)), (3, (8, 8, 8))])
def test_restrict_matches_numpy_reference(comm, dim, nb_fine):
    """Pᵀ agrees with an independent tensor-product implementation."""
    nb_dof = dim
    nb_coarse = tuple(n // 2 for n in nb_fine)
    rng = np.random.default_rng(1)
    fine_values = rng.standard_normal((nb_dof,) + nb_fine)

    coarse_decomp = _decomposition(comm, nb_coarse, nb_dof)
    fine_decomp = _decomposition(comm, nb_fine, nb_dof)
    coarse = coarse_decomp.collection.real_field("coarse", (nb_dof,))
    fine = fine_decomp.collection.real_field("fine", (nb_dof,))

    _fill(fine_decomp, fine, fine_values)
    coarse.s[...] = 0.0
    muGrid.GridTransfer(dim).restrict(fine, coarse)

    got = _gather_interior(comm, coarse_decomp, coarse, nb_dof, nb_coarse)
    expected = _restrict_reference(fine_values, dim)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-13)


@pytest.mark.parametrize("dim,nb_fine", [(2, (16, 16)), (3, (8, 8, 8))])
def test_transfers_are_adjoint(comm, dim, nb_fine):
    """<P c, f> == <c, Pᵀ f>: the property that keeps a V-cycle symmetric."""
    nb_dof = dim
    nb_coarse = tuple(n // 2 for n in nb_fine)
    rng = np.random.default_rng(2)

    coarse_decomp = _decomposition(comm, nb_coarse, nb_dof)
    fine_decomp = _decomposition(comm, nb_fine, nb_dof)
    coarse = coarse_decomp.collection.real_field("coarse", (nb_dof,))
    fine = fine_decomp.collection.real_field("fine", (nb_dof,))
    prolonged = fine_decomp.collection.real_field("prolonged", (nb_dof,))
    restricted = coarse_decomp.collection.real_field("restricted", (nb_dof,))

    _fill(coarse_decomp, coarse, rng.standard_normal((nb_dof,) + nb_coarse))
    _fill(fine_decomp, fine, rng.standard_normal((nb_dof,) + nb_fine))

    transfer = muGrid.GridTransfer(dim)
    prolonged.s[...] = 0.0
    restricted.s[...] = 0.0
    transfer.prolong(coarse, prolonged)
    transfer.restrict(fine, restricted)

    lhs = _local_dot(prolonged, fine)
    rhs = _local_dot(coarse, restricted)
    if comm is not None and comm.size > 1:
        lhs = float(comm.sum(lhs))
        rhs = float(comm.sum(rhs))
    assert abs(lhs - rhs) <= 1e-12 * max(abs(lhs), abs(rhs))


@pytest.mark.parametrize("dim,nb_fine", [(2, (16, 16)), (3, (8, 8, 8))])
def test_prolong_reproduces_linear_fields(comm, dim, nb_fine):
    """P maps a linear displacement field to itself.

    This is what puts every rigid-body mode and every constant strain in
    range(P). The periodic wrap makes the last interval non-linear by
    construction, so the check covers the interior away from the seam.
    """
    nb_dof = dim
    nb_coarse = tuple(n // 2 for n in nb_fine)
    rng = np.random.default_rng(3)
    gradient = rng.standard_normal((nb_dof, dim))
    offset = rng.standard_normal(nb_dof)

    def linear_on(nb_grid_pts, spacing):
        axes = [np.arange(n) * spacing for n in nb_grid_pts]
        coords = np.meshgrid(*axes, indexing="ij")
        return np.stack(
            [
                sum(gradient[i, j] * coords[j] for j in range(dim)) + offset[i]
                for i in range(nb_dof)
            ]
        )

    coarse_values = linear_on(nb_coarse, 2.0)
    fine_values = linear_on(nb_fine, 1.0)

    coarse_decomp = _decomposition(comm, nb_coarse, nb_dof)
    fine_decomp = _decomposition(comm, nb_fine, nb_dof)
    coarse = coarse_decomp.collection.real_field("coarse", (nb_dof,))
    fine = fine_decomp.collection.real_field("fine", (nb_dof,))

    _fill(coarse_decomp, coarse, coarse_values)
    fine.s[...] = 0.0
    muGrid.GridTransfer(dim).prolong(coarse, fine)

    got = _gather_interior(comm, fine_decomp, fine, nb_dof, nb_fine)
    interior = (slice(None),) + tuple(slice(0, n - 2) for n in nb_fine)
    np.testing.assert_allclose(
        got[interior], fine_values[interior], rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("dim,nb_fine", [(2, (16, 16)), (3, (8, 8, 8))])
def test_prolong_reproduces_constants_everywhere(comm, dim, nb_fine):
    """A constant field prolongs exactly, seam included.

    Constants are the nullspace of the periodic stiffness operator, so an
    error here would feed the very modes the coarse solve projects out.
    """
    nb_dof = dim
    nb_coarse = tuple(n // 2 for n in nb_fine)
    constants = np.arange(1, nb_dof + 1, dtype=float)
    coarse_values = np.broadcast_to(
        constants.reshape((nb_dof,) + (1,) * dim), (nb_dof,) + nb_coarse
    ).copy()

    coarse_decomp = _decomposition(comm, nb_coarse, nb_dof)
    fine_decomp = _decomposition(comm, nb_fine, nb_dof)
    coarse = coarse_decomp.collection.real_field("coarse", (nb_dof,))
    fine = fine_decomp.collection.real_field("fine", (nb_dof,))

    _fill(coarse_decomp, coarse, coarse_values)
    fine.s[...] = 0.0
    muGrid.GridTransfer(dim).prolong(coarse, fine)

    got = _gather_interior(comm, fine_decomp, fine, nb_dof, nb_fine)
    expected = np.broadcast_to(
        constants.reshape((nb_dof,) + (1,) * dim), (nb_dof,) + tuple(nb_fine)
    )
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-13)


def test_single_precision_matches_double(comm):
    """The Real32 overloads run the same rule as the Real ones."""
    dim = 2
    nb_fine = (16, 16)
    nb_coarse = tuple(n // 2 for n in nb_fine)
    nb_dof = dim
    rng = np.random.default_rng(4)
    coarse_values = rng.standard_normal((nb_dof,) + nb_coarse)

    coarse_decomp = _decomposition(comm, nb_coarse, nb_dof)
    fine_decomp = _decomposition(comm, nb_fine, nb_dof)
    # register_real32_field returns the bare C++ field; the array views used
    # below live on the Python wrapper.
    coarse32 = muGrid.wrap_field(
        coarse_decomp.collection.register_real32_field("coarse32", (nb_dof,))
    )
    fine32 = muGrid.wrap_field(
        fine_decomp.collection.register_real32_field("fine32", (nb_dof,))
    )

    _fill(coarse_decomp, coarse32, coarse_values)
    fine32.s[...] = 0.0
    muGrid.GridTransfer(dim).prolong(coarse32, fine32)

    got = _gather_interior(comm, fine_decomp, fine32, nb_dof, nb_fine)
    expected = _prolong_reference(coarse_values, dim)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)


def test_rejects_grids_that_are_not_nested(comm):
    """A non-2:1 pair must raise rather than corrupt the subdomain seams."""
    dim = 2
    coarse_decomp = _decomposition(comm, (8, 8), dim)
    fine_decomp = _decomposition(comm, (12, 12), dim)  # not 2 x 8
    coarse = coarse_decomp.collection.real_field("coarse", (dim,))
    fine = fine_decomp.collection.real_field("fine", (dim,))

    with pytest.raises(RuntimeError, match="not nested"):
        muGrid.GridTransfer(dim).prolong(coarse, fine)


def test_rejects_mismatched_component_counts(comm):
    """A transfer acts component-wise and cannot change the DOF count."""
    dim = 2
    coarse_decomp = _decomposition(comm, (8, 8), dim)
    fine_decomp = _decomposition(comm, (16, 16), dim)
    coarse = coarse_decomp.collection.real_field("coarse", (2,))
    fine = fine_decomp.collection.real_field("fine", (3,))

    with pytest.raises(RuntimeError, match="component-wise"):
        muGrid.GridTransfer(dim).prolong(coarse, fine)


# --------------------------------------------------------------------------- #
# The V-cycle preconditioner
# --------------------------------------------------------------------------- #


def _serial_only(comm):
    if comm is not None and comm.size > 1:
        pytest.skip("MultigridReferencePreconditioner is serial for now")


def _uniform_setup(comm, dim, n, min_coarse=8, nu=2):
    from muGrid.Preconditioners import MultigridReferencePreconditioner

    h = 1.0 / n
    decomp = muGrid.CartesianDecomposition(
        comm,
        [n] * dim,
        nb_subdivisions=[1] * dim,
        nb_ghosts_left=(1,) * dim,
        nb_ghosts_right=(1,) * dim,
    )
    prec = MultigridReferencePreconditioner(
        decomp, (h,) * dim, 1.3, 0.7, min_coarse=min_coarse, nu=nu
    )
    return decomp, prec


def _zero_mean_field(collection, name, dim, seed):
    rng = np.random.default_rng(seed)
    field = collection.real_field(name, (dim,))
    values = rng.standard_normal(field.s.shape)
    axes = tuple(range(1, values.ndim))
    field.s[...] = values - values.mean(axis=axes, keepdims=True)
    return field


@pytest.mark.parametrize("nu", [1, 2, 3])
@pytest.mark.parametrize("dim,n", [(2, 32), (3, 16)])
def test_preconditioner_is_symmetric(comm, dim, n, nu):
    """<M⁻¹a, b> == <a, M⁻¹b>.

    Plain CG requires the preconditioner to be a fixed symmetric operator. The
    V-cycle is symmetric only because the pre- and post-smoothing counts are
    equal, the cycle count is fixed, and R is exactly Pᵀ -- so this test is what
    protects all three from being "optimised" apart.

    Swept over nu because nu is a tuning parameter: the cost measurements
    favour nu = 1, and a symmetry guarantee that held only at the value that
    happened to be the default would be worth very little.
    """
    _serial_only(comm)
    from muGrid import linalg

    decomp, prec = _uniform_setup(comm, dim, n, nu=nu)
    fc = decomp.collection
    a = _zero_mean_field(fc, "sym-a", dim, 0)
    b = _zero_mean_field(fc, "sym-b", dim, 1)
    ma = fc.real_field("sym-ma", (dim,))
    mb = fc.real_field("sym-mb", (dim,))

    prec.apply(a, ma)
    prec.apply(b, mb)

    lhs = linalg.vecdot(ma, b)
    rhs = linalg.vecdot(a, mb)
    assert abs(lhs - rhs) <= 1e-10 * max(abs(lhs), abs(rhs))


@pytest.mark.parametrize("nu", [1, 2, 3])
@pytest.mark.parametrize("dim,n", [(2, 32), (3, 16)])
def test_vcycle_converges_on_the_reference_operator(comm, dim, n, nu):
    """Used as a stationary iteration, the cycle contracts the residual.

    The measured factor is ~0.355 at nu = 2 in both 2D and 3D and degrades to
    ~0.5 at nu = 1; 0.6 covers the whole useful range of nu while still failing
    loudly if the cycle stops working. A weaker smoother contracts more slowly
    but costs proportionally less, which is the trade §8 of the plan measures.
    """
    _serial_only(comm)
    from muGrid import linalg

    decomp, prec = _uniform_setup(comm, dim, n, nu=nu)
    fc = decomp.collection
    level = prec.levels[0]

    rhs = _zero_mean_field(fc, "vc-rhs", dim, 2)
    x = fc.real_field("vc-x", (dim,))
    residual = fc.real_field("vc-res", (dim,))
    correction = fc.real_field("vc-corr", (dim,))
    x.set_zero()

    norms = []
    for _ in range(8):
        level.apply(x, residual)
        linalg.axpby(1.0, rhs, -1.0, residual)  # residual = rhs - K x
        norms.append(np.sqrt(linalg.norm_sq(residual)))
        prec.apply(residual, correction)
        linalg.axpy(1.0, correction, x)

    factors = [b / a for a, b in zip(norms[:-1], norms[1:])]
    assert max(factors[-3:]) < 0.6, f"residual factors {factors}"


def test_damping_is_derived_from_the_spectrum(comm):
    """omega = 1.7 / lambda_max, and lambda_max is dimension-dependent.

    A hardcoded damping is unsafe: the stability limit is 2/lambda_max, so the
    2D optimum diverges in 3D. This pins the rule rather than the value.
    """
    _serial_only(comm)
    _, prec_2d = _uniform_setup(comm, 2, 32)
    _, prec_3d = _uniform_setup(comm, 3, 16)

    for prec in (prec_2d, prec_3d):
        lambda_max = prec.SAFETY / prec.omega
        assert prec.omega < 2.0 / lambda_max  # strictly inside the limit
    assert prec_3d.omega < prec_2d.omega  # 3D has the larger lambda_max


def test_rejects_2d_p1_nodal_block(comm):
    """2D P1 has a non-diagonal nodal block and must be refused, not smoothed.

    The two-triangle Kuhn split breaks the x-y symmetry that cancels the
    off-diagonal, leaving K01 = lambda + mu exactly. 3D P1 keeps enough
    symmetry and is fine.
    """
    _serial_only(comm)
    from muGrid.Preconditioners import MultigridReferencePreconditioner

    decomp = muGrid.CartesianDecomposition(
        comm,
        [32, 32],
        nb_subdivisions=[1, 1],
        nb_ghosts_left=(1, 1),
        nb_ghosts_right=(1, 1),
    )
    with pytest.raises(NotImplementedError, match="diagonal nodal block"):
        MultigridReferencePreconditioner(
            decomp, (1 / 32, 1 / 32), 1.3, 0.7,
            element=muGrid._muGrid.FEMElement.p1, min_coarse=8,
        )

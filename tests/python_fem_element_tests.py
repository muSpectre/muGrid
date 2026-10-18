"""
The P1 tetrahedral element against its derivation.

``P1Tet3D`` in ``fem_element.hh`` is the five-tetrahedra voxel of R. Leute's
thesis (Sec. 3.5.3, Eqs. (3.63)-(3.65)). These tests rebuild every
tetrahedron's constant gradient from its node coordinates alone, so the
tables cannot drift from the geometry they encode, and pin the Laplacian
stencil the element assembles to the values the same split is used for in
every voxel (see the element's documentation for why it is not alternated).
"""

import itertools

import numpy as np
import pytest

import muGrid

# Thesis Eq. (3.63): element k -> its four nodes as unit-cube corners. The
# first node is the element's origin r0, which the thesis' local coordinate
# system starts from.
LEUTE_ELEMENTS = [
    [(1, 0, 0), (1, 1, 1), (0, 1, 0), (0, 0, 1)],  # T(0), central
    [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],  # T(1)
    [(1, 1, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)],  # T(2)
    [(1, 0, 1), (0, 0, 1), (1, 1, 1), (1, 0, 0)],  # T(3)
    [(0, 1, 1), (1, 1, 1), (0, 0, 1), (0, 1, 0)],  # T(4)
]
VOLUME_FRACTIONS = [1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6]


def _node_index(corner):
    """muGrid's binary corner index, x fastest."""
    return corner[0] + 2 * corner[1] + 4 * corner[2]


def _gradient_rows(corners, h):
    """The constant gradient of the linear interpolant on the tetrahedron
    with the given corners, as an (3, 8) matrix over the voxel's nodes:
    solve [1 x] a = u for the linear coefficients."""
    X = np.array([np.asarray(c) * h for c in corners], dtype=float)
    A = np.hstack([np.ones((4, 1)), X])
    G_local = np.linalg.inv(A)[1:, :]  # rows: d/dx, d/dy, d/dz
    G = np.zeros((3, 8))
    for col, c in enumerate(corners):
        G[:, _node_index(c)] = G_local[:, col]
    return G


def _volume(corners, h):
    X = np.array([np.asarray(c) * h for c in corners], dtype=float)
    return abs(np.linalg.det(X[1:] - X[0])) / 6.0


@pytest.mark.parametrize("h", [(1.0, 1.0, 1.0), (0.5, 2.0, 0.25)])
def test_p1_gradient_matches_the_thesis_elements(h):
    """apply() on a random nodal field gives, per voxel and tetrahedron,
    the gradient of the linear interpolant through that tetrahedron's four
    corner values -- for cubic and anisotropic voxels."""
    op = muGrid.FEMGradientOperator(3, list(h), muGrid.FEMElement.p1)
    assert op.nb_quad_pts == 5
    n = (4, 3, 5)
    fc = muGrid.GlobalFieldCollection(
        list(n), sub_pts={"quad": 5}, nb_ghosts_left=(1, 1, 1),
        nb_ghosts_right=(1, 1, 1))
    u = fc.real_field("u")
    g = fc.real_field("g", (3,), "quad")
    rng = np.random.default_rng(0)
    values = rng.standard_normal(n)
    u.pg[...] = np.pad(values, 1, mode="wrap")
    op.apply(u, g)
    grad = np.asarray(g.s)  # (3, 5, *n)

    for q, corners in enumerate(LEUTE_ELEMENTS):
        G = _gradient_rows(corners, h)
        for i, j, k in itertools.product(*(range(m) for m in n)):
            nodal = np.array([
                values[(i + c[0]) % n[0], (j + c[1]) % n[1], (k + c[2]) % n[2]]
                for c in itertools.product((0, 1), repeat=3)])
            # itertools gives z fastest; reorder to x fastest
            nodal = nodal.reshape(2, 2, 2).transpose(2, 1, 0).ravel()
            np.testing.assert_allclose(grad[:, q, i, j, k], G @ nodal,
                                       rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("h", [(1.0, 1.0, 1.0), (0.5, 2.0, 0.25)])
def test_p1_quadrature_weights_are_the_tetrahedra_volumes(h):
    """Central tetrahedron 1/3 of the voxel, each corner one 1/6 (thesis:
    |J(0)| = 2 dx dy dz, |J(e)| = dx dy dz)."""
    op = muGrid.FEMGradientOperator(3, list(h), muGrid.FEMElement.p1)
    volume = float(np.prod(h))
    np.testing.assert_allclose(
        op.quadrature_weights, [f * volume for f in VOLUME_FRACTIONS],
        rtol=1e-14)
    np.testing.assert_allclose(
        [_volume(c, h) for c in LEUTE_ELEMENTS],
        [f * volume for f in VOLUME_FRACTIONS], rtol=1e-14)


def test_p1_laplacian_is_the_symmetric_19_point_stencil():
    """transpose(apply(.)) of the P1 element on a unit grid is one
    translation-invariant stencil: 5 on the centre, -2/3 on the six face
    neighbours, -1/12 on the twelve edge neighbours, 0 on the corners. It is
    reflection symmetric in every axis although the split itself is not,
    and equals the average of the two stencils of the conforming
    (alternating) split."""
    op = muGrid.FEMGradientOperator(3, [1.0, 1.0, 1.0], muGrid.FEMElement.p1)
    n = (5, 5, 5)
    fc = muGrid.GlobalFieldCollection(
        list(n), sub_pts={"quad": 5}, nb_ghosts_left=(1, 1, 1),
        nb_ghosts_right=(1, 1, 1))
    u = fc.real_field("u")
    g = fc.real_field("g", (3,), "quad")
    lap = fc.real_field("lap")
    impulse = np.zeros(n)
    impulse[2, 2, 2] = 1.0
    u.pg[...] = np.pad(impulse, 1, mode="wrap")
    op.apply(u, g)
    # periodic ghosts of the gradient by hand: the element below each first
    # node lives in the left ghost layer
    g.sg[...] = np.pad(np.asarray(g.s), [(0, 0), (0, 0)] + [(1, 1)] * 3,
                       mode="wrap")
    op.transpose(g, lap)
    stencil = np.asarray(lap.p)[1:4, 1:4, 1:4]

    expected = np.zeros((3, 3, 3))
    expected[1, 1, 1] = 5.0
    for axis in range(3):
        for shift in (0, 2):
            idx = [1, 1, 1]
            idx[axis] = shift
            expected[tuple(idx)] = -2.0 / 3.0
    for a, b in itertools.combinations(range(3), 2):
        for sa in (0, 2):
            for sb in (0, 2):
                idx = [1, 1, 1]
                idx[a], idx[b] = sa, sb
                expected[tuple(idx)] = -1.0 / 12.0
    np.testing.assert_allclose(stencil, expected, atol=1e-13)
    assert np.allclose(np.asarray(lap.p)[[0, 4]], 0.0)
    assert np.allclose(np.asarray(lap.p)[:, [0, 4]], 0.0)
    assert np.allclose(np.asarray(lap.p)[:, :, [0, 4]], 0.0)

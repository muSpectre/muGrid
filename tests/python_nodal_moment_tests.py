"""
Tests for NodalMomentOperator: the cell moments ∫_e rho^k dx (k = 2, 3, 4) of a
nodal Q1 interpolant and their nodal gradients.

Copyright © 2026 Lars Pastewka

µGrid is free software; you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, either version 3, or (at your option) any later version.

µGrid is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
details.

You should have received a copy of the GNU Lesser General Public License along
with µGrid; see the file COPYING. If not, write to the Free Software
Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
"""

import itertools
import unittest

import numpy as np

import muGrid


def _gauss3():
    """3-point Gauss-Legendre rule on [0, 1] — the reference implementation's
    copy of the rule the operator uses."""
    g = np.sqrt(3.0 / 5.0)
    return (np.array([0.5 - 0.5 * g, 0.5, 0.5 + 0.5 * g]),
            np.array([5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0]))


def _tohost(view):
    """NumPy copy of a host or device field view."""
    return view.get() if hasattr(view, "get") else np.asarray(view)


def _asmodule(view, values):
    """`values` in the array module of `view` (cupy for a device field)."""
    if hasattr(view, "get"):
        import cupy
        return cupy.asarray(values)
    return values


def reference_moments(rho, dim, cell_volume):
    """Moments and nodal gradients by explicit quadrature over a periodic grid.

    Deliberately written the slow, obvious way -- a Python loop over corners
    and quadrature points on periodically rolled copies of the field -- so it
    shares no structure with the kernel it checks.
    """
    xi, w = _gauss3()
    nb_nodes = 2 ** dim
    ks = np.arange(2, 2 + muGrid.NodalMomentOperator3D.nb_moments)
    moments = np.zeros(rho.shape + (len(ks),))
    grads = np.zeros_like(moments)

    def offset(n, d):
        return (n >> d) & 1

    # Corner c of cell (node - offset(c)) is this node; rolling by -offset
    # brings that cell's corner values onto the node's own index.
    for c in range(nb_nodes):
        shift_c = tuple(-offset(c, d) for d in range(dim))
        corners = []
        for n in range(nb_nodes):
            sh = tuple(shift_c[d] + offset(n, d) for d in range(dim))
            corners.append(np.roll(rho, tuple(-s for s in sh),
                                   axis=tuple(range(dim))))
        for qidx in itertools.product(range(3), repeat=dim):
            N = np.array([
                np.prod([xi[qidx[d]] if offset(n, d) else 1.0 - xi[qidx[d]]
                         for d in range(dim)])
                for n in range(nb_nodes)])
            wq = float(np.prod([w[qidx[d]] for d in range(dim)]))
            rho_q = sum(N[n] * corners[n] for n in range(nb_nodes))
            for j, k in enumerate(ks):
                grads[..., j] += wq * N[c] * k * rho_q ** (k - 1)
                if c == 0:
                    moments[..., j] += wq * rho_q ** k
    return moments * cell_volume, grads * cell_volume


class NodalMomentCheck(unittest.TestCase):
    def _run(self, nb_grid_pts, spacing, dtype=np.float64, device=False,
             element=None):
        dim = len(nb_grid_pts)
        fc = muGrid.GlobalFieldCollection(
            nb_grid_pts, nb_ghosts_left=(1,) * dim, nb_ghosts_right=(1,) * dim,
            **({"device": muGrid.Device.gpu()} if device else {}))
        rng = np.random.default_rng(3)
        rho_np = rng.random(nb_grid_pts).astype(dtype)

        # register_real32_field is register-only and returns the bare C++
        # field; wrap_field gives it the same .p/.pg views real_field has.
        def field(name, nb_components=1):
            if dtype == np.float64:
                return fc.real_field(name, nb_components)
            return muGrid.wrap_field(
                fc.register_real32_field(name, nb_components))

        rho, mom, grd = field("rho"), field("mom", 3), field("grd", 3)

        # Periodic ghosts, filled by hand: the operator requires them
        # communicated, and this test has no communicator. A device field's
        # view is a cupy array, which refuses implicit conversion, so assign
        # through the view's own array module in both cases.
        pg = rho.pg
        pg[...] = _asmodule(pg, np.pad(rho_np, 1, mode="wrap").reshape(
            pg.shape))

        element = muGrid.FEMElement.q1 if element is None else element
        op = (muGrid.NodalMomentOperator2D if dim == 2
              else muGrid.NodalMomentOperator3D)(list(spacing), element)
        op.compute(rho, mom, grd)

        ref_m, ref_g = reference_moments(rho_np.astype(np.float64), dim,
                                         float(np.prod(spacing)))
        got_m = np.moveaxis(
            _tohost(mom.p).reshape((3,) + tuple(nb_grid_pts)), 0, -1)
        got_g = np.moveaxis(
            _tohost(grd.p).reshape((3,) + tuple(nb_grid_pts)), 0, -1)
        tol = 1e-12 if dtype == np.float64 else 2e-5
        np.testing.assert_allclose(got_m, ref_m, rtol=tol, atol=tol)
        np.testing.assert_allclose(got_g, ref_g, rtol=tol, atol=tol)
        return got_m, got_g

    def test_p1_matches_closed_form(self):
        """P1 moments against the exact integral of a linear function's k-th
        power over a simplex, d! |T| k!/(k+d)! h_k(corner values), summed over
        muGrid's sub-simplex decomposition. Independent of the quadrature."""
        from math import factorial
        simp = {2: [((0, 1, 2), 0.5), ((1, 2, 3), 0.5)],
                3: [((1, 2, 4, 7), 1 / 3), ((0, 1, 2, 4), 1 / 6),
                    ((1, 2, 3, 7), 1 / 6), ((1, 4, 5, 7), 1 / 6),
                    ((2, 4, 6, 7), 1 / 6)]}
        for dim, n in ((2, 7), (3, 5)):
            h = 0.4
            fc = muGrid.GlobalFieldCollection(
                (n,) * dim, nb_ghosts_left=(1,) * dim,
                nb_ghosts_right=(1,) * dim)
            rho, mom, grd = (fc.real_field("rho"), fc.real_field("m", 3),
                             fc.real_field("g", 3))
            r = np.random.default_rng(4).random((n,) * dim)
            pg = np.asarray(rho.pg)
            pg[...] = np.pad(r, 1, mode="wrap").reshape(pg.shape)
            op = (muGrid.NodalMomentOperator2D if dim == 2
                  else muGrid.NodalMomentOperator3D)(
                      [h] * dim, muGrid.FEMElement.p1)
            op.compute(rho, mom, grd)
            self.assertEqual(op.nb_quad, 18 if dim == 2 else 135)
            got = np.asarray(mom.p).reshape((3, -1)).sum(axis=1)

            off = lambda nd, d: (nd >> d) & 1  # noqa: E731
            ref = np.zeros(3)
            for k in (2, 3, 4):
                C = factorial(dim) * factorial(k) / factorial(k + dim)
                acc = 0.0
                for nodes, frac in simp[dim]:
                    av = [np.roll(r, tuple(-off(nd, d) for d in range(dim)),
                                  axis=tuple(range(dim))) for nd in nodes]
                    ps = [sum(a ** j for a in av) for j in range(5)]
                    hk = [np.ones_like(r)]
                    for m in range(1, 5):
                        hk.append(sum(ps[j] * hk[m - j]
                                      for j in range(1, m + 1)) / m)
                    acc = acc + frac * C * hk[k]
                ref[k - 2] = acc.sum() * h ** dim
            np.testing.assert_allclose(got, ref, rtol=1e-13, atol=1e-13)

    def test_p1_weights_are_positive(self):
        """A constant rho must give exactly rho^k per cell. With a negative
        quadrature weight that still holds, so also check no cell energy of
        the double well W = rho^2 (1-rho)^2 >= 0 comes out negative."""
        n, h = 5, 0.4
        fc = muGrid.GlobalFieldCollection(
            (n,) * 3, nb_ghosts_left=(1,) * 3, nb_ghosts_right=(1,) * 3)
        rho, mom, grd = (fc.real_field("rho"), fc.real_field("m", 3),
                         fc.real_field("g", 3))
        rng = np.random.default_rng(5)
        for _ in range(20):
            r = rng.random((n,) * 3)
            pg = np.asarray(rho.pg)
            pg[...] = np.pad(r, 1, mode="wrap").reshape(pg.shape)
            muGrid.NodalMomentOperator3D([h] * 3,
                                         muGrid.FEMElement.p1).compute(
                                             rho, mom, grd)
            m = np.asarray(mom.p).reshape((3, -1))
            well = m[0] - 2 * m[1] + m[2]
            self.assertGreaterEqual(well.min(), -1e-15)

    def test_2d(self):
        self._run((7, 5), (0.3, 0.7))

    def test_3d(self):
        self._run((5, 6, 4), (0.3, 0.7, 1.1))

    def test_constant_field_is_exact(self):
        """A constant rho has ∫_e rho^k = rho^k |e| exactly, and a gradient of
        k rho^(k-1) |e| per node -- the quadrature must reproduce both to
        round-off, which pins the weights and the partition of unity."""
        dim, n, h = 3, 4, 0.5
        fc = muGrid.GlobalFieldCollection(
            (n,) * dim, nb_ghosts_left=(1,) * dim, nb_ghosts_right=(1,) * dim)
        rho = fc.real_field("rho")
        mom = fc.real_field("mom", 3)
        grd = fc.real_field("grd", 3)
        np.asarray(rho.pg)[...] = 0.25
        muGrid.NodalMomentOperator3D([h] * dim).compute(rho, mom, grd)
        vol = h ** dim
        got_m = np.asarray(mom.p).reshape((3, -1))
        got_g = np.asarray(grd.p).reshape((3, -1))
        for j, k in enumerate((2, 3, 4)):
            np.testing.assert_allclose(got_m[j], 0.25 ** k * vol, rtol=1e-14)
            # Every node belongs to 2^dim cells and carries weight 1/2^dim of
            # each, so the gradients sum to k rho^(k-1) |e|.
            np.testing.assert_allclose(got_g[j],
                                       k * 0.25 ** (k - 1) * vol, rtol=1e-13)

    def test_gradient_matches_finite_differences(self):
        """The nodal gradient is the derivative of the summed moments."""
        dim, n, h = 3, 4, 0.5
        rng = np.random.default_rng(11)
        rho_np = rng.random((n,) * dim)

        def total(field):
            fc = muGrid.GlobalFieldCollection(
                (n,) * dim, nb_ghosts_left=(1,) * dim,
                nb_ghosts_right=(1,) * dim)
            r, m, g = (fc.real_field("rho"), fc.real_field("mom", 3),
                       fc.real_field("grd", 3))
            np.asarray(r.pg)[...] = np.pad(field, 1, mode="wrap").reshape(
                np.asarray(r.pg).shape)
            muGrid.NodalMomentOperator3D([h] * dim).compute(r, m, g)
            return (np.asarray(m.p).reshape((3, -1)).sum(axis=1),
                    np.asarray(g.p).reshape((3, -1)))

        _, grad = total(rho_np)
        eps = 1e-6
        for idx in ((0, 0, 0), (2, 1, 3), (3, 3, 3)):
            flat = np.ravel_multi_index(idx, (n,) * dim)
            up, dn = rho_np.copy(), rho_np.copy()
            up[idx] += eps
            dn[idx] -= eps
            fd = (total(up)[0] - total(dn)[0]) / (2 * eps)
            np.testing.assert_allclose(grad[:, flat], fd, rtol=1e-6, atol=1e-8)

    @unittest.skipUnless(muGrid.has_gpu, "no GPU backend in this build")
    def test_device_matches_host(self):
        """The device kernel must agree with the host kernel bit-for-bit up to
        round-off: they are the same arithmetic in a different traversal."""
        for dtype in (np.float64, np.float32):
            host_m, host_g = self._run((5, 6, 4), (0.3, 0.7, 1.1), dtype)
            dev_m, dev_g = self._run((5, 6, 4), (0.3, 0.7, 1.1), dtype,
                                     device=True)
            tol = 1e-13 if dtype == np.float64 else 1e-6
            np.testing.assert_allclose(dev_m, host_m, rtol=tol, atol=tol)
            np.testing.assert_allclose(dev_g, host_g, rtol=tol, atol=tol)


if __name__ == "__main__":
    unittest.main()

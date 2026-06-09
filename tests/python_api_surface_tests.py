#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file     python_api_surface_tests.py

@author  Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>

@date    09 Jun 2026

@brief   regression tests for Python API surface fixes (package exports,
         operator wrappers and bound increment methods)

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

import importlib
import unittest

import numpy as np

import muGrid


class StarImportTest(unittest.TestCase):
    def test_star_import(self):
        """`from muGrid import *` must not reference undefined names.

        __all__ previously listed "Verbosity" (never defined) and
        unconditionally listed "OpenMode" (only defined on NetCDF builds),
        either of which broke a star import.
        """
        namespace = {}
        exec("from muGrid import *", namespace)  # noqa: S102
        for name in muGrid.__all__:
            self.assertTrue(
                hasattr(muGrid, name),
                msg=f"__all__ lists '{name}' but muGrid has no such attribute",
            )


class FEMGradientIncrementTest(unittest.TestCase):
    def test_increment_methods_bound(self):
        """FEMGradientOperator must expose apply_increment/transpose_increment.

        These wrapper methods called C++ methods that were never bound.
        """
        op = muGrid.FEMGradientOperator(2)
        self.assertTrue(hasattr(op, "apply_increment"))
        self.assertTrue(hasattr(op, "transpose_increment"))

    def test_increment_matches_apply(self):
        nn = 6
        grid_spacing = [0.25, 0.25]
        op = muGrid.FEMGradientOperator(2, grid_spacing)

        decomposition = muGrid.CartesianDecomposition(
            muGrid.Communicator(),
            (nn, nn),
            nb_subdivisions=(1, 1),
            nb_ghosts_left=(1, 1),
            nb_ghosts_right=(1, 1),
            nb_sub_pts={"quad": op.nb_quad_pts},
        )
        displacement = decomposition.real_field("displacement", (2,))
        np.random.seed(42)
        displacement.p[...] = np.random.rand(*displacement.p.shape)
        decomposition.communicate_ghosts(displacement)

        grad_a = decomposition.real_field("grad_a", (2, 2), "quad")
        grad_b = decomposition.real_field("grad_b", (2, 2), "quad")

        op.apply(displacement, grad_a)
        grad_b.s[...] = 0.0
        # apply_increment from zero must reproduce apply (it was bound to the
        # wrong C++ method before the fix).
        op.apply_increment(displacement, 1.0, grad_b)
        np.testing.assert_allclose(grad_a.s, grad_b.s, atol=1e-12)


class IsotropicStiffnessWrapperTest(unittest.TestCase):
    def test_accepts_wrapped_fields(self):
        """The exported IsotropicStiffnessOperator must accept wrapped Fields.

        The raw pybind classes reject the pure-Python Field wrapper; the
        package now exports unwrapping wrappers.
        """
        nx, ny = 6, 6
        grid_spacing = [1.0 / (nx - 1), 1.0 / (ny - 1)]
        op = muGrid.IsotropicStiffnessOperator2D(grid_spacing)

        fc = muGrid.GlobalFieldCollection(
            (nx, ny), nb_ghosts_left=(1, 1), nb_ghosts_right=(1, 1)
        )
        displacement = fc.real_field("displacement", (2,))
        force = fc.real_field("force", (2,))
        fc_mat = muGrid.GlobalFieldCollection(
            (nx, ny), nb_ghosts_left=(1, 1), nb_ghosts_right=(1, 1)
        )
        lam = fc_mat.real_field("lambda", (1,))
        mu = fc_mat.real_field("mu", (1,))
        lam.pg[...] = 1.0
        mu.pg[...] = 1.0
        displacement.pg[...] = 0.0

        # Should not raise (previously a TypeError on the raw pybind class).
        op.apply(displacement, lam, mu, force)
        # Zero displacement -> zero force.
        np.testing.assert_allclose(force.s, 0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()

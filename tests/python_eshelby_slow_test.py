#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   python_eshelby_slow_test.py

@author Richard Leute <richard.leute@imtek.uni-freiburg.de>

@date   06 Jun 2019

@brief  Tests for the computation of the Eshelby tensor and the computation
        of the interrior and exterrior stresses and strains in an infinite
        matrix with a single eshelby inclusion/inhomogeneity.

Copyright © 2019 Till Junge

µSpectre is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µSpectre is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with µSpectre; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.

Additional permission under GNU GPL version 3 section 7

If you modify this Program, or any covered work, by linking or combining it
with proprietary FFT implementations or numerical libraries, containing parts
covered by the terms of those libraries' licenses, the licensors of this
Program grant you additional permission to convey the resulting work.
"""

import unittest
import warnings
import numpy as np
import scipy as sp
from python_test_imports import µ

from muSpectre.gradient_integration import Voigt_vector_to_full_matrix as vv_fm


def S_spherical_inclusion(nu):
    """
    Eshelby tensor for spherical inclusion (a=b=c)
    (T. Mura, Micromechanics of defects in solids (1982) eq. 11.21)

    Keyword Arguments:
    nu   -- float, Poissons ratio of the matrix material; -1 <= nu <= 0.5.

    Returns:
    S_spherical   -- np.ndarray of shape (3,3,3,3). The Eshelby tensor for a
                     spherical inclusion. (dtype = float)
    """
    factor = 1 / (15*(1-nu))
    S_spherical = np.zeros((3,)*4)

    # S₁₁₁₁ = S₂₂₂₂ = S₃₃₃₃ = (7-5ν)/(15(1-ν))
    iiii = tuple(np.array([(i, i, i, i) for i in [0, 1, 2]]).T)
    S_spherical[iiii] = factor * (7 - 5*nu)

    # S₁₁₂₂ = S₂₂₃₃ = S₃₃₁₁ = S₁₁₃₃ = S₂₂₁₁ = S₃₃₂₂ = (5ν-1)/(15(1-ν))
    iijj = tuple(np.array([(i, i, j, j)
                           for i, j in zip([0, 1, 2], [1, 2, 0])]).T)
    S_spherical[iijj] = factor * (5*nu - 1)

    iikk = tuple(np.array([(i, i, k, k)
                           for i, k in zip([0, 1, 2], [2, 0, 1])]).T)
    S_spherical[iikk] = factor * (5*nu - 1)

    # S₁₂₁₂ = S₂₃₂₃ = S₃₁₃₁ = (4-5ν)/(15(1-ν))
    ijij = tuple(np.array([(i, j, i, j)
                           for i, j in zip([0, 1, 2], [1, 2, 0])]).T)
    S_spherical[ijij] = factor * (4 - 5*nu)
    # use symmetry Sⱼᵢₖₗ = Sᵢⱼₖₗ and Sᵢⱼₗₖ = Sᵢⱼₖₗ and Sⱼᵢₗₖ=Sᵢⱼₖₗ
    jiij = tuple(np.array([(j, i, i, j)
                           for i, j in zip([0, 1, 2], [1, 2, 0])]).T)
    ijji = tuple(np.array([(i, j, j, i)
                           for i, j in zip([0, 1, 2], [1, 2, 0])]).T)
    jiji = tuple(np.array([(j, i, j, i)
                           for i, j in zip([0, 1, 2], [1, 2, 0])]).T)
    S_spherical[jiij] = S_spherical[ijij]
    S_spherical[ijji] = S_spherical[ijij]
    S_spherical[jiji] = S_spherical[ijij]

    return S_spherical


def S_elliptic_cylindrical_inclusion_a_inf(a, b, c, nu):
    """
    inclusion shaped like an elliptic cylinder with (a→∞)
    (T. Mura, Micromechanics of defects in solids (1982) eq. 11.22)
    Here we turn a→∞ to satisfy a>b>c, so in all formulas of T. Mura we have to
    exchange a ↔ c and 0 ↔ 2.

    Keyword Arguments:
    a, b and c -- Three floats giving the principal half axes of the
                  ellipsoidal inclusion, where the half axes are alligned with
                  the coordinate axes and the order "a > b > c" is fulfilled.
                  The principal half axes a should fulfill a >> b, c!
    nu         -- float, Poissons ratio of the matrix material; -1 <= nu <= 0.5
    """
    if a < b or b < c or a < b*c*1e2:
        raise ValueError("The ellipsoidal pricipal half axes should satisfy:" +
                         "\na > b > c\nBut you gave: a="+str(a)+", b="+str(b) +
                         " c="+str(c)+".\nEspecially a>>b,c should be "
                         "satisfied.")
    fac = 1/(2*(1-nu))  # often ocuring factor
    S_cylinder = np.zeros((3,)*4)

    # analytic exact
    S_cylinder[2, 2, 2, 2] = fac * \
        ((b**2 + 2*c*b)/(c+b)**2 + (1-2*nu) * b/(c+b))
    S_cylinder[1, 1, 1, 1] = fac * \
        ((c**2 + 2*c*b)/(c+b)**2 + (1-2*nu) * c/(c+b))
    S_cylinder[0, 0, 0, 0] = 0

    S_cylinder[2, 2, 1, 1] = fac * ((b**2)/(c+b)**2 - (1-2*nu) * b/(c+b))
    S_cylinder[1, 1, 0, 0] = fac * (2*nu*c)/(c+b)
    S_cylinder[0, 0, 2, 2] = 0
    S_cylinder[2, 2, 0, 0] = fac * (2*nu*b)/(c+b)
    S_cylinder[1, 1, 2, 2] = fac * ((c**2)/(c+b)**2 - (1-2*nu) * c/(c+b))
    S_cylinder[0, 0, 1, 1] = 0

    S_cylinder[2, 1, 2, 1] = fac * ((c**2 + b**2)/(2*(c+b)**2) + (1 - 2*nu)/2)
    S_cylinder[1, 0, 1, 0] = c / (2*(c + b))
    S_cylinder[0, 2, 0, 2] = b / (2*(c + b))
    # use symmetries
    # 1.) Sⱼᵢₖₗ = Sᵢⱼₖₗ
    S_cylinder[1, 2, 2, 1] = S_cylinder[2, 1, 2, 1]
    S_cylinder[0, 1, 1, 0] = S_cylinder[1, 0, 1, 0]
    S_cylinder[2, 0, 0, 2] = S_cylinder[0, 2, 0, 2]
    # 2.) Sᵢⱼₗₖ = Sᵢⱼₖₗ
    S_cylinder[2, 1, 1, 2] = S_cylinder[2, 1, 2, 1]
    S_cylinder[1, 0, 0, 1] = S_cylinder[1, 0, 1, 0]
    S_cylinder[0, 2, 2, 0] = S_cylinder[0, 2, 0, 2]
    # 3.) Sⱼᵢₗₖ=Sᵢⱼₖₗ
    S_cylinder[1, 2, 1, 2] = S_cylinder[2, 1, 2, 1]
    S_cylinder[0, 1, 0, 1] = S_cylinder[1, 0, 1, 0]
    S_cylinder[2, 0, 2, 0] = S_cylinder[0, 2, 0, 2]

    return S_cylinder


def stiffness_tensor_voigt(Young, Poisson):
    """
    σᵢ = Cᵢⱼεⱼ
    where σᵢ=(σ₁₁,σ₂₂,σ₃₃,σ₁₂,σ₂₃,σ₁₃) and εⱼ=(ε₁₁,ε₂₂,ε₃₃,ε₁₂,ε₂₃,ε₁₃)
    """
    C_ijkl = µ.eshelby_slow.stiffness_tensor(Young, Poisson)
    ordering = np.array([[0, 0], [1, 1], [2, 2], [0, 1], [1, 2], [0, 2]])
    C_ab = np.zeros((6, 6))
    for a, ij in enumerate(ordering):
        i, j = ij
        for b, kl in enumerate(ordering):
            k, l = kl
            C_ab[a, b] = C_ijkl[i, j, k, l]
    return C_ab


class MuSpectre_Eshelby_Slow_Check(unittest.TestCase):
    """
    Check the integration of the Eshelby tensor and the computed stresses and
    strains in muSpectre.eshelby_slow.
    """

    def setUp(self):
        self.a = 3
        self.b = 2
        self.c = 1
        self.nu = 0.35  # Poisson ratio of the matrix
        self.E = 20  # Young modulus of the matrix
        self.nu_I = 0.25  # Poisson ratio of the inhomogeneity
        self.E_I = 30  # Young modulus of the inhomogeneity

        # a random point lying in D-Ω
        self.x = 4
        self.y = 3
        self.z = 2

        # arbitrary λ, largest positive square root of the ellipsoidal equation
        # x²/(a²+λ)·y²/(b²+λ)·z²/(c²+λ)=1
        self.lam = 5.0

        self.tol = 1e-12
        self.verbose = 0  # 0-non verbose, 1-some details, 2-all details

        # first and second lame constants
        def lame_first(E, nu): return E * nu / ((1 + nu) * (1 - 2*nu))
        def lame_second(E, nu): return E / (2 * (1 + nu))
        self.lame_first = lame_first(self.E, self.nu)
        self.lame_second = lame_second(self.E, self.nu)
        self.lame_first_I = lame_first(self.E_I, self.nu_I)
        self.lame_second_I = lame_second(self.E_I, self.nu_I)

    def test_F(self):
        """
        Check the computed value for the incomplete elliptic integral of the
        first kind.
        """
        # analytic
        phi = np.pi/4
        m = 0.5
        K = sp.special.ellipkinc(phi, m)

        # with muSpectre.eshelby_slow.F(a,b,c)
        a = 1
        b = 0.866025 * a
        c = 0.707107 * a  # with this b and c you get phi=pi/4 and m=0.5
        F = µ.eshelby_slow.F(a, b, c)
        self.assertAlmostEqual(K, F, places=6)

    def test_E(self):
        """
        Check the computed value for the incomplete elliptic integral of the
        second kind.
        """
        # analytic
        phi = np.pi/4
        m = 0.5
        E_comp = sp.special.ellipeinc(phi, m)

        # with muSpectre.eshelby_slow.E(a,b,c)
        a = 1
        b = 0.866025 * a
        c = 0.707107 * a  # with this b and c you get phi=pi/4 and m=0.5
        E = µ.eshelby_slow.E(a, b, c)
        self.assertAlmostEqual(E_comp, E, places=6)

    def test_I_integrals(self):
        """
        Check sum rules and analytic values if a=b=c (sphere) for the I's.
        """
        # Proof sum rules, e.g. T. Mura, Micromechanics of defects in solids
        # (1982) eq. 11.19
        # I_a + I_b + I_c = 4π
        sum1 = µ.eshelby_slow.I_a(self.a, self.b, self.c) \
            + µ.eshelby_slow.I_b(self.a, self.b, self.c) \
            + µ.eshelby_slow.I_c(self.a, self.b, self.c)
        self.assertAlmostEqual(4*np.pi, sum1, places=14)

        # 3I_aa + I_ab + I_ac = 4π/a²
        sum2 = 3*µ.eshelby_slow.I_aa(self.a, self.b, self.c) \
            + µ.eshelby_slow.I_ab(self.a, self.b, self.c) \
            + µ.eshelby_slow.I_ac(self.a, self.b, self.c)
        self.assertAlmostEqual(4*np.pi/(self.a**2), sum2, places=14)
        # cyclic 1
        sum2_c1 = 3*µ.eshelby_slow.I_bb(self.a, self.b, self.c) \
            + µ.eshelby_slow.I_bc(self.a, self.b, self.c) \
            + µ.eshelby_slow.I_ba(self.a, self.b, self.c)
        self.assertAlmostEqual(4*np.pi/(self.b**2), sum2_c1, places=14)
        # cyclic 2
        sum2_c2 = 3*µ.eshelby_slow.I_cc(self.a, self.b, self.c) \
            + µ.eshelby_slow.I_ca(self.a, self.b, self.c) \
            + µ.eshelby_slow.I_cb(self.a, self.b, self.c)
        self.assertAlmostEqual(4*np.pi/(self.c**2), sum2_c2, places=14)

        # 3a²I_aa + b²I_ab + c²I_ac = 3I_a
        sum3 = 3*self.a**2 * µ.eshelby_slow.I_aa(self.a, self.b, self.c) \
            + self.b**2 * µ.eshelby_slow.I_ab(self.a, self.b, self.c) \
            + self.c**2 * µ.eshelby_slow.I_ac(self.a, self.b, self.c)
        self.assertAlmostEqual(3*µ.eshelby_slow.I_a(self.a, self.b, self.c),
                               sum3, places=14)
        # cyclic 1
        sum3_c1 = 3*self.b**2 * µ.eshelby_slow.I_bb(self.a, self.b, self.c) \
            + self.c**2 * µ.eshelby_slow.I_bc(self.a, self.b, self.c) \
            + self.a**2 * µ.eshelby_slow.I_ba(self.a, self.b, self.c)
        self.assertAlmostEqual(3*µ.eshelby_slow.I_b(self.a, self.b, self.c),
                               sum3_c1, places=14)
        # cyclic 2
        sum3_c2 = 3*self.c**2 * µ.eshelby_slow.I_cc(self.a, self.b, self.c) \
            + self.a**2 * µ.eshelby_slow.I_ca(self.a, self.b, self.c) \
            + self.b**2 * µ.eshelby_slow.I_cb(self.a, self.b, self.c)
        self.assertAlmostEqual(3*µ.eshelby_slow.I_c(self.a, self.b, self.c),
                               sum3_c2, places=14)

        # Proof analytic values of I's for a=b=c,
        # e.g. T. Mura, Micromechanics of defects in solids (1982) eq. 11.21
        deviation = 1e-4
        a = self.a + deviation
        b = self.a
        c = self.a - deviation

        tol = 1.7e-4

        # I_a = I_b = I_c = 4π/3
        c1 = 4*np.pi/3
        self.assertLess(np.abs(µ.eshelby_slow.I_a(a, b, c) - c1), tol)
        self.assertLess(np.abs(µ.eshelby_slow.I_b(a, b, c) - c1), tol)
        self.assertLess(np.abs(µ.eshelby_slow.I_c(a, b, c) - c1), tol)

        # I_aa = I_bb = I_cc = I_ab = I_bc = I_ca = 4π/(5 a*a)
        # Here we use b*b instead of a*a because b is the mean length
        # (1/3*(a+b+c)) and in the ideal case a=b=c is true.
        c2 = 4*np.pi/(5*b**2)
        self.assertLess(np.abs(µ.eshelby_slow.I_aa(a, b, c) - c2), tol)
        self.assertLess(np.abs(µ.eshelby_slow.I_bb(a, b, c) - c2), tol)
        self.assertLess(np.abs(µ.eshelby_slow.I_cc(a, b, c) - c2), tol)
        self.assertLess(np.abs(µ.eshelby_slow.I_ab(a, b, c) - c2), tol)
        self.assertLess(np.abs(µ.eshelby_slow.I_bc(a, b, c) - c2), tol)
        self.assertLess(np.abs(µ.eshelby_slow.I_ca(a, b, c) - c2), tol)
        # test also for I_ba = I_cb = I_ac = 4π/(5 a*a)
        self.assertLess(np.abs(µ.eshelby_slow.I_ba(a, b, c) - c2), tol)
        self.assertLess(np.abs(µ.eshelby_slow.I_cb(a, b, c) - c2), tol)
        self.assertLess(np.abs(µ.eshelby_slow.I_ac(a, b, c) - c2), tol)

    def test_Il_integrals(self):
        """
        -> Check sum rules for I(λ)'s at λ=0.
        """
        lam = 0.0
        # lam = µ.eshelby_slow.compute_lambda(self.x, self.y, self.z,
        #                                     self.a, self.b, self.c)
        # Proof sum rules, e.g. T. Mura, Micromechanics of defects in solids
        # (1982) eq. 11.19
        # I_a(λ) + I_b(λ) + I_c(λ) = 4π
        sum1 = µ.eshelby_slow.Il_a(self.a, self.b, self.c, lam) \
            + µ.eshelby_slow.Il_b(self.a, self.b, self.c, lam) \
            + µ.eshelby_slow.Il_c(self.a, self.b, self.c, lam)
        self.assertAlmostEqual(4*np.pi, sum1, places=14)

        # 3*I_aa(λ) + I_ab(λ) + I_ac(λ) = 4π/(a²)
        sum2 = 3*µ.eshelby_slow.Il_aa(self.a, self.b, self.c, lam) \
            + µ.eshelby_slow.Il_ab(self.a, self.b, self.c, lam) \
            + µ.eshelby_slow.Il_ac(self.a, self.b, self.c, lam)
        self.assertAlmostEqual(4*np.pi/(self.a**2), sum2, places=14)
        # cyclic 1
        sum2_c1 = 3*µ.eshelby_slow.Il_bb(self.a, self.b, self.c, lam) \
            + µ.eshelby_slow.Il_bc(self.a, self.b, self.c, lam) \
            + µ.eshelby_slow.Il_ba(self.a, self.b, self.c, lam)
        self.assertAlmostEqual(4*np.pi/(self.b**2), sum2_c1, places=14)
        # cyclic 2
        sum2_c2 = 3*µ.eshelby_slow.Il_cc(self.a, self.b, self.c, lam) \
            + µ.eshelby_slow.Il_ca(self.a, self.b, self.c, lam) \
            + µ.eshelby_slow.Il_cb(self.a, self.b, self.c, lam)
        self.assertAlmostEqual(4*np.pi/(self.c**2), sum2_c2, places=14)

        # 3a²I_aa(λ) + b²I_ab(λ) + c²I_ac(λ) = 3I_a(λ)
        sum3 = 3*self.a**2 * µ.eshelby_slow.Il_aa(self.a, self.b, self.c, lam)\
            + self.b**2 * µ.eshelby_slow.Il_ab(self.a, self.b, self.c, lam)\
            + self.c**2 * µ.eshelby_slow.Il_ac(self.a, self.b, self.c, lam)
        self.assertAlmostEqual(
            3*µ.eshelby_slow.Il_a(self.a, self.b, self.c, lam),
            sum3, places=14)
        # cyclic 1
        sum3_c1 = 3*self.b**2 \
            * µ.eshelby_slow.Il_bb(self.a, self.b, self.c, lam) \
            + self.c**2 * µ.eshelby_slow.Il_bc(self.a, self.b, self.c, lam) \
            + self.a**2 * µ.eshelby_slow.Il_ba(self.a, self.b, self.c, lam)
        self.assertAlmostEqual(
            3*µ.eshelby_slow.Il_b(self.a, self.b, self.c, lam),
            sum3_c1, places=14)
        # cyclic 2
        sum3_c2 = 3*self.c**2 \
            * µ.eshelby_slow.Il_cc(self.a, self.b, self.c, lam) \
            + self.a**2 * µ.eshelby_slow.Il_ca(self.a, self.b, self.c, lam) \
            + self.b**2 * µ.eshelby_slow.Il_cb(self.a, self.b, self.c, lam)
        self.assertAlmostEqual(
            3*µ.eshelby_slow.Il_c(self.a, self.b, self.c, lam),
            sum3_c2, places=14)

    def test_I_vs_I_lambda(self):
        """
        Test if the Integrals fulfill I(λ) = I for λ=0
        """
        lam = 0.0
        # Iᵢ
        self.assertAlmostEqual(
            µ.eshelby_slow.I_a(self.a, self.b, self.c),
            µ.eshelby_slow.Il_a(self.a, self.b, self.c, lam),
            places=12)
        self.assertAlmostEqual(
            µ.eshelby_slow.I_b(self.a, self.b, self.c),
            µ.eshelby_slow.Il_b(self.a, self.b, self.c, lam),
            places=12)
        self.assertAlmostEqual(
            µ.eshelby_slow.I_c(self.a, self.b, self.c),
            µ.eshelby_slow.Il_c(self.a, self.b, self.c, lam),
            places=12)
        # Iᵢⱼ i≠j
        self.assertAlmostEqual(
            µ.eshelby_slow.I_ab(self.a, self.b, self.c),
            µ.eshelby_slow.Il_ab(self.a, self.b, self.c, lam),
            places=12)
        self.assertAlmostEqual(
            µ.eshelby_slow.I_ba(self.a, self.b, self.c),
            µ.eshelby_slow.Il_ba(self.a, self.b, self.c, lam),
            places=12)
        self.assertAlmostEqual(
            µ.eshelby_slow.I_bc(self.a, self.b, self.c),
            µ.eshelby_slow.Il_bc(self.a, self.b, self.c, lam),
            places=12)
        self.assertAlmostEqual(
            µ.eshelby_slow.I_cb(self.a, self.b, self.c),
            µ.eshelby_slow.Il_cb(self.a, self.b, self.c, lam),
            places=12)
        self.assertAlmostEqual(
            µ.eshelby_slow.I_ca(self.a, self.b, self.c),
            µ.eshelby_slow.Il_ca(self.a, self.b, self.c, lam),
            places=12)
        self.assertAlmostEqual(
            µ.eshelby_slow.I_ac(self.a, self.b, self.c),
            µ.eshelby_slow.Il_ac(self.a, self.b, self.c, lam),
            places=12)
        # Iᵢᵢ
        self.assertAlmostEqual(
            µ.eshelby_slow.I_aa(self.a, self.b, self.c),
            µ.eshelby_slow.Il_aa(self.a, self.b, self.c, lam),
            places=12)
        self.assertAlmostEqual(
            µ.eshelby_slow.I_bb(self.a, self.b, self.c),
            µ.eshelby_slow.Il_bb(self.a, self.b, self.c, lam),
            places=12)
        self.assertAlmostEqual(
            µ.eshelby_slow.I_cc(self.a, self.b, self.c),
            µ.eshelby_slow.Il_cc(self.a, self.b, self.c, lam),
            places=12)

    def test_S(self):
        """
        –> Test the symmetry properties Sᵢⱼₖₗ=Sⱼᵢₖₗ=Sᵢⱼₗₖ (T. Mura eq. (11.16))
        –> Test the computed Eshelby tensor for a=b=c against the analytic
           solution for a spherical inclusion (T. Mura, Micromechanics of
           defects in solids (1982) eq. 11.21)
        –> Test the computed Eshelby tensor for c→∞ against the analytic
           solution for an ellyptic cylinder (T. Mura, Micromechanics of
           defects in solids (1982) eq. 11.22)
        """
        # Symmetry porperties #
        S_eshelby_sym = µ.eshelby_slow.S(self.a, self.b, self.c, self.nu)
        # Sᵢⱼₖₗ=Sⱼᵢₖₗ
        self.assertLess(np.linalg.norm(S_eshelby_sym -
                                       S_eshelby_sym.transpose((1, 0, 2, 3))),
                        self.tol)
        # Sᵢⱼₖₗ==Sᵢⱼₗₖ
        self.assertLess(np.linalg.norm(S_eshelby_sym -
                                       S_eshelby_sym.transpose((0, 1, 3, 2))),
                        self.tol)

        # Spherical inclusion (a=b=c) #
        # You can not choose a=b=c because it will lead to devisions by zero.
        # Therefore choose a small deviation from equality, with a > b > c.
        S_spherical = S_spherical_inclusion(self.nu)  # exact solution
        deviation = 1e-4
        a = self.a + deviation
        b = self.a
        c = self.a - deviation
        S_eshelby_spherical = µ.eshelby_slow.S(a, b, c, self.nu)
        self.assertLess(
            np.linalg.norm(S_spherical - S_eshelby_spherical), 1.435e-4)

        # Elliptic cylinder (a→∞) #
        # (T. Mura, Micromechanics of defects in solids (1982) eq. 11.22)
        # Turn a→∞ to satisfy a>b>c, So in all formulas exchange a↔c and 0↔2
        a = 1e7  # (a→∞)
        S_cylinder = S_elliptic_cylindrical_inclusion_a_inf(
            a, self.b, self.c, self.nu)  # exact solution
        S_eshelby_cylinder = µ.eshelby_slow.S(a, self.b, self.c, self.nu)
        self.assertLess(
            np.linalg.norm(S_cylinder - S_eshelby_cylinder), self.tol)

    def test_SL(self):
        """
        -> Test the symmetry properties Sᵢⱼₖₗ=Sⱼᵢₖₗ=Sᵢⱼₗₖ (T. Mura eq. (11.16))
        -> Test the computed Eshelby tensor Sᵢⱼₖₗ(λ) against the previous
           computed Sᵢⱼₖₗ. They should be the same for λ=0!
        –> Test the computed Eshelby tensor for a=b=c at λ=0 against the
           analytic solution for a spherical inclusion (T. Mura, Micromechanics
           of defects in solids (1982) eq. 11.21)
        –> Test the computed Eshelby tensor for c→∞ at λ=0 against the analytic
           solution for an ellyptic cylinder (T. Mura, Micromechanics of
           defects in solids (1982) eq. 11.22)
        """
        # Symmetry porperties #
        Sl_eshelby_sym = µ.eshelby_slow.Sl(
            self.a, self.b, self.c, self.nu, self.lam)
        # Sᵢⱼₖₗ=Sⱼᵢₖₗ
        self.assertLess(np.linalg.norm(Sl_eshelby_sym -
                                       Sl_eshelby_sym.transpose((1, 0, 2, 3))),
                        self.tol)
        # Sᵢⱼₖₗ==Sᵢⱼₗₖ
        self.assertLess(np.linalg.norm(Sl_eshelby_sym -
                                       Sl_eshelby_sym.transpose((0, 1, 3, 2))),
                        self.tol)

        # Sᵢⱼₖₗ(λ=0) vs Sᵢⱼₖₗ #
        lam = 0  # λ=0 thus Sᵢⱼₖₗ(λ) = Sᵢⱼₖₗ
        S = µ.eshelby_slow.S(self.a, self.b, self.c, self.nu)
        Sl = µ.eshelby_slow.Sl(self.a, self.b, self.c, self.nu, lam)
        self.assertLess(np.linalg.norm(S-Sl), self.tol)

        # Sᵢⱼₖₗ(λ=1e-4) vs Sᵢⱼₖₗ #
        lam = 1e-4  # λ=10⁻⁴ thus Sᵢⱼₖₗ(λ) ≈ Sᵢⱼₖₗ
        S = µ.eshelby_slow.S(self.a, self.b, self.c, self.nu)
        Sl = µ.eshelby_slow.Sl(self.a, self.b, self.c, self.nu, lam)
        self.assertLess(np.linalg.norm(S-Sl), 1.6183e-4)

        # Spherical inclusion (a=b=c) #
        # You can not choose a=b=c because it will lead to devisions by zero.
        # Therefore choose a small deviation from equality, with a > b > c.
        S_spherical = S_spherical_inclusion(self.nu)  # exact solution
        lam = 0
        deviation = 1e-4
        a = self.a + deviation
        b = self.a
        c = self.a - deviation
        S_eshelby_spherical = µ.eshelby_slow.Sl(a, b, c, self.nu, lam)
        self.assertLess(
            np.linalg.norm(S_spherical - S_eshelby_spherical), 1.435e-4)

        # Elliptic cylinder (a→∞) #
        # (T. Mura, Micromechanics of defects in solids (1982) eq. 11.22)
        # Turn a→∞ to satisfy a>b>c, So in all formulas exchange a↔c and 0↔2
        a = 1e7  # (a→∞)
        lam = 0
        S_cylinder = S_elliptic_cylindrical_inclusion_a_inf(
            a, self.b, self.c, self.nu)  # exact solution
        S_eshelby_cylinder = µ.eshelby_slow.Sl(a, self.b, self.c, self.nu, lam)
        self.assertLess(
            np.linalg.norm(S_cylinder - S_eshelby_cylinder), self.tol)

    def test_compute_lambda(self):
        """
        Tests if the computed lambda solves the ellipsoidal equation
        """
        lam = µ.eshelby_slow.compute_lambda(self.x, self.y, self.z,
                                            self.a, self.b, self.c)
        deviation = self.x**2/(self.a**2+lam) + self.y**2/(self.b**2+lam) \
            + self.z**2/(self.c**2+lam) - 1
        self.assertLess(deviation, self.tol)

        # 2.)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Trigger the warning.
            µ.eshelby_slow.compute_lambda(self.a/2, self.b/2, self.c/2,
                                          self.a, self.b, self.c)
            # Verify for correct message
            self.assertTrue(issubclass(w[-1].category, UserWarning))
            self.assertTrue("Your point x={}, y={}, z={} lies inside the "
                            "inclusion!".format(self.a/2, self.b/2, self.c/2)
                            in str(w[-1].message))

    def test_D(self):
        """
        -> Test the symmetry properties Dᵢⱼₖₗ=Dⱼᵢₖₗ=Dᵢⱼₗₖ, which should hold
           analogous to the symmetry for Sᵢⱼₖₗ.
        """
        D_sym_test = µ.eshelby_slow.D(self.x, self.y, self.z,
                                      self.a, self.b, self.c, self.nu)
        # Dᵢⱼₖₗ=Dⱼᵢₖₗ
        self.assertLess(np.linalg.norm(D_sym_test -
                                       D_sym_test.transpose((1, 0, 2, 3))),
                        self.tol)
        # Dᵢⱼₖₗ==Dᵢⱼₗₖ
        self.assertLess(np.linalg.norm(D_sym_test -
                                       D_sym_test.transpose((0, 1, 3, 2))),
                        self.tol)

    def test_stiffness_tensor(self):
        """
        --> Test symmetries: Cᵢⱼₖₗ=Cₖₗᵢⱼ=Cⱼᵢₖₗ=Cᵢⱼₗₖ
        """
        # Symmetries #
        C_sym_test = µ.eshelby_slow.stiffness_tensor(self.E, self.nu)
        # Cᵢⱼₖₗ=Cₖₗᵢⱼ
        self.assertLess(np.linalg.norm(C_sym_test -
                                       C_sym_test.transpose((2, 3, 0, 1))),
                        self.tol)
        # Cᵢⱼₖₗ=Cⱼᵢₖₗ
        self.assertLess(np.linalg.norm(C_sym_test -
                                       C_sym_test.transpose((1, 0, 2, 3))),
                        self.tol)
        # Cᵢⱼₖₗ==Cᵢⱼₗₖ
        self.assertLess(np.linalg.norm(C_sym_test -
                                       C_sym_test.transpose((0, 1, 3, 2))),
                        self.tol)

    def test_get_equivalent_eigenstrain(self):
        """
        -> Tests if the function is executable without deeper meaning.
        -> Test if the solution eps_eq_eig fulfills the fourth order tensor eq.
        """
        # 1.)
        eps_0 = np.zeros((3, 3))
        eps_0 += np.eye(3)
        if self.verbose == 2:
            print("\n\nTest 'get_equivalent_eigenstrain()'")
            print("strain at boundaries, ε⁰ₖₗ:\n", eps_0)
        eps_eq_eig = µ.eshelby_slow.get_equivalent_eigenstrain(
            self.E, self.nu, self.E_I, self.nu_I,
            self.a, self.b, self.c, eps_0)
        if self.verbose == 2:
            print("equivalent eigenstrain, ε*ₘₙ:\n", eps_eq_eig)

        # 2.)
        eps_0 = np.random.random((3, 3))  # random boundary strain
        eps_0 = 1/2 * (eps_0 + eps_0.T)

        eps_p_0 = np.zeros((3, 3))  # zero initial eigenstrain

        eps_eq_eig = µ.eshelby_slow.get_equivalent_eigenstrain(
            self.E, self.nu, self.E_I, self.nu_I,
            self.a, self.b, self.c, eps_0)
        # does it fulfills
        # (ΔCᵢⱼₖₗSₖₗₘₙ - Cᵢⱼₘₙ)ε*ₘₙ = - ΔCᵢⱼₖₗε⁰ₖₗ - C*ᵢⱼₖₗεᵖₖₗ?
        C_ijkl_out = µ.eshelby_slow.stiffness_tensor(self.E, self.nu)
        C_ijkl_in = µ.eshelby_slow.stiffness_tensor(self.E_I, self.nu_I)
        Del_C_ijkl = C_ijkl_out - C_ijkl_in
        S_klmn = µ.eshelby_slow.S(self.a, self.b, self.c, self.nu_I)
        lh = np.einsum("ijkl, klmn -> ijmn", Del_C_ijkl, S_klmn) - C_ijkl_out
        rh = - np.einsum("ijkl, kl -> ij", Del_C_ijkl, eps_0) \
             - np.einsum("ijkl, kl -> ij", C_ijkl_in, eps_p_0)

        self.assertLess(np.linalg.norm(
            np.einsum("ijmn, mn -> ij", lh, eps_eq_eig) - rh), self.tol)

    def test_get_stress_and_strain_in(self):
        """
        Tests if the function is executable without deeper meaning.
        """
        eps_0 = np.zeros((3, 3))
        eps_0 += np.eye(3)
        if self.verbose == 2:
            print("\n\nTest 'get_stress_and_strain_in()'")
            print("strain at boundaries, ε⁰ₖₗ:\n", eps_0)
        sigma_in, eps_in = µ.eshelby_slow.get_stress_and_strain_in(
            self.E, self.nu, self.E_I, self.nu_I,
            self.a, self.b, self.c, eps_0)
        if self.verbose == 2:
            print("strain in Ω:\n", eps_in)
            print("stress in Ω:\n", sigma_in)

    def test_get_stress_and_strain_out(self):
        """
        Tests if the function is executable without deeper meaning.
        """
        eps_0 = np.zeros((3, 3))
        eps_0 += np.eye(3)
        sigma_0 = np.einsum("ijkl, kl -> ij",
                            µ.eshelby_slow.stiffness_tensor(self.E, self.nu),
                            eps_0)
        if self.verbose == 2:
            print("\n\nTest 'get_stress_and_strain_out()'")
            print("strain at boundaries, ε⁰ₖₗ:\n", eps_0)
            print("stress at boundaries, ε⁰ₖₗ:\n", sigma_0)
        sigma_out, eps_out = µ.eshelby_slow.get_stress_and_strain_out(
            self.x, self.y, self.z, self.E, self.nu, self.E_I, self.nu_I,
            self.a, self.b, self.c, eps_0)
        if self.verbose == 2:
            print("strain in D-Ω, at x=", self.y,
                  self.y, self.z, "\n", eps_out)
            print("stress in D-Ω, at x=", self.y,
                  self.y, self.z, "\n", sigma_out)

    def test_get_displacement_field(self):
        """
        Tests if the function is executable without deeper meaning.
        """
        eps_0 = np.zeros((3, 3))
        eps_0 += np.eye(3)
        if self.verbose == 2:
            print("\n\nTest 'get_displacement_field()'")
            print("strain at boundaries, ε⁰ₖₗ:\n", eps_0)
        displacement = µ.eshelby_slow.get_displacement_field(
            self.x, self.y, self.z, self.E, self.nu, self.E_I, self.nu_I,
            self.a, self.b, self.c, eps_0)
        if self.verbose == 2:
            print("displacement:\n", displacement)

    def test_muSpectre_eshelby_vs_Esh3D(self):
        """
        Compute the stress, strain and displacements by the eshelby solution as
        implemented in muSpectre and compare the results with tabulated results
        from an Esh3D computation.
        """
        # ESHELBY INCLUSION #
        # read reference data and initialise inclusion parameters
        reference = np.load("reference_computations/eshelby_inclusion.ref.npz")

        inclusion_parameters = reference["ellip"]
        x_0 = inclusion_parameters[0:3]*1e-3  # center of the inclusion
        a_ = inclusion_parameters[3:6]*1e-3  # ellipsoidal half axes
        a, b, c = a_
        # Young and Poisson of the inclusion
        E, nu = inclusion_parameters[9:11]
        E_I, nu_I = E, nu  # inclusion ⇒ E_I = E, nu_I = nu
        # eigen strain of the inclusion
        eps_p_0_voigt = inclusion_parameters[11:17]
        eps_p_0 = vv_fm(eps_p_0_voigt, order="esh3d")
        sigma_0_voigt = np.zeros((6,))  # initial stress at boundaries
        eps_0 = np.zeros((3, 3))  # initial strain at boundaries

        # results from Esh3D computation
        x_ = reference["ocoord"][:]
        displ_esh3d = reference["odat"][:, :3]*1e-3
        sigma_esh3d = reference["odat"][:, 3:]
        # correct stress by stress at infinity
        sigma_esh3d += sigma_0_voigt

        # compute strain from stress
        eps_esh3d = np.linalg.solve(
            stiffness_tensor_voigt(E, nu).reshape((1, 6, 6)), sigma_esh3d)
        # correct strain by eigen strain inside the inclusion
        # and off diagonal elements by factor 1/2
        eps_esh3d[:, 3:6] *= 1/2
        inside_inclusion = np.where(
            ((x_-x_0)**2 / a_**2).sum(axis=-1) - 1 <= 0)
        eps_esh3d[inside_inclusion] += eps_p_0_voigt

        for i, position in enumerate(x_):
            x, y, z = position - x_0
            if (x/a)**2 + (y/b)**2 + (z/c)**2 - 1 < 0:
                sigma, eps = µ.eshelby_slow.get_stress_and_strain_in(
                    E, nu, E_I, nu_I, a, b, c, eps_0, eps_p_0)
            else:
                sigma, eps = µ.eshelby_slow.get_stress_and_strain_out(
                    x, y, z, E, nu, E_I, nu_I, a, b, c, eps_0, eps_p_0)

            displ = µ.eshelby_slow.get_displacement_field(
                x, y, z, E, nu, E_I, nu_I, a, b, c, eps_0, eps_p_0)

            self.assertLess(np.linalg.norm(sigma - vv_fm(sigma_esh3d[i],
                                                         order="esh3d")), 1e-8)
            self.assertLess(
                np.linalg.norm(eps - vv_fm(eps_esh3d[i], order="esh3d")), 1e-8)
            self.assertLess(np.linalg.norm(displ - displ_esh3d[i]), 1e-8)

        # ESHELBY INHOMOGENEITY #
        # read reference data and initialise inclusion parameters
        reference = np.load(
            "reference_computations/eshelby_inhomogeneity.ref.npz")

        inclusion_parameters = reference["ellip"]
        x_0 = inclusion_parameters[0:3]*1e-3  # center of the inclusion
        a_ = inclusion_parameters[3:6]*1e-3  # ellipsoidal half axes
        a, b, c = a_
        # Young and Poisson of the inclusion
        E_I, nu_I = inclusion_parameters[9:11]
        E, nu = E_I/2, nu_I
        # eigen strain of the inclusion
        eps_p_0_voigt = inclusion_parameters[11:17]
        eps_p_0 = vv_fm(eps_p_0_voigt, order="esh3d")
        # initial strain at boundaries from initial stress at boundaries
        # (σ₁₁,σ₂₂,σ₃₃,σ₁₂,σ₂₃,σ₁₃)
        sigma_0_voigt = np.array([0.1, 0.05, 0.2, 0.15, 0.02, 0.08])
        # compute strain from stress
        eps_0_voigt = np.linalg.solve(
            stiffness_tensor_voigt(E, nu), sigma_0_voigt)
        # correct strain off diagonal elements by factor 1/2
        eps_0_voigt[3:6] *= 1/2
        eps_0 = vv_fm(eps_0_voigt, order="esh3d")

        # results from Esh3D computation
        x_ = reference["ocoord"][:]
        displ_esh3d = reference["odat"][:, :3]*1e-3
        sigma_esh3d = reference["odat"][:, 3:]
        # correct stress by stress at infinity
        sigma_esh3d += sigma_0_voigt

        # compute strain from stress
        eps_esh3d = np.linalg.solve(
            stiffness_tensor_voigt(E, nu).reshape((1, 6, 6)), sigma_esh3d)
        # correct strain by eigen strain and different material inside the
        # inclusion and off diagonal elements by factor 1/2
        inside_inclusion = np.where(
            ((x_-x_0)**2 / a_**2).sum(axis=-1) - 1 <= 0)
        eps_esh3d[inside_inclusion] = np.linalg.solve(
            stiffness_tensor_voigt(E_I, nu_I).reshape((1, 6, 6)),
            sigma_esh3d[inside_inclusion])
        eps_esh3d[:, 3:6] *= 1/2
        eps_esh3d[inside_inclusion] += eps_p_0_voigt

        for i, position in enumerate(x_):
            x, y, z = position - x_0
            if (x/a)**2 + (y/b)**2 + (z/c)**2 - 1 < 0:
                sigma, eps = µ.eshelby_slow.get_stress_and_strain_in(
                    E, nu, E_I, nu_I, a, b, c, eps_0, eps_p_0)
            else:
                sigma, eps = µ.eshelby_slow.get_stress_and_strain_out(
                    x, y, z, E, nu, E_I, nu_I, a, b, c, eps_0, eps_p_0)

            displ = µ.eshelby_slow.get_displacement_field(
                x, y, z, E, nu, E_I, nu_I, a, b, c, eps_0, eps_p_0)

            self.assertLess(np.linalg.norm(sigma - vv_fm(sigma_esh3d[i],
                                                         order="esh3d")), 1e-8)
            self.assertLess(np.linalg.norm(eps - vv_fm(eps_esh3d[i],
                                                       order="esh3d")), 1e-8)
            self.assertLess(np.linalg.norm(displ - displ_esh3d[i]), 1e-8)

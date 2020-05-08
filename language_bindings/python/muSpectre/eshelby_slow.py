#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file   eshelby_slow.py

@author Richard Leute <richard.leute@imtek.uni-freiburg.de>

@date   05 Jun 2019

@brief  Eshelby tensor and functions to compute the stress and strain of a
        single Eshelby inclusion or inhomogeneity. The ellipsoid inclusion
        with the main axes a, b, c along the coordinate directions have to
        fulfill a > b > c.

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

import numpy as np
import warnings
from scipy.special import ellipkinc, ellipeinc
from scipy.linalg import solve
from itertools import product


##########   INCOMPLETE ELLIPTIC INTEGRALS OF FIRST AND SECOND KIND   #########
def F(a, b, c):
    """
    Incomplete elliptic integral of the first kind for an Eshelby inclusion.
    The parameters a, b, c are the principal half axes of the ellipsoide with
    a > b > c.

    F(θ,k) = ∫₀ᶿ(1 - k²sin²(w))^{-1/2} dw
    θ = arcsin( sqrt(1 - c²/a²) )
    k = [ (a²-b²) / (a²-c²) ]^{1/2}
    """
    theta = np.arcsin(np.sqrt(1 - c**2/a**2))
    m = (a**2 - b**2) / (a**2 - c**2)  # m=k²
    return ellipkinc(theta, m)


def E(a, b, c):
    """
    Incomplete elliptic integral of the second kind for an Eshelby inclusion.
    The parameters a, b, c are the principal half axes of the ellipsoide with
    a > b > c.

    E(θ,k) = ∫₀ᶿ(1 - k²sin²(w))^{1/2} dw
    θ = arcsin( sqrt(1 - c²/a²) )
    k = [ (a²-b²) / (a²-c²) ]^{1/2}
    """
    theta = np.arcsin(np.sqrt(1 - c**2/a**2))
    m = (a**2 - b**2) / (a**2 - c**2)  # m=k²
    return ellipeinc(theta, m)


def Fl(a, b, c, lam):
    """
    Incomplete elliptic integral of the first kind for an Eshelby inclusion.
    The parameters a, b, c are the principal half axes of the ellipsoide with
    a > b > c, lam is the lower integration boundary given by the largest
    positive square root of the equation x²/(a²+λ) + y²/(b²+λ)+z²/(c²+λ) = 1

    F(θ,k) = ∫ₗₐₘᶿ(1 - k²sin²(w))^{-1/2} dw
    θ = arcsin( sqrt((a²-c²)/(a²+λ)) )
    k = [ (a²-b²) / (a²-c²) ]^{1/2}
    """
    theta = np.arcsin(np.sqrt((a**2 - c**2)/(a**2 + lam)))
    m = (a**2 - b**2) / (a**2 - c**2)  # m=k²
    return ellipkinc(theta, m)


def El(a, b, c, lam):
    """
    Incomplete elliptic integral of the second kind for an Eshelby inclusion.
    The parameters a, b, c are the principal half axes of the ellipsoide with
    a > b > c, lam is the lower integration boundary given by the largest
    positive square root of the equation x²/(a²+λ) + y²/(b²+λ)+z²/(c²+λ) = 1

    E(θ,k) = ∫ₗₐₘᶿ(1 - k²sin²(w))^{1/2} dw
    θ = arcsin( sqrt((a²-c²)/(a²+λ)) )
    k = [ (a²-b²) / (a²-c²) ]^{1/2}
    """
    theta = np.arcsin(np.sqrt((a**2 - c**2)/(a**2 + lam)))
    m = (a**2 - b**2) / (a**2 - c**2)  # m=k²
    return ellipeinc(theta, m)


###############################################################################
##########################    "INNER REGION"    Ω    ##########################
###############################################################################

####################   I-INTEGRALS FOR THE ESHELBY TENSOR   ###################
def I_a(a, b, c):
    """
    Eshelby 57' eq. (3·9) first line
    I_a = 4πabc/((a²-b²)*sqrt(a²-c²)) * (F(θ,k)-E(θ,k))
    """
    return (4*np.pi*a*b*c)/((a**2-b**2)*np.sqrt(a**2-c**2)) \
        * (F(a, b, c) - E(a, b, c))


def I_b(a, b, c):
    """
    Eshelby 57' eq. (3·10)
    I_a + I_b + I_c = 4π  ==>  I_b = 4π - I_a - I_c
    """
    return 4*np.pi - I_a(a, b, c) - I_c(a, b, c)


def I_c(a, b, c):
    """
    Eshelby 57' eq. (3·9) second line
    I_c = 4πabc/((b²-c²)*sqrt(a²-c²)) * ( (b*sqrt(a²-c²))/(a*c) - E(θ,k))
    """
    return (4*np.pi*a*b*c)/((b**2-c**2)*np.sqrt(a**2-c**2)) \
        * ((b*np.sqrt(a**2 - c**2))/(a*c) - E(a, b, c))


def I_ab(a, b, c):
    """
    Eshelby 57' eq. (3·13) corrected by factor 3 to end up with T. Muras
    definition of I_12 in eq (11.14)
    I_ab = (I_b - I_a)/(a²-b²)
    """
    return (I_b(a, b, c) - I_a(a, b, c)) / (a**2 - b**2)


def I_bc(a, b, c):
    """
    Eshelby 57' eq. (3·13) cyclic counterpart (one permutation) corrected by
    factor 3 to end up with T. Muras definition of I_12 in eq (11.14)
    I_bc = (I_c - I_b)/(b²-c²)
    """
    return (I_c(a, b, c) - I_b(a, b, c)) / (b**2 - c**2)


def I_ca(a, b, c):
    """
    Eshelby 57' eq. (3·13) cyclic counterpart (two permutations) corrected by
    factor 3 to end up with T. Muras definition of I_12 in eq (11.14)
    I_ca = (I_a - I_c)/(c²-a²)
    """
    return (I_a(a, b, c) - I_c(a, b, c)) / (c**2 - a**2)


def I_ba(a, b, c):
    return I_ab(a, b, c)


def I_cb(a, b, c):
    return I_bc(a, b, c)


def I_ac(a, b, c):
    return I_ca(a, b, c)


def I_aa(a, b, c):
    """
    Eshelby 57' eq. (3·14) corrected by factor 3 to take note of T. Muras
    different definition of I_12 in eq (11.14), we use T. Mura (11.19,2)
    I_aa = 1/3 (4π/a² - I_ab - I_ac)
    """
    return 1/3 * ((4*np.pi) / a**2 - I_ab(a, b, c) - I_ac(a, b, c))


def I_bb(a, b, c):
    """
    Eshelby 57' eq. (3·14) cyclic counterpart (one permutation) corrected by
    factor 3 to take note of T. Muras different definition of I_12 in eq.
    (11.14) we use T. Mura (11.19,2)
    I_bb = 1/3 (4π/b² - I_bc - I_ba)
    """
    return 1/3 * ((4*np.pi) / b**2 - I_bc(a, b, c) - I_ba(a, b, c))


def I_cc(a, b, c):
    """
    Eshelby 57' eq. (3·14) cyclic counterpart (two permutations) corrected by
    factor 3 to take note of T. Muras different definition of I_12 in eq.
    (11.14) we use T. Mura (11.19,2)
    I_cc = 1/3 (4π/c² - I_ca - I_cb)
    """
    return 1/3 * ((4*np.pi) / c**2 - I_ca(a, b, c) - I_cb(a, b, c))


###########################   ESHELBY TENSOR Sᵢⱼₖₗ   ##########################
def S(a, b, c, nu):
    """
    Eshelby tensor Sᵢⱼₖₗ, computed as described in T. Mura, Micromechanics of
    defects in solids (1982) eq. 11.16.

    Keyword Arguments:
    a, b and c -- Three floats giving the principal half axes of the
                  ellipsoidal inclusion, where the half axes are alligned with
                  the coordinate axes and the order "a > b > c" is fulfilled.
    nu         -- float, Poissons ratio of the matrix material; -1 <= nu <= 0.5

    Returns:
    S   -- Eshelby Tensor Sᵢⱼₖₗ belonging to the given ellipsoidal inclusion
    """
    if a < b or b < c:
        raise ValueError("The ellipsoidal pricipal half axes should satisfy:" +
                         "\na > b > c\nBut you gave: a="+str(a)+", b="+str(b) +
                         " c="+str(c))

    # initialize prefactors
    f1 = 3 / (8*np.pi*(1-nu))
    f2 = (1-2*nu) / (8*np.pi*(1-nu))
    f3 = 1 / (8*np.pi*(1-nu))
    f4 = 1/2 * f3
    f5 = 1/2 * f2

    # compute I's
    _I_a = I_a(a, b, c)
    _I_b = I_b(a, b, c)
    _I_c = I_c(a, b, c)
    _I_ab = I_ab(a, b, c)
    _I_bc = I_bc(a, b, c)
    _I_ca = I_ca(a, b, c)
    _I_ac = _I_ca
    _I_ba = _I_ab
    _I_cb = _I_bc
    _I_aa = I_aa(a, b, c)
    _I_bb = I_bb(a, b, c)
    _I_cc = I_cc(a, b, c)

    # initialize S
    S = np.zeros((3,)*4)

    # fill S, Mura 82' eq. 11.16
    S[0, 0, 0, 0] = f1*a**2*_I_aa + f2*_I_a  # S_1111
    S[1, 1, 1, 1] = f1*b**2*_I_bb + f2*_I_b  # S_2222
    S[2, 2, 2, 2] = f1*c**2*_I_cc + f2*_I_c  # S_3333

    S[0, 0, 1, 1] = f3*b**2*_I_ab - f2*_I_a  # S_1122
    S[1, 1, 2, 2] = f3*c**2*_I_bc - f2*_I_b  # S_2233
    S[2, 2, 0, 0] = f3*a**2*_I_ca - f2*_I_c  # S_3311

    S[0, 0, 2, 2] = f3*c**2*_I_ac - f2*_I_a  # S_1133
    S[1, 1, 0, 0] = f3*a**2*_I_ba - f2*_I_b  # S_2211
    S[2, 2, 1, 1] = f3*b**2*_I_cb - f2*_I_c  # S_3322

    S[0, 1, 0, 1] = f4*(a**2+b**2)*_I_ab + f5*(_I_a+_I_b)  # S_1212
    S[1, 0, 0, 1] = S[0, 1, 0, 1]  # use symmetry Sⱼᵢₖₗ = Sᵢⱼₖₗ
    S[0, 1, 1, 0] = S[0, 1, 0, 1]  # use symmetry Sᵢⱼₗₖ = Sᵢⱼₖₗ
    S[1, 0, 1, 0] = S[0, 1, 0, 1]  # use both symmetries Sⱼᵢₗₖ = Sᵢⱼₖₗ
    S[1, 2, 1, 2] = f4*(b**2+c**2)*_I_bc + f5*(_I_b+_I_c)  # S_2323
    S[2, 1, 1, 2] = S[1, 2, 1, 2]  # use symmetry Sⱼᵢₖₗ = Sᵢⱼₖₗ
    S[1, 2, 2, 1] = S[1, 2, 1, 2]  # use symmetry Sᵢⱼₗₖ = Sᵢⱼₖₗ
    S[2, 1, 2, 1] = S[1, 2, 1, 2]  # use both symmetries Sⱼᵢₗₖ = Sᵢⱼₖₗ
    S[2, 0, 2, 0] = f4*(c**2+a**2)*_I_ca + f5*(_I_c+_I_a)  # S_3131
    S[0, 2, 2, 0] = S[2, 0, 2, 0]  # use symmetry Sⱼᵢₖₗ = Sᵢⱼₖₗ
    S[2, 0, 0, 2] = S[2, 0, 2, 0]  # use symmetry Sᵢⱼₗₖ = Sᵢⱼₖₗ
    S[0, 2, 0, 2] = S[2, 0, 2, 0]  # use both symmetries Sⱼᵢₗₖ = Sᵢⱼₖₗ

    return S


###############################################################################
##########################    "OUTER REGION"  D-Ω    ##########################
###############################################################################

################    I(λ)-INTEGRALS FOR THE D ESHELBY TENSOR    ################
def Il(a, b, c, lam):
    """
    I(λ) = 4πabc·F(θ(λ),k) / sqrt(a²-c²)
    From Meng et al. (2012) eq. (24)
    """
    return 4*np.pi*a*b*c*Fl(a, b, c, lam) / np.sqrt(a**2 - c**2)


def Il_a(a, b, c, lam):
    """
    Meng et al. (2012) eq. (15) first line
    I_a(λ) = 4πabc/((a²-b²)*sqrt(a²-c²)) * (F(θ(λ),k)-E(θ(λ),k))
    """
    return (4*np.pi*a*b*c)/((a**2-b**2)*np.sqrt(a**2-c**2)) \
        * (Fl(a, b, c, lam) - El(a, b, c, lam))


def Il_b(a, b, c, lam):
    """
    Meng et al. (2012) eq. (15) third line
    I_b(λ) = 4πabc/sqrt((a²+λ)*(b²+λ)*(c²+λ)) - I_a(λ) - I_c(λ)
    """
    return (4*np.pi*a*b*c)/np.sqrt((a**2+lam)*(b**2+lam)*(c**2+lam)) \
        - Il_a(a, b, c, lam) - Il_c(a, b, c, lam)


def Il_c(a, b, c, lam):
    """
    Meng et al. (2012) eq. (15) second line
    I_c(λ) = 4πabc/((b²-c²)*sqrt(a²-c²)) *
             {((b²+λ)*sqrt(a²-c²))/sqrt((a²+λ)*(b²+λ)*(c²+λ)) - E(θ(λ),k)}
    """
    return \
        (4*np.pi*a*b*c)/((b**2-c**2)*np.sqrt(a**2-c**2)) *\
        (((b**2+lam)*np.sqrt(a**2-c**2)) /
         np.sqrt((a**2+lam)*(b**2+lam)*(c**2+lam)) - El(a, b, c, lam))


def Il_ab(a, b, c, lam):
    """
    Meng et al. (2012) eq. (15) fourth line
    I_ab(λ) = (I_b(λ) - I_a(λ))/(a²-b²)
    """
    return (Il_b(a, b, c, lam) - Il_a(a, b, c, lam)) / (a**2 - b**2)


def Il_bc(a, b, c, lam):
    """
    Meng et al. (2012) eq. (15) fourth line cyclic counterpart
    (one permutation)
    I_bc(λ) = (I_c(λ) - I_b(λ))/(b²-c²)
    """
    return (Il_c(a, b, c, lam) - Il_b(a, b, c, lam)) / (b**2 - c**2)


def Il_ca(a, b, c, lam):
    """
    Meng et al. (2012) eq. (15) fourth line cyclic counterpart
    (two permutations)
    I_ca(λ) = (I_a(λ) - I_c(λ))/(c²-a²)
    """
    return (Il_a(a, b, c, lam) - Il_c(a, b, c, lam)) / (c**2 - a**2)


def Il_ba(a, b, c, lam):
    return Il_ab(a, b, c, lam)


def Il_cb(a, b, c, lam):
    return Il_bc(a, b, c, lam)


def Il_ac(a, b, c, lam):
    return Il_ca(a, b, c, lam)


def Il_aa(a, b, c, lam):
    """
    Meng et al. (2012) eq. (15) fifth line
    I_aa(λ) = 4πabc / (3*(a²+λ)*sqrt((a²+λ)*(b²+λ)*(c²+λ)))
              - (I_ab(λ)+I_ac(λ))/3
    """
    return (4*np.pi*a*b*c) /  \
        (3*(a**2+lam) * np.sqrt((a**2+lam)*(b**2+lam)*(c**2+lam)))  \
        - 1/3 * (Il_ab(a, b, c, lam) + Il_ac(a, b, c, lam))


def Il_bb(a, b, c, lam):
    """
    Meng et al. (2012) eq. (15) fifth line cyclic counterpart (one permutation)
    I_bb(λ) = 4πabc / (3*(b²+λ)*sqrt((a²+λ)*(b²+λ)*(c²+λ)))
              - (I_bc(λ)+I_ba(λ))/3
    """
    return (4*np.pi*a*b*c) /  \
        (3*(b**2+lam) * np.sqrt((a**2+lam)*(b**2+lam)*(c**2+lam)))  \
        - 1/3 * (Il_bc(a, b, c, lam) + Il_ba(a, b, c, lam))


def Il_cc(a, b, c, lam):
    """
    Meng et al. (2012) eq. (15) fifth line cyclic counterpart
    (two permutations)
    I_cc(λ) = 4πabc / (3*(c²+λ)*sqrt((a²+λ)*(b²+λ)*(c²+λ)))
              - (I_ca(λ)+I_cb(λ))/3
    """
    return (4*np.pi*a*b*c) /  \
        (3*(c**2+lam) * np.sqrt((a**2+lam)*(b**2+lam)*(c**2+lam)))  \
        - 1/3 * (Il_ca(a, b, c, lam) + Il_cb(a, b, c, lam))


#########   DERIVATIVES OF THE I-INTEGRALS FOR THE D ESHELBY TENSOR   #########

# helper functions for derivatives of lambda (λ,ᵢ , λ,ᵢⱼ)
def Fi(_x, _a, lam, i):
    """
    Following Meng et al. (2012) eq. (19)
    Fᵢ = 2xᵢ / (a_I²+λ)
    """
    return (2*_x[i]) / (_a[i]**2 + lam)


def C(_x, _a, lam):
    """
    Following Meng et al. (2012) eq. (19)
    C = xᵢxᵢ / (a_I²+λ)²
    """
    return (_x**2 / (_a**2+lam)**2).sum()


def Fi_d_j(_x, _a, lam, i, j):
    """
    Computing the derivative of Meng et al. (2012) eq. (19)
    Fᵢ,ⱼ = ∂Fᵢ/∂xⱼ = {2δᵢⱼ / (a_I²+λ)} - {2xᵢλ,ⱼ / (a_I²+λ)²}
    """
    if i == j:
        return (2 / (_a[i]**2 + lam)) \
            - (2*_x[i]*lambda_d_i(_x, _a, lam, j)) / (_a[i]**2+lam)**2
    else:
        return -(2*_x[i]*lambda_d_i(_x, _a, lam, j)) / (_a[i]**2+lam)**2


def C_d_i(_x, _a, lam, i):
    """
    Computing the derivative of Meng et al. (2012) eq. (19)
    C,ᵢ = ∂C/∂xᵢ = {2xᵢ / (a_I²+λ)²} - {2λ,ᵢ · xₘxₘ/(a_M²+λ)³}
    """
    return 2*_x[i] / (_a[i]**2 + lam)**2 \
        - 2*lambda_d_i(_x, _a, lam, i) * (_x**2/(_a**2+lam)**3).sum()


# computing first and second derivative of λ(_x)
def lambda_d_i(_x, _a, lam, i):
    """
    First derivative ov λ(_x) with respect to xᵢ
    Following Meng et al. (2012) eq. (18)
    λ,ᵢ = Fᵢ/C
    """
    return Fi(_x, _a, lam, i) / C(_x, _a, lam)


def lambda_d_ij(_x, _a, lam, i, j):
    """
    Second derivative ov λ(_x) with respect to xᵢ and xⱼ
    Following Meng et al. (2012) eq. (18)
    λ,ᵢⱼ = (Fᵢ,ⱼ - λ,i · C,ⱼ) / C = (Fᵢ,ⱼ - Fᵢ/C · C,ⱼ) / C
    """
    return (Fi_d_j(_x, _a, lam, i, j) - Fi(_x, _a, lam, i)/C(_x, _a, lam) *
            C_d_i(_x, _a, lam, j)) / C(_x, _a, lam)


# Computing the I derivatives
def Il_i_d_j(_x, _a, lam, i, j):
    """
    Following Meng et al. (2012) eq. (17) first line
    Iᵢ,ⱼ(λ) = -2πabc / sqrt((a²+λ)(b²+λ)(c²+λ)) * λ,ⱼ/(_aᵢ²+λ)
    """
    return (-2*np.pi*_a.prod()) / (np.sqrt(_a**2+lam)).prod() * \
        lambda_d_i(_x, _a, lam, j) / (_a[i]**2+lam)


def Il_ij_d_k(_x, _a, lam, i, j, k):
    """
    Following Meng et al. (2012) eq. (17) second line
    Iᵢⱼ,ₖ(λ) = -2πabc / sqrt((a²+λ)(b²+λ)(c²+λ)) * λ,ₖ/((_aᵢ²+λ)(_aⱼ²+λ))
    """
    return (-2*np.pi*_a.prod()) / (np.sqrt(_a**2+lam)).prod() * \
        lambda_d_i(_x, _a, lam, k) / ((_a[i]**2+lam)*(_a[j]**2+lam))


def Il_i_d_jk(_x, _a, lam, i, j, k):
    """
    Following Meng et al. (2012) eq. (17) second line
    Iᵢ,ⱼₖ(λ) = -2πabc / ((_aᵢ²+λ)·sqrt((a²+λ)(b²+λ)(c²+λ))) ·
              [ λ,ⱼₖ - { 1/(_aᵢ²+λ) + 1/2·∑ₙ(1/(_aₙ²+λ))}·λ,ⱼ·λ,ₖ ]
    """
    return (-2*np.pi*_a.prod()) / ((_a[i]**2+lam)*(np.sqrt(_a**2+lam)).prod())\
        * (lambda_d_ij(_x, _a, lam, j, k) - (1/(_a[i]**2+lam) +
                                             1/2 * (1/(_a**2+lam)).sum())
           * lambda_d_i(_x, _a, lam, j) * lambda_d_i(_x, _a, lam, k))


def Il_ij_d_kl(_x, _a, lam, i, j, k, l):
    """
    Following Meng et al. (2012) eq. (17) second line
    Iᵢⱼ,ₖₗ(λ) = -2πabc / ((_aᵢ²+λ)(_aⱼ²+λ)·sqrt((a²+λ)(b²+λ)(c²+λ))) ·
             [ λ,ₖₗ - { 1/(_aᵢ²+λ) + 1/(_aⱼ²+λ) + 1/2·∑ₙ(1/(_aₙ²+λ))}·λ,ₖ·λ,ₗ ]
    """
    return (-2*np.pi*_a.prod()) / \
        ((_a[i]**2+lam) * (_a[j]**2+lam) * (np.sqrt(_a**2+lam)).prod()) \
        * (lambda_d_ij(_x, _a, lam, k, l)
           - (1/(_a[i]**2+lam) + 1/(_a[j]**2+lam)
              + 1/2 * (1/(_a**2+lam)).sum())
           * lambda_d_i(_x, _a, lam, k) * lambda_d_i(_x, _a, lam, l))


######################   ESHELBY TENSOR Sᵢⱼₖₗ(λ) in D-Ω   #####################
def Sl(a, b, c, nu, lam):
    """
    Eshelby Tensor in the outer region D-Ω depending additionally on λ and _x
    8π(1-ν) Sᵢⱼₖₗ(λ) = δᵢⱼδₖₗ[2ν I_I(λ) - I_K(λ) + a_I² I_KI(λ)]
                  +(δᵢₖδⱼₗ+δⱼₖδᵢₗ){a_I²I_IJ(λ) - I_J(λ) + (1-ν)[I_K(λ)+I_L(λ)]}
    See T. Mura 82' eq. (11.42)
    Hint: 11.42 breaks the symmetry of Sᵢⱼₖₗ, we instead use (11.16)
    """
    if a < b or b < c:
        raise ValueError("The ellipsoidal pricipal half axes should satisfy:" +
                         "\na > b > c\nBut you gave: a="+str(a)+", b="+str(b) +
                         " c="+str(c))
    # initialize Sᵢⱼₖₗ(λ)
    Sl = np.zeros((3,)*4)

    # Iᵢ(λ)
    Il_I = np.array([Il_a(a, b, c, lam), Il_b(
        a, b, c, lam), Il_c(a, b, c, lam)])

    # Iᵢⱼ(λ)
    Il_IJ = np.array(
        [[Il_aa(a, b, c, lam), Il_ab(a, b, c, lam), Il_ac(a, b, c, lam)],
         [Il_ba(a, b, c, lam), Il_bb(a, b, c, lam), Il_bc(a, b, c, lam)],
         [Il_ca(a, b, c, lam), Il_cb(a, b, c, lam), Il_cc(a, b, c, lam)]])

    # Different way to compute Sᵢⱼₖₗ fulfilling symmetry! (T. Mura eq. 11.16)
    # initialize prefactors
    f1 = 3 / (8*np.pi*(1-nu))
    f2 = (1-2*nu) / (8*np.pi*(1-nu))
    f3 = 1 / (8*np.pi*(1-nu))
    f4 = 1/2 * f3
    f5 = 1/2 * f2

    # fill S, Mura 82' eq. 11.16
    Sl[0, 0, 0, 0] = f1*a**2*Il_IJ[0, 0] + f2*Il_I[0]  # S_1111
    Sl[1, 1, 1, 1] = f1*b**2*Il_IJ[1, 1] + f2*Il_I[1]  # S_2222
    Sl[2, 2, 2, 2] = f1*c**2*Il_IJ[2, 2] + f2*Il_I[2]  # S_3333

    Sl[0, 0, 1, 1] = f3*b**2*Il_IJ[0, 1] - f2*Il_I[0]  # S_1122
    Sl[1, 1, 2, 2] = f3*c**2*Il_IJ[1, 2] - f2*Il_I[1]  # S_2233
    Sl[2, 2, 0, 0] = f3*a**2*Il_IJ[2, 0] - f2*Il_I[2]  # S_3311

    Sl[0, 0, 2, 2] = f3*c**2*Il_IJ[0, 2] - f2*Il_I[0]  # S_1133
    Sl[1, 1, 0, 0] = f3*a**2*Il_IJ[1, 0] - f2*Il_I[1]  # S_2211
    Sl[2, 2, 1, 1] = f3*b**2*Il_IJ[2, 1] - f2*Il_I[2]  # S_3322

    Sl[0, 1, 0, 1] = f4*(a**2+b**2)*Il_IJ[0, 1] + f5 * \
        (Il_I[0] + Il_I[1])  # S_1212
    Sl[1, 0, 0, 1] = Sl[0, 1, 0, 1]  # use symmetry Sⱼᵢₖₗ = Sᵢⱼₖₗ
    Sl[0, 1, 1, 0] = Sl[0, 1, 0, 1]  # use symmetry Sᵢⱼₗₖ = Sᵢⱼₖₗ
    Sl[1, 0, 1, 0] = Sl[0, 1, 0, 1]  # use both symmetries Sⱼᵢₗₖ = Sᵢⱼₖₗ
    Sl[1, 2, 1, 2] = f4*(b**2+c**2)*Il_IJ[1, 2] + f5 * \
        (Il_I[1] + Il_I[2])  # S_2323
    Sl[2, 1, 1, 2] = Sl[1, 2, 1, 2]  # use symmetry Sⱼᵢₖₗ = Sᵢⱼₖₗ
    Sl[1, 2, 2, 1] = Sl[1, 2, 1, 2]  # use symmetry Sᵢⱼₗₖ = Sᵢⱼₖₗ
    Sl[2, 1, 2, 1] = Sl[1, 2, 1, 2]  # use both symmetries Sⱼᵢₗₖ = Sᵢⱼₖₗ
    Sl[2, 0, 2, 0] = f4*(c**2+a**2)*Il_IJ[2, 0] + f5 * \
        (Il_I[2] + Il_I[0])  # S_3131
    Sl[0, 2, 2, 0] = Sl[2, 0, 2, 0]  # use symmetry Sⱼᵢₖₗ = Sᵢⱼₖₗ
    Sl[2, 0, 0, 2] = Sl[2, 0, 2, 0]  # use symmetry Sᵢⱼₗₖ = Sᵢⱼₖₗ
    Sl[0, 2, 0, 2] = Sl[2, 0, 2, 0]  # use both symmetries Sⱼᵢₗₖ = Sᵢⱼₖₗ

    return Sl


#######   HARMONIC- Φ AND BYHARMONIC POTENTIAL Ψ AND THEIR DERIVATIVES   ######
def Phi(_x, _a, lam):
    """
    Φ(xₙ,λ) = 1/2 {I(λ) - xₙxₙ I_N(λ)}
    From Meng et al. (2012) eq. (23) first line
    """
    a, b, c = _a
    # Iᵢ(λ)
    Il_I = np.array([Il_a(a, b, c, lam),
                     Il_b(a, b, c, lam),
                     Il_c(a, b, c, lam)])

    return 1/2 * (Il(a, b, c, lam) - (_x**2 * Il_I).sum())


def Psi_d_i(_x, _a, lam, i):
    """
    Ψ,ᵢ(xₙ,λ) = 1/2xᵢ {I(λ) - xₙxₙI_N(λ) - a_I²[I_I(λ)-xₙxₙI_IN(λ)]}
              = 1/2xᵢ {2*Φ(xₙ,λ) - a_I²[I_I(λ)-xₙxₙI_IN(λ)]}
    From Meng et al. (2012) eq. (24) second line
    """
    a, b, c = _a
    # Iᵢ(λ)
    Il_I = np.array([Il_a(a, b, c, lam),
                     Il_b(a, b, c, lam),
                     Il_c(a, b, c, lam)])
    # Iᵢⱼ(λ)
    Il_IJ = np.array(
        [[Il_aa(a, b, c, lam), Il_ab(a, b, c, lam), Il_ac(a, b, c, lam)],
         [Il_ba(a, b, c, lam), Il_bb(a, b, c, lam), Il_bc(a, b, c, lam)],
         [Il_ca(a, b, c, lam), Il_cb(a, b, c, lam), Il_cc(a, b, c, lam)]])

    return - 1/2*_x[i] * (2*Phi(_x, _a, lam)
                          - _a[i]**2*(Il_I[i] - (_x**2 * Il_IJ[i, :]).sum()))


def Phi_d_i(_x, _a, lam, i):
    """
    Φ,ᵢ(xᵢ,λ) = - xᵢ·I_I(λ)
    From Meng et al. (2012) eq. (25) first line
    """
    # Iᵢ(λ)
    a, b, c = _a
    Il_I = np.array([Il_a(a, b, c, lam), Il_b(
        a, b, c, lam), Il_c(a, b, c, lam)])

    return -_x[i] * Il_I[i]


def Psi_d_ijk(_x, _a, lam, i, j, k):
    """
    Ψ,ᵢⱼₖ(xᵢ,λ) = - δᵢⱼxₖ{I_K(λ)-a_I²I_IK(λ)}
                 - xᵢxⱼ{I_J,ₖ(λ)-a_I²I_IJ,ₖ(λ)}
                 - {δᵢₖxⱼ+δⱼₖxᵢ}·{I_J(λ)-a_I²I_IJ(λ)}
    From Meng et al. (2012) eq. (25) second line,
    dragged the derivatives into parenthesis.
    """
    a, b, c = _a
    delta = np.eye(3)  # δᵢⱼ
    # Iᵢ(λ)
    Il_I = np.array([Il_a(a, b, c, lam), Il_b(
        a, b, c, lam), Il_c(a, b, c, lam)])
    # Iᵢⱼ(λ)
    Il_IJ = np.array(
        [[Il_aa(a, b, c, lam), Il_ab(a, b, c, lam), Il_ac(a, b, c, lam)],
         [Il_ba(a, b, c, lam), Il_bb(a, b, c, lam), Il_bc(a, b, c, lam)],
         [Il_ca(a, b, c, lam), Il_cb(a, b, c, lam), Il_cc(a, b, c, lam)]])
    return \
        -delta[i, j]*_x[k]*(Il_I[k] - _a[i]**2*Il_IJ[i, k]) \
        - _x[i]*_x[j]*(Il_i_d_j(_x, _a, lam, j, k)
                       - _a[i]**2*Il_ij_d_k(_x, _a, lam, i, j, k)) \
        - (delta[i, k]*_x[j] + delta[j, k]*_x[i]) * \
        (Il_I[j] - _a[i]**2*Il_IJ[i, j])


###################   Dᵢⱼₖₗ FOR THE EXTERIOR ELASTIC FIELD   ##################
def compute_lambda(x, y, z, a, b, c, NoWarning=False):
    """
    Computes λ the largest positiv root of the equation
    x₁²/(a²+λ) + x₂²/(b²+λ) + x₃²/(c²+λ) = 1
    Therefore it is written as a polynomial in λ and solved by numpy.roots()
    λ³ -L*λ² + M*λ - N = 0; compare Eshelby 59' eq 3·5
    Caution: there are wrong signs in the equation for "M" in Eshelby 59'.
    """
    r_square = x**2 + y**2 + z**2
    R_square = a**2 + b**2 + c**2
    L = r_square - R_square
    # Caution the last four signs in te equation for M should be opposite!!!
    # M corrected
    M = a**2*x**2 + b**2*y**2 + c**2*z**2 \
        + a**2*b**2 + b**2*c**2 + c**2*a**2 - r_square*R_square
    N = a**2*b**2*c**2 * (x**2/a**2 + y**2/b**2 + z**2/c**2 - 1)

    roots = np.roots([1, -L, M, -N])
    if (x**2/a**2 + y**2/b**2 + z**2/c**2 <= 1):
        # point inside the inclusion/inhomogeneity ==> λ=0
        if not NoWarning:
            warnings.warn("Your point x={}, y={}, z={} lies inside the "
                          "inclusion!".format(x, y, z), category=UserWarning)
        lam = 0
    elif (x**2/a**2 + y**2/b**2 + z**2/c**2 > 1):
        # point outside of the inclusion/inhomogeneity
        # ==> λ=λₘₐₓ (largest positive root)
        lam = np.amax(roots[np.where(roots.imag == 0)])

    return lam.real


def D(x, y, z, a, b, c, nu, test_case=False):
    """
    Compute the "Eshelby tensor D for exterior points. See chap. 11, T. Mura
    82'. We use eq. (11.41) and pull the derivatives into all brackets
    8π(1-ν) Dᵢⱼₖₗ(x) = 8π(1-ν)Sᵢⱼₖₗ(λ) + 2ν δₖₗxᵢ I_I,ⱼ(λ)
                     + (1-ν) · {δᵢₗxₖ I_K,ⱼ(λ) + δⱼₗxₖ I_K,ᵢ(λ)
                                + δᵢₖxₗ I_L,ⱼ(λ) + δⱼₖxₗ I_L,ᵢ(λ)}
                     - δᵢⱼxₖ · [I_K,ₗ(λ) - a_I²I_KI,ₗ(λ)]
                     - (δᵢₖxⱼ+δⱼₖxᵢ) · [I_J,ₗ(λ) - a_I²I_IJ,ₗ(λ)]
                     - (δᵢₗxⱼ+δⱼₗxᵢ) · [I_J,ₖ(λ) - a_I²I_IJ,ₖ(λ)]
                     - xᵢxⱼ · [I_J,ₗₖ(λ) - a_I²I_IJ,ₗₖ(λ)]
    """
    # short hands for positions "x,y,z" and half axes "a,b,c" of the ellipsoide
    _x = np.array([x, y, z])
    _a = np.array([a, b, c])
    delta = np.eye(3)  # δᵢⱼ
    lam = compute_lambda(x, y, z, a, b, c)

    if lam == 0:
        warnings.warn("It seems like your point x=" + str(x) + " y=" + str(y)
                      + " z=" + str(z) + " lies inside the inclusion!\nThere"
                      " you should compute Sᵢⱼₖₗ which is constant in the "
                      "inclusion.", category=UserWarning)

    # store the computed integrals and their derivatives as tensors
    # Iᵢ,ⱼ(λ)
    Il_I_d_J = np.zeros((3,)*2)
    for i in range(3):
        for j in range(3):
            Il_I_d_J[i, j] = Il_i_d_j(_x, _a, lam, i, j)

    # Iᵢⱼ,ₖ(λ)
    Il_IJ_d_K = np.zeros((3,)*3)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                Il_IJ_d_K[i, j, k] = Il_ij_d_k(_x, _a, lam, i, j, k)

    # Iᵢ,ⱼₖ(λ)
    Il_I_d_JK = np.zeros((3,)*3)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                Il_I_d_JK[i, j, k] = Il_i_d_jk(_x, _a, lam, i, j, k)

    # Iᵢⱼ,ₖₗ(λ)
    Il_IJ_d_KL = np.zeros((3,)*4)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    Il_IJ_d_KL[i, j, k, l] = Il_ij_d_kl(
                        _x, _a, lam, i, j, k, l)

    # initialize Dᵢⱼₖₗ(_x,_a,ν) with 8π(1-ν)Sᵢⱼₖₗ(λ)
    D = 8*np.pi*(1-nu) * Sl(a, b, c, nu, lam)

    # 2ν δₖₗxᵢ I_I,ⱼ(λ)
    D1 = 2 * nu * delta[np.newaxis, np.newaxis, :, :] * \
        _x[:, np.newaxis, np.newaxis, np.newaxis] *\
        Il_I_d_J[:, :, np.newaxis, np.newaxis]

    # (1-ν)·{δᵢₗxₖ I_K,ⱼ(λ) + δⱼₗxₖ I_K,ᵢ(λ) + δᵢₖxₗ I_L,ⱼ(λ) + δⱼₖxₗ I_L,ᵢ(λ)}
    D2 = (1-nu) * (
        delta[:, np.newaxis, np.newaxis, :] *
        _x[np.newaxis, np.newaxis, :, np.newaxis] *
        Il_I_d_J.T[np.newaxis, :, :, np.newaxis]
        +
        delta[np.newaxis, :, np.newaxis, :] *
        _x[np.newaxis, np.newaxis, :, np.newaxis] *
        Il_I_d_J.T[:, np.newaxis, :, np.newaxis]
        +
        delta[:, np.newaxis, :, np.newaxis] *
        _x[np.newaxis, np.newaxis, np.newaxis, :] *
        Il_I_d_J.T[np.newaxis, :, np.newaxis, :]
        +
        delta[np.newaxis, :, :, np.newaxis] *
        _x[np.newaxis, np.newaxis, np.newaxis, :] *
        Il_I_d_J.T[:, np.newaxis, np.newaxis, :])

    # -δᵢⱼxₖ[I_K,ₗ(λ) - a_I²I_KI,ₗ(λ)]
    D3 = - delta[:, :, np.newaxis, np.newaxis] * \
        _x[np.newaxis, np.newaxis, :, np.newaxis] * (
        Il_I_d_J[np.newaxis, np.newaxis, :, :] -
        _a[:, np.newaxis, np.newaxis, np.newaxis]**2 *
        Il_IJ_d_K[:, np.newaxis, :, :].transpose((2, 1, 0, 3)))

    # -(δᵢₖxⱼ+δⱼₖxᵢ)[I_J,ₗ(λ) - a_I²I_IJ,ₗ(λ)]
    D4 = - (delta[:, np.newaxis, :, np.newaxis] *
            _x[np.newaxis, :, np.newaxis, np.newaxis] +
            delta[np.newaxis, :, :, np.newaxis] *
            _x[:, np.newaxis, np.newaxis, np.newaxis]) * (
                Il_I_d_J[np.newaxis, :, np.newaxis, :] -
                _a[:, np.newaxis, np.newaxis, np.newaxis]**2 *
                Il_IJ_d_K[:, :, np.newaxis, :])

    # -(δᵢₗxⱼ+δⱼₗxᵢ)[I_J,ₖ(λ) - a_I²I_IJ,ₖ(λ)]
    D5 = - (delta[:, np.newaxis, np.newaxis, :] *
            _x[np.newaxis, :, np.newaxis, np.newaxis] +
            delta[np.newaxis, :, np.newaxis, :] *
            _x[:, np.newaxis, np.newaxis, np.newaxis]) * (
                Il_I_d_J[np.newaxis, :, :, np.newaxis] -
                _a[:, np.newaxis, np.newaxis, np.newaxis]**2 *
                Il_IJ_d_K[:, :, :, np.newaxis])

    # - xᵢxⱼ[I_J,ₗₖ(λ) - a_I²I_IJ,ₗₖ(λ)]
    D6 = -(_x[:, np.newaxis, np.newaxis, np.newaxis] *
           _x[np.newaxis, :, np.newaxis, np.newaxis]) * (
               Il_I_d_JK[np.newaxis, :, :, :].transpose((0, 1, 3, 2)) -
               _a[:, np.newaxis, np.newaxis, np.newaxis]**2 *
               Il_IJ_d_KL[:, :, :, :].transpose((0, 1, 3, 2)))

    if test_case:
        D0 = D
        return D0, D1, D2, D3, D4, D5, D6
    else:
        D += D1 + D2 + D3 + D4 + D5 + D6
        return D / (8 * np.pi * (1 - nu))


#   COMPUTE STRESS σᵢⱼ(x), STRAIN εᵢⱼ(x) AND DISPLACEMENT uᵢⱼ(x) ON D-Ω & Ω   #
# STIFFNESS TENSOR Cᵢⱼₖₗ
def stiffness_tensor(Young, Poisson):
    """
    Cᵢⱼₖₗ = λ·δᵢⱼδₖₗ + μ·(δᵢₖδⱼₗ + δᵢₗδⱼₖ)
    """
    E = Young
    nu = Poisson
    lame_1 = E*nu / ((1+nu)*(1-2*nu))  # Lamé's first parameter (λ)
    mu = E / (2*(1+nu))  # shear modulus (μ)
    delta = np.eye(3)  # δᵢⱼ
    # δᵢⱼδₖₗ
    d_ij_kl = delta[:, :, np.newaxis, np.newaxis] * \
        delta[np.newaxis, np.newaxis, :, :]
    # (δᵢₖδⱼₗ + δᵢₗδⱼₖ)
    I_sym = delta[:, np.newaxis, :, np.newaxis] \
        * delta[np.newaxis, :, np.newaxis, :] \
        + delta[:, np.newaxis, np.newaxis, :] \
        * delta[np.newaxis, :, :, np.newaxis]

    return lame_1 * d_ij_kl + mu * I_sym


def get_equivalent_eigenstrain(E, nu, E_I, nu_I, a, b, c, eps_0, eps_p_0=None,
                               S_ijkl=None):
    """
    Computes the equivalent eigenstrain for an ellipsoidal inhomogeneity in an
    infinte matrix by the Eshelby solution, see Eshelby 57' 59' 61'.
    The equivalent eigenstrain "ε*ₘₙ" is found by solving the equation:
    (ΔCᵢⱼₖₗSₖₗₘₙ - Cᵢⱼₘₙ)ε*ₘₙ = -ΔCᵢⱼₖₗε⁰ₖₗ - C*ᵢⱼₖₗεᵖₖₗ
    where:
    ΔCᵢⱼₖₗ=Cᵢⱼₖₗ-C*ᵢⱼₖₗ, is the difference in the stiffness tensors
    Cᵢⱼₘₙ   -- is the stifness tensor of the matrix (region D-Ω)
    C*ᵢⱼₖₗ  -- is the stifness tensor of the inhomogeneity (region Ω)
    Sₖₗₘₙ  -- is the Eshelby tensor of the inclusion (region Ω)
    ε⁰ₖₗ   -- is the remote strain applied on the boundary ∂D
    εᵖₖₗ   -- is an optional initial eigenstrain of the inhomogeneity(region Ω)

    Keyword Arguments:
    E          -- Young modulus of the matrix, region D-Ω
    nu         -- Poisson ratio of the matrix, region D-Ω
    E_I        -- Young modulus of the inhomogeneity, region Ω
    nu_I       -- Poisson ratio of the inhomogeneity, region Ω
    a, b and c -- Three floats giving the principal half axes of the
                  ellipsoidal inclusion, where the half axes are alligned with
                  the coordinate axes and the order "a > b > c" is fulfilled.
    eps_0      -- applied strain at the boundaries of D, (ε⁰ₖₗ)
    eps_p_0    -- initial eigenstrain of the inhomogeneity, (εᵖₖₗ)
                  optional; default: np.zeros((3,3)).
    S_ijkl     -- Eshelby tensor as computed by S(a, b, c, nu_I). Can be given
                  to avoid double calculation.
                  optional; default: None, S_ijkl is internal computed

    Returns:
    eps_eq_eig -- equivalent eigenstrain, (ε*ₘₙ)
    """
    # check if eps_0 and eps_p_0 are small strains, i.e. symmetric
    if (eps_0 - eps_0.T != 0).any():
        raise ValueError("The applied strain at infinity ε⁰ₖₗ should be a "
                         "symmetric small strain tensor. You gave ε⁰ₖₗ = {}"
                         .format(eps_0))
    if eps_p_0 is None:
        eps_p_0 = np.zeros((3, 3))
    elif (eps_p_0 - eps_p_0.T != 0).any():
        raise ValueError("The eigenstrain of the inclusion εᵖₖₗ should be a "
                         "symmetric small strain tensor. You gave ε⁰ₖₗ = {}"
                         .format(eps_p_0))

    C_ijkl_out = stiffness_tensor(E, nu)
    C_ijkl_in = stiffness_tensor(E_I, nu_I)
    Del_C_ijkl = C_ijkl_out - C_ijkl_in
    if S_ijkl is None:
        # default, compute S_ijkl
        S_klmn = S(a, b, c, nu_I)
    else:
        S_klmn = S_ijkl

    # rewrite (ΔCᵢⱼₖₗSₖₗₘₙ - Cᵢⱼₘₙ)ε*ₘₙ = - ΔCᵢⱼₖₗε⁰ₖₗ - C*ᵢⱼₖₗεᵖₖₗ
    # into a matrix vector equation:
    # Aᵢⱼ ε*ⱼ = bⱼ
    A_ijmn = np.einsum("ijkl, klmn -> ijmn", Del_C_ijkl, S_klmn) - C_ijkl_out
    b_ij = - np.einsum("ijkl, kl -> ij", Del_C_ijkl, eps_0) \
        - np.einsum("ijkl, kl -> ij", C_ijkl_in, eps_p_0)
    # check if b_ij and Aijmn are symmetric
    if np.linalg.norm(b_ij-b_ij.T) > 1e-12:
        raise ValueError("The right hand side of:\n (ΔCᵢⱼₖₗSₖₗₘₙ - Cᵢⱼₘₙ)ε*ₘₙ "
                         "= - ΔCᵢⱼₖₗε⁰ₖₗ - C*ᵢⱼₖₗεᵖₖₗ\nis not symmetric in "
                         "'ij' which should never happen!")
    if ((np.linalg.norm(A_ijmn - A_ijmn.transpose((1, 0, 2, 3))) > 1e-12) or
            (np.linalg.norm(A_ijmn - A_ijmn.transpose((0, 1, 3, 2))) > 1e-12)):
        raise ValueError("The left hand side of:\n (ΔCᵢⱼₖₗSₖₗₘₙ - Cᵢⱼₘₙ)ε*ₘₙ ="
                         " - ΔCᵢⱼₖₗε⁰ₖₗ - C*ᵢⱼₖₗεᵖₖₗ\nis not symmetric in 'ij'"
                         " which should never happen!")

    # fill symmetric parts of b_ij and A_ijmn in b_j and A_jn
    dim = 3  # spatial dimension of the problem
    b_i = np.zeros((6,))
    for i in range(dim):
        for j in range(i, dim):
            index_i = int(i*dim + j - i*(i+1)/2)
            b_i[index_i] = b_ij[i, j]

    A_im = np.zeros((6, 6))
    for i in range(dim):
        for j in range(i, dim):
            for m in range(dim):
                for n in range(m, dim):
                    index_i = int(i*dim + j - i*(i+1)/2)
                    index_m = int(m*dim + n - m*(m+1)/2)
                    A_im[index_i, index_m] = A_ijmn[i, j, m, n]

    eps_eq_j = solve(A_im, b_i)

    # map eps_eq_j back into 3x3 matrix
    eps_eq_ij = np.zeros((3, 3))
    for i in range(dim):
        for j in range(i, dim):
            index_i = int(i*dim + j - i*(i+1)/2)
            eps_eq_ij[i, j] = eps_eq_j[index_i]
    eps_eq_ij = 1/2*(eps_eq_ij + eps_eq_ij.T)

    return eps_eq_ij


def get_stress_and_strain_in(E, nu, E_I, nu_I, a, b, c, eps_0, eps_p_0=None,
                             return_eps_eq_eig=False):
    """
    Compute the stress and strain response in the inner region Ω.
    From Meng et al. (2012) eq. (27) first and second line
    or T. Mura 82' eq. (22.8)-(22.13):
    εᵢⱼ = ε⁰ᵢⱼ + Sᵢⱼₘₙε*ₘₙ             in Ω
    σᵢⱼ = σ⁰ᵢⱼ + Cᵢⱼₖₗ(Sₖₗₘₙε*ₘₙ-ε*ₘₙ)   in Ω

    Keyword Arguments:
    E          -- Young modulus of the matrix, region D-Ω
    nu         -- Poisson ratio of the matrix, region D-Ω
    E_I        -- Young modulus of the inhomogeneity, region Ω
    nu_I       -- Poisson ratio of the inhomogeneity, region Ω
    a, b and c -- Three floats giving the principal half axes of the
                  ellipsoidal inclusion, where the half axes are alligned with
                  the coordinate axes and the order "a > b > c" is fulfilled.
    eps_0      -- applied strain at the boundaries of D, (ε⁰ₖₗ)
    eps_p_0    -- initial eigenstrain of the inhomogeneity, (εᵖₖₗ)
                  optional; default: np.zeros((3,3)).

    Returns:
    sigma_in   -- stress in the inclusion Ω, "σᵢⱼ"
    eps_in     -- strain in the inclusion Ω, "εᵢⱼ"
    """
    S_ijkl = S(a, b, c, nu)
    eps_eq_eig = get_equivalent_eigenstrain(E, nu, E_I, nu_I,
                                            a, b, c, eps_0, eps_p_0, S_ijkl)
    C_ijkl_out = stiffness_tensor(E, nu)
    sigma_0 = np.einsum("ijkl, kl -> ij", C_ijkl_out, eps_0)
    _eps = np.einsum("klmn, mn -> kl", S_ijkl, eps_eq_eig)

    eps_in = eps_0 + _eps
    sigma_in = sigma_0 + \
        np.einsum("ijkl,kl->ij", C_ijkl_out, _eps - eps_eq_eig)

    if return_eps_eq_eig:
        return sigma_in, eps_in, eps_eq_eig
    return sigma_in, eps_in


def get_stress_and_strain_out(x, y, z,
                              E, nu, E_I, nu_I, a, b, c, eps_0, eps_p_0=None,
                              return_eps_eq_eig=False):
    """
    Compute the stress and strain response in the outer region D-Ω.
    From Meng et al. (2012) eq. (27) third and fourth line
    or T. Mura 82' eq. (22.8)-(22.13):
    εᵢⱼ(x) = ε⁰ᵢⱼ + Dᵢⱼₘₙ(x)ε*ₘₙ      for x ∈ D-Ω
    σᵢⱼ(x) = σ⁰ᵢⱼ + CᵢⱼₖₗDₖₗₘₙ(x)ε*ₘₙ   for x ∈ D-Ω

    Keyword Arguments:
    x, y and z -- x, y and z coordinate of the position where the stress and
                  strain is computed. The point (x,y,z) should be located
                  outside the inclusion Ω.
    E          -- Young modulus of the matrix, region D-Ω
    nu         -- Poisson ratio of the matrix, region D-Ω
    E_I        -- Young modulus of the inhomogeneity, region Ω
    nu_I       -- Poisson ratio of the inhomogeneity, region Ω
    a, b and c -- Three floats giving the principal half axes of the
                  ellipsoidal inclusion, where the half axes are alligned with
                  the coordinate axes and the order "a > b > c" is fulfilled.
    eps_0      -- applied strain at the boundaries of D, (ε⁰ₖₗ)
    eps_p_0    -- initial eigenstrain of the inhomogeneity, (εᵖₖₗ)
                  optional; default: np.zeros((3,3)).

    Returns:
    sigma_out  -- stress at the position x ∈ D-Ω, σᵢⱼ(x)
    eps_out    -- strain at the position x ∈ D-Ω, εᵢⱼ(x)
    """
    D_ijkl = D(x, y, z, a, b, c, nu)
    eps_eq_eig = get_equivalent_eigenstrain(E, nu, E_I, nu_I,
                                            a, b, c, eps_0, eps_p_0)
    C_ijkl_out = stiffness_tensor(E, nu)
    sigma_0 = np.einsum("ijkl, kl -> ij", C_ijkl_out, eps_0)
    _eps = np.einsum("klmn, mn -> kl", D_ijkl, eps_eq_eig)

    eps_out = eps_0 + np.einsum("ijkl, kl -> ij", D_ijkl, eps_eq_eig)
    sigma_out = sigma_0 + np.einsum("ijkl, kl -> ij", C_ijkl_out, _eps)

    if return_eps_eq_eig:
        return sigma_out, eps_out, eps_eq_eig
    return sigma_out, eps_out


def get_displacement_field(x, y, z,
                           E, nu, E_I, nu_I, a, b, c, eps_0, eps_p_0=None):
    """
    Compute the displacement field at the position (x,y,z)
    From Meng et al. (2012) eq. (21) or T. Mura 82' eq. (11.30):
    uᵢ(x,y,z) = 1/(8π(1-ν))·{Ψ,ⱼₗᵢ ε*ⱼₗ - 2νε*ₘₘΦ,ᵢ - 4(1-ν)ε*ᵢₗΦ,ₗ}

    Keyword Arguments:
    x, y and z -- x, y and z coordinate of the position where the stress and
                  strain is computed. The point (x,y,z) should be located
                  outside the inclusion Ω.
    E          -- Young modulus of the matrix, region D-Ω
    nu         -- Poisson ratio of the matrix, region D-Ω
    E_I        -- Young modulus of the inhomogeneity, region Ω
    nu_I       -- Poisson ratio of the inhomogeneity, region Ω
    a, b and c -- Three floats giving the principal half axes of the
                  ellipsoidal inclusion, where the half axes are alligned with
                  the coordinate axes and the order "a > b > c" is fulfilled.
    eps_0      -- applied strain at the boundaries of D, (ε⁰ₖₗ)
    eps_p_0    -- initial eigenstrain of the inhomogeneity, (εᵖₖₗ)
                  optional; default: np.zeros((3,3)).

    Returns:
    uᵢ(x,y,z)   -- displacement field at position (x,y,z), "uᵢ(x,y,z)"
    """
    if a < b or b < c:
        raise ValueError("The ellipsoidal pricipal half axes should satisfy:" +
                         "\na > b > c\nBut you gave: a="+str(a)+", b="+str(b) +
                         " c="+str(c))

    eps_eq_eig = get_equivalent_eigenstrain(E, nu, E_I, nu_I,
                                            a, b, c, eps_0, eps_p_0)  # ε*
    lam = compute_lambda(x, y, z, a, b, c, NoWarning=True)
    _x = np.array([x, y, z])
    _a = np.array([a, b, c])

    # compute tensors
    Psi_d_IJK = np.zeros((3, 3, 3))
    for i, j, k in product(range(3), repeat=3):
        Psi_d_IJK[i, j, k] = Psi_d_ijk(_x, _a, lam, i, j, k)

    Phi_d_I = np.zeros((3,))
    for i in range(3):
        Phi_d_I[i] = Phi_d_i(_x, _a, lam, i)

    # uᵢ(x,y,z) = 1/(8π(1-ν))·{Ψ,ⱼₗᵢ ε*ⱼₗ - 2νε*ₘₘΦ,ᵢ - 4(1-ν)ε*ᵢₗΦ,ₗ}
    return 1/(8*np.pi*(1-nu)) * (
        np.einsum("jli, jl -> i", Psi_d_IJK.transpose((1, 2, 0)), eps_eq_eig)
        - 2 * nu * eps_eq_eig.diagonal().sum() * Phi_d_I
        - 4 * (1-nu) * eps_eq_eig.dot(Phi_d_I))

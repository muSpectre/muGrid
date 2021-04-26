#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file    python_test_1d_scipy_damage_classes.py

@author Till Junge <till.junge@altermail.ch>

@date   18 Jan 2018

@brief  prepares sys.path to load muSpectre

Copyright © 2018 Till Junge

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

import sys
import os
import numpy as np

class mat_lin_undam:
    """
    This class mimics 1D material linear elastic material
    It can calculate the stress (spring force) and takes a constant k as its
    stiffness (tangent).
    It can also calculate the elastic energy of the spring according to its
    deformation (strain)
    """

    def __init__(self, k):
        self.k = k

    def stress_tangent_cal(self, e, v=False):
        if v:
            print("NON DAMAGE MATERIAL")
        return self.k * e, self.k

    def energy_cal(self, e):
        return 0.5 * self.stress_tangent_cal(e)[0] * e


class mat_lin_dam:
    """
    This class mimics 1D material linear elastic material with a linear damage
    part after yield
    According to the deformation applied it calculates the damage caused by the
    deformation which will affect the Force (stress) it carries as well as its
    stiffness and elastic energy.
    It can calculate the stress (spring force) and takes a constant k as its
    undamaged stiffness (tangent).
    It can also calculate the elastic energy of the spring according to its
    deformation (strain) and the damage extent it experiences.
    """

    def __init__(self, k_pos, k_neg, e0):
        self.e0 = e0
        self.e_init = e0
        self.k_pos = k_pos
        self.k_pos_init = k_pos
        self.k_neg = k_neg

    def reset(self):
        self.e0 = self.e_init
        self.k_pos = self.k_pos_init

    def update_internal(self, e0_new, k_new):
        self.e0 = e0_new
        self.k_pos = k_new

    def stress_tangent_cal(self, e, v=False):
        if abs(e) <= self.e0:
            if v:
                print("NO DAMAGE")
            return self.k_pos * e, self.k_pos
        else:
            ret_stress = 0.0
            if e > 0.0:
                ret_stress = (self.k_pos * self.e0 +
                              self.k_neg * (e - self.e0))
            cor_stress = ret_stress if (ret_stress * e) > 0 else 0.0
            k_update = cor_stress / e
            self.update_internal(e, k_update)
            if v:
                print(
                    "DAMAGE MATERIAL Reduction factor is : " +
                    "{}".format((self.k_pos_init - k_update) / self.k_pos_init))
            return cor_stress, self.k_neg

    def energy_cal(self, e):
        return 0.5 * self.stress_tangent_cal(e)[0] * e


class func_calculations:
    """
    This class can be used to pile up the springs of the mat_lin or mat_lin_dam
    in a serial fashion and given a macroscopic mean strain value will calculate
    regarding forces (jacs) and stifnesses of the system.
    it also can calculate the energy of the system as a whole.
    This class is used as an equivalent of the cell concept in muSpectre.
    It will be used to make the jac and hess functions fed to scipy trust_region
    solver and the equilibrium of the system defined as an object of the class
    will be calculated by scipy.minimize() with 'trust-ncg' method.
    """

    def __init__(self, k_pos, k_neg, e_init,
                 mats_dam_neg_slope_coeff=[1.0, 0.0, 0.0],
                 e_mac=0.0):
        self.k_pos = k_pos
        self.k_neg = k_neg
        self.e_init = e_init
        self.e_mac = e_mac
        self.mats_dam_neg_slope_coeff = mats_dam_neg_slope_coeff

    def mat_dam_make(self, coeff=1):
        return mat_lin_dam(self.k_pos,
                           coeff * self.k_neg,
                           self.e_init)

    def mat_undam_make(self):
        return mat_lin_undam(self.k_pos)

    def mats_make(self):
        ret_tup = list()
        for i, mat_dam_neg_slope_coeff in enumerate(
                self.mats_dam_neg_slope_coeff):
            if mat_dam_neg_slope_coeff == 0:
                ret_tup.append(self.mat_undam_make())
            else:
                ret_tup.append(self.mat_dam_make(mat_dam_neg_slope_coeff))
        return tuple(ret_tup)

    def cal_e_last(self, es):
        return (3.0 * self.e_mac-np.sum(es))

    def tot_energy(self, es):
        es_loc = np.array([es[0], es[1], self.cal_e_last(es)])
        mats = self.mats_make()
        ret_energy = 0.0
        for i, e_loc in enumerate(es_loc):
            ret_energy += mats[i].energy_cal(e_loc)
        return ret_energy

    def tot_jac(self, es):
        es_loc = np.array([es[0], es[1],
                           self.cal_e_last(es)])
        mats = self.mats_make()
        jacs = np.zeros_like(es_loc)
        for i, e_loc in enumerate(es_loc):
            jacs[i] = mats[i].stress_tangent_cal(e_loc)[0]
        return (jacs[:2] - self.k_pos * es_loc[2])

    def tot_stress(self, es):
        es_loc = np.array([es[0], es[1], self.cal_e_last(es)])
        mats = self.mats_make()
        jacs = np.zeros_like(es_loc)
        for i, e_loc in enumerate(es_loc):
            jacs[i] = mats[i].stress_tangent_cal(e_loc)[0]
        return (jacs[:2])

    def tot_hess(self, es):
        es_loc = np.array([es[0], es[1], self.cal_e_last(es)])
        mats = self.mats_make()
        hesses = np.zeros_like(es_loc)
        for i, e_loc in enumerate(es_loc):
            hesses[i] = mats[i].stress_tangent_cal(e_loc)[1]
        return (np.diag(hesses[:2]) +
                self.k_pos)

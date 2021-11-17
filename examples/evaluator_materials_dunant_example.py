#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@file evaluator_materials_dunant_example.py

@author Ali Falsafi <afalsafi@epfl.ch>

@date   15 Jul 2020

@brief this file is an example showing how to use the damage
material viscoelatic in deviatoric and elastic in bulk loading

Copyright © 2020 Ali Falsafi

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
import numpy as np
import argparse
try:
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib_found = True
except ImportError:
    matplotlib_found = False


from python_example_imports import muSpectre_vtk_export as vt_ex
from python_example_imports import muSpectre_gradient_integration as gi
from python_example_imports import muSpectre as µ


def compute_evaluator(nb_steps, young, kappa, alpha, dE,
                      mat, eval, mat_c, eval_c, material_name):
    """
    description: this function takes a material, its evaluator, and its
    correspondent constants and using the output of the evaluator plots the
    strain-stress curve of the material under tensile and compressive loads

    @param nb_steps:      number of steps for applying the
                          tensile and the compressive load
    @param young:         The young modulus of the intact material
    @param kappa:         The strain (or maybe other damage criterion) threshold
    @param alpha:         The ratio of the slope of the damage part of the
                          strain-stress curve wrt to the elastic part (positive)
    @param dE:            The strain application step
    @param mat:           The material to be evaluated in the tensile regime
    @praam eval:          The material evaluator to be used in the tensile  regime
    @param mat_c:         The material to be evaluated in the compressive regime
    @praam eval_c:        The material evaluator to be used in compressive  regime
    @param material_name: The name o the material for plot title

    @return None
    """
    # making load steps:
    nb_steps_tot = np.sum(nb_steps)
    nb_steps_acc = np.cumsum(nb_steps)
    Es = np.empty((nb_steps_tot, 2, 2))
    for i in range(nb_steps[1]):
        Es[i, ...] = i * dE
    for i in range(nb_steps[2]):
        Es[nb_steps[1] + i, ...] = (nb_steps[1] - i - 0.5) * dE
    for i in range(nb_steps[3]):
        Es[nb_steps[1] + nb_steps[2] +
            i, ...] = (nb_steps[1] - nb_steps[2] + i) * dE
    for i in range(nb_steps[4]):
        Es[nb_steps[1] + nb_steps[2] + nb_steps[3] +
            i, ...] = (nb_steps[1] - nb_steps[2] + nb_steps[3] - i - 0.5) * dE
    for i in range(nb_steps[5]):
        Es[nb_steps[1] + nb_steps[2] + nb_steps[3] + nb_steps[4] +
            i, ...] = \
            (nb_steps[1] - nb_steps[2] + nb_steps[3] - nb_steps[4] + i)*dE

    # making variables
    Ss = np.empty((nb_steps_tot, 2, 2))
    Ss_c = np.empty((nb_steps_tot, 2, 2))
    S_norms = np.empty((nb_steps_tot))
    S_norms_c = np.empty((nb_steps_tot))
    E_norms = np.empty((nb_steps_tot))
    S_evals = np.empty((nb_steps_tot, 2))
    Es_c = -2.0 * Es

    # applying strain steps and save responses
    for i in range(nb_steps_tot):
        Ss[i, ...] = eval.evaluate_stress(Es[i, ...],
                                          μ.Formulation.small_strain)
        Ss_c[i, ...] = eval_c.evaluate_stress(Es_c[i, ...],
                                              μ.Formulation.small_strain)
        S_norms[i] = np.linalg.norm(Ss[i, ...])
        S_norms_c[i] = -np.linalg.norm(Ss_c[i, ...])
        S_evals[i, ...], a = np.linalg.eig(Ss[i, ...])
        E_norms[i] = np.linalg.norm(Es[i, ...])
        mat.save_history_variables()
        mat_c.save_history_variables()
    # plotting the output
    # prevent visual output during ctest
    if len(sys.argv[:]) == 2:
        if sys.argv[1] != 1:
            pass
    else:
        # and len(sys.argv[:]) != 2 and (~sys.argv[1] == 1):
        if matplotlib_found:
            font = {'family': 'DejaVu Sans',
                    'weight': 'bold',
                    'size': 36}
            matplotlib.rc('font', **font)
            jet = plt.get_cmap('Greys')
            colors = iter(jet(np.linspace(0.8, 1, 10)))
            figure_size = [20, 12]
            fig = plt.figure(figsize=figure_size)
            ax = fig.add_subplot(111)
            ax.set_xlabel("$\epsilon$", fontsize=44)
            s_id = np.array([0, 0])
            ax.set_ylabel("$\sigma$", fontsize=44)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.yaxis.set_label_coords(0.78, 0.94)
            ax.xaxis.set_label_coords(0.94, 0.78)
            # Move left y-axis and bottom x-axis to center,
            # passing through (0,0)
            ax.spines['left'].set_position('zero')
            ax.spines['bottom'].set_position('zero')

            # Eliminate upper and right axes
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            # ax.set_xlim([-2.5e-5, 5e-5])
            for i in range(1, len(nb_steps)):
                if i == 1:
                    ax.plot(Es[nb_steps_acc[i-1]:nb_steps_acc[i],
                               s_id[0], s_id[1]],
                            Ss[nb_steps_acc[i-1]:nb_steps_acc[i],
                               s_id[0], s_id[1]],
                            '-.', linewidth=4.0, color=next(colors),
                            label="tension")
                else:
                    ax.plot(Es[nb_steps_acc[i-1]:nb_steps_acc[i],
                               s_id[0], s_id[1]],
                            Ss[nb_steps_acc[i-1]:nb_steps_acc[i],
                               s_id[0], s_id[1]],
                            '-.', linewidth=4.0, color=next(colors))
            for i in range(1, len(nb_steps)):
                if i == 1:
                    ax.plot(Es_c[nb_steps_acc[i-1]:nb_steps_acc[i],
                                 s_id[0], s_id[1]],
                            Ss_c[nb_steps_acc[i-1]:nb_steps_acc[i],
                                 s_id[0], s_id[1]],
                            '.', linewidth=4.0, color=next(colors),
                            label="compression")
                else:
                    ax.plot(Es_c[nb_steps_acc[i-1]:nb_steps_acc[i],
                                 s_id[0], s_id[1]],
                            Ss_c[nb_steps_acc[i-1]:nb_steps_acc[i],
                                 s_id[0], s_id[1]],
                            '.', linewidth=4.0, color=next(colors))
            ax.legend()
            plt.title(material_name)
            plt.show()
            fig.savefig(
                "damage_material_kappa_{}_alpha_{}_new_tc.png"
                .format(kappa, alpha))


def compute():
    dE = np.array([[2.000, 0.000],
                   [0.000, 2.000]]) * 1.e-6

    young = 2.2876e10
    alphas = np.array([2])
    for alpha in alphas:
        for kappa in [1.e-4]:
            # do calculation and plot for material_dunant
            material_name = "Material Dunant"
            nb_steps = np.array([0, 60, 15, 23, 15, 30])
            mat, eval = µ.material.MaterialDunant_2d.make_evaluator(
                young, .33, kappa, alpha)
            mat_c, eval_c = µ.material.MaterialDunant_2d.make_evaluator(
                young, .33, kappa, alpha)
            mat.add_pixel(0)
            mat_c.add_pixel(0)
            compute_evaluator(nb_steps, young, kappa, alpha, 0.55 * dE,
                              mat, eval, mat_c, eval_c, material_name)
            # do calculation and plot for material_dunant_t
            material_name = "Material Dunant T"
            nb_steps = np.array([0, 60, 15, 23, 15, 30])
            mat, eval = µ.material.MaterialDunantT_2d.make_evaluator(
                young, .33, kappa, alpha)
            mat_c, eval_c = µ.material.MaterialDunantT_2d.make_evaluator(
                young, .33, kappa, alpha)
            mat.add_pixel(0)
            mat_c.add_pixel(0)
            compute_evaluator(nb_steps, young, kappa, alpha, dE,
                              mat, eval, mat_c, eval_c, material_name)

            # do calculation and plot for material_dunant_tc
            material_name = "Material Dunant TC assym."
            nb_steps2 = np.array([0, 45, 15, 23, 15, 30])
            mat2, eval2 = µ.material.MaterialDunantTC_2d.make_evaluator(
                young, .33, kappa, alpha, 0.2, 1.0)
            mat2_c, eval2_c = µ.material.MaterialDunantTC_2d.make_evaluator(
                young, .33, kappa, alpha, 0.2, 1.0)
            mat2.add_pixel(0)
            mat2_c.add_pixel(0)
            compute_evaluator(nb_steps2, young, kappa, alpha, dE,
                              mat2, eval2, mat2_c, eval2_c, material_name)


def main():
    compute()


if __name__ == "__main__":
    main()

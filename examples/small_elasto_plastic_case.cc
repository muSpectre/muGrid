/**
 * @file   small_elasto_plastic_case.cc
 *
 * @author Indre Joedicke <indre.joedicke@imtek.uni-freiburg.de>
 *
 * @date   12 Jan 2018
 *
 * @brief  small case for debugging elasto-plasticity
 *
 * Copyright © 2018 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#include "common/muSpectre_common.hh"
#include "cell/cell_factory.hh"
#include "materials/material_hyper_elasto_plastic1.hh"
#include "materials/stress_transformations_Kirchhoff.hh"
#include "solver/solvers.hh"
#include "solver/krylov_solver_cg.hh"
#include <libmugrid/iterators.hh>

#include <iostream>

using namespace muSpectre;

int main() {
  constexpr Index_t Dim{twoD};

  DynCcoord_t nb_grid_pts{5, 5};

  DynRcoord_t lengths{7, 5};
  Formulation form{Formulation::finite_strain};

  auto rve{make_cell(nb_grid_pts, lengths, form)};

  // material constants
  float K{0.833};
  float mu{0.386};
  float H{0.04};  // Low values of H worsen the condition number of the
                  // stiffness-matrix
  float tauy0{0.006};
  float Young, Poisson;
  Young = 9*K*mu / (3*K+mu);
  Poisson = (3*K-2*mu) / (2*(3*K+mu));

  auto & hard{MaterialHyperElastoPlastic1<Dim>::make(rve, "hard", Young,
                                                     Poisson, 2*tauy0, 2*H)};
  auto & soft{MaterialHyperElastoPlastic1<Dim>::make(rve, "soft", Young,
                                                     Poisson, tauy0, H)};

  for (auto && i : rve.get_pixel_indices()) {
    if (i < 3) {
      hard.add_pixel(i);
    } else {
      soft.add_pixel(i);
      }
  }

  rve.initialise();

  Real tol{1e-5};
  Real equi_tol{1e-5};
  Eigen::MatrixXd Del0(Dim, Dim);
  Del0 << 0, 0, 0, 3e-2;

  Uint maxiter{401};
  Verbosity verbose{Verbosity::Detailed};

  KrylovSolverCG cg{rve, tol, maxiter, verbose};
  auto res = de_geus(rve, Del0, cg, tol, equi_tol, verbose);
  std::cout << res.grad.transpose() << std::endl;
  return 0;
}

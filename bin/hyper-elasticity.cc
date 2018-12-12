/**
 * @file   hyper-elasticity.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   16 Jan 2018
 *
 * @brief  Recreation of GooseFFT's hyper-elasticity.py calculation
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
 * General Public License for more details.
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
 */

#include "cell/cell_factory.hh"
#include "materials/material_linear_elastic1.hh"
#include "solver/solvers.hh"
#include "solver/solver_cg.hh"

#include <iostream>
#include <iomanip>

using namespace muSpectre;

int main() {
  constexpr Dim_t dim{3};
  constexpr Ccoord_t<dim> N{CcoordOps::get_cube<dim>(11)};
  constexpr Rcoord_t<dim> lens{CcoordOps::get_cube<dim>(1.)};
  constexpr Dim_t incl_size{3};

  auto cell{make_cell(N, lens, Formulation::small_strain)};

  // constexpr Real K_hard{8.33}, K_soft{.833};
  // constexpr Real mu_hard{3.86}, mu_soft{.386};
  // auto E = [](Real K, Real G) {return 9*K*G / (3*K+G);}; //G is mu
  // auto nu= [](Real K, Real G) {return (3*K-2*G) / (2*(3*K+G));};

  // auto & hard{MaterialLinearElastic1<dim, dim>::make(cell, "hard",
  //                                                   E(K_hard, mu_hard),
  //                                                   nu(K_hard, mu_hard))};
  // auto & soft{MaterialLinearElastic1<dim, dim>::make(cell, "soft",
  //                                                   E(K_soft, mu_soft),
  //                                                   nu(K_soft, mu_soft))};
  Real ex{1e-5};
  using Mat_t = MaterialLinearElastic1<dim, dim>;
  auto &hard{Mat_t::make(cell, "hard", 210. * ex, .33)};
  auto &soft{Mat_t::make(cell, "soft", 70. * ex, .33)};

  for (auto pixel : cell) {
    if ((pixel[0] >= N[0] - incl_size) && (pixel[1] < incl_size) &&
        (pixel[2] >= N[2] - incl_size)) {
      hard.add_pixel(pixel);
    } else {
      soft.add_pixel(pixel);
    }
  }
  std::cout << hard.size() << " pixels in the inclusion" << std::endl;
  cell.initialise();
  constexpr Real cg_tol{1e-8}, newton_tol{1e-5};
  constexpr Dim_t maxiter{200};
  constexpr Dim_t verbose{1};

  Eigen::MatrixXd dF_bar{Eigen::MatrixXd::Zero(dim, dim)};
  dF_bar(0, 1) = 1.;
  SolverCG cg{cell, cg_tol, maxiter, verbose};
  auto optimize_res = de_geus(cell, dF_bar, cg, newton_tol, verbose);

  std::cout << "nb_cg: " << optimize_res.nb_fev << std::endl;
  std::cout << optimize_res.grad.transpose().block(0, 0, 10, 9) << std::endl;
  return 0;
}

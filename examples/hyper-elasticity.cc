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
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#include "cell/cell_factory.hh"
#include "materials/material_linear_elastic1.hh"
#include "solver/solvers.hh"
#include "solver/krylov_solver_cg.hh"

#include <iostream>
#include <iomanip>

using namespace muSpectre;
using namespace muGrid;

int main() {
  constexpr Index_t Dim{3};
  DynCcoord_t N{CcoordOps::get_cube<Dim>(Index_t{11})};
  DynRcoord_t lens{CcoordOps::get_cube<Dim>(1.)};
  constexpr Index_t incl_size{3};

  auto cell{make_cell(N, lens, Formulation::finite_strain)};

  Real ex{1e-5};
  using Mat_t = MaterialLinearElastic1<Dim>;
  auto & hard{Mat_t::make(cell, "hard", 210. * ex, .33)};
  auto & soft{Mat_t::make(cell, "soft", 70. * ex, .33)};

  for (auto && index_pixel : cell.get_pixels().enumerate()) {
    auto && index{std::get<0>(index_pixel)};
    auto && pixel{std::get<1>(index_pixel)};
    if ((pixel[0] >= N[0] - incl_size) && (pixel[1] < incl_size) &&
        (pixel[2] >= N[2] - incl_size)) {
      hard.add_pixel(index);
    } else {
      soft.add_pixel(index);
    }
  }
  std::cout << hard.size() << " pixels in the inclusion" << std::endl;
  cell.initialise();
  constexpr Real cg_tol{1e-8}, newton_tol{1e-5}, equil_tol{1e-8};
  constexpr Index_t maxiter{200};
  constexpr Verbosity verbose{Verbosity::Some};

  Eigen::MatrixXd dF_bar{Eigen::MatrixXd::Zero(Dim, Dim)};
  dF_bar(0, 1) = 1.;
  KrylovSolverCG cg{cell, cg_tol, maxiter, verbose};
  auto optimize_res = de_geus(cell, dF_bar, cg, newton_tol, equil_tol, verbose);

  std::cout << "nb_cg: " << optimize_res.nb_fev << std::endl;
  std::cout << optimize_res.grad.transpose().block(0, 0, 1, 1) << std::endl;
  return 0;}

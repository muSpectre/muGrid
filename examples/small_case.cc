/**
 * @file   small_case.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   12 Jan 2018
 *
 * @brief  small case for debugging
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
#include "materials/material_linear_elastic1.hh"
#include "solver/solvers.hh"
#include "solver/krylov_solver_cg.hh"
#include <libmugrid/iterators.hh>

#include <iostream>

using namespace muSpectre;

int main() {
  constexpr Index_t Dim{twoD};

  DynCcoord_t nb_grid_pts{11, 11};

  DynRcoord_t lengths{
      muGrid::CcoordOps::get_cube<Dim>(11.)};  // {5.2e-9, 8.3e-9, 8.3e-9};
  Formulation form{Formulation::finite_strain};

  auto rve{make_cell(nb_grid_pts, lengths, form)};

  auto & hard{MaterialLinearElastic1<Dim>::make(rve, "hard", 210., .33)};
  auto & soft{MaterialLinearElastic1<Dim>::make(rve, "soft", 70., .33)};

  for (auto && i : rve.get_pixel_indices()) {
    if (i < 3) {
      hard.add_pixel(i);
    } else {
      soft.add_pixel(i);
    }
  }

  rve.initialise();

  Real tol{1e-6};
  Real equi_tol{0};
  Eigen::MatrixXd Del0(Dim, Dim);
  Del0 << 0, .1, 0, 0;

  Uint maxiter{31};
  Verbosity verbose{Verbosity::Detailed};

  KrylovSolverCG cg{rve, tol, maxiter, verbose};
  auto res{newton_cg(rve, Del0, cg, tol, equi_tol, verbose)};
  std::cout << res.grad.transpose() << std::endl;
  return 0;
}

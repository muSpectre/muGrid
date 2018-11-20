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

#include "common/common.hh"
#include "common/iterators.hh"
#include "cell/cell_factory.hh"
#include "materials/material_linear_elastic1.hh"
#include "solver/solvers.hh"
#include "solver/solver_cg.hh"

#include <iostream>


using namespace muSpectre;


int main()
{
  constexpr Dim_t dim{twoD};

  Ccoord_t<dim> resolution{11, 11};

  Rcoord_t<dim> lengths{CcoordOps::get_cube<dim>(11.)};//{5.2e-9, 8.3e-9, 8.3e-9};
  Formulation form{Formulation::finite_strain};

  auto rve{make_cell(resolution,
                     lengths,
                     form)};

  auto & hard{MaterialLinearElastic1<dim, dim>::make
      (rve, "hard", 210., .33)};
  auto & soft{MaterialLinearElastic1<dim, dim>::make
      (rve, "soft",  70., .33)};

  for (auto && tup: akantu::enumerate(rve)) {
    auto & i = std::get<0>(tup);
    auto & pixel = std::get<1>(tup);
    if (i < 3) {
      hard.add_pixel(pixel);
    } else {
      soft.add_pixel(pixel);
    }
  }

  rve.initialise();

  Real tol{1e-6};
  Eigen::MatrixXd Del0{};
  Del0 <<  0, .1,
           0,  0;

  Uint maxiter{31};
  Dim_t verbose{3};

  SolverCG cg{rve, tol, maxiter, bool(verbose)};
  auto res = de_geus(rve, Del0, cg, tol, verbose);
  std::cout << res.grad.transpose() << std::endl;
  return 0;
}

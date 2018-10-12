/**
* @file   demonstrator1.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   03 Jan 2018
 *
 * @brief  larger problem to show off
 *
 * Copyright © 2018 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#include <iostream>
#include <memory>
#include <chrono>

#include "common/common.hh"
#include "common/ccoord_operations.hh"
#include "cell/cell_factory.hh"
#include "materials/material_linear_elastic1.hh"
#include "solver/solvers.hh"
#include "solver/solver_cg.hh"

using namespace muSpectre;

int main()
{
  banner("demonstrator1", 2018, "Till Junge <till.junge@epfl.ch>");
  constexpr Dim_t dim{2};

  constexpr Formulation form{Formulation::finite_strain};

  const Rcoord_t<dim> lengths{5.2, 8.3};
  const Ccoord_t<dim> resolutions{5, 7};

  auto cell{make_cell<dim, dim>(resolutions, lengths, form)};

  constexpr Real E{1.0030648180242636};
  constexpr Real nu{0.29930675909878679};

  using Material_t = MaterialLinearElastic1<dim, dim>;
  auto & soft{Material_t::make(cell, "soft",    E, nu)};
  auto & hard{Material_t::make(cell, "hard", 10*E, nu)};

  int counter{0};
  for (const auto && pixel:cell) {
    if (counter < 3) {
      hard.add_pixel(pixel);
      counter++;
    } else {
      soft.add_pixel(pixel);
    }
  }
  std::cout << counter << " Pixel out of " << cell.size()
            << " are in the hard material" << std::endl;

  cell.initialise();

  constexpr Real newton_tol{1e-4};
  constexpr Real cg_tol{1e-7};
  const size_t maxiter = 100;

  Eigen::MatrixXd DeltaF{Eigen::MatrixXd::Zero(dim, dim)};
  DeltaF(0, 1) = .1;
  Dim_t verbose {1};

  auto start = std::chrono::high_resolution_clock::now();
  SolverCG cg{cell, cg_tol, maxiter, bool(verbose)};
  auto res = de_geus(cell, DeltaF, cg, newton_tol, verbose);
  std::chrono::duration<Real> dur = std::chrono::high_resolution_clock::now() - start;
  std::cout << "Resolution time = " << dur.count() << "s" << std::endl;

  std::cout << res.grad.transpose() << std::endl;
  return 0;
}

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
#include "external/cxxopts.hpp"

#include "common/common.hh"
#include "common/ccoord_operations.hh"
#include "cell/cell_factory.hh"
#include "materials/material_linear_elastic1.hh"
#include "solver/solvers.hh"
#include "solver/solver_cg.hh"


using opt_ptr = std::unique_ptr<cxxopts::Options>;


opt_ptr parse_args(int argc, char **argv) {
  opt_ptr options =
    std::make_unique<cxxopts::Options>(argv[0],  "Tests MPI fft scalability");

  try {
    options->add_options()
      ("0,N0", "number of rows", cxxopts::value<int>(), "N0")
      ("h,help", "print help")
      ("positional",
       "Positional arguments: these are the arguments that are entered "
       "without an option", cxxopts::value<std::vector<std::string>>());
    options->parse_positional(std::vector<std::string>{"N0", "positional"});
    options->parse(argc, argv);
    if (options->count("help")) {
      std::cout << options->help({"", "Group"}) << std::endl;
      exit(0);
    }
    if (options->count("N0") != 1 ) {
      throw cxxopts::OptionException("Parameter N0 missing");
    } else if ((*options)["N0"].as<int>()%2 != 1) {
      throw cxxopts::OptionException("N0 must be odd");
    } else if (options->count("positional") > 0) {
      throw cxxopts::OptionException("There are too many positional arguments");
    }
  } catch (const cxxopts::OptionException & e) {
    std::cout << "Error parsing options: " << e.what() << std::endl;
    exit(1);
  }
  return options;
}


using namespace muSpectre;

int main(int argc, char *argv[])
{
  banner("demonstrator1", 2018, "Till Junge <till.junge@epfl.ch>");
  auto options{parse_args(argc, argv)};
  auto & opt{*options};
  const Dim_t size{opt["N0"].as<int>()};
  constexpr Real fsize{1.};
  constexpr Dim_t dim{3};
  const Dim_t nb_dofs{ipow(size, dim)*ipow(dim, 2)};
  std::cout << "Number of dofs: " <<  nb_dofs << std::endl;

  constexpr Formulation form{Formulation::finite_strain};

  const Rcoord_t<dim> lengths{CcoordOps::get_cube<dim>(fsize)};
  const Ccoord_t<dim> resolutions{CcoordOps::get_cube<dim>(size)};

  auto cell{make_cell<dim, dim>(resolutions, lengths, form)};

  constexpr Real E{1.0030648180242636};
  constexpr Real nu{0.29930675909878679};

  using Material_t = MaterialLinearElastic1<dim, dim>;
  auto & Material_soft{Material_t::make(cell, "soft",    E, nu)};
  auto & Material_hard{Material_t::make(cell, "hard", 10*E, nu)};

  int counter{0};
  for (const auto && pixel:cell) {

    int sum = 0;
    for (Dim_t i = 0; i < dim; ++i) {
      sum  += pixel[i]*2 / resolutions[i];
    }

    if (sum == 0) {
      Material_hard.add_pixel(pixel);
      counter ++;
    } else {
      Material_soft.add_pixel(pixel);
    }
  }
  std::cout << counter << " Pixel out of " << cell.size()
            << " are in the hard material" << std::endl;

  cell.initialise(FFT_PlanFlags::measure);

  constexpr Real newton_tol{1e-4};
  constexpr Real cg_tol{1e-7};
  const Uint maxiter = nb_dofs;

  Eigen::MatrixXd DeltaF{Eigen::MatrixXd::Zero(dim, dim)};
  DeltaF(0, 1) = .1;
  Dim_t verbose {1};

  auto start = std::chrono::high_resolution_clock::now();
  LoadSteps_t loads{DeltaF};
  SolverCG cg{cell, cg_tol, maxiter, bool(verbose)};
  newton_cg(cell, loads, cg, newton_tol, verbose);
  std::chrono::duration<Real> dur = std::chrono::high_resolution_clock::now() - start;
  std::cout << "Resolution time = " << dur.count() << "s" << std::endl;

  return 0;
}

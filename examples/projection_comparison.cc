/**
 * @file   projection_comparison.cc
 *
 * @author Martin Ladecký <m.ladecky@gmail.com>
 *
 * @date   01 Feb 2020
 *
 * @brief  proof of concept for preconditioner
 *
 * Copyright © 2020 Martin Ladecký
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
#include "common/muSpectre_common.hh"
#include "cell/cell_factory.hh"
#include "materials/material_linear_elastic1.hh"
#include "solver/solvers.hh"
#include "solver/krylov_solver_cg.hh"
#include "projection/projection_approx_Green_operator.hh"

#include <libmugrid/ccoord_operations.hh>

#include "external/cxxopts.hpp"

#include <iostream>
#include <memory>
#include <chrono>

using opt_ptr = std::unique_ptr<cxxopts::Options>;

opt_ptr parse_args2(int argc, char ** argv) {
  opt_ptr options =
      std::make_unique<cxxopts::Options>(argv[0], "Tests MPI fft scalability");
  std::cout << argv[0] << std::endl;
  try {
    options->add_options()("0,N0", "number of rows", cxxopts::value<int>(),
                           "N0")(
        "p,proj",
        "projection kind (0 priconditioned by approximed Green function, 1 NO "
        "preconditioner)",
        cxxopts::value<int>(), "proj")("h,help", "print help")(
        "positional",
        "Positional arguments: these are the arguments that are entered "
        "without an option",
        cxxopts::value<std::vector<std::string>>());

    options->parse_positional(
        std::vector<std::string>{"N0", "proj", "positional"});

    options->parse(argc, argv);

    if (options->count("help")) {
      std::cout << options->help({"", "Group"}) << std::endl;
      exit(0);
    }

    if (options->count("N0") != 1) {
      throw cxxopts::OptionException("Parameter N0 missing");
    } else if ((*options)["N0"].as<int>() % 2 != 1) {
      throw cxxopts::OptionException("N0 must be odd");
    }

    if (options->count("proj") != 1) {
      throw cxxopts::OptionException("Parameter projection_kind missing");
    } else if ((*options)["proj"].as<int>() != 0 &&
               (*options)["proj"].as<int>() != 1) {
      throw cxxopts::OptionException("Proj must be 0 or 1");
    }
  } catch (const cxxopts::OptionException & e) {
    std::cout << "Error parsing options: " << e.what() << std::endl;
    exit(1);
  }
  return options;
}

using namespace muSpectre;
using namespace muGrid;
using namespace muFFT;

int small_sym(int argc, char * argv[]) {
  auto options{parse_args2(argc, argv)};
  auto & opt{*options};

  const Index_t size{opt["N0"].as<int>()};
  const int operator_kind{opt["proj"].as<int>()};

  constexpr Real fsize{1.};
  constexpr Index_t Dim{3};
  const Index_t nb_dofs{ipow(size, Dim) * ipow(Dim, 2)};

  std::cout << "Projection operator : " << opt["proj"].as<int>() << std::endl;
  std::cout << "Number of dofs: " << nb_dofs << std::endl;

  constexpr Real E{1.0030648180242636};
  constexpr Real nu{0.29930675909878679};
  const DynRcoord_t lengths{CcoordOps::get_cube<Dim>(fsize)};
  const DynCcoord_t nb_grid_pts{CcoordOps::get_cube<Dim>(size)};

  auto && C_ref(muGrid::Matrices::Iiden<Dim>());

  auto && fft_pointer(
      std::make_unique<muFFT::FFTWEngine>(DynCcoord_t(nb_grid_pts)));

  std::unique_ptr<ProjectionBase> projection_ptr{};

  if (operator_kind == 0) {
    projection_ptr = std::make_unique<ProjectionApproxGreenOperator<Dim>>(
        std::move(fft_pointer), DynRcoord_t(lengths), C_ref);
  } else {
    projection_ptr = std::make_unique<ProjectionSmallStrain<Dim>>(
        std::move(fft_pointer), DynRcoord_t(lengths));
  }

  ProjectionApproxGreenOperator<Dim> * projector_ptr{
      dynamic_cast<ProjectionApproxGreenOperator<Dim> *>(projection_ptr.get())};

  Cell cell(std::move(projection_ptr));

  using Material_t = MaterialLinearElastic1<Dim>;
  auto & material_soft{Material_t::make(cell, "soft", E, nu)};
  auto & material_hard{Material_t::make(cell, "hard", 10 * E, nu)};

  int counter{0};

  for (auto && id_pixel :
       akantu::zip(cell.get_pixel_indices(), cell.get_pixels())) {
    const auto & pixel_index{std::get<0>(id_pixel)};
    const auto & pixel{std::get<1>(id_pixel)};
    int sum{0};
    for (Index_t i = 0; i < Dim; ++i) {
      sum += pixel[i] * 2 / nb_grid_pts[i];
    }

    if (sum == 0) {
      material_hard.add_pixel(pixel_index);
      counter++;
    } else {
      material_soft.add_pixel(pixel_index);
    }
  }

  cell.initialise();

  std::cout << counter << " Pixel out of " << cell.get_nb_pixels()
            << " are in the hard material" << std::endl;

  cell.get_strain().set_zero();
  cell.evaluate_stress_tangent();
  auto Cref{cell.get_tangent().get_sub_pt_map(Dim * Dim).mean()};
  if (operator_kind == 0) {
    projector_ptr->reinitialise(Cref);
  }

  constexpr Real newton_tol{1e-4};
  constexpr Real cg_tol{1e-7};
  constexpr Real stress_tol{0};
  const Uint maxiter{static_cast<Uint>(nb_dofs)};

  Eigen::MatrixXd DeltaF{Eigen::MatrixXd::Zero(Dim, Dim)};
  DeltaF(0, 1) = .1;
  DeltaF(1, 0) = .1;
  Verbosity verbose{Verbosity::Some};

  auto start = std::chrono::high_resolution_clock::now();
  LoadSteps_t loads{DeltaF};

  KrylovSolverCG cg{cell, cg_tol, maxiter, verbose};
  newton_cg(cell, loads, cg, newton_tol, stress_tol, verbose);

  std::chrono::duration<Real> dur =
      std::chrono::high_resolution_clock::now() - start;
  std::cout << "Resolution time = " << dur.count() << "s" << std::endl;

  return 0;
}

int main(int argc, char * argv[]) {
  banner("projection_comparison", 2020,
         "Martin Ladecký <martin.ladecky@gmail.com>");

  small_sym(argc, argv);

  return 0;
}

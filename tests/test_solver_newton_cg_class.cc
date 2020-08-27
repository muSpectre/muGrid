/**
 * @file   test_solver_newton_cg_class.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   24 Jul 2020
 *
 * @brief  Tests for the new class-based Newton-Raphson + Conjugate Gradient
 * solver
 *
 * Copyright © 2020 Till Junge
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

#include "tests.hh"
#include "test_cell_data.hh"
#include "libmugrid/test_goodies.hh"

#include "solver/solver_newton_cg.hh"
#include "solver/solvers.hh"
#include "materials/material_linear_elastic1.hh"

#include <cell/cell_factory.hh>
#include <solver/krylov_solver_cg.hh>

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(newton_cg_solverclass);

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(constructor_test, Fix, CellDataFixtures,
                                   Fix) {
    this->cell_data->set_nb_quad_pts(OneQuadPt);
    MaterialLinearElastic1<Fix::SpatialDim>::make(this->cell_data, "material",
                                                  4, .3);
    auto krylov_solver{std::make_shared<KrylovSolverCG>(1e-8, 100)};
    auto solver{std::make_shared<SolverNewtonCG>(this->cell_data, krylov_solver,
                                                 muGrid::Verbosity::Full, 1e-10,
                                                 1e-10, 100)};
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(solver_test, Fix, CellDataFixtures, Fix) {
    const Real young_soft{4};
    const Real young_hard{8};
    const Real poisson{.3};
    using Mat_t = MaterialLinearElastic1<Fix::SpatialDim>;
    this->cell_data->set_nb_quad_pts(OneQuadPt);
    auto & soft{Mat_t::make(this->cell_data, "soft", young_soft, poisson)};
    auto & hard{Mat_t::make(this->cell_data, "hard", young_hard, poisson)};

    auto legacy_cell{make_cell(this->cell_data->get_nb_domain_grid_pts(),
                               this->cell_data->get_domain_lengths(),
                               Formulation::small_strain)};
    auto & legacy_soft{Mat_t::make(legacy_cell, "soft", young_soft, poisson)};
    auto & legacy_hard{Mat_t::make(legacy_cell, "hard", young_hard, poisson)};

    {
      bool first{true};
      for (auto && index_pixel : this->cell_data->get_pixels().enumerate()) {
        auto && index{std::get<0>(index_pixel)};
        if (first) {
          first = false;
          hard.add_pixel(index);
          legacy_hard.add_pixel(index);
        } else {
          soft.add_pixel(index);
          legacy_soft.add_pixel(index);
        }
      }
    }

    BOOST_TEST_CHECKPOINT("after material assignment");

    constexpr Real cg_tol{1e-8}, newton_tol{1e-5}, equil_tol{1e-10};
    const Uint maxiter{static_cast<Uint>(this->cell_data->get_spatial_dim()) *
                       10};
    constexpr Verbosity verbose{Verbosity::Full};

    auto krylov_solver{
        std::make_shared<KrylovSolverCG>(cg_tol, maxiter, verbose)};
    auto solver{std::make_shared<SolverNewtonCG>(this->cell_data, krylov_solver,
                                                 verbose, newton_tol, equil_tol,
                                                 maxiter)};
    auto && symmetric{[](Eigen::MatrixXd mat) -> Eigen::MatrixXd {
      return 0.5 * (mat + mat.transpose());
    }};
    const Eigen::MatrixXd strain{
        symmetric(Eigen::MatrixXd::Random(Fix::SpatialDim, Fix::SpatialDim)) /
        2};

    solver->set_formulation(Formulation::small_strain);
    solver->initialise_cell();

    BOOST_TEST_CHECKPOINT("before load increment");
    std::cout << std::endl
              << "strain:" << std::endl
              << strain << std::endl
              << std::endl;
    std::cout << std::endl
              << "symmetric(strain):" << std::endl
              << symmetric(strain) << std::endl
              << std::endl;
    auto && new_result{solver->solve_load_increment(strain)};
    BOOST_TEST_CHECKPOINT("after load increment");

    KrylovSolverCG legacy_krylov_solver{legacy_cell, cg_tol, maxiter, verbose};
    auto && legacy_result{newton_cg(legacy_cell, strain, legacy_krylov_solver,
                                    newton_tol, equil_tol, verbose)};

    Eigen::Map<Eigen::ArrayXXd> legacy_stress_map{legacy_result.stress.data(),
                                                  new_result.stress.rows(),
                                                  new_result.stress.cols()};
    auto && error{
        muGrid::testGoodies::rel_error(new_result.stress, legacy_stress_map)};
    BOOST_CHECK_LE(error, tol);
    if (not(error < tol)) {
      std::cout << "legacy stress result" << std::endl
                << legacy_stress_map.transpose() << std::endl;
      std::cout << "new stress result" << std::endl
                << new_result.stress.transpose() << std::endl;
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muSpectre

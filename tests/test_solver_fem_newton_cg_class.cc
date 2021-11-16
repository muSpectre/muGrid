/**
 * @file   test_solver_fem_newton_cg_class.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 * @author Martin Ladecký <m.ladecky@gmail.com>
 *
 * @date   01 Sep 2020
 *
 * @brief  tests for the un-preconditioned newton-cg FEM solver class
 *
 * Copyright © 2020 Till Junge, Martin Ladecký
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

#include "projection/fem_library.hh"
#include "projection/discretisation.hh"
#include "solver/solver_fem_newton_cg.hh"
#include "solver/solver_fem_newton_pcg.hh"
#include "solver/solvers.hh"
#include "materials/material_linear_elastic1.hh"

#include <cell/cell_factory.hh>
#include <solver/krylov_solver_cg.hh>
#include <solver/krylov_solver_pcg.hh>

#include <cstdlib>

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(fem_newton_cg_solverclass);

  BOOST_FIXTURE_TEST_CASE(constructor_test, CellDataFixture<twoD>) {
    using Fix = CellDataFixture<twoD>;
    auto stencil{FEMLibrary::linear_triangle_straight(this->cell_data)};
    auto discretisation{std::make_shared<Discretisation>(stencil)};
    MaterialLinearElastic1<Fix::SpatialDim>::make(this->cell_data, "material",
                                                  4, .3);
    auto krylov_solver{std::make_shared<KrylovSolverCG>(1e-8, 100)};
    BOOST_TEST_CHECKPOINT("Before constructor");
    auto solver{std::make_shared<SolverFEMNewtonCG>(
        discretisation, krylov_solver, muGrid::Verbosity::Silent, 1e-10, 1e-10,
        100)};
  }

  BOOST_FIXTURE_TEST_CASE(solver_test, CellDataFixture<twoD>) {
    constexpr Formulation Form{Formulation::small_strain};
    using Fix = CellDataFixture<twoD>;
    const Real young_soft{4};
    const Real young_hard{8};
    const Real poisson{.3};
    using Mat_t = MaterialLinearElastic1<Fix::SpatialDim>;
    auto stencil{FEMLibrary::bilinear_quadrangle(this->cell_data)};
    auto discretisation{std::make_shared<Discretisation>(stencil)};

    auto & soft{Mat_t::make(this->cell_data, "soft", young_soft, poisson)};
    auto & hard{Mat_t::make(this->cell_data, "hard", young_hard, poisson)};

    auto legacy_cell{make_cell(this->cell_data->get_nb_domain_grid_pts(),
                               this->cell_data->get_domain_lengths(), Form)};
    auto & legacy_soft{Mat_t::make(legacy_cell, "soft", young_soft, poisson)};
    auto & legacy_hard{Mat_t::make(legacy_cell, "hard", young_hard, poisson)};
    {
      Index_t nb_hard{this->cell_data->get_nb_domain_grid_pts()[0]};
      for (auto && index_pixel : this->cell_data->get_pixels().enumerate()) {
        auto && index{std::get<0>(index_pixel)};
        if (nb_hard) {
          --nb_hard;
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
                       20};
    constexpr Verbosity verbose{Verbosity::Silent};

    auto krylov_solver{
        std::make_shared<KrylovSolverCG>(cg_tol, maxiter, verbose)};
    auto solver{std::make_shared<SolverFEMNewtonCG>(
        discretisation, krylov_solver, verbose, newton_tol, equil_tol,
        maxiter)};

    auto strain_maker = [](auto && spatial_dim,
                           auto && formulation) -> Eigen::MatrixXd {
      // std::srand(42);
      Eigen::MatrixXd retval{0.01 *
                             Eigen::MatrixXd::Random(spatial_dim, spatial_dim)};
      switch (formulation) {
      case Formulation::finite_strain: {
        return retval;
        break;
      }
      case Formulation::small_strain: {
        return .5 * (retval + retval.transpose());
        break;
      }
      default:
        throw std::runtime_error("bad formulation");
        break;
      }
    };

    const Eigen::MatrixXd grad{strain_maker(Fix::SpatialDim, Form)};

    solver->set_formulation(Form);
    solver->initialise_cell();

    BOOST_TEST_CHECKPOINT("before load increment");

    KrylovSolverCG legacy_krylov_solver{legacy_cell, cg_tol, maxiter, verbose};
    auto && legacy_result{newton_cg(legacy_cell, grad, legacy_krylov_solver,
                                    newton_tol, equil_tol, verbose)};
    std::cout << "Done with legacy solver" << std::endl;
    std::cout << "legacy Newton-CG converged in "
              << legacy_krylov_solver.get_counter() << " CG steps and "
              << legacy_result.nb_it << " Newton Steps." << std::endl;

    auto && new_result{solver->solve_load_increment(grad)};
    std::cout << "Newton-CG converged in " << krylov_solver->get_counter()
              << " CG steps and " << new_result.nb_it << " Newton steps."
              << std::endl;
    BOOST_TEST_CHECKPOINT("after load increment");
    Index_t nb_vals{stencil->get_gradient_operator()->get_nb_pixel_quad_pts()};
    Eigen::Map<Eigen::ArrayXXd> legacy_stress_map{
        legacy_result.stress.data(), grad.size(),
        legacy_result.stress.size() / grad.size()};
    // all pixels have uniform strain, so the stress can be stored per pixel for
    // comparison with legacy stress
    Eigen::ArrayXXd new_stress_pixel{legacy_stress_map.rows(),
                                     legacy_stress_map.cols()};
    for (Index_t i{0}; i < legacy_stress_map.cols(); ++i) {
      Eigen::VectorXd col_stress{Eigen::VectorXd::Zero(grad.size())};
      for (Index_t j{1}; j < nb_vals; ++j) {
        Real error{muGrid::testGoodies::rel_error(
            new_result.stress.col(i * nb_vals),
            new_result.stress.col(i * nb_vals + j))};
        BOOST_CHECK_LE(error, tol);
      }
      new_stress_pixel.col(i) = new_result.stress.col(i * nb_vals);
    }
    auto && error{
        muGrid::testGoodies::rel_error(new_stress_pixel, legacy_stress_map)};

    BOOST_CHECK_LE(error, tol);
    if (not(error < tol)) {
      std::cout << "legacy stress result" << std::endl
                << legacy_stress_map.transpose() << std::endl;
      std::cout << "new stress result" << std::endl
                << new_result.stress.transpose() << std::endl;

      for (Index_t i{0}; i < new_result.stress.cols(); ++i) {
        std::cout << "E = " << new_result.grad.col(i).transpose() << std::endl;
        std::cout << "P = " << new_result.stress.col(i).transpose() << std::endl
                  << std::endl;
      }
    }
  }

  BOOST_FIXTURE_TEST_CASE(solver_test_3d, CellDataFixture<threeD>) {
    constexpr Formulation Form{Formulation::small_strain};
    using Fix = CellDataFixture<threeD>;
    const Real young_soft{4};
    const Real young_hard{8};
    const Real poisson{.3};
    using Mat_t = MaterialLinearElastic1<Fix::SpatialDim>;
    auto stencil{FEMLibrary::trilinear_hexahedron(this->cell_data)};
    BOOST_TEST_CHECKPOINT("after stencil creation");
    auto discretisation{std::make_shared<Discretisation>(stencil)};

    BOOST_TEST_CHECKPOINT("after discretisation creation");
    auto & soft{Mat_t::make(this->cell_data, "soft", young_soft, poisson)};
    auto & hard{Mat_t::make(this->cell_data, "hard", young_hard, poisson)};

    auto legacy_cell{make_cell(this->cell_data->get_nb_domain_grid_pts(),
                               this->cell_data->get_domain_lengths(), Form)};
    auto & legacy_soft{Mat_t::make(legacy_cell, "soft", young_soft, poisson)};
    auto & legacy_hard{Mat_t::make(legacy_cell, "hard", young_hard, poisson)};

    BOOST_TEST_CHECKPOINT("after legacy cell creation");
    {
      Index_t nb_hard{this->cell_data->get_nb_domain_grid_pts()[0] *
                      this->cell_data->get_nb_domain_grid_pts()[1]};
      for (auto && index_pixel : this->cell_data->get_pixels().enumerate()) {
        auto && index{std::get<0>(index_pixel)};
        if (nb_hard) {
          --nb_hard;
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
                       20};
    constexpr Verbosity verbose{Verbosity::Silent};

    auto krylov_solver{
        std::make_shared<KrylovSolverCG>(cg_tol, maxiter, verbose)};
    auto solver{std::make_shared<SolverFEMNewtonCG>(
        discretisation, krylov_solver, verbose, newton_tol, equil_tol,
        maxiter)};

    auto strain_maker = [](auto && spatial_dim,
                           auto && formulation) -> Eigen::MatrixXd {
      Eigen::MatrixXd retval{Eigen::MatrixXd::Random(spatial_dim, spatial_dim)};
      switch (formulation) {
      case Formulation::finite_strain: {
        return retval;
        break;
      }
      case Formulation::small_strain: {
        return .5 * (retval + retval.transpose());
        break;
      }
      default:
        throw std::runtime_error("bad formulation");
        break;
      }
    };

    const Eigen::MatrixXd grad{strain_maker(Fix::SpatialDim, Form)};

    solver->set_formulation(Form);
    solver->initialise_cell();

    BOOST_TEST_CHECKPOINT("before load increment");

    KrylovSolverCG legacy_krylov_solver{legacy_cell, cg_tol, maxiter, verbose};
    auto && legacy_result{newton_cg(legacy_cell, grad, legacy_krylov_solver,
                                    newton_tol, equil_tol, verbose)};
    std::cout << "Done with legacy solver" << std::endl;
    std::cout << "legacy Newton-CG converged in "
              << legacy_krylov_solver.get_counter() << " CG steps and "
              << legacy_result.nb_it << " Newton Steps." << std::endl;

    auto && new_result{solver->solve_load_increment(grad)};
    std::cout << "Newton-CG converged in " << krylov_solver->get_counter()
              << " CG steps and " << new_result.nb_it << " Newton steps."
              << std::endl;
    BOOST_TEST_CHECKPOINT("after load increment");
    Index_t nb_quad{stencil->get_gradient_operator()->get_nb_pixel_quad_pts()};
    Eigen::Map<Eigen::ArrayXXd> legacy_stress_map{
        legacy_result.stress.data(), grad.size(),
        legacy_result.stress.size() / grad.size()};
    // all pixels have uniform strain, so the stress can be stored per pixel for
    // comparison with legacy stress
    Eigen::ArrayXXd new_stress_pixel{legacy_stress_map.rows(),
                                     legacy_stress_map.cols()};
    for (Index_t i{0}; i < legacy_stress_map.cols(); ++i) {
      Eigen::VectorXd col_stress{Eigen::VectorXd::Zero(grad.size())};
      for (Index_t j{1}; j < nb_quad; ++j) {
        Real error{muGrid::testGoodies::rel_error(
            new_result.stress.col(i * nb_quad),
            new_result.stress.col(i * nb_quad + j))};
        BOOST_CHECK_LE(error, tol);
      }
      new_stress_pixel.col(i) = new_result.stress.col(i * nb_quad);
    }
    auto && error{
        muGrid::testGoodies::rel_error(new_stress_pixel, legacy_stress_map)};

    BOOST_CHECK_LE(error, tol);
    if (not(error < tol)) {
      std::cout << "legacy stress result" << std::endl
                << legacy_stress_map.transpose() << std::endl;
      std::cout << "new stress result" << std::endl
                << new_result.stress.transpose() << std::endl;

      for (Index_t i{0}; i < new_result.stress.cols(); ++i) {
        std::cout << "E = " << new_result.grad.col(i).transpose() << std::endl;
        std::cout << "P = " << new_result.stress.col(i).transpose() << std::endl
                  << std::endl;
      }
    }
  }

  BOOST_FIXTURE_TEST_CASE(solver_test_preconditioned, CellDataFixture<twoD>) {
    using Fix = CellDataFixture<twoD>;
    const Real young_soft{4};
    const Real young_hard{8};
    const Real poisson{.3};
    using Mat_t = MaterialLinearElastic1<Fix::SpatialDim>;
    auto stencil{FEMLibrary::bilinear_quadrangle(this->cell_data)};
    auto discretisation{std::make_shared<Discretisation>(stencil)};

    auto & soft{Mat_t::make(this->cell_data, "soft", young_soft, poisson)};
    auto & hard{Mat_t::make(this->cell_data, "hard", young_hard, poisson)};

    constexpr Formulation Form{Formulation::finite_strain};
    auto legacy_cell{make_cell(this->cell_data->get_nb_domain_grid_pts(),
                               this->cell_data->get_domain_lengths(), Form)};
    auto & legacy_soft{Mat_t::make(legacy_cell, "soft", young_soft, poisson)};
    auto & legacy_hard{Mat_t::make(legacy_cell, "hard", young_hard, poisson)};
    {
      Index_t nb_hard{this->cell_data->get_nb_domain_grid_pts()[0]};
      for (auto && index_pixel : this->cell_data->get_pixels().enumerate()) {
        auto && index{std::get<0>(index_pixel)};
        if (nb_hard) {
          --nb_hard;
          hard.add_pixel(index);
          legacy_hard.add_pixel(index);
        } else {
          soft.add_pixel(index);
          legacy_soft.add_pixel(index);
        }
      }
    }

    BOOST_TEST_CHECKPOINT("after material assignment");

    constexpr Real cg_tol{1e-8}, newton_tol{1e-8}, equil_tol{1e-10};
    const Uint maxiter{static_cast<Uint>(this->cell_data->get_spatial_dim()) *
                       50};
    constexpr Verbosity verbose{Verbosity::Silent};

    auto krylov_solver{
        std::make_shared<KrylovSolverPCG>(cg_tol, maxiter, verbose)};
    auto solver{std::make_shared<SolverFEMNewtonPCG>(
        discretisation, krylov_solver, verbose, newton_tol, equil_tol,
        maxiter)};
    std::srand(42);
    auto strain_maker = [](auto && spatial_dim,
                           auto && formulation) -> Eigen::MatrixXd {
      Eigen::MatrixXd retval{.1 *
                             Eigen::MatrixXd::Random(spatial_dim, spatial_dim)};
      switch (formulation) {
      case Formulation::finite_strain: {
        return retval;
        break;
      }
      case Formulation::small_strain: {
        return .5 * (retval + retval.transpose());
        break;
      }
      default:
        throw std::runtime_error("bad formulation");
        break;
      }
    };

    const Eigen::MatrixXd grad{strain_maker(Fix::SpatialDim, Form)};

    solver->set_formulation(Form);
    solver->initialise_cell();
    solver->get_set_eval_grad().get_map() =
        Eigen::MatrixXd::Identity(twoD, twoD);
    solver->clear_last_step_nonlinear();
    solver->evaluate_stress_tangent();
    auto ref_material{solver->get_tangent().get_map().mean()};
    solver->set_reference_material(ref_material);

    BOOST_TEST_CHECKPOINT("before load increment");
    std::cout << std::endl << "Gradient:" << std::endl << grad << std::endl;
    std::cout << "Formulation = " << Form << std::endl;

    KrylovSolverCG legacy_krylov_solver{legacy_cell, cg_tol, maxiter, verbose};
    auto && legacy_result{newton_cg(legacy_cell, grad, legacy_krylov_solver,
                                    newton_tol, equil_tol, verbose)};
    std::cout << "Done with legacy solver" << std::endl;

    Eigen::MatrixXd load_step{grad};

    // std::cout << "grad:\n" << grad << "\n";
    BOOST_TEST_CHECKPOINT("before fem solver call");
    auto && new_result{solver->solve_load_increment(load_step)};
    std::cout << "Newton-PCG converged in " << krylov_solver->get_counter()
              << " PCG steps and " << new_result.nb_it << " Newton steps."
              << std::endl;

    BOOST_TEST_CHECKPOINT("after load increment");
    Index_t nb_vals{stencil->get_nb_pixel_quad_pts()};
    Eigen::Map<Eigen::ArrayXXd> legacy_stress_map{
        legacy_result.stress.data(), grad.size(),
        legacy_result.stress.size() / grad.size()};
    // all pixels have uniform strain, so the stress can be stored per pixel for
    // comparison with legacy stress
    Eigen::ArrayXXd new_stress_pixel{legacy_stress_map.rows(),
                                     legacy_stress_map.cols()};
    for (Index_t i{0}; i < legacy_stress_map.cols(); ++i) {
      Eigen::VectorXd col_stress{Eigen::VectorXd::Zero(grad.size())};
      for (Index_t j{1}; j < nb_vals; ++j) {
        Real error{muGrid::testGoodies::rel_error(
            new_result.stress.col(i * nb_vals),
            new_result.stress.col(i * nb_vals + j))};
        BOOST_CHECK_LE(error, finite_diff_tol);
      }
      new_stress_pixel.col(i) = new_result.stress.col(i * nb_vals);
    }
    auto && error{
        muGrid::testGoodies::rel_error(new_stress_pixel, legacy_stress_map)};
    BOOST_CHECK_LE(error, finite_diff_tol);
    if (not(error < tol)) {
      std::cout << "legacy stress result" << std::endl
                << legacy_stress_map.transpose() << std::endl;
      std::cout << "new stress result" << std::endl
                << new_result.stress.transpose() << std::endl;
    }
    for (Index_t i{0}; i < new_result.stress.cols(); ++i) {
      std::cout << "ε = " << new_result.grad.col(i).transpose() << std::endl;
      std::cout << "σ = " << new_result.stress.col(i).transpose() << std::endl
                << std::endl;
    }
  }

  BOOST_AUTO_TEST_SUITE_END();
}  // namespace muSpectre

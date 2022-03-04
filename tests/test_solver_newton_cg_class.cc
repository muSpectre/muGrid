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
#include "materials/material_linear_diffusion.hh"

#include <cell/cell_factory.hh>
#include <solver/krylov_solver_eigen.hh>
#include <solver/krylov_solver_cg.hh>

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(newton_cg_solverclass);

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(constructor_test, Fix, CellDataFixtures,
                                   Fix) {
    this->cell_data->set_nb_quad_pts(OneQuadPt);
    MaterialLinearElastic1<Fix::SpatialDim>::make(this->cell_data, "material",
                                                  4, .3);
    auto krylov_solver{std::make_shared<KrylovSolverCGEigen>(1e-8, 100)};
    auto solver{std::make_shared<SolverNewtonCG>(this->cell_data, krylov_solver,
                                                 muGrid::Verbosity::Full, 1e-10,
                                                 1e-10, 100)};
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(scalar_problem_test, Fix, CellDataFixtures,
                                   Fix) {
    const Real high_conductivity{3.};
    const Real low_conductivity{2.};
    using Mat_t = MaterialLinearDiffusion<Fix::SpatialDim>;

    this->cell_data->set_nb_quad_pts(OneQuadPt);
    auto & low{Mat_t::make(this->cell_data, "low", low_conductivity)};
    auto & high{Mat_t::make(this->cell_data, "high", high_conductivity)};

    {
      bool first{true};
      for (auto && index_pixel : this->cell_data->get_pixels().enumerate()) {
        auto && index{std::get<0>(index_pixel)};
        if (first) {
          first = false;
          high.add_pixel(index);
        } else {
          low.add_pixel(index);
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
    const Eigen::VectorXd load{Eigen::VectorXd::Random(Fix::SpatialDim)};

    solver->initialise_cell();

    BOOST_TEST_CHECKPOINT("before load increment");
    std::cout << std::endl
              << "load:" << std::endl
              << load << std::endl
              << std::endl;
    solver->solve_load_increment(load);
  }

  /* ----------------------------------------------------------------------- */
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
        std::make_shared<KrylovSolverCGEigen>(cg_tol, maxiter, verbose)};
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

    KrylovSolverCGEigen legacy_krylov_solver{legacy_cell, cg_tol, maxiter,
                                             verbose};
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

  /* ----------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(solver_homogenizer_test, Fix,
                                   CellDataFixturesSmall, Fix) {
    const Real young_soft{1.};
    const Real young_hard{2.};
    const Real poisson{0.0};

    using T2_t = Eigen::Matrix<Real, Fix::SpatialDim, Fix::SpatialDim>;
    using T4_t = muGrid::T4Mat<Real, Fix::SpatialDim>;
    using Mat_t = MaterialLinearElastic1<Fix::SpatialDim>;

    auto && ref_parallel_2_s_1_h{[](const Real & v_h, const Real & v_s) {
      return (1. / ((2. / v_s) + (1. / v_h))) * 3.;
    }};

    auto && ref_serial_2_s_1_h{[](const Real & v_h, const Real & v_s) {
      return ((2. * v_s) + (1. * v_h)) / 3.;
    }};

    this->cell_data->set_nb_quad_pts(OneQuadPt);
    auto & soft{Mat_t::make(this->cell_data, "soft", young_soft, poisson)};
    auto & hard{Mat_t::make(this->cell_data, "hard", young_hard, poisson)};

    // making evaluator for materials soft and hard  to directly obtain their
    // stiffness
    auto mat_eval_soft = Mat_t::make_evaluator(young_soft, poisson);
    auto & mat_soft = *std::get<0>(mat_eval_soft);
    auto & evaluator_soft = std::get<1>(mat_eval_soft);
    mat_soft.add_pixel({});

    auto mat_eval_hard = Mat_t::make_evaluator(young_hard, poisson);
    auto & mat_hard = *std::get<0>(mat_eval_hard);
    auto & evaluator_hard = std::get<1>(mat_eval_hard);
    mat_hard.add_pixel({});

    T2_t sigma_soft, sigma_hard;
    T4_t C_soft, C_hard;

    BOOST_TEST_CHECKPOINT("before material assignment");
    {
      for (auto && index_pixel : this->cell_data->get_pixels().enumerate()) {
        auto && index{std::get<0>(index_pixel)};
        auto & pixel{std::get<1>(index_pixel)};
        if (pixel[1] == 0) {
          hard.add_pixel(index);
        } else {
          soft.add_pixel(index);
        }
      }
    }

    BOOST_TEST_CHECKPOINT("after material assignment");

    constexpr Real cg_tol{1e-8}, newton_tol{1e-5}, equil_tol{1e-10};
    const Uint maxiter{static_cast<Uint>(this->cell_data->get_spatial_dim()) *
                       10};
    constexpr Verbosity verbose{Verbosity::Silent};

    auto krylov_solver{
        std::make_shared<KrylovSolverCGEigen>(cg_tol, maxiter, verbose)};
    auto solver{std::make_shared<SolverNewtonCG>(this->cell_data, krylov_solver,
                                                 verbose, newton_tol, equil_tol,
                                                 maxiter)};
    auto && symmetric{[](Eigen::MatrixXd mat) -> Eigen::MatrixXd {
      return 0.5 * (mat + mat.transpose());
    }};

    Eigen::MatrixXd strain{
        symmetric(Eigen::MatrixXd::Identity(Fix::SpatialDim, Fix::SpatialDim))};
    strain(0, 0) *= -1.2367;

    std::tie(sigma_soft, C_soft) = evaluator_soft.evaluate_stress_tangent(
        strain, Formulation::small_strain);

    std::tie(sigma_hard, C_hard) = evaluator_hard.evaluate_stress_tangent(
        strain, Formulation::small_strain);

    solver->set_formulation(Formulation::small_strain);
    solver->initialise_cell();

    BOOST_TEST_CHECKPOINT("before load increment");
    auto && new_result{solver->solve_load_increment(strain)};
    BOOST_TEST_CHECKPOINT("after load increment");

    auto && C_eff{solver->compute_effective_stiffness()};
    BOOST_TEST_CHECKPOINT("after effective tangent calculation");

    auto && error{muGrid::testGoodies::rel_error(C_eff, C_eff.transpose())};
    BOOST_CHECK_LE(error, tol);
    if (not(error < tol)) {
      std::cout << "The calculated C_eff is not symmetric"
                << "C_eff:" << std::endl
                << C_eff << std::endl;
    }

    for (Dim_t i{0}; i < Fix::SpatialDim * Fix::SpatialDim; i++) {
      for (Dim_t j{0}; j < Fix::SpatialDim * Fix::SpatialDim; j++) {
        Real ref_value{};
        if (i == 0 and j == 0) {
          ref_value = ref_serial_2_s_1_h(C_hard(i, j), C_soft(i, j));
        } else {
          ref_value = ref_parallel_2_s_1_h(C_hard(i, j), C_soft(i, j));
        }
        auto && error{ref_value - C_eff(i, j)};
        BOOST_CHECK_LE(error, tol);
        if (not(error < tol)) {
          std::cout << "C_soft:\n" << C_soft << std::endl;
          std::cout << "C_hard:\n" << C_hard << std::endl;
          std::cout << "C_eff:\n" << C_eff << std::endl;
        }
      }
    }
  }

  /* ----------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(solver_homogenizer_test_stress_control, Fix,
                                   CellDataFixturesSmall, Fix) {
    const Real young_soft{1.};
    const Real young_hard{2.};
    const Real poisson{0.0};

    using T2_t = Eigen::Matrix<Real, Fix::SpatialDim, Fix::SpatialDim>;
    using T4_t = muGrid::T4Mat<Real, Fix::SpatialDim>;
    using Mat_t = MaterialLinearElastic1<Fix::SpatialDim>;

    auto && ref_parallel_2_s_1_h{[](const Real & v_h, const Real & v_s) {
      return (1. / ((2. / v_s) + (1. / v_h))) * 3.;
    }};

    auto && ref_serial_2_s_1_h{[](const Real & v_h, const Real & v_s) {
      return ((2. * v_s) + (1. * v_h)) / 3.;
    }};

    this->cell_data->set_nb_quad_pts(OneQuadPt);
    auto & soft{Mat_t::make(this->cell_data, "soft", young_soft, poisson)};
    auto & hard{Mat_t::make(this->cell_data, "hard", young_hard, poisson)};

    // making evaluator for materials soft and hard  to directly obtain their
    // stiffness
    auto mat_eval_soft = Mat_t::make_evaluator(young_soft, poisson);
    auto & mat_soft = *std::get<0>(mat_eval_soft);
    auto & evaluator_soft = std::get<1>(mat_eval_soft);
    mat_soft.add_pixel({});

    auto mat_eval_hard = Mat_t::make_evaluator(young_hard, poisson);
    auto & mat_hard = *std::get<0>(mat_eval_hard);
    auto & evaluator_hard = std::get<1>(mat_eval_hard);
    mat_hard.add_pixel({});

    T2_t sigma_soft, sigma_hard;
    T4_t C_soft, C_hard;

    BOOST_TEST_CHECKPOINT("before material assignment");
    {
      for (auto && index_pixel : this->cell_data->get_pixels().enumerate()) {
        auto && index{std::get<0>(index_pixel)};
        auto & pixel{std::get<1>(index_pixel)};
        if (pixel[1] == 0) {
          hard.add_pixel(index);
        } else {
          soft.add_pixel(index);
        }
      }
    }

    BOOST_TEST_CHECKPOINT("after material assignment");

    constexpr Real cg_tol{1e-8}, newton_tol{1e-5}, equil_tol{1e-10};
    const Uint maxiter{static_cast<Uint>(this->cell_data->get_spatial_dim()) *
                       10};
    constexpr Verbosity verbose{Verbosity::Silent};
    constexpr MeanControl mean_control{MeanControl::StressControl};
    auto krylov_solver{
        std::make_shared<KrylovSolverCGEigen>(cg_tol, maxiter, verbose)};
    auto solver{std::make_shared<SolverNewtonCG>(this->cell_data, krylov_solver,
                                                 verbose, newton_tol, equil_tol,
                                                 maxiter, mean_control)};

    // In case of stress control solver, we need to define a new strain control
    // solver for the cell because in the algorithm derived needs to use a
    // strain control projection operator to zero out σ_eq.
    auto homo_krylov_solver{
        std::make_shared<KrylovSolverCGEigen>(cg_tol, maxiter, verbose)};
    auto homo_solver{std::make_shared<SolverNewtonCG>(
        this->cell_data, homo_krylov_solver, verbose, newton_tol, equil_tol,
        maxiter)};

    auto && symmetric{[](Eigen::MatrixXd mat) -> Eigen::MatrixXd {
      return 0.5 * (mat + mat.transpose());
    }};

    Eigen::MatrixXd strain{
        symmetric(Eigen::MatrixXd::Identity(Fix::SpatialDim, Fix::SpatialDim))};

    strain(0, 0) *= -0.91236;

    std::tie(sigma_soft, C_soft) = evaluator_soft.evaluate_stress_tangent(
        strain, Formulation::small_strain);

    std::tie(sigma_hard, C_hard) = evaluator_hard.evaluate_stress_tangent(
        strain, Formulation::small_strain);

    solver->set_formulation(Formulation::small_strain);
    solver->initialise_cell();

    homo_solver->set_formulation(Formulation::small_strain);
    homo_solver->initialise_cell();

    BOOST_TEST_CHECKPOINT("before load increment");
    auto && new_result{solver->solve_load_increment(strain)};
    BOOST_TEST_CHECKPOINT("after load increment");

    BOOST_CHECK_THROW(solver->compute_effective_stiffness(), SolverError);

    BOOST_TEST_CHECKPOINT("before effective tangent calculation");
    auto && C_eff{homo_solver->compute_effective_stiffness()};
    BOOST_TEST_CHECKPOINT("after effective tangent calculation");

    auto && error{muGrid::testGoodies::rel_error(C_eff, C_eff.transpose())};
    BOOST_CHECK_LE(error, tol);
    if (not(error < tol)) {
      std::cout << "The calculated C_eff is not symmetric"
                << "C_eff:" << std::endl
                << C_eff << std::endl;
    }

    for (Dim_t i{0}; i < Fix::SpatialDim * Fix::SpatialDim; i++) {
      for (Dim_t j{0}; j < Fix::SpatialDim * Fix::SpatialDim; j++) {
        Real ref_value{};
        if (i == 0 and j == 0) {
          ref_value = ref_serial_2_s_1_h(C_hard(i, j), C_soft(i, j));
        } else {
          ref_value = ref_parallel_2_s_1_h(C_hard(i, j), C_soft(i, j));
        }
        auto && error{ref_value - C_eff(i, j)};
        BOOST_CHECK_LE(error, tol);
        if (not(error < tol)) {
          std::cout << "C_soft:\n" << C_soft << std::endl;
          std::cout << "C_hard:\n" << C_hard << std::endl;
          std::cout << "C_eff:\n" << C_eff << std::endl;
        }
      }
    }
  }

  /* ----------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(solver_homogenizer_test_symmetry, Fix,
                                   CellDataFixturesSmall, Fix) {
    const Real young_soft{1.8462834};
    const Real young_hard{2 * young_soft};
    const Real poisson{.25};
    using Mat_t = MaterialLinearElastic1<Fix::SpatialDim>;

    this->cell_data->set_nb_quad_pts(OneQuadPt);
    auto & soft{Mat_t::make(this->cell_data, "soft", young_soft, poisson)};
    auto & hard{Mat_t::make(this->cell_data, "hard", young_hard, poisson)};

    {
      bool first{true};
      for (auto && index_pixel : this->cell_data->get_pixels().enumerate()) {
        auto && index{std::get<0>(index_pixel)};
        if (first) {
          hard.add_pixel(index);
          first = false;
        } else {
          soft.add_pixel(index);
        }
      }
    }

    BOOST_TEST_CHECKPOINT("after material assignment");

    constexpr Real cg_tol{1e-8}, newton_tol{1e-5}, equil_tol{1e-10};
    const Uint maxiter{static_cast<Uint>(this->cell_data->get_spatial_dim()) *
                       10};
    constexpr Verbosity verbose{Verbosity::Silent};

    auto krylov_solver{
        std::make_shared<KrylovSolverCGEigen>(cg_tol, maxiter, verbose)};
    auto solver{std::make_shared<SolverNewtonCG>(this->cell_data, krylov_solver,
                                                 verbose, newton_tol, equil_tol,
                                                 maxiter)};
    auto && symmetric{[](Eigen::MatrixXd mat) -> Eigen::MatrixXd {
      return 0.5 * (mat + mat.transpose());
    }};

    const Eigen::MatrixXd strain{
        symmetric(Eigen::MatrixXd::Identity(Fix::SpatialDim, Fix::SpatialDim))};

    solver->set_formulation(Formulation::small_strain);
    solver->initialise_cell();

    BOOST_TEST_CHECKPOINT("before load increment");
    auto && new_result{solver->solve_load_increment(strain)};
    BOOST_TEST_CHECKPOINT("after load increment");

    Eigen::MatrixXd C_eff{solver->compute_effective_stiffness()};

    BOOST_TEST_CHECKPOINT("after effective tangent calculation");

    Real error{0.0};

    error = muGrid::testGoodies::rel_error(C_eff, C_eff.transpose());
    BOOST_CHECK_LE(error, tol);
    if (not(error < tol)) {
      std::cout << "The calculated C_eff is not symmetric"
                << "C_eff:" << std::endl
                << C_eff << std::endl;
    }

    // check for minor symmetries:
    std::vector<int> ind_shear{};
    std::vector<int> ind_shear_transpose{};
    std::vector<int> ind_norm{};
    switch (Fix::SpatialDim) {
    case twoD: {
      ind_shear.insert(ind_shear.end(), {1, 2});
      ind_shear_transpose.insert(ind_shear_transpose.end(), {2, 1});
      ind_norm.insert(ind_norm.end(), {0, 3});
      break;
    }
    case threeD: {
      ind_shear.insert(ind_shear.end(), {1, 2, 3, 5, 6, 7});
      ind_shear_transpose.insert(ind_shear_transpose.end(), {3, 6, 1, 7, 2, 5});
      ind_norm.insert(ind_norm.end(), {0, 4, 8});
      break;
    }
    default: {
      std::cout << "dimension not defined"
                << "\n";
      break;
    }
    }

    // normal components:
    for (auto && i : ind_norm) {
      for (auto && j : ind_norm) {
        if (i == j) {
          error = C_eff(0, 0) - C_eff(i, j);
          BOOST_CHECK_LE(error, tol);
          if (not(error < tol)) {
            std::cout << "C_eff(0, 0): " << C_eff(0, 0) << std::endl
                      << "C(" << i << ", " << j << ")" << C_eff(i, j)
                      << std::endl
                      << "so C_eff does not satisfy minor symmetry";
          }
        } else {
          error = poisson * C_eff(0, 0) - C_eff(i, j);
          BOOST_CHECK_LE(error, tol);
        }
      }
    }

    // shear components:
    for (auto && tup : akantu::enumerate(ind_shear)) {
      auto & n{std::get<0>(tup)};
      auto & i{std::get<1>(tup)};
      for (auto && j : ind_shear) {
        if (i == j) {
          error = C_eff(1, 1) - C_eff(i, j);
          BOOST_CHECK_LE(error, tol);
          if (not(error < tol)) {
            std::cout << "C_eff(1, 1): " << C_eff(1, 1) << std::endl
                      << "C(" << i << ", " << j << ")" << C_eff(i, j)
                      << std::endl
                      << "so C_eff does not satisfy minor symmetry";
          }
        } else if (j == ind_shear_transpose[n]) {
          error = C_eff(1, 1) - C_eff(i, j);
          BOOST_CHECK_LE(error, tol);
          if (not(error < tol)) {
            std::cout << "C_eff(1, 1): " << C_eff(1, 1) << std::endl
                      << "C(" << i << ", " << j << ")" << C_eff(i, j)
                      << std::endl
                      << "so C_eff does not satisfy minor symmetry";
          }
        } else {
          BOOST_CHECK_LE(C_eff(i, j), tol);
        }
      }
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muSpectre

/**
 * @file   test_eigen_strain_solver_classes.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   13 Jan 2021
 *
 * @brief  Testing the eigen strain handling of solver classes against eigen
 * strain handling of materials
 *
 * Copyright © 2021 Ali Falsafi
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
#include "solver/solver_newton_cg.hh"
#include "solver/solver_fem_newton_cg.hh"
#include "solver/solvers.hh"
#include "solver/krylov_solver_cg.hh"
#include "solver/krylov_solver_eigen.hh"
#include "projection/projection_finite_strain_fast.hh"
#include "materials/material_linear_elastic1.hh"
#include "materials/material_linear_elastic2.hh"
#include "cell/cell_factory.hh"

#include <libmugrid/iterators.hh>
#include <libmugrid/ccoord_operations.hh>
#include <libmufft/pocketfft_engine.hh>
#include <libmugrid/exception.hh>

#include <boost/mpl/list.hpp>
#include <functional>
#include <iomanip>

namespace muSpectre {

  using muGrid::testGoodies::rel_error;

  BOOST_AUTO_TEST_SUITE(eigen_strain_solver_classes);

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(constructor_test, Fix,
                                   CellDataFixtureEigenStrains, Fix) {
    this->cell_data->set_nb_quad_pts(OneQuadPt);

    MaterialLinearElastic1<Fix::SpatialDim>::make(this->cell_data, "material",
                                                  4, .3);
    auto krylov_solver{std::make_shared<KrylovSolverCG>(1e-8, 100)};
    auto solver{std::make_shared<SolverNewtonCG>(this->cell_data, krylov_solver,
                                                 muGrid::Verbosity::Silent,
                                                 1e-10, 1e-10, 100)};
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(solver_test_single_step, Fix,
                                   CellDataFixtureEigenStrains, Fix) {
    const Real young_soft{4};
    const Real young_hard{8};
    const Real poisson{.3};
    constexpr Index_t Dim{Fix::SpatialDim};
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

    using Func_t =
        std::function<void(const size_t &, muGrid::TypedFieldBase<Real> &)>;

    using Func_new_t = std::function<void(muGrid::TypedFieldBase<Real> &)>;

    using Matrix_t = typename Fix::Matrix_t;
    Matrix_t F_eigen{Fix::F_eigen};

    // The function which is responsible for assigning eigen strain
    Func_t eigen_func_legacy{
        [this, &F_eigen](const size_t & /*step*/,
                         muGrid::TypedFieldBase<Real> & eval_field) {
          auto && eigen_field_map{muGrid::FieldMap<Real, Mapping::Mut>(
              eval_field, Dim, muGrid::IterUnit::SubPt)};
          for (auto && tup :
               akantu::zip(eigen_field_map.enumerate_indices(),
                           this->cell_data->get_pixels().enumerate())) {
            auto && index_eigen{std::get<0>(tup)};
            auto && index_pixel{std::get<1>(tup)};
            // auto && index{std::get<0>(index_pixel)};
            auto && pixel{std::get<1>(index_pixel)};
            auto && eigen{std::get<1>(index_eigen)};
            DynCcoord_t cell_lengths{Fix::get_size()};
            DynCcoord_t mid_pixel{};
            for (Dim_t i{0}; i < Dim; ++i) {
              mid_pixel[i] = std::floor(cell_lengths[i]) / 2 + 1;
            }
            bool is_mid_pixel{true};
            for (Dim_t i{0}; i < Dim; ++i) {
              if (mid_pixel[i] != pixel[i])
                is_mid_pixel = false;
            }

            if (is_mid_pixel) {
              eigen -= F_eigen;
            }
          }
        }};

    // The function which is responsible for assigning eigen strain
    Func_new_t eigen_func_new{
        [this, &eigen_func_legacy](muGrid::TypedFieldBase<Real> & eval_field) {
          size_t step{this->step_nb};
          eigen_func_legacy(step, eval_field);
        }};

    BOOST_TEST_CHECKPOINT("after material assignment");

    constexpr Real cg_tol{1e-8}, newton_tol{1e-5}, equil_tol{1e-10};
    const Uint maxiter{static_cast<Uint>(this->cell_data->get_spatial_dim()) *
                       10};
    constexpr Verbosity verbose{Verbosity::Silent};

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

    auto && new_result{solver->solve_load_increment(strain, eigen_func_new)};
    BOOST_TEST_CHECKPOINT("after load increment");

    KrylovSolverCG legacy_krylov_solver{legacy_cell, cg_tol, maxiter, verbose};
    auto && legacy_result{newton_cg(legacy_cell, strain, legacy_krylov_solver,
                                    newton_tol, equil_tol, verbose,
                                    IsStrainInitialised::False,
                                    StoreNativeStress::no, eigen_func_legacy)};

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

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(solver_test_multiple_step_homogen_cell, Fix,
                                   CellDataFixtureEigenStrains, Fix) {
    constexpr Index_t Dim{Fix::SpatialDim};
    // const Real young_soft{4};
    const Real young_hard{8};
    const Real poisson{.3};
    using Mat_t = MaterialLinearElastic1<Fix::SpatialDim>;
    this->cell_data->set_nb_quad_pts(OneQuadPt);

    auto & hard{Mat_t::make(this->cell_data, "hard", young_hard, poisson)};

    auto legacy_cell{make_cell(this->cell_data->get_nb_domain_grid_pts(),
                               this->cell_data->get_domain_lengths(),
                               Formulation::small_strain)};

    auto & legacy_hard{Mat_t::make(legacy_cell, "hard", young_hard, poisson)};

    for (auto && index_pixel : this->cell_data->get_pixels().enumerate()) {
      auto && index{std::get<0>(index_pixel)};
      hard.add_pixel(index);
      legacy_hard.add_pixel(index);
    }

    using Func_t =
        std::function<void(const size_t &, muGrid::TypedFieldBase<Real> &)>;

    using Func_new_t = std::function<void(muGrid::TypedFieldBase<Real> &)>;

    using Matrix_t = typename Fix::Matrix_t;
    Matrix_t F_eigen{Fix::F_eigen};

    // The function which is responsible for assigning eigen strain
    Func_t eigen_func_legacy{
        [this, &F_eigen](const size_t & step,
                         muGrid::TypedFieldBase<Real> & eval_field) {
          auto && stress_coeff{step + 1};
          auto && eigen_field_map{muGrid::FieldMap<Real, Mapping::Mut>(
              eval_field, Dim, muGrid::IterUnit::SubPt)};
          for (auto && tup :
               akantu::zip(eigen_field_map.enumerate_indices(),
                           this->cell_data->get_pixels().enumerate())) {
            auto && index_eigen{std::get<0>(tup)};
            auto && index_pixel{std::get<1>(tup)};
            // auto && index{std::get<0>(index_pixel)};
            auto && pixel{std::get<1>(index_pixel)};
            auto && eigen{std::get<1>(index_eigen)};
            // DynCcoord_t cell_lengths{this->cell_data->get_domain_lengths()};
            DynCcoord_t cell_lengths{Fix::get_size()};
            DynCcoord_t mid_pixel{};
            for (Dim_t i{0}; i < Dim; ++i) {
              mid_pixel[i] = cell_lengths[i] / 2 + 1;
            }
            bool is_mid_pixel{true};
            for (Dim_t i{0}; i < Dim; ++i) {
              if (mid_pixel[i] != pixel[i])
                is_mid_pixel = false;
            }

            if (is_mid_pixel) {
              eigen -= (stress_coeff) * (F_eigen);
            }
          }
        }};

    Func_new_t eigen_func_new{
        [this, &eigen_func_legacy](muGrid::TypedFieldBase<Real> & eval_field) {
          size_t step{this->step_nb};
          eigen_func_legacy(step, eval_field);
        }};

    BOOST_TEST_CHECKPOINT("after material assignment");

    constexpr Real cg_tol{1e-8}, newton_tol{1e-8}, equil_tol{1e-12};
    const Uint maxiter{static_cast<Uint>(this->cell_data->get_spatial_dim()) *
                       10};
    constexpr Verbosity verbose{Verbosity::Silent};

    auto krylov_solver{
        std::make_shared<KrylovSolverCG>(cg_tol, maxiter, verbose)};
    auto solver{std::make_shared<SolverNewtonCG>(this->cell_data, krylov_solver,
                                                 verbose, newton_tol, equil_tol,
                                                 maxiter)};

    auto homo_solver{std::make_shared<SolverNewtonCG>(
        this->cell_data, krylov_solver, verbose, newton_tol, equil_tol,
        maxiter)};

    auto && symmetric{[](Eigen::MatrixXd mat) -> Eigen::MatrixXd {
      return 0.5 * (mat + mat.transpose());
    }};
    const Eigen::MatrixXd strain{
        symmetric(Eigen::MatrixXd::Random(Fix::SpatialDim, Fix::SpatialDim)) /
        2};

    solver->set_formulation(Formulation::small_strain);
    solver->initialise_cell();

    homo_solver->set_formulation(Formulation::small_strain);
    homo_solver->initialise_cell();

    BOOST_TEST_CHECKPOINT("before load increment");
    std::cout << std::endl
              << "strain:" << std::endl
              << strain << std::endl
              << std::endl;
    std::cout << std::endl
              << "symmetric(strain):" << std::endl
              << symmetric(strain) << std::endl
              << std::endl;
    auto && new_result{solver->solve_load_increment(strain, eigen_func_new)};
    BOOST_TEST_CHECKPOINT("after load increment");

    using LoadSteps_t = std::vector<Eigen::MatrixXd>;
    const int nb_steps{10};

    LoadSteps_t delFs{};
    for (int step{0}; step < nb_steps; ++step) {
      delFs.push_back(strain);
    }

    KrylovSolverCG legacy_krylov_solver{legacy_cell, cg_tol, maxiter, verbose};
    auto && legacy_result{newton_cg(legacy_cell, delFs, legacy_krylov_solver,
                                    newton_tol, equil_tol, verbose,
                                    IsStrainInitialised::False,
                                    StoreNativeStress::no, eigen_func_legacy)};

    for (int i{0}; i < nb_steps; ++i) {
      this->step_nb = i;
      auto && legacy_res{legacy_result[i]};
      Eigen::Map<Eigen::ArrayXXd> legacy_stress_map{legacy_res.stress.data(),
                                                    new_result.stress.rows(),
                                                    new_result.stress.cols()};
      auto && new_result{solver->solve_load_increment(strain, eigen_func_new)};
      auto && error{
          muGrid::testGoodies::rel_error(new_result.stress, legacy_stress_map)};
      BOOST_CHECK_LE(error, tol);
      if (not(error < tol)) {
        std::cout << "legacy stress result" << std::endl
                  << legacy_stress_map.transpose() << std::endl;
        std::cout << "new stress result" << std::endl
                  << new_result.stress.transpose() << std::endl;
        std::cout << "legacy stress result - new stress result" << std::endl
                  << legacy_stress_map.transpose() -
                         new_result.stress.transpose()
                  << std::endl;
      }
      auto && C_eff{homo_solver->compute_effective_stiffness()};
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(solver_test_multiple_step_heterogen_cell,
                                   Fix, CellDataFixtureEigenStrains, Fix) {
    constexpr Index_t Dim{Fix::SpatialDim};
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

    using Func_t =
        std::function<void(const size_t &, muGrid::TypedFieldBase<Real> &)>;

    using Func_new_t = std::function<void(muGrid::TypedFieldBase<Real> &)>;

    using Matrix_t = typename Fix::Matrix_t;
    Matrix_t F_eigen{Fix::F_eigen};

    // The function which is responsible for assigning eigen strain
    Func_t eigen_func_legacy{
        [this, &F_eigen](const size_t & step,
                         muGrid::TypedFieldBase<Real> & eval_field) {
          auto && stress_coeff{step + 1};
          auto && eigen_field_map{muGrid::FieldMap<Real, Mapping::Mut>(
              eval_field, Dim, muGrid::IterUnit::SubPt)};
          for (auto && tup :
               akantu::zip(eigen_field_map.enumerate_indices(),
                           this->cell_data->get_pixels().enumerate())) {
            auto && index_eigen{std::get<0>(tup)};
            auto && index_pixel{std::get<1>(tup)};
            // auto && index{std::get<0>(index_pixel)};
            auto && pixel{std::get<1>(index_pixel)};
            auto && eigen{std::get<1>(index_eigen)};
            // DynCcoord_t cell_lengths{this->cell_data->get_domain_lengths()};
            DynCcoord_t cell_lengths{Fix::get_size()};
            DynCcoord_t mid_pixel{};
            for (Dim_t i{0}; i < Dim; ++i) {
              mid_pixel[i] = cell_lengths[i] / 2 + 1;
            }

            bool is_mid_pixel{true};
            for (Dim_t i{0}; i < Dim; ++i) {
              if (mid_pixel[i] != pixel[i])
                is_mid_pixel = false;
            }

            if (is_mid_pixel) {
              eigen -= (stress_coeff) * (F_eigen);
            }
          }
        }};

    Func_new_t eigen_func_new{
        [this, &eigen_func_legacy](muGrid::TypedFieldBase<Real> & eval_field) {
          size_t step{this->step_nb};
          eigen_func_legacy(step, eval_field);
        }};

    BOOST_TEST_CHECKPOINT("after material assignment");

    constexpr Real cg_tol{1e-8}, newton_tol{1e-8}, equil_tol{1e-12};
    const Uint maxiter{static_cast<Uint>(this->cell_data->get_spatial_dim()) *
                       10};
    constexpr Verbosity verbose{Verbosity::Silent};

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

    BOOST_TEST_CHECKPOINT("before load increment");
    std::cout << std::endl
              << "strain:" << std::endl
              << strain << std::endl
              << std::endl;
    std::cout << std::endl
              << "symmetric(strain):" << std::endl
              << symmetric(strain) << std::endl
              << std::endl;
    solver->solve_load_increment(strain, eigen_func_new);
    BOOST_TEST_CHECKPOINT("after load increment");

    using LoadSteps_t = std::vector<Eigen::MatrixXd>;
    const int nb_steps{10};

    LoadSteps_t delFs{};
    for (int step{0}; step < nb_steps; ++step) {
      delFs.push_back(strain);
    }

    BOOST_TEST_CHECKPOINT("before legacy solver call");

    KrylovSolverCG legacy_krylov_solver{legacy_cell, cg_tol, maxiter, verbose};
    auto && legacy_result{newton_cg(legacy_cell, delFs, legacy_krylov_solver,
                                    newton_tol, equil_tol, verbose,
                                    IsStrainInitialised::False,
                                    StoreNativeStress::no, eigen_func_legacy)};

    BOOST_TEST_CHECKPOINT("after legacy solver call");

    for (int i{1}; i < nb_steps; ++i) {
      this->step_nb = i;
      auto && new_result{solver->solve_load_increment(strain, eigen_func_new)};
      auto && legacy_res{legacy_result[i]};
      Eigen::Map<Eigen::ArrayXXd> legacy_stress_map{legacy_res.stress.data(),
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
        std::cout << "legacy stress result - new stress result" << std::endl
                  << legacy_stress_map.transpose() -
                         new_result.stress.transpose()
                  << std::endl;
      }
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(solver_test_multiple_step_heterogen_cell_fem,
                                   Fix, CellDataFixtureEigenStrains2D, Fix) {
    constexpr Index_t Dim{Fix::SpatialDim};
    const Real young_soft{4};
    const Real young_hard{8};
    const Real poisson{.3};
    using Mat_t = MaterialLinearElastic1<Fix::SpatialDim>;
    auto stencil{FEMLibrary::bilinear_quadrangle(this->cell_data)};
    auto discretisation{std::make_shared<Discretisation>(stencil)};

    auto stencil_eigen{FEMLibrary::bilinear_quadrangle(this->cell_data_eigen)};
    auto discretisation_eigen{std::make_shared<Discretisation>(stencil_eigen)};

    auto & soft_eigen{
        Mat_t::make(this->cell_data_eigen, "soft", young_soft, poisson)};
    auto & hard_eigen{
        Mat_t::make(this->cell_data_eigen, "hard", young_hard, poisson)};

    auto & soft{Mat_t::make(this->cell_data, "soft", young_soft, poisson)};
    auto & hard{Mat_t::make(this->cell_data, "hard", young_hard, poisson)};

    {
      bool first{true};
      for (auto && index_pixel : this->cell_data->get_pixels().enumerate()) {
        auto && index{std::get<0>(index_pixel)};
        if (first) {
          first = false;
          hard.add_pixel(index);
          hard_eigen.add_pixel(index);
        } else {
          soft.add_pixel(index);
          soft_eigen.add_pixel(index);
        }
      }
    }

    using Func_t =
        std::function<void(const size_t &, muGrid::TypedFieldBase<Real> &)>;

    using Func_eigen_t = std::function<void(muGrid::TypedFieldBase<Real> &)>;

    using Matrix_t = typename Fix::Matrix_t;
    Matrix_t F_eigen{Fix::F_eigen};

    // The function which is responsible for assigning eigen strain
    Func_t eigen_func_legacy{
        [&F_eigen](const size_t & step,
                   muGrid::TypedFieldBase<Real> & eval_field) {
          auto && stress_coeff{step + 1};
          auto && eigen_field_map{muGrid::FieldMap<Real, Mapping::Mut>(
              eval_field, Dim, muGrid::IterUnit::SubPt)};
          for (auto && tup : eigen_field_map.enumerate_indices()) {
            auto && eigen{std::get<1>(tup)};
            eigen += (stress_coeff) * (F_eigen);
          }
        }};

    Func_eigen_t eigen_func_eigen{
        [this, &eigen_func_legacy](muGrid::TypedFieldBase<Real> & eval_field) {
          size_t step{this->step_nb};
          eigen_func_legacy(step, eval_field);
        }};

    BOOST_TEST_CHECKPOINT("after material assignment");

    constexpr Real cg_tol{1e-8}, newton_tol{1e-8}, equil_tol{1e-12};
    const Uint maxiter{static_cast<Uint>(this->cell_data->get_spatial_dim()) *
                       10};
    constexpr Verbosity verbose{Verbosity::Silent};

    auto krylov_solver{
        std::make_shared<KrylovSolverCG>(cg_tol, maxiter, verbose)};

    auto krylov_solver_eigen{
        std::make_shared<KrylovSolverCG>(cg_tol, maxiter, verbose)};

    auto solver{std::make_shared<SolverFEMNewtonCG>(
        discretisation, krylov_solver, verbose, newton_tol, equil_tol,
        maxiter)};

    auto solver_eigen{std::make_shared<SolverFEMNewtonCG>(
        discretisation_eigen, krylov_solver_eigen, verbose, newton_tol,
        equil_tol, maxiter)};

    const Eigen::MatrixXd strain{Fix::F_eigen};
    solver->set_formulation(Formulation::small_strain);
    solver_eigen->set_formulation(Formulation::small_strain);

    const Eigen::MatrixXd zero_strain{
        Eigen::MatrixXd::Zero(Fix::SpatialDim, Fix::SpatialDim)};

    BOOST_TEST_CHECKPOINT("Solvers ready");
    const int nb_steps{10};
    for (int i{0}; i < nb_steps; ++i) {
      std::cout << "step_nb: " << i << "\n";
      this->step_nb = i;

      auto && strain_app{(this->step_nb + 1) * strain};
      auto && old_result{solver->solve_load_increment(strain_app)};
      auto && eigen_result{
          solver_eigen->solve_load_increment(zero_strain, eigen_func_eigen)};
      auto && error{muGrid::testGoodies::rel_error(eigen_result.stress,
                                                   old_result.stress)};
      BOOST_CHECK_LE(error, tol);
      if (not(error < tol)) {
        std::cout << "old stress result" << std::endl
                  << old_result.stress.transpose() << std::endl;
        std::cout << "eigen stress result" << std::endl
                  << eigen_result.stress.transpose() << std::endl;
        std::cout << "delta stress result" << std::endl
                  << (eigen_result.stress.transpose() -
                      old_result.stress.transpose())
                  << std::endl;
      }
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muSpectre

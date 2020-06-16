/**
 * @file   test_eigen_strain_solver.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   27 May 2020
 *
 * @brief  Testing the eigen strain handling of solvers against eigen strain
 * handling of materials
 *
 * Copyright © 2020 Ali Falsafi
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
#include "libmugrid/test_goodies.hh"

#include "solver/solvers.hh"
#include "solver/krylov_solver_cg.hh"
#include "solver/krylov_solver_eigen.hh"
#include "projection/projection_finite_strain_fast.hh"
#include "materials/material_linear_elastic1.hh"
#include "materials/material_linear_elastic2.hh"
#include "cell/cell_factory.hh"

#include <libmugrid/iterators.hh>
#include <libmugrid/ccoord_operations.hh>
#include <libmufft/fftw_engine.hh>
#include <libmugrid/exception.hh>

#include <boost/mpl/list.hpp>
#include <functional>
#include <iomanip>

namespace muSpectre {

  using muGrid::testGoodies::rel_error;

  BOOST_AUTO_TEST_SUITE(eigen_strain_hendling_solvers);

  template <Index_t Dim>
  struct DimFixture {
    constexpr static Index_t Mdim{Dim};
    using Matrix_t = Eigen::Matrix<Real, Dim, Dim>;
    DimFixture() : F_eigen{F_eigen_maker()}, F0{Matrix_t::Zero()} {};

    Matrix_t F_eigen_maker() {
      const Real eps{1.0e-6};
      switch (Dim) {
      case twoD: {
        return (Matrix_t() << eps, 0.0, 0.0, eps).finished();
        break;
      }
      case threeD: {
        return (Matrix_t() << eps, 0.0, 0.0, 0.0, eps, 0.0, 0.0, 0.0, eps)
            .finished();
        break;
      }
      default:
        throw muGrid::RuntimeError("The dimension is invalid");
        break;
      }
    }

    Matrix_t F_eigen;
    Matrix_t F0;
  };

  using strains = boost::mpl::list<DimFixture<twoD>, DimFixture<threeD>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_single_step, Fix, strains, Fix) {
    constexpr Index_t Dim{Fix::Mdim};

    using Mat1_t = MaterialLinearElastic1<Dim>;
    using Mat2_t = MaterialLinearElastic2<Dim>;
    using Matrix_t = typename Fix::Matrix_t;

    const DynCcoord_t nb_grid_pts{muGrid::CcoordOps::get_cube<Dim>(Index_t{3})};
    const DynRcoord_t lengths{muGrid::CcoordOps::get_cube<Dim>(1.)};
    constexpr Formulation form{Formulation::small_strain};

    auto cell_solver{make_cell(nb_grid_pts, lengths, form)};
    auto cell_material{make_cell(nb_grid_pts, lengths, form)};

    constexpr Real Young{2.}, Poisson{.33};

    auto & material_1_material{
        Mat1_t::make(cell_material, "material_1_material", Young, Poisson)};
    auto & material_2_material{
        Mat2_t::make(cell_material, "material_2_material", Young, Poisson)};
    auto & material_1_solver{
        Mat1_t::make(cell_solver, "material_1_solver", Young, Poisson)};

    for (const auto && index_pixel : cell_solver.get_pixels().enumerate()) {
      auto && index{std::get<0>(index_pixel)};
      material_1_solver.add_pixel(index);
    }

    Matrix_t F_eigen{Fix::F_eigen};
    Matrix_t delF0{Fix::F0};
    constexpr Real cg_tol{1e-8}, newton_tol{1e-5}, equi_tol{1e-8};
    constexpr Dim_t maxiter{100};
    constexpr Verbosity verbose{Verbosity::Silent};

    for (const auto && index_pixel : cell_material.get_pixels().enumerate()) {
      auto && index{std::get<0>(index_pixel)};
      if (index == Index_t(cell_material.get_nb_pixels() / 2 + 1)) {
        material_2_material.add_pixel(index, F_eigen);
      } else {
        material_1_material.add_pixel(index);
      }
    }

    using Func_t =
        std::function<void(const size_t &, muGrid::TypedFieldBase<Real> &)>;

    // The function which is responsible for assigning eigen strain
    Func_t eigen_func{
        [&cell_material, &F_eigen](const size_t & /*step*/,
                                   muGrid::TypedFieldBase<Real> & eval_field) {
          auto shape{cell_material.get_strain_shape()};
          auto && eigen_field_map{muGrid::FieldMap<Real, Mapping::Mut>(
              eval_field, shape[0], muGrid::IterUnit::SubPt)};
          for (auto && tup : eigen_field_map.enumerate_indices()) {
            auto && index{std::get<0>(tup)};
            auto && eigen{std::get<1>(tup)};
            if (index == cell_material.get_nb_pixels() / 2 + 1) {
              eigen -= F_eigen;
            }
          }
        }};

    cell_material.initialise();
    cell_solver.initialise();

    KrylovSolverCGEigen cg_material{cell_material, cg_tol, maxiter, verbose};
    KrylovSolverCGEigen cg_solver{cell_solver, cg_tol, maxiter, verbose};

    auto && res_material{newton_cg(cell_material, delF0, cg_material,
                                   newton_tol, equi_tol, verbose,
                                   IsStrainInitialised::False)};

    auto && res_solver{newton_cg(cell_solver, delF0, cg_solver, newton_tol,
                                 equi_tol, verbose, IsStrainInitialised::False,
                                 eigen_func)};

    auto && stress_solver{res_solver.stress};
    auto && stress_material{res_material.stress};

    auto && strain_solver{res_solver.grad};
    auto && strain_material{res_material.grad};

    auto && diff_stress{rel_error(stress_solver, stress_material)};
    auto && diff_strain{rel_error(strain_solver, strain_material)};

    BOOST_CHECK_LE(diff_stress, cg_tol);
    BOOST_CHECK_LE(diff_strain, cg_tol);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_multiple_step, Fix, strains, Fix) {
    constexpr Index_t Dim{Fix::Mdim};

    using Mat1_t = MaterialLinearElastic1<Dim>;
    using Mat2_t = MaterialLinearElastic2<Dim>;
    using Matrix_t = typename Fix::Matrix_t;

    const DynCcoord_t nb_grid_pts{muGrid::CcoordOps::get_cube<Dim>(Index_t{3})};
    const DynRcoord_t lengths{muGrid::CcoordOps::get_cube<Dim>(1.)};
    constexpr Formulation form{Formulation::small_strain};

    auto cell_solver{make_cell(nb_grid_pts, lengths, form)};
    auto cell_material{make_cell(nb_grid_pts, lengths, form)};

    constexpr Real Young{2.0e9}, Poisson{.33};
    Matrix_t F_eigen{Fix::F_eigen};
    Matrix_t delF0{Fix::F0};
    const int nb_steps{10};
    constexpr Real cg_tol{1e-8}, newton_tol{1e-5}, equi_tol{1e-8};
    constexpr Dim_t maxiter{10};
    constexpr Verbosity verbose{Verbosity::Silent};

    auto & material_1_material{
        Mat1_t::make(cell_material, "material_1_material", Young, Poisson)};
    auto & material_2_material{
        Mat2_t::make(cell_material, "material_2_material", Young, Poisson)};
    auto & material_1_solver{
        Mat1_t::make(cell_solver, "material_1_solver", Young, Poisson)};

    for (const auto && index_pixel : cell_solver.get_pixels().enumerate()) {
      auto && index{std::get<0>(index_pixel)};
      material_1_solver.add_pixel(index);
    }

    for (const auto && index_pixel : cell_material.get_pixels().enumerate()) {
      auto && index{std::get<0>(index_pixel)};
      if (index == Index_t(cell_material.get_nb_pixels() / 2)) {
        material_2_material.add_pixel(index, F_eigen);
      } else {
        material_1_material.add_pixel(index);
      }
    }

    using Func_t =
        std::function<void(const size_t &, muGrid::TypedFieldBase<Real> &)>;

    // The function which is responsible for assigning eigen strain
    Func_t eigen_func{
        [&cell_material, &F_eigen](const size_t & step,
                                   muGrid::TypedFieldBase<Real> & eval_field) {
          auto && stress_ratio{step + 1};
          auto shape{cell_material.get_strain_shape()};
          auto && eval_field_map{muGrid::FieldMap<Real, Mapping::Mut>(
              eval_field, shape[0], muGrid::IterUnit::SubPt)};
          for (auto && tup : eval_field_map.enumerate_indices()) {
            auto && index{std::get<0>(tup)};
            auto && eval{std::get<1>(tup)};
            if (index == cell_material.get_nb_pixels() / 2) {
              eval -= (stress_ratio) * (F_eigen);
            }
          }
        }};

    cell_material.initialise();
    cell_solver.initialise();

    KrylovSolverCG cg_material{cell_material, cg_tol, maxiter, verbose};
    KrylovSolverCG cg_solver{cell_solver, cg_tol, maxiter, verbose};

    muSpectre::LoadSteps_t delF0s{};
    muSpectre::LoadSteps_t delFs{};
    for (int step{0}; step < nb_steps; ++step) {
      // delFs.push_back((step + 1) * delF0);
      delF0s.push_back(delF0);
      delFs.push_back(delF0);
    }

    auto && ress_material{newton_cg(cell_material, delF0s, cg_material,
                                    newton_tol, equi_tol, verbose,
                                    IsStrainInitialised::False)};

    auto && ress_solver{newton_cg(cell_solver, delFs, cg_solver, newton_tol,
                                  equi_tol, verbose, IsStrainInitialised::False,
                                  eigen_func)};

    int step_out{0};
    for (auto && tup : akantu::zip(ress_solver, ress_material)) {
      auto && res_solver{std::get<0>(tup)};
      auto && res_material{std::get<1>(tup)};
      auto && stress_ratio{step_out + 1};

      auto && stress_solver{res_solver.stress};
      auto && strain_solver{res_solver.grad};
      auto && stress_material{res_material.stress};
      auto && strain_material{res_material.grad};

      auto && diff_stress{
          rel_error(stress_solver, stress_ratio * stress_material)};
      auto && diff_strain{
          rel_error(strain_solver, stress_ratio * strain_material)};

      BOOST_CHECK_LE(diff_strain, cg_tol);
      BOOST_CHECK_LE(diff_stress, cg_tol);
      step_out++;
    }
  }

  BOOST_AUTO_TEST_SUITE_END();
}  // namespace muSpectre

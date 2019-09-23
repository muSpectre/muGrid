/**
 * @file   split_test_laminate_solver.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   19 Oct 2018
 *
 * @brief  Tests for the large-strain/ small-strain, laminate homogenisation
 * algorithm
 *
 * Copyright © 2017 Till Junge
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
#include "tests/libmugrid/test_goodies.hh"

#include "materials/material_linear_elastic1.hh"
#include "materials/material_linear_elastic2.hh"
#include "materials/material_linear_orthotropic.hh"
#include "materials/laminate_homogenisation.hh"
#include "materials/materials_toolbox.hh"
#include "materials/stress_transformations.hh"
#include "libmugrid/field_collection.hh"
#include "libmugrid/iterators.hh"
#include "libmugrid/tensor_algebra.hh"

#include <type_traits>
#include <typeinfo>
#include <boost/mpl/list.hpp>
#include <boost/range/combine.hpp>

namespace muSpectre {
  BOOST_AUTO_TEST_SUITE(laminate_homogenisation);

  template <class Mat_t, Dim_t Dim, Dim_t c>
  struct MaterialFixture;

  /*--------------------------------------------------------------------------*/
  //! Material orthotropic fixture
  template <Dim_t Dim, Dim_t c>
  struct MaterialFixture<MaterialLinearAnisotropic<Dim, Dim>, Dim, c> {
    MaterialFixture() : mat("Name_aniso", aniso_inp_maker()) {}

    std::vector<Real> aniso_inp_maker() {
      std::vector<Real> aniso_inp{};
      switch (Dim) {
      case (twoD): {
        aniso_inp = {c * 1.1, c * 2.1, c * 3.1, c * 4.1, c * 5.1, c * 6.1};
        break;
      }
      case (threeD): {
        aniso_inp = {c * 1.1,  c * 2.1,  c * 3.1,  c * 4.1,  c * 5.1,  c * 6.1,
                     c * 7.1,  c * 8.1,  c * 9.1,  c * 10.1, c * 11.1, c * 12.1,
                     c * 13.1, c * 14.1, c * 15.1, c * 16.1, c * 17.1, c * 18.1,
                     c * 19.1, c * 20.1, c * 21.1};
        break;
      }
      default:
        throw std::runtime_error("Unknown dimension");
      }
      return aniso_inp;
    }

    MaterialLinearAnisotropic<Dim, Dim> mat;
  };

  /*--------------------------------------------------------------------------*/
  //! Material orthotropic fixture
  template <Dim_t Dim, Dim_t c>
  struct MaterialFixture<MaterialLinearOrthotropic<Dim, Dim>, Dim, c> {
    MaterialFixture() : mat("Name_ortho", ortho_inp_maker()) {}
    std::vector<Real> ortho_inp_maker() {
      std::vector<Real> ortho_inp{};
      switch (Dim) {
      case (twoD): {
        ortho_inp = {c * 1.1, c * 2.1, c * 3.1, c * 4.1};
        break;
      }
      case (threeD): {
        ortho_inp = {c * 1.1, c * 2.1, c * 3.1, c * 4.1, c * 5.1,
                     c * 6.1, c * 7.1, c * 8.1, c * 9.1};
        break;
      }
      default:
        throw std::runtime_error("Unknown dimension");
      }
      return ortho_inp;
    }

    MaterialLinearOrthotropic<Dim, Dim> mat;
  };

  /*--------------------------------------------------------------------------*/
  //! Material linear elastic fixture
  template <Dim_t Dim, Dim_t c>
  struct MaterialFixture<MaterialLinearElastic1<Dim, Dim>, Dim, c> {
    MaterialFixture()
        : mat("Name_LinElastic1", young_maker(), poisson_maker()) {}

    Real young_maker() {
      Real lambda{c * 2}, mu{c * 1.5};
      Real young{mu * (3 * lambda + 2 * mu) / (lambda + mu)};
      return young;
    }

    Real poisson_maker() {
      Real lambda{c * 2}, mu{c * 1.5};
      Real poisson{lambda / (2 * (lambda + mu))};
      return poisson;
    }
    MaterialLinearElastic1<Dim, Dim> mat;
  };

  /*--------------------------------------------------------------------------*/
  //! Material pair fixture
  template <class Mat1_t, class Mat2_t, Dim_t Dim, Dim_t c1, Dim_t c2>
  struct MaterialPairFixture {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Vec_t = Eigen::Matrix<Real, Dim, 1>;
    using Stiffness_t = muGrid::T4Mat<Real, Dim>;
    using Serial_stiffness_t = Eigen::Matrix<Real, 2 * Dim - 1, 2 * Dim - 1>;
    using Parallel_stiffness_t = muGrid::T4Mat<Real, (Dim - 1)>;

    using Strain_t = Eigen::Matrix<Real, Dim, Dim>;
    using Serial_strain_t = Eigen::Matrix<Real, 2 * Dim - 1, 1>;
    using Parallel_strain_t = Eigen::Matrix<Real, (Dim - 1) * (Dim - 1), 1>;

    using Stress_t = Strain_t;
    using Serial_stress_t = Serial_strain_t;
    using Paralle_stress_t = Parallel_strain_t;

    using Output_t = std::tuple<Stress_t, Stiffness_t>;
    using Function_t =
        std::function<Output_t(const Eigen::Ref<const Strain_t> &)>;

    // constructor :
    MaterialPairFixture() {
      this->normal_vec = this->normal_vec / this->normal_vec.norm();
    }

    constexpr Dim_t get_dim() { return Dim; }
    MaterialFixture<Mat1_t, Dim, c1> mat_fix_1{};
    MaterialFixture<Mat2_t, Dim, c2> mat_fix_2{};
    Vec_t normal_vec{Vec_t::Random()};
    // Vec_t normal_vec { Vec_t::UnitX() + Vec_t::UnitY() };
    Real ratio{0.5 + 0.5 * static_cast<double>(std::rand() / (RAND_MAX))};
    static constexpr Dim_t fix_dim{Dim};
  };

  /*--------------------------------------------------------------------------*/
  using fix_list = boost::mpl::list<
      MaterialPairFixture<MaterialLinearElastic1<threeD, threeD>,
                          MaterialLinearElastic1<threeD, threeD>, threeD, 1, 1>,
      MaterialPairFixture<MaterialLinearOrthotropic<threeD, threeD>,
                          MaterialLinearOrthotropic<threeD, threeD>, threeD, 1,
                          1>,
      MaterialPairFixture<MaterialLinearAnisotropic<threeD, threeD>,
                          MaterialLinearAnisotropic<threeD, threeD>, threeD, 1,
                          1>,
      MaterialPairFixture<MaterialLinearElastic1<twoD, twoD>,
                          MaterialLinearElastic1<twoD, twoD>, twoD, 1, 1>,
      MaterialPairFixture<MaterialLinearOrthotropic<twoD, twoD>,
                          MaterialLinearOrthotropic<twoD, twoD>, twoD, 1, 1>,
      MaterialPairFixture<MaterialLinearAnisotropic<twoD, twoD>,
                          MaterialLinearAnisotropic<twoD, twoD>, twoD, 1, 1>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(identical_material, Fix, fix_list, Fix) {
    auto & mat1{Fix::mat_fix_1.mat};
    auto & mat2{Fix::mat_fix_2.mat};

    using Strain_t = typename Fix::Strain_t;
    Real amp{1.0};
    Strain_t F{Strain_t::Identity() + amp * Strain_t::Random()};
    Strain_t E{MatTB::convert_strain<StrainMeasure::Gradient,
                                     StrainMeasure::GreenLagrange>(F)};
    using Function_t = typename Fix::Function_t;

    Function_t mat1_evaluate_stress_func{
        [&mat1](const Eigen::Ref<const Strain_t> & strain) {
          return mat1.evaluate_stress_tangent(std::move(strain));
        }};

    Function_t mat2_evaluate_stress_func{
        [&mat2](const Eigen::Ref<const Strain_t> & strain) {
          return mat2.evaluate_stress_tangent(std::move(strain));
        }};

    auto && S_C_lam{
        LamHomogen<Fix::fix_dim, Formulation::small_strain>::
            evaluate_stress_tangent(E, mat1_evaluate_stress_func,
                                    mat2_evaluate_stress_func, Fix::ratio,
                                    Fix::normal_vec, 1e-8, 10)};

    auto && S_lam{std::get<0>(S_C_lam)};
    auto && C_lam{std::get<1>(S_C_lam)};

    auto && S_C_ref{mat1_evaluate_stress_func(E)};
    auto && S_ref{std::get<0>(S_C_ref)};
    auto && C_ref{std::get<1>(S_C_ref)};

    auto && err_S{(S_lam - S_ref).norm() / S_ref.norm()};
    auto && err_C{(C_lam - C_ref).norm() / C_ref.norm()};
    auto && solution_parameters =
        LamHomogen<Fix::fix_dim, Formulation::small_strain>::laminate_solver(
            E, mat1_evaluate_stress_func, mat2_evaluate_stress_func, Fix::ratio,
            Fix::normal_vec, 1e-8, 10);
    auto && iters{std::get<0>(solution_parameters)};
    auto && del_energy{std::get<1>(solution_parameters)};
    BOOST_CHECK_EQUAL(1, iters);
    BOOST_CHECK_LT(std::abs(del_energy), tol);

    auto && S_ref_2 = Matrices::tensmult(C_lam, E);
    auto && err_S_2{((S_lam - S_ref_2).norm()) / S_ref_2.norm()};
    BOOST_CHECK_LT(err_S_2, tol);

    BOOST_CHECK_LT(err_S, tol);
    BOOST_CHECK_LT(err_C, tol);
  };
  // /*--------------------------------------------------------------------------*/

  using fix_list2 = boost::mpl::list<
      MaterialPairFixture<MaterialLinearElastic1<threeD, threeD>,
                          MaterialLinearElastic1<threeD, threeD>, threeD, 7, 3>,
      MaterialPairFixture<MaterialLinearElastic1<threeD, threeD>,
                          MaterialLinearElastic1<threeD, threeD>, threeD, 3, 7>,
      MaterialPairFixture<MaterialLinearElastic1<twoD, twoD>,
                          MaterialLinearElastic1<twoD, twoD>, twoD, 3, 2>,
      MaterialPairFixture<MaterialLinearElastic1<twoD, twoD>,
                          MaterialLinearElastic1<twoD, twoD>, twoD, 2, 3>,
      MaterialPairFixture<MaterialLinearOrthotropic<threeD, threeD>,
                          MaterialLinearOrthotropic<threeD, threeD>, threeD, 3,
                          1>,
      MaterialPairFixture<MaterialLinearOrthotropic<twoD, twoD>,
                          MaterialLinearOrthotropic<twoD, twoD>, twoD, 3, 1>,
      MaterialPairFixture<MaterialLinearAnisotropic<threeD, threeD>,
                          MaterialLinearAnisotropic<threeD, threeD>, threeD, 3,
                          1>,
      MaterialPairFixture<MaterialLinearAnisotropic<twoD, twoD>,
                          MaterialLinearAnisotropic<twoD, twoD>, twoD, 3, 1>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(different_material_linear_elastic, Fix,
                                   fix_list2, Fix) {
    auto & mat1{Fix::mat_fix_1.mat};
    auto & mat2{Fix::mat_fix_2.mat};
    using Strain_t = typename Fix::Strain_t;
    Real amp{1.0};
    Strain_t F{Strain_t::Identity() + amp * Strain_t::Random()};
    Strain_t E{MatTB::convert_strain<StrainMeasure::Gradient,
                                     StrainMeasure::GreenLagrange>(F)};

    using Function_t = typename Fix::Function_t;
    Function_t mat1_evaluate_stress_func{
        [&mat1](const Eigen::Ref<const Strain_t> & strain) {
          return mat1.evaluate_stress_tangent(std::move(strain));
        }};

    Function_t mat2_evaluate_stress_func{
        [&mat2](const Eigen::Ref<const Strain_t> & strain) {
          return mat2.evaluate_stress_tangent(std::move(strain));
        }};

    auto && S_C_lam{
        LamHomogen<Fix::fix_dim, Formulation::small_strain>::
            evaluate_stress_tangent(E, mat1_evaluate_stress_func,
                                    mat2_evaluate_stress_func, Fix::ratio,
                                    Fix::normal_vec, 1e-9, 10)};
    auto && solution_parameters{
        LamHomogen<Fix::fix_dim, Formulation::small_strain>::laminate_solver(
            E, mat1_evaluate_stress_func, mat2_evaluate_stress_func, Fix::ratio,
            Fix::normal_vec, 1e-8, 10)};

    auto && iters{std::get<0>(solution_parameters)};
    auto && del_energy{std::get<1>(solution_parameters)};
    BOOST_CHECK_EQUAL(1, iters);
    BOOST_CHECK_LT(std::abs(del_energy), tol);

    auto && S_lam{std::get<0>(S_C_lam)};
    auto && C_lam{std::get<1>(S_C_lam)};

    auto && S_ref_2{Matrices::tensmult(C_lam, E)};
    auto && err_S_2{((S_lam - S_ref_2).norm()) / S_ref_2.norm()};
    BOOST_CHECK_LT(err_S_2, tol);
  }

  // /*--------------------------------------------------------------------------*/
  // /**
  //  * material linear elastic contrast && inb finite strain
  //  */
  using fix_list3 = boost::mpl::list<
      MaterialPairFixture<MaterialLinearElastic1<threeD, threeD>,
                          MaterialLinearElastic1<threeD, threeD>, threeD, 1, 1>,
      MaterialPairFixture<MaterialLinearElastic1<twoD, twoD>,
                          MaterialLinearElastic1<twoD, twoD>, twoD, 1, 1>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(identical_material_finite_strain, Fix,
                                   fix_list3, Fix) {
    auto & mat1{Fix::mat_fix_1.mat};
    auto & mat2{Fix::mat_fix_2.mat};
    using Strain_t = typename Fix::Strain_t;
    Real amp{1e-1};
    Strain_t F{Strain_t::Identity() + amp * Strain_t::Random()};
    Strain_t E{MatTB::convert_strain<StrainMeasure::Gradient,
                                     StrainMeasure::GreenLagrange>(F)};
    using Function_t = typename Fix::Function_t;

    Function_t mat1_evaluate_stress_func{
        [&mat1](const Eigen::Ref<const Strain_t> & strain) {
          // here the material calculates the stress in its own traits
          // conversion from strain to stress:
          // here we extract the type of strain and stress that mat1 deals
          // with
          using traits =
              typename std::remove_reference_t<decltype(mat1)>::traits;

          auto && mat1_strain =
              MatTB::convert_strain<StrainMeasure::Gradient,
                                    traits::strain_measure>(strain);

          auto && mat_stress_tgt =
              mat1.evaluate_stress_tangent(std::move(mat1_strain));

          auto && ret_P_K =
              MatTB::PK1_stress<traits::stress_measure, traits::strain_measure>(
                  std::move(strain), std::move(std::get<0>(mat_stress_tgt)),
                  std::move(std::get<1>(mat_stress_tgt)));

          return std::move(ret_P_K);
        }};

    Function_t mat2_evaluate_stress_func{
        [&mat2](const Eigen::Ref<const Strain_t> & strain) {
          // here the material calculates the stress in its own traits
          // conversion from strain to stress:
          // here we extract the type of strain and stress that mat2 deals
          // with
          using traits =
              typename std::remove_reference_t<decltype(mat2)>::traits;

          auto && mat2_strain =
              MatTB::convert_strain<StrainMeasure::Gradient,
                                    traits::strain_measure>(strain);

          auto && mat_stress_tgt =
              mat2.evaluate_stress_tangent(std::move(mat2_strain));

          auto && ret_P_K =
              MatTB::PK1_stress<traits::stress_measure, traits::strain_measure>(
                  std::move(strain), std::move(std::get<0>(mat_stress_tgt)),
                  std::move(std::get<1>(mat_stress_tgt)));

          return std::move(ret_P_K);
        }};

    auto && P_K_lam{
        LamHomogen<Fix::fix_dim, Formulation::finite_strain>::
            evaluate_stress_tangent(F, mat1_evaluate_stress_func,
                                    mat2_evaluate_stress_func, Fix::ratio,
                                    Fix::normal_vec, 1e-9, 10)};
    auto && solution_parameters{
        LamHomogen<Fix::fix_dim, Formulation::finite_strain>::laminate_solver(
            F, mat1_evaluate_stress_func, mat2_evaluate_stress_func, Fix::ratio,
            Fix::normal_vec, 1e-8, 10)};

    auto && iter{std::get<0>(solution_parameters)};
    BOOST_CHECK_EQUAL(iter, 1);

    auto && del_energy{std::get<1>(solution_parameters)};
    BOOST_CHECK_LT(std::abs(del_energy), tol);

    auto && P_lam{std::get<0>(P_K_lam)};
    auto && K_lam{std::get<1>(P_K_lam)};

    auto && S_C_lam{
        MatTB::PK2_stress<StressMeasure::PK1, StrainMeasure::Gradient>(F, P_lam,
                                                                       K_lam)};
    auto && S_lam{std::get<0>(S_C_lam)};
    auto && C_lam{std::get<1>(S_C_lam)};
    auto && P_ref_1{std::get<0>(mat1_evaluate_stress_func(F))};

    auto && S_ref_1{
        MatTB::PK2_stress<StressMeasure::PK1, StrainMeasure::Gradient>(
            F, P_ref_1)};
    auto && S_ref_2{muGrid::Matrices::tensmult(C_lam, E)};

    auto err_S_1{((S_lam - S_ref_1).norm()) / S_ref_1.norm()};
    auto err_S_2{((S_lam - S_ref_2).norm()) / S_ref_2.norm()};

    BOOST_CHECK_LT(err_S_1, tol);
    BOOST_CHECK_LT(err_S_2, tol);
  }

  /**
   * material linear elastic contrast && finite strain
   */
  using fix_list4 = boost::mpl::list<
      MaterialPairFixture<MaterialLinearElastic1<threeD, threeD>,
                          MaterialLinearElastic1<threeD, threeD>, threeD, 1, 3>,
      MaterialPairFixture<MaterialLinearOrthotropic<threeD, threeD>,
                          MaterialLinearOrthotropic<threeD, threeD>, threeD, 3,
                          7>,
      MaterialPairFixture<MaterialLinearAnisotropic<threeD, threeD>,
                          MaterialLinearAnisotropic<threeD, threeD>, threeD, 3,
                          7>,
      MaterialPairFixture<MaterialLinearElastic1<twoD, twoD>,
                          MaterialLinearElastic1<twoD, twoD>, twoD, 3, 1>,
      MaterialPairFixture<MaterialLinearElastic1<twoD, twoD>,
                          MaterialLinearElastic1<twoD, twoD>, twoD, 3, 7>,
      MaterialPairFixture<MaterialLinearOrthotropic<twoD, twoD>,
                          MaterialLinearOrthotropic<twoD, twoD>, twoD, 3, 7>,
      MaterialPairFixture<MaterialLinearAnisotropic<twoD, twoD>,
                          MaterialLinearAnisotropic<twoD, twoD>, twoD, 3, 7>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(different_material_finite_strain, Fix,
                                   fix_list4, Fix) {
    auto & mat1{Fix::mat_fix_1.mat};
    auto & mat2{Fix::mat_fix_2.mat};
    using Stiffness_t = typename Fix::Stiffness_t;
    using Strain_t = typename Fix::Strain_t;
    using Stress_t = Strain_t;

    Real amp{1e-1};
    Real amp_del{1e-6};
    Strain_t F{Strain_t::Identity() + amp * Strain_t::Random()};
    Strain_t del_F{amp_del * Strain_t::Random()};

    Strain_t F_del{F + del_F};
    Strain_t E{MatTB::convert_strain<StrainMeasure::Gradient,
                                     StrainMeasure::GreenLagrange>(F)};

    Strain_t E_del{
        MatTB::convert_strain<StrainMeasure::Gradient,
                              StrainMeasure::GreenLagrange>(F + del_F)};
    auto && del_E{E_del - E};

    using Function_t = typename Fix::Function_t;
    using Mat_ret_t = typename std::tuple<Stress_t, Stiffness_t>;
    Function_t mat1_evaluate_stress_func{
        [&mat1](const Eigen::Ref<const Strain_t> & strain) {
          // here the material calculates the stress in its own traits
          // conversion from strain to stress:
          // here we extract the type of strain and stress that mat1 deals
          // with
          using traits =
              typename std::remove_reference_t<decltype(mat1)>::traits;

          auto && mat1_strain =
              MatTB::convert_strain<StrainMeasure::Gradient,
                                    traits::strain_measure>(strain);

          auto && mat_stress_tgt =
              mat1.evaluate_stress_tangent(std::move(mat1_strain));

          Mat_ret_t ret_P_K =
              MatTB::PK1_stress<traits::stress_measure, traits::strain_measure>(
                  std::move(strain), std::move(std::get<0>(mat_stress_tgt)),
                  std::move(std::get<1>(mat_stress_tgt)));

          return ret_P_K;
        }};

    Function_t mat2_evaluate_stress_func{
        [&mat2](const Eigen::Ref<const Strain_t> & strain) {
          // here the material calculates the stress in its own traits
          // conversion from strain to stress:
          // here we extract the type of strain and stress that mat2 deals
          // with
          using traits =
              typename std::remove_reference_t<decltype(mat2)>::traits;

          auto && mat2_strain =
              MatTB::convert_strain<StrainMeasure::Gradient,
                                    traits::strain_measure>(strain);

          auto && mat_stress_tgt =
              mat2.evaluate_stress_tangent(std::move(mat2_strain));

          Mat_ret_t ret_P_K =
              MatTB::PK1_stress<traits::stress_measure, traits::strain_measure>(
                  std::move(strain), std::move(std::get<0>(mat_stress_tgt)),
                  std::move(std::get<1>(mat_stress_tgt)));

          return ret_P_K;
        }};

    auto && P_K_lam_del{
        LamHomogen<Fix::fix_dim, Formulation::finite_strain>::
            evaluate_stress_tangent(F_del, mat1_evaluate_stress_func,
                                    mat2_evaluate_stress_func, Fix::ratio,
                                    Fix::normal_vec, 1e-9, 100)};
    auto && P_K_lam{
        LamHomogen<Fix::fix_dim, Formulation::finite_strain>::
            evaluate_stress_tangent(F, mat1_evaluate_stress_func,
                                    mat2_evaluate_stress_func, Fix::ratio,
                                    Fix::normal_vec, 1e-9, 1000)};
    auto && solution_parameters{
        LamHomogen<Fix::fix_dim, Formulation::finite_strain>::laminate_solver(
            F, mat1_evaluate_stress_func, mat2_evaluate_stress_func, Fix::ratio,
            Fix::normal_vec, 1e-9, 1000)};

    auto && del_energy{std::get<1>(solution_parameters)};
    BOOST_CHECK_LT(std::abs(del_energy), 1e-9);

    auto && P_lam{std::get<0>(P_K_lam)};
    auto && K_lam{std::get<1>(P_K_lam)};

    auto && P_lam_del{std::get<0>(P_K_lam_del)};
    auto && K_lam_del{std::get<1>(P_K_lam_del)};

    auto && S_C_lam{
        MatTB::PK2_stress<StressMeasure::PK1, StrainMeasure::Gradient>(F, P_lam,
                                                                       K_lam)};
    auto && S_C_lam_del{
        MatTB::PK2_stress<StressMeasure::PK1, StrainMeasure::Gradient>(
            F + del_F, P_lam_del, K_lam_del)};

    auto && S_lam{std::get<0>(S_C_lam)};
    auto && C_lam{std::get<1>(S_C_lam)};

    auto && S_lam_del{std::get<0>(S_C_lam_del)};

    auto && del_S{muGrid::Matrices::tensmult(C_lam, del_E)};

    Real err_S{(S_lam_del - S_lam - del_S).norm() / S_lam.norm()};
    BOOST_CHECK_LT(err_S, amp);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muSpectre

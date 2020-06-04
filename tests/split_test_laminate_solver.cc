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

#include "materials/material_linear_elastic2.hh"
#include "materials/material_linear_orthotropic.hh"
#include "materials/laminate_homogenisation.hh"

#include <libmugrid/field_collection.hh>
#include <libmugrid/iterators.hh>
#include <libmugrid/tensor_algebra.hh>

#include <type_traits>
#include <typeinfo>
#include <boost/mpl/list.hpp>
#include <boost/range/combine.hpp>

namespace muSpectre {

  using muGrid::testGoodies::rel_error;

  BOOST_AUTO_TEST_SUITE(laminate_homogenisation);

  constexpr Index_t OneContrast{
      1};  //! The values used to introduce material contrast
  constexpr Index_t TwoContrast{
      2};  //! The values used to introduce material contrast
  constexpr Index_t ThreeContrast{
      3};  //! The values used to introduce material contrast
  constexpr Index_t SevenContrast{
      7};  //! The values used to introduce material contrast

  constexpr Real Tol{1e-8};
  constexpr Real MaxIter{100};

  template <class Mat_t, Index_t Dim, Index_t c>
  struct MaterialFixture;

  /*--------------------------------------------------------------------------*/
  //! Material orthotropic fixture
  template <Index_t Dim, Index_t c>
  struct MaterialFixture<MaterialLinearAnisotropic<Dim>, Dim, c> {
    MaterialFixture() : mat("Name_aniso", Dim, OneQuadPt, aniso_inp_maker()) {}

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
        throw muGrid::RuntimeError("Unknown dimension");
      }
      return aniso_inp;
    }

    MaterialLinearAnisotropic<Dim> mat;
  };

  /*--------------------------------------------------------------------------*/
  //! Material orthotropic fixture
  template <Index_t Dim, Index_t c>
  struct MaterialFixture<MaterialLinearOrthotropic<Dim>, Dim, c> {
    MaterialFixture() : mat("Name_ortho", Dim, OneQuadPt, ortho_inp_maker()) {}
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
        throw muGrid::RuntimeError("Unknown dimension");
      }
      return ortho_inp;
    }

    MaterialLinearOrthotropic<Dim> mat;
  };

  /*--------------------------------------------------------------------------*/
  //! Material linear elastic fixture
  template <Index_t Dim, Index_t c>
  struct MaterialFixture<MaterialLinearElastic1<Dim>, Dim, c> {
    MaterialFixture()
        : mat("Name_LinElastic1", Dim, OneQuadPt, young_maker(),
              poisson_maker()) {}

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
    MaterialLinearElastic1<Dim> mat;
  };

  /*--------------------------------------------------------------------------*/
  //! Material pair fixture
  template <class Mat1_t, class Mat2_t, Index_t Dim, Index_t c1, Index_t c2>
  struct MaterialPairFixture {
    using Vec_t = Eigen::Matrix<Real, Dim, 1>;
    using Strain_t = Eigen::Matrix<Real, Dim, Dim>;
    using Stress_t = Eigen::Matrix<Real, Dim, Dim>;
    using Stiffness_t = muGrid::T4Mat<Real, Dim>;
    using Output_t = std::tuple<Stress_t, Stiffness_t>;
    using Function_t =
        std::function<Output_t(const Eigen::Ref<const Strain_t> &)>;

    // constructor :
    MaterialPairFixture()
        : normal_vec_holder{std::make_unique<Vec_t>(Vec_t::Random())},
          /*normal_vec_holder{std::make_unique<Vec_t>(Vec_t::Random())}*/
          normal_vec{*normal_vec_holder} {
      this->normal_vec = this->normal_vec / this->normal_vec.norm();
    }

    constexpr Index_t get_dim() { return Dim; }

   protected:
    MaterialFixture<Mat1_t, Dim, c1> mat_fix_1{};
    MaterialFixture<Mat2_t, Dim, c2> mat_fix_2{};

    std::unique_ptr<Vec_t> normal_vec_holder;
    Vec_t & normal_vec;

    Real ratio{0.5 + 0.5 * static_cast<double>(std::rand() / (RAND_MAX))};
    static constexpr Index_t fix_dim{Dim};
  };

  /*--------------------------------------------------------------------------*/
  using fix_list =
      boost::mpl::list<MaterialPairFixture<MaterialLinearElastic1<threeD>,
                                           MaterialLinearElastic1<threeD>,
                                           threeD, OneContrast, OneContrast>,
                       MaterialPairFixture<MaterialLinearOrthotropic<threeD>,
                                           MaterialLinearOrthotropic<threeD>,
                                           threeD, OneContrast, OneContrast>,
                       MaterialPairFixture<MaterialLinearAnisotropic<threeD>,
                                           MaterialLinearAnisotropic<threeD>,
                                           threeD, OneContrast, OneContrast>,
                       MaterialPairFixture<MaterialLinearElastic1<twoD>,
                                           MaterialLinearElastic1<twoD>, twoD,
                                           OneContrast, OneContrast>,
                       MaterialPairFixture<MaterialLinearOrthotropic<twoD>,
                                           MaterialLinearOrthotropic<twoD>,
                                           twoD, OneContrast, OneContrast>,
                       MaterialPairFixture<MaterialLinearAnisotropic<twoD>,
                                           MaterialLinearAnisotropic<twoD>,
                                           twoD, OneContrast, OneContrast>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(identical_material, Fix, fix_list, Fix) {
    auto & mat1{Fix::mat_fix_1.mat};
    auto & mat2{Fix::mat_fix_2.mat};
    constexpr Real OneRatio{1.0};
    using Strain_t = typename Fix::Strain_t;
    Real amp{1.0};
    Strain_t F{Strain_t::Identity() + amp * Strain_t::Random()};
    Strain_t E{MatTB::convert_strain<StrainMeasure::Gradient,
                                     StrainMeasure::GreenLagrange>(F)};
    using Function_t = typename Fix::Function_t;

    Function_t mat1_evaluate_stress_func = [&mat1](
        const Eigen::Ref<const Strain_t> & strain)
        -> std::tuple<typename Fix::Stress_t, typename Fix::Stiffness_t> {
      return mat1.evaluate_stress_tangent(std::move(strain), OneRatio);
    };

    Function_t mat2_evaluate_stress_func = [&mat2](
        const Eigen::Ref<const Strain_t> & strain)
        -> std::tuple<typename Fix::Stress_t, typename Fix::Stiffness_t> {
      return mat2.evaluate_stress_tangent(std::move(strain), OneRatio);
    };

    auto && S_C_lam{
        LamHomogen<Fix::fix_dim, Formulation::small_strain>::
            evaluate_stress_tangent(E, mat1_evaluate_stress_func,
                                    mat2_evaluate_stress_func, Fix::ratio,
                                    Fix::normal_vec, Tol, MaxIter)};

    auto && S_lam{std::get<0>(S_C_lam)};
    auto && C_lam{std::get<1>(S_C_lam)};

    auto && S_C_ref{mat1_evaluate_stress_func(E)};
    auto && S_ref{std::get<0>(S_C_ref)};
    auto && C_ref{std::get<1>(S_C_ref)};

    Real err_S{rel_error(S_lam, S_ref)};
    Real err_C{rel_error(C_lam, C_ref)};

    BOOST_CHECK_LT(err_S, tol);
    BOOST_CHECK_LT(err_C, tol);

    auto && solution_parameters =
        LamHomogen<Fix::fix_dim, Formulation::small_strain>::laminate_solver(
            E, mat1_evaluate_stress_func, mat2_evaluate_stress_func, Fix::ratio,
            Fix::normal_vec, Tol, MaxIter);
    auto && iters{std::get<0>(solution_parameters)};
    auto && del_energy{std::get<1>(solution_parameters)};

    BOOST_CHECK_EQUAL(1, iters);
    BOOST_CHECK_LT(std::abs(del_energy), tol);

    auto && S_ref_2 = Matrices::tensmult(C_lam, E);
    Real err_S_2{rel_error(S_lam, S_ref_2)};

    BOOST_CHECK_LT(err_S_2, tol);
  };
  /*--------------------------------------------------------------------------*/

  using fix_list2 = boost::mpl::list<
      MaterialPairFixture<MaterialLinearElastic1<threeD>,
                          MaterialLinearElastic1<threeD>, threeD, SevenContrast,
                          ThreeContrast>,
      MaterialPairFixture<MaterialLinearElastic1<threeD>,
                          MaterialLinearElastic1<threeD>, threeD, ThreeContrast,
                          SevenContrast>,
      MaterialPairFixture<MaterialLinearElastic1<twoD>,
                          MaterialLinearElastic1<twoD>, twoD, ThreeContrast,
                          TwoContrast>,
      MaterialPairFixture<MaterialLinearElastic1<twoD>,
                          MaterialLinearElastic1<twoD>, twoD, TwoContrast,
                          ThreeContrast>,
      MaterialPairFixture<MaterialLinearOrthotropic<threeD>,
                          MaterialLinearOrthotropic<threeD>, threeD,
                          ThreeContrast, OneContrast>,
      MaterialPairFixture<MaterialLinearOrthotropic<twoD>,
                          MaterialLinearOrthotropic<twoD>, twoD, ThreeContrast,
                          OneContrast>,
      MaterialPairFixture<MaterialLinearAnisotropic<threeD>,
                          MaterialLinearAnisotropic<threeD>, threeD,
                          ThreeContrast, OneContrast>,
      MaterialPairFixture<MaterialLinearAnisotropic<twoD>,
                          MaterialLinearAnisotropic<twoD>, twoD, ThreeContrast,
                          OneContrast>>;

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
          return mat1.evaluate_stress_tangent(std::move(strain), 1);
        }};

    Function_t mat2_evaluate_stress_func{
        [&mat2](const Eigen::Ref<const Strain_t> & strain) {
          return mat2.evaluate_stress_tangent(std::move(strain), 1);
        }};

    auto && S_C_lam{
        LamHomogen<Fix::fix_dim, Formulation::small_strain>::
            evaluate_stress_tangent(E, mat1_evaluate_stress_func,
                                    mat2_evaluate_stress_func, Fix::ratio,
                                    Fix::normal_vec, 1e-9, MaxIter)};
    auto && solution_parameters{
        LamHomogen<Fix::fix_dim, Formulation::small_strain>::laminate_solver(
            E, mat1_evaluate_stress_func, mat2_evaluate_stress_func, Fix::ratio,
            Fix::normal_vec, Tol, MaxIter)};

    auto && iters{std::get<0>(solution_parameters)};
    auto && del_energy{std::get<1>(solution_parameters)};
    BOOST_CHECK_EQUAL(1, iters);
    BOOST_CHECK_LT(std::abs(del_energy), tol);

    auto && S_lam{std::get<0>(S_C_lam)};
    auto && C_lam{std::get<1>(S_C_lam)};

    auto && S_ref_2{Matrices::tensmult(C_lam, E)};

    Real err_S_2{rel_error(S_lam, S_ref_2)};
    BOOST_CHECK_LT(err_S_2, tol);
  }

  /*--------------------------------------------------------------------------*/
  /**
   * material linear elastic contrast && inb finite strain
   */
  using fix_list3 =
      boost::mpl::list<MaterialPairFixture<MaterialLinearElastic1<threeD>,
                                           MaterialLinearElastic1<threeD>,
                                           threeD, OneContrast, OneContrast>,
                       MaterialPairFixture<MaterialLinearElastic1<twoD>,
                                           MaterialLinearElastic1<twoD>, twoD,
                                           OneContrast, OneContrast>>;

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
              mat1.evaluate_stress_tangent(std::move(mat1_strain), 1);

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
              mat2.evaluate_stress_tangent(std::move(mat2_strain), 1);

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
                                    Fix::normal_vec, 1e-9, MaxIter)};
    auto && solution_parameters{
        LamHomogen<Fix::fix_dim, Formulation::finite_strain>::laminate_solver(
            F, mat1_evaluate_stress_func, mat2_evaluate_stress_func, Fix::ratio,
            Fix::normal_vec, Tol, MaxIter)};

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

    Real err_S_1{rel_error(S_lam, S_ref_1)};
    Real err_S_2{rel_error(S_lam, S_ref_2)};

    BOOST_CHECK_LT(err_S_1, tol);
    BOOST_CHECK_LT(err_S_2, tol);
  }

  /**
   * material linear elastic contrast && finite strain
   */
  using fix_list4 = boost::mpl::list<
      MaterialPairFixture<MaterialLinearElastic1<threeD>,
                          MaterialLinearElastic1<threeD>, threeD, OneContrast,
                          ThreeContrast>,
      MaterialPairFixture<MaterialLinearOrthotropic<threeD>,
                          MaterialLinearOrthotropic<threeD>, threeD,
                          ThreeContrast, SevenContrast>,
      MaterialPairFixture<MaterialLinearAnisotropic<threeD>,
                          MaterialLinearAnisotropic<threeD>, threeD,
                          ThreeContrast, SevenContrast>,
      MaterialPairFixture<MaterialLinearElastic1<twoD>,
                          MaterialLinearElastic1<twoD>, twoD, ThreeContrast,
                          OneContrast>,
      MaterialPairFixture<MaterialLinearElastic1<twoD>,
                          MaterialLinearElastic1<twoD>, twoD, ThreeContrast,
                          SevenContrast>,
      MaterialPairFixture<MaterialLinearOrthotropic<twoD>,
                          MaterialLinearOrthotropic<twoD>, twoD, ThreeContrast,
                          SevenContrast>,
      MaterialPairFixture<MaterialLinearAnisotropic<twoD>,
                          MaterialLinearAnisotropic<twoD>, twoD, ThreeContrast,
                          SevenContrast>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(different_material_finite_strain, Fix,
                                   fix_list4, Fix) {
    auto & mat1{Fix::mat_fix_1.mat};
    auto & mat2{Fix::mat_fix_2.mat};
    using Stiffness_t = typename Fix::Stiffness_t;
    using Strain_t = typename Fix::Strain_t;
    using Stress_t = Strain_t;

    constexpr Real OneRatio{1.0};
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
              mat1.evaluate_stress_tangent(std::move(mat1_strain), OneRatio);

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
              mat2.evaluate_stress_tangent(std::move(mat2_strain), 1);

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
                                    Fix::normal_vec, 1e-9, MaxIter)};
    auto && P_K_lam{
        LamHomogen<Fix::fix_dim, Formulation::finite_strain>::
            evaluate_stress_tangent(F, mat1_evaluate_stress_func,
                                    mat2_evaluate_stress_func, Fix::ratio,
                                    Fix::normal_vec, 1e-9, MaxIter)};
    auto && solution_parameters{
        LamHomogen<Fix::fix_dim, Formulation::finite_strain>::laminate_solver(
            F, mat1_evaluate_stress_func, mat2_evaluate_stress_func, Fix::ratio,
            Fix::normal_vec, 1e-9, MaxIter)};

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

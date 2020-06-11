/**
 * @file   test_material_hyper_elasto_plastic2.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *         Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *
 * @date   09 Jul 2019
 *
 * @brief  Tests for the large-strain Simo-type plastic law implemented
 *         using MaterialMuSpectre. Copy of the tests
 *         test_material_hyper_elasto_plastic1.cc which are slightly modified to
 *         fit to the changes in material_hyper_elasto_plastic2.
 *
 * Copyright © 2018 Till Junge
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
 * * Boston, MA 02111-1307, USA.
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

#include <materials/stress_transformations_Kirchhoff.hh>
#include <materials/material_hyper_elasto_plastic2.hh>
#include <materials/materials_toolbox.hh>

#include <boost/mpl/list.hpp>

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(material_hyper_elasto_plastic_2);

  template <class Mat_t>
  struct MaterialFixture {
    using Mat = Mat_t;

    constexpr static Dim_t mdim() { return Mat_t::MaterialDimension(); }
    constexpr static Dim_t sdim() { return mdim(); }
    constexpr static Dim_t NbQuadPts() { return 2; }

    MaterialFixture() : mat("Name", mdim(), NbQuadPts()) {}

    Mat_t mat;
  };

  template <class Mat1_t, class Mat2_t>
  struct MaterialPairFixture {
    using Mat1 = Mat1_t;
    using Mat2 = Mat2_t;
    constexpr static Dim_t mdim() { return Mat1_t::MaterialDimension(); }
    constexpr static Dim_t sdim() { return mdim(); }
    constexpr static Dim_t NbQuadPts() { return 2; }

    MaterialPairFixture()
        : mat1("Name", mdim(), NbQuadPts()), mat2("Name", mdim(), NbQuadPts()) {
    }

    Mat1_t mat1;
    Mat2_t mat2;
  };

  /* ---------------------------------------------------------------------- */
  using mats =
      boost::mpl::list<MaterialFixture<MaterialHyperElastoPlastic2<twoD>>,
                       MaterialFixture<MaterialHyperElastoPlastic2<threeD>>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_constructor, Fix, mats, Fix) {
    BOOST_CHECK_EQUAL("Name", Fix::mat.get_name());
    auto & mat{Fix::mat};
    auto mdim{Fix::mdim()};
    BOOST_CHECK_EQUAL(mdim, mat.MaterialDimension());
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_evaluate_stress, Fix, mats, Fix) {
    // This test uses precomputed reference values (computed using
    // elasto-plasticity.py) for the 3d case only

    // need higher tol because of printout precision of reference solutions
    constexpr Real hi_tol{1e-8};
    constexpr Dim_t mdim{Fix::mdim()}, sdim{Fix::sdim()};
    constexpr bool has_precomputed_values{(mdim == sdim) && (mdim == threeD)};
    constexpr bool verbose{false};
    using Strain_t = Eigen::Matrix<Real, mdim, mdim>;

    // initialise pixel
    constexpr static Real K{.833};       // bulk modulus
    constexpr static Real mu{.386};      // shear modulus
    constexpr static Real H{.004};       // hardening modulus
    constexpr static Real tau_y0{.003};  // initial yield stress
    constexpr static Real young{MatTB::convert_elastic_modulus<
        ElasticModulus::Young, ElasticModulus::Bulk, ElasticModulus::Shear>(
        K, mu)};
    constexpr static Real poisson{MatTB::convert_elastic_modulus<
        ElasticModulus::Poisson, ElasticModulus::Bulk, ElasticModulus::Shear>(
        K, mu)};
    constexpr static Real lambda{MatTB::convert_elastic_modulus<
        ElasticModulus::lambda, ElasticModulus::Bulk, ElasticModulus::Shear>(
        K, mu)};
    Fix::mat.add_pixel({0}, young, poisson, tau_y0, H);
    Fix::mat.initialise();

    // create statefields
    muGrid::LocalFieldCollection::SubPtMap_t map{};
    map[QuadPtTag] = Fix::NbQuadPts();
    muGrid::LocalFieldCollection coll{sdim, map};
    coll.add_pixel({0});
    coll.initialise();

    muGrid::MappedT2StateField<Real, Mapping::Mut, mdim, IterUnit::SubPt>
      F_{"previous gradient", coll, QuadPtTag};
    muGrid::MappedT2StateField<Real, Mapping::Mut, mdim, IterUnit::SubPt>
        be_{"previous elastic strain", coll, QuadPtTag};
    muGrid::MappedScalarStateField<Real, Mapping::Mut, IterUnit::SubPt>
        eps_{"plastic flow", coll, QuadPtTag};

    auto & F_prev{F_.get_map()};
    F_prev[0].current() = Strain_t::Identity();
    auto & be_prev{be_.get_map()};
    be_prev[0].current() = Strain_t::Identity();
    auto & eps_prev{eps_.get_map()};
    eps_prev[0].current() = 0;
    // elastic deformation
    Strain_t F{Strain_t::Identity()};
    F(0, 1) = 1e-5;

    F_.get_state_field().cycle();
    be_.get_state_field().cycle();
    eps_.get_state_field().cycle();
    Strain_t stress{Fix::mat.evaluate_stress(
        F, F_prev[0], be_prev[0], eps_prev[0], lambda, mu, tau_y0, H)};

    if (has_precomputed_values) {
      Strain_t tau_ref{};
      tau_ref << 1.92999522e-11, 3.86000000e-06, 0.00000000e+00, 3.86000000e-06,
          -1.93000510e-11, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          -2.95741950e-17;
      Real error{(tau_ref - stress).norm() / tau_ref.norm()};
      BOOST_CHECK_LT(error, hi_tol);

      Strain_t be_ref{};
      be_ref << 1.00000000e+00, 1.00000000e-05, 0.00000000e+00, 1.00000000e-05,
          1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          1.00000000e+00;
      error = (be_ref - be_prev[0].current()).norm() / be_ref.norm();
      BOOST_CHECK_LT(error, hi_tol);

      Real ep_ref{0};
      error = ep_ref - eps_prev[0].current();
      BOOST_CHECK_LT(error, hi_tol);
    }
    if (verbose) {
      std::cout << "τ  =" << std::endl << stress << std::endl;
      std::cout << "F  =" << std::endl << F << std::endl;
      std::cout << "Fₜ =" << std::endl << F_prev[0].current() << std::endl;
      std::cout << "bₑ =" << std::endl << be_prev[0].current() << std::endl;
      std::cout << "εₚ =" << std::endl << eps_prev[0].current() << std::endl;
    }
    F_.get_state_field().cycle();
    be_.get_state_field().cycle();
    eps_.get_state_field().cycle();

    // plastic deformation
    F(0, 1) = .2;
    stress = Fix::mat.evaluate_stress(F, F_prev[0], be_prev[0], eps_prev[0],
                                      lambda, mu, tau_y0, H);

    if (has_precomputed_values) {
      Strain_t tau_ref{};
      tau_ref << 1.98151335e-04, 1.98151335e-03, 0.00000000e+00, 1.98151335e-03,
          -1.98151335e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          1.60615155e-16;
      Real error{(tau_ref - stress).norm() / tau_ref.norm()};
      BOOST_CHECK_LT(error, hi_tol);

      Strain_t be_ref{};
      be_ref << 1.00052666, 0.00513348, 0., 0.00513348, 0.99949996, 0., 0., 0.,
          1.;
      error = (be_ref - be_prev[0].current()).norm() / be_ref.norm();
      BOOST_CHECK_LT(error, hi_tol);

      Real ep_ref{0.11229988};
      error = (ep_ref - eps_prev[0].current()) / ep_ref;
      BOOST_CHECK_LT(error, hi_tol);
    }
    if (verbose) {
      std::cout << "Post Cycle" << std::endl;
      std::cout << "τ  =" << std::endl
                << stress << std::endl
                << "F  =" << std::endl
                << F << std::endl
                << "Fₜ =" << std::endl
                << F_prev[0].current() << std::endl
                << "bₑ =" << std::endl
                << be_prev[0].current() << std::endl
                << "εₚ =" << std::endl
                << eps_prev[0].current() << std::endl;
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_evaluate_stiffness, Fix, mats, Fix) {
    // This test uses precomputed reference values (computed using
    // elasto-plasticity.py) for the 3d case only

    // need higher tol because of printout precision of reference solutions
    constexpr Real hi_tol{2e-7};
    constexpr Dim_t mdim{Fix::mdim()}, sdim{Fix::sdim()};
    constexpr bool has_precomputed_values{(mdim == sdim) && (mdim == threeD)};
    constexpr bool verbose{has_precomputed_values && false};
    using Strain_t = Eigen::Matrix<Real, mdim, mdim>;
    using Stiffness_t = muGrid::T4Mat<Real, mdim>;

    // initialise pixel
    constexpr static Real K{.833};       // bulk modulus
    constexpr static Real mu{.386};      // shear modulus
    constexpr static Real H{.004};       // hardening modulus
    constexpr static Real tau_y0{.003};  // initial yield stress
    constexpr static Real young{MatTB::convert_elastic_modulus<
        ElasticModulus::Young, ElasticModulus::Bulk, ElasticModulus::Shear>(
        K, mu)};
    constexpr static Real poisson{MatTB::convert_elastic_modulus<
        ElasticModulus::Poisson, ElasticModulus::Bulk, ElasticModulus::Shear>(
        K, mu)};
    constexpr static Real lambda{MatTB::convert_elastic_modulus<
        ElasticModulus::lambda, ElasticModulus::Bulk, ElasticModulus::Shear>(
        K, mu)};
    Fix::mat.add_pixel({0}, young, poisson, tau_y0, H);
    Fix::mat.initialise();

    // create statefields
    muGrid::LocalFieldCollection coll{sdim};
    coll.set_nb_sub_pts(QuadPtTag, Fix::NbQuadPts());
    coll.add_pixel({0});
    coll.initialise();

    muGrid::MappedT2StateField<Real, Mapping::Mut, mdim, IterUnit::SubPt>
      F_{"previous gradient", coll, QuadPtTag};
    muGrid::MappedT2StateField<Real, Mapping::Mut, mdim, IterUnit::SubPt>
        be_{"previous elastic strain", coll, QuadPtTag};
    muGrid::MappedScalarStateField<Real, Mapping::Mut, IterUnit::SubPt>
        eps_{"plastic flow", coll, QuadPtTag};

    auto & F_prev{F_.get_map()};
    F_prev[0].current() = Strain_t::Identity();
    auto & be_prev{be_.get_map()};
    be_prev[0].current() = Strain_t::Identity();
    auto & eps_prev{eps_.get_map()};
    eps_prev[0].current() = 0;
    // elastic deformation
    Strain_t F{Strain_t::Identity()};
    F(0, 1) = 1e-5;

    F_.get_state_field().cycle();
    be_.get_state_field().cycle();
    eps_.get_state_field().cycle();
    Strain_t stress{};
    Stiffness_t stiffness{};

    std::tie(stress, stiffness) = Fix::mat.evaluate_stress_tangent(
        F, F_prev[0], be_prev[0], eps_prev[0], lambda, mu, tau_y0, H, K);

    if (has_precomputed_values) {
      Strain_t tau_ref{};
      tau_ref << 1.92999522e-11, 3.86000000e-06, 0.00000000e+00, 3.86000000e-06,
          -1.93000510e-11, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          -2.95741950e-17;
      Real error{(tau_ref - stress).norm() / tau_ref.norm()};
      BOOST_CHECK_LT(error, hi_tol);

      Strain_t be_ref{};
      be_ref << 1.00000000e+00, 1.00000000e-05, 0.00000000e+00, 1.00000000e-05,
          1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          1.00000000e+00;
      error = (be_ref - be_prev[0].current()).norm() / be_ref.norm();
      BOOST_CHECK_LT(error, hi_tol);

      Real ep_ref{0};
      error = ep_ref - eps_prev[0].current();
      BOOST_CHECK_LT(error, hi_tol);

      Stiffness_t temp;
      temp << 1.34766667e+00, 3.86000000e-06, 0.00000000e+00, -3.86000000e-06,
          5.75666667e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          5.75666667e-01, -3.61540123e-17, 3.86000000e-01, 0.00000000e+00,
          3.86000000e-01, 7.12911684e-17, 0.00000000e+00, 0.00000000e+00,
          0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          3.86000000e-01, 0.00000000e+00, 0.00000000e+00, -1.93000000e-06,
          3.86000000e-01, 1.93000000e-06, 0.00000000e+00, -3.61540123e-17,
          3.86000000e-01, 0.00000000e+00, 3.86000000e-01, 7.12911684e-17,
          0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          5.75666667e-01, -3.86000000e-06, 0.00000000e+00, 3.86000000e-06,
          1.34766667e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          5.75666667e-01, 0.00000000e+00, 0.00000000e+00, -1.93000000e-06,
          0.00000000e+00, 0.00000000e+00, 3.86000000e-01, 1.93000000e-06,
          3.86000000e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          3.86000000e-01, 0.00000000e+00, 0.00000000e+00, -1.93000000e-06,
          3.86000000e-01, 1.93000000e-06, 0.00000000e+00, 0.00000000e+00,
          0.00000000e+00, -1.93000000e-06, 0.00000000e+00, 0.00000000e+00,
          3.86000000e-01, 1.93000000e-06, 3.86000000e-01, 0.00000000e+00,
          5.75666667e-01, 2.61999996e-17, 0.00000000e+00, 2.61999996e-17,
          5.75666667e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          1.34766667e+00;
      Stiffness_t K4b_ref{muGrid::testGoodies::from_numpy(temp)};

      error = (K4b_ref - stiffness).norm() / K4b_ref.norm();
      BOOST_CHECK_LT(error, hi_tol);
      if (not(error < hi_tol)) {
        std::cout << "stiffness reference:\n" << K4b_ref << std::endl;
        std::cout << "stiffness computed:\n" << stiffness << std::endl;
      }
    }
    if (verbose) {
      std::cout << "C₄  =" << std::endl << stiffness << std::endl;
    }
    F_.get_state_field().cycle();
    be_.get_state_field().cycle();
    eps_.get_state_field().cycle();

    // plastic deformation
    F(0, 1) = .2;
    std::tie(stress, stiffness) = Fix::mat.evaluate_stress_tangent(
        F, F_prev[0], be_prev[0], eps_prev[0], lambda, mu, tau_y0, H, K);

    if (has_precomputed_values) {
      Strain_t tau_ref{};
      tau_ref << 1.98151335e-04, 1.98151335e-03, 0.00000000e+00, 1.98151335e-03,
          -1.98151335e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          1.60615155e-16;
      Real error{(tau_ref - stress).norm() / tau_ref.norm()};
      BOOST_CHECK_LT(error, hi_tol);

      Strain_t be_ref{};
      be_ref << 1.00052666, 0.00513348, 0., 0.00513348, 0.99949996, 0., 0., 0.,
          1.;
      error = (be_ref - be_prev[0].current()).norm() / be_ref.norm();
      BOOST_CHECK_LT(error, hi_tol);

      Real ep_ref{0.11229988};
      error = (ep_ref - eps_prev[0].current()) / ep_ref;
      BOOST_CHECK_LT(error, hi_tol);

      Stiffness_t temp{};
      temp << 8.46343327e-01, 1.11250597e-03, 0.00000000e+00, -2.85052074e-03,
          8.26305692e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          8.26350980e-01, -8.69007382e-04, 1.21749295e-03, 0.00000000e+00,
          1.61379562e-03, 8.69007382e-04, 0.00000000e+00, 0.00000000e+00,
          0.00000000e+00, 5.58242059e-18, 0.00000000e+00, 0.00000000e+00,
          9.90756677e-03, 0.00000000e+00, 0.00000000e+00, -9.90756677e-04,
          1.01057181e-02, 9.90756677e-04, 0.00000000e+00, -8.69007382e-04,
          1.21749295e-03, 0.00000000e+00, 1.61379562e-03, 8.69007382e-04,
          0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.58242059e-18,
          8.26305692e-01, -1.11250597e-03, 0.00000000e+00, 2.85052074e-03,
          8.46343327e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          8.26350980e-01, 0.00000000e+00, 0.00000000e+00, -9.90756677e-04,
          0.00000000e+00, 0.00000000e+00, 1.01057181e-02, 9.90756677e-04,
          9.90756677e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          9.90756677e-03, 0.00000000e+00, 0.00000000e+00, -9.90756677e-04,
          1.01057181e-02, 9.90756677e-04, 0.00000000e+00, 0.00000000e+00,
          0.00000000e+00, -9.90756677e-04, 0.00000000e+00, 0.00000000e+00,
          1.01057181e-02, 9.90756677e-04, 9.90756677e-03, 0.00000000e+00,
          8.26350980e-01, 0.00000000e+00, 0.00000000e+00, 1.38777878e-17,
          8.26350980e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          8.46298039e-01;

      Stiffness_t K4b_ref{muGrid::testGoodies::from_numpy(temp)};
      error = (K4b_ref - stiffness).norm() / K4b_ref.norm();

      error = (K4b_ref - stiffness).norm() / K4b_ref.norm();
      BOOST_CHECK_LT(error, hi_tol);
      if (not(error < hi_tol)) {
        std::cout << "stiffness reference:\n" << K4b_ref << std::endl;
        std::cout << "stiffness computed:\n" << stiffness << std::endl;
      }

      // check also whether pull_back is correct

      Stiffness_t intermediate{stiffness};
      Stiffness_t zero_mediate{Stiffness_t::Zero()};
      for (int i{0}; i < mdim; ++i) {
        for (int j{0}; j < mdim; ++j) {
          for (int m{0}; m < mdim; ++m) {
            const auto & k{i};
            const auto & l{j};
            // k,m inverted for right transpose
            muGrid::get(zero_mediate, i, j, k, m) -= stress(l, m);
            muGrid::get(intermediate, i, j, k, m) -= stress(l, m);
          }
        }
      }

      temp << 8.46145176e-01, -8.69007382e-04, 0.00000000e+00, -2.85052074e-03,
          8.26305692e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          8.26350980e-01, -2.85052074e-03, 1.41564428e-03, 0.00000000e+00,
          1.61379562e-03, 8.69007382e-04, 0.00000000e+00, 0.00000000e+00,
          0.00000000e+00, 5.58242059e-18, 0.00000000e+00, 0.00000000e+00,
          9.90756677e-03, 0.00000000e+00, 0.00000000e+00, -9.90756677e-04,
          1.01057181e-02, 9.90756677e-04, 0.00000000e+00, -8.69007382e-04,
          1.21749295e-03, 0.00000000e+00, 1.41564428e-03, -1.11250597e-03,
          0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.58242059e-18,
          8.26305692e-01, -1.11250597e-03, 0.00000000e+00, 8.69007382e-04,
          8.46541479e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          8.26350980e-01, 0.00000000e+00, 0.00000000e+00, -9.90756677e-04,
          0.00000000e+00, 0.00000000e+00, 1.01057181e-02, 9.90756677e-04,
          9.90756677e-03, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          9.90756677e-03, 0.00000000e+00, 0.00000000e+00, -9.90756677e-04,
          9.90756677e-03, -9.90756677e-04, 0.00000000e+00, 0.00000000e+00,
          0.00000000e+00, -9.90756677e-04, 0.00000000e+00, 0.00000000e+00,
          1.01057181e-02, -9.90756677e-04, 1.01057181e-02, 0.00000000e+00,
          8.26350980e-01, 0.00000000e+00, 0.00000000e+00, 1.38777878e-17,
          8.26350980e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          8.46298039e-01;

      Stiffness_t K4c_ref{muGrid::testGoodies::from_numpy(temp)};
      error = (K4b_ref + zero_mediate - K4c_ref).norm() / zero_mediate.norm();
      BOOST_CHECK_LT(error, hi_tol);  // rel error on small difference between
                                      // inexacly read doubles
      if (not(error < hi_tol)) {
        std::cout << "decrement reference:\n" << K4c_ref - K4b_ref << std::endl;
        std::cout << "zero_mediate computed:\n" << zero_mediate << std::endl;
      }

      error = (K4c_ref - intermediate).norm() / K4c_ref.norm();
      BOOST_CHECK_LT(error, hi_tol);
      if (not(error < hi_tol)) {
        std::cout << "stiffness reference:\n" << K4c_ref << std::endl;
        std::cout << "stiffness computed:\n" << intermediate << std::endl;
        std::cout << "zero-mediate computed:\n" << zero_mediate << std::endl;
        std::cout << "difference:\n" << K4c_ref - intermediate << std::endl;
      }
    }
    if (verbose) {
      std::cout << "Post Cycle" << std::endl;
      std::cout << "C₄  =" << std::endl << stiffness << std::endl;
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muSpectre

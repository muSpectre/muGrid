/**
 * @file   test_material_hyper_elasto_plastic1.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   25 Feb 2018
 *
 * @brief  Tests for the large-strain Simo-type plastic law implemented
 *         using MaterialMuSpectre
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
 * along with GNU Emacs; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#include "boost/mpl/list.hpp"

#include "materials/material_hyper_elasto_plastic1.hh"
#include "materials/materials_toolbox.hh"
#include "tests.hh"

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(material_hyper_elasto_plastic_1);

  template <class Mat_t>
  struct MaterialFixture
  {
    using Mat = Mat_t;
    constexpr static Real K{.833};      // bulk modulus
    constexpr static Real mu{.386};     // shear modulus
    constexpr static Real H{.004};      // hardening modulus
    constexpr static Real tau_y0{.003}; // initial yield stress
    constexpr static Real young{
      MatTB::convert_elastic_modulus<ElasticModulus::Young,
                                     ElasticModulus::Bulk,
                                     ElasticModulus::Shear>(K, mu)};
    constexpr static Real poisson{
      MatTB::convert_elastic_modulus<ElasticModulus::Poisson,
                                     ElasticModulus::Bulk,
                                     ElasticModulus::Shear>(K, mu)};
    MaterialFixture():mat("Name", young, poisson, tau_y0, H){};
    constexpr static Dim_t sdim{Mat_t::sdim()};
    constexpr static Dim_t mdim{Mat_t::mdim()};

    Mat_t mat;
  };

  using mats = boost::mpl::list<MaterialFixture<MaterialHyperElastoPlastic1<  twoD,   twoD>>,
                                MaterialFixture<MaterialHyperElastoPlastic1<  twoD, threeD>>,
                                MaterialFixture<MaterialHyperElastoPlastic1<threeD, threeD>>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_constructor, Fix, mats, Fix) {
    BOOST_CHECK_EQUAL("Name", Fix::mat.get_name());
    auto & mat{Fix::mat};
    auto sdim{Fix::sdim};
    auto mdim{Fix::mdim};
    BOOST_CHECK_EQUAL(sdim, mat.sdim());
    BOOST_CHECK_EQUAL(mdim, mat.mdim());
  }


  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_evaluate_stress, Fix, mats, Fix) {

    // This test uses precomputed reference values (computed using
    // elasto-plasticity.py) for the 3d case only

    // need higher tol because of printout precision of reference solutions
    constexpr Real hi_tol{1e-8};
    constexpr Dim_t mdim{Fix::mdim}, sdim{Fix::sdim};
    constexpr bool has_precomputed_values{(mdim == sdim) && (mdim == threeD)};
    constexpr bool verbose{false};
    using Strain_t = Eigen::Matrix<Real, mdim, mdim>;
    using traits = MaterialMuSpectre_traits<MaterialHyperElastoPlastic1<sdim, mdim>>;
    using LColl_t = typename traits::LFieldColl_t;
    using StrainStField_t = StateField<
      TensorField<LColl_t, Real, secondOrder, mdim>>;
    using FlowStField_t = StateField<
      ScalarField<LColl_t, Real>>;

    // using StrainStRef_t = typename traits::LStrainMap_t::reference;
    // using ScalarStRef_t = typename traits::LScalarMap_t::reference;

    // create statefields
    LColl_t coll{};
    coll.add_pixel({0});
    coll.initialise();

    auto & F_{make_statefield<StrainStField_t>("previous gradient", coll)};
    auto & be_{make_statefield<StrainStField_t>("previous elastic strain", coll)};
    auto & eps_{make_statefield<FlowStField_t>("plastic flow", coll)};

    auto F_prev{F_.get_map()};
    F_prev[0].current() = Strain_t::Identity();
    auto be_prev{be_.get_map()};
    be_prev[0].current() = Strain_t::Identity();
    auto eps_prev{eps_.get_map()};
    eps_prev[0].current() = 0;
    // elastic deformation
    Strain_t F{Strain_t::Identity()};
    F(0, 1) = 1e-5;

    F_.cycle();
    be_.cycle();
    eps_.cycle();
    Strain_t stress{Fix::mat.evaluate_stress(F,
                                             F_prev[0],
                                             be_prev[0],
                                             eps_prev[0])};

    if (has_precomputed_values) {
      Strain_t tau_ref{};
      tau_ref <<
        1.92999522e-11,  3.86000000e-06,  0.00000000e+00,
        3.86000000e-06, -1.93000510e-11,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00, -2.95741950e-17;
      Real error{(tau_ref-stress).norm()};
      BOOST_CHECK_LT(error, hi_tol);

      Strain_t be_ref{};
      be_ref <<
        1.00000000e+00,   1.00000000e-05,   0.00000000e+00,
        1.00000000e-05,   1.00000000e+00,   0.00000000e+00,
        0.00000000e+00,   0.00000000e+00,   1.00000000e+00;
      error = (be_ref-be_prev[0].current()).norm();
      BOOST_CHECK_LT(error, hi_tol);

      Real ep_ref{0};
      error = ep_ref-eps_prev[0].current();
      BOOST_CHECK_LT(error, hi_tol);

    }
    if (verbose) {
      std::cout << "τ  =" << std::endl << stress << std::endl;
      std::cout << "F  =" << std::endl << F << std::endl;
      std::cout << "Fₜ =" << std::endl << F_prev[0].current() << std::endl;
      std::cout << "bₑ =" << std::endl << be_prev[0].current() << std::endl;
      std::cout << "εₚ =" << std::endl << eps_prev[0].current() << std::endl;
    }
    F_.cycle();
    be_.cycle();
    eps_.cycle();

    // plastic deformation
    F(0, 1) = .2;
    stress = Fix::mat.evaluate_stress(F,
                                      F_prev[0],
                                      be_prev[0],
                                      eps_prev[0]);

    if (has_precomputed_values) {
      Strain_t tau_ref{};
      tau_ref <<
        1.98151335e-04,   1.98151335e-03,   0.00000000e+00,
        1.98151335e-03,  -1.98151335e-04,   0.00000000e+00,
        0.00000000e+00,   0.00000000e+00,   1.60615155e-16;
      Real error{(tau_ref-stress).norm()};
      BOOST_CHECK_LT(error, hi_tol);

      Strain_t be_ref{};
      be_ref <<
        1.00052666,  0.00513348,  0.,
        0.00513348,  0.99949996,  0.,
        0.,          0.,          1.;
      error = (be_ref-be_prev[0].current()).norm();
      BOOST_CHECK_LT(error, hi_tol);

      Real ep_ref{0.11229988};
      error = ep_ref-eps_prev[0].current();
      BOOST_CHECK_LT(error, hi_tol);

    }
    if (verbose) {
      std::cout << "Post Cycle" << std::endl;
      std::cout << "τ  =" << std::endl << stress << std::endl
                << "F  =" << std::endl << F << std::endl
                << "Fₜ =" << std::endl << F_prev[0].current() << std::endl
                << "bₑ =" << std::endl << be_prev[0].current() << std::endl
                << "εₚ =" << std::endl << eps_prev[0].current() << std::endl;
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_evaluate_stiffness, Fix, mats, Fix) {

    // This test uses precomputed reference values (computed using
    // elasto-plasticity.py) for the 3d case only

    // need higher tol because of printout precision of reference solutions
    constexpr Real hi_tol{1e-8};
    constexpr Dim_t mdim{Fix::mdim}, sdim{Fix::sdim};
    constexpr bool has_precomputed_values{(mdim == sdim) && (mdim == threeD)};
    constexpr bool verbose{has_precomputed_values && false};
    using Strain_t = Eigen::Matrix<Real, mdim, mdim>;
    using Stiffness_t = T4Mat<Real, mdim>;
    using traits = MaterialMuSpectre_traits<MaterialHyperElastoPlastic1<sdim, mdim>>;
    using LColl_t = typename traits::LFieldColl_t;
    using StrainStField_t = StateField<
      TensorField<LColl_t, Real, secondOrder, mdim>>;
    using FlowStField_t = StateField<
      ScalarField<LColl_t, Real>>;

    // using StrainStRef_t = typename traits::LStrainMap_t::reference;
    // using ScalarStRef_t = typename traits::LScalarMap_t::reference;

    // create statefields
    LColl_t coll{};
    coll.add_pixel({0});
    coll.initialise();

    auto & F_{make_statefield<StrainStField_t>("previous gradient", coll)};
    auto & be_{make_statefield<StrainStField_t>("previous elastic strain", coll)};
    auto & eps_{make_statefield<FlowStField_t>("plastic flow", coll)};

    auto F_prev{F_.get_map()};
    F_prev[0].current() = Strain_t::Identity();
    auto be_prev{be_.get_map()};
    be_prev[0].current() = Strain_t::Identity();
    auto eps_prev{eps_.get_map()};
    eps_prev[0].current() = 0;
    // elastic deformation
    Strain_t F{Strain_t::Identity()};
    F(0, 1) = 1e-5;

    F_.cycle();
    be_.cycle();
    eps_.cycle();
    Strain_t stress{};
    Stiffness_t stiffness{};
    std::tie(stress, stiffness) =
      Fix::mat.evaluate_stress_tangent(F,
                                       F_prev[0],
                                       be_prev[0],
                                       eps_prev[0]);

    if (has_precomputed_values) {
      Strain_t tau_ref{};
      tau_ref <<
        1.92999522e-11,  3.86000000e-06,  0.00000000e+00,
        3.86000000e-06, -1.93000510e-11,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00, -2.95741950e-17;
      Real error{(tau_ref-stress).norm()};
      BOOST_CHECK_LT(error, hi_tol);

      Strain_t be_ref{};
      be_ref <<
        1.00000000e+00,   1.00000000e-05,   0.00000000e+00,
        1.00000000e-05,   1.00000000e+00,   0.00000000e+00,
        0.00000000e+00,   0.00000000e+00,   1.00000000e+00;
      error = (be_ref-be_prev[0].current()).norm();
      BOOST_CHECK_LT(error, hi_tol);

      Real ep_ref{0};
      error = ep_ref-eps_prev[0].current();
      BOOST_CHECK_LT(error, hi_tol);

      Stiffness_t C4_ref{};
      C4_ref <<
        0.67383333,  0.,          0.,          0.,          0.28783333,  0.,          0.,      0.,       0.28783333,
        0.,          0.193,       0.,          0.193,       0.,          0.,          0.,      0.,       0.,
        0.,          0.,          0.193,       0.,          0.,          0.,          0.193,   0.,       0.,
        0.,          0.193,       0.,          0.193,       0.,          0.,          0.,      0.,       0.,
        0.28783333,  0.,          0.,          0.,          0.67383333,  0.,          0.,      0.,       0.28783333,
        0.,          0.,          0.,          0.,          0.,          0.193,       0.,   0.193,       0.,
        0.,          0.,          0.193,       0.,          0.,          0.,          0.193,   0.,       0.,
        0.,          0.,          0.,          0.,          0.,          0.193,       0.,   0.193,       0.,
        0.28783333,  0.,          0.,          0.,          0.28783333,  0.,          0.,      0.,       0.67383333;
      error = (C4_ref - stiffness).norm();
      BOOST_CHECK_LT(error, hi_tol);

    }
    if (verbose) {
      std::cout << "C₄  =" << std::endl << stiffness << std::endl;
    }
    F_.cycle();
    be_.cycle();
    eps_.cycle();

    // plastic deformation
    F(0, 1) = .2;
    std::tie(stress, stiffness) =
      Fix::mat.evaluate_stress_tangent(F,
                                       F_prev[0],
                                       be_prev[0],
                                       eps_prev[0]);

    if (has_precomputed_values) {
      Strain_t tau_ref{};
      tau_ref <<
        1.98151335e-04,   1.98151335e-03,   0.00000000e+00,
        1.98151335e-03,  -1.98151335e-04,   0.00000000e+00,
        0.00000000e+00,   0.00000000e+00,   1.60615155e-16;
      Real error{(tau_ref-stress).norm()};
      BOOST_CHECK_LT(error, hi_tol);

      Strain_t be_ref{};
      be_ref <<
        1.00052666,  0.00513348,  0.,
        0.00513348,  0.99949996,  0.,
        0.,          0.,          1.;
      error = (be_ref-be_prev[0].current()).norm();
      BOOST_CHECK_LT(error, hi_tol);

      Real ep_ref{0.11229988};
      error = ep_ref-eps_prev[0].current();
      BOOST_CHECK_LT(error, hi_tol);

      Stiffness_t C4_ref{};
      C4_ref <<
        +4.23106224e-01,  -4.27959704e-04,   0.00000000e+00,  -4.27959704e-04,    4.13218286e-01,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,    4.13175490e-01,
        -4.27959704e-04,   7.07167743e-04,   0.00000000e+00,   7.07167743e-04,    4.27959704e-04,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,    2.79121029e-18,
        +0.00000000e+00,   0.00000000e+00,   4.98676478e-03,   0.00000000e+00,    0.00000000e+00,   0.00000000e+00,   4.98676478e-03,   0.00000000e+00,    0.00000000e+00,
        -4.27959704e-04,   7.07167743e-04,   0.00000000e+00,   7.07167743e-04,    4.27959704e-04,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,    2.79121029e-18,
        +4.13218286e-01,   4.27959704e-04,   0.00000000e+00,   4.27959704e-04,    4.23106224e-01,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,    4.13175490e-01,
        +0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,    0.00000000e+00,   4.98676478e-03,   0.00000000e+00,   4.98676478e-03,    0.00000000e+00,
        +0.00000000e+00,   0.00000000e+00,   4.98676478e-03,   0.00000000e+00,    0.00000000e+00,   0.00000000e+00,   4.98676478e-03,   0.00000000e+00,    0.00000000e+00,
        +0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,    0.00000000e+00,   4.98676478e-03,   0.00000000e+00,   4.98676478e-03,    0.00000000e+00,
        +4.13175490e-01,   2.79121029e-18,   0.00000000e+00,   2.79121029e-18,    4.13175490e-01,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00,    4.23149020e-01;
      error = (C4_ref - stiffness).norm();
      BOOST_CHECK_LT(error, hi_tol);

    }
    if (verbose) {
      std::cout << "Post Cycle" << std::endl;
      std::cout << "C₄  =" << std::endl << stiffness << std::endl;
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre

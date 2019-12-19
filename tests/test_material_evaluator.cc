/**
 * @file   test_material_evaluator.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   13 Jan 2019
 *
 * @brief  tests for the material evaluator mechanism
 *
 * Copyright © 2019 Till Junge
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
#include "materials/material_linear_elastic2.hh"
#include "materials/material_evaluator.hh"

#include <libmugrid/T4_map_proxy.hh>

#include "Eigen/Dense"

namespace muSpectre {

  using muGrid::testGoodies::rel_error;

  BOOST_AUTO_TEST_SUITE(material_evaluator_tests);

  BOOST_AUTO_TEST_CASE(without_adding_a_pixel) {
    using Mat_t = MaterialLinearElastic1<twoD>;

    constexpr Real Young{210e9};
    constexpr Real Poisson{.33};

    auto mat_eval = Mat_t::make_evaluator(Young, Poisson);

    auto & evaluator = std::get<1>(mat_eval);

    using T2_t = Eigen::Matrix<Real, twoD, twoD>;
    const T2_t F{(T2_t::Random() - (T2_t::Ones() * .5)) * 1e-4 +
                 T2_t::Identity()};
    const T2_t eps{
        .5 * ((F - T2_t::Identity()) + (F - T2_t::Identity()).transpose())};

    /*
     * at this point, the evaluator has been created, but the underlying
     * material still has zero pixels. Evaluation is not yet possible, and
     * trying to do so has to fail with an explicit error message
     */
    BOOST_CHECK_THROW(evaluator.evaluate_stress(eps, Formulation::small_strain),
                      std::runtime_error);
  }

  BOOST_AUTO_TEST_CASE(multiple_pixels) {
    using Mat_t = MaterialLinearElastic1<twoD>;

    constexpr Real Young{210e9};
    constexpr Real Poisson{.33};

    auto mat_eval = Mat_t::make_evaluator(Young, Poisson);

    auto & mat = *std::get<0>(mat_eval);
    auto & evaluator = std::get<1>(mat_eval);

    using T2_t = Eigen::Matrix<Real, twoD, twoD>;
    const T2_t F{(T2_t::Random() - (T2_t::Ones() * .5)) * 1e-4 +
                 T2_t::Identity()};
    const T2_t eps{
        .5 * ((F - T2_t::Identity()) + (F - T2_t::Identity()).transpose())};

    mat.add_pixel(0);
    mat.add_pixel(1);
    /*
     * at this point, the evaluator has been created, but the underlying
     * material has two pixels. Evaluation would be ambiguous, and
     * trying to do so has to fail with an explicit error message
     */
    BOOST_CHECK_THROW(evaluator.evaluate_stress(eps, Formulation::small_strain),
                      std::runtime_error);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(without_per_pixel_data) {
    using Mat_t = MaterialLinearElastic1<twoD>;

    constexpr Real Young{210e9};
    constexpr Real Poisson{.33};

    auto mat_eval = Mat_t::make_evaluator(Young, Poisson);
    auto & mat = *std::get<0>(mat_eval);

    auto & evaluator = std::get<1>(mat_eval);

    using T2_t = Eigen::Matrix<Real, twoD, twoD>;
    using T4_t = muGrid::T4Mat<Real, twoD>;
    const T2_t F{(T2_t::Random() - (T2_t::Ones() * .5)) * 1e-4 +
                 T2_t::Identity()};
    const T2_t eps{
        .5 * ((F - T2_t::Identity()) + (F - T2_t::Identity()).transpose())};

    mat.add_pixel({});

    const T2_t sigma{evaluator.evaluate_stress(eps, Formulation::small_strain)};
    const T2_t P{evaluator.evaluate_stress(F, Formulation::finite_strain)};

    auto J{F.determinant()};
    auto P_reconstruct{J * sigma * F.inverse().transpose()};

    auto error_comp{[](const auto & a, const auto & b) {
      return (a - b).norm() / (a + b).norm();
    }};
    auto error{error_comp(P, P_reconstruct)};

    constexpr Real small_strain_tol{1e-3};
    if (not(error <= small_strain_tol)) {
      std::cout << "F =" << std::endl << F << std::endl;
      std::cout << "ε =" << std::endl << eps << std::endl;
      std::cout << "P =" << std::endl << P << std::endl;
      std::cout << "σ =" << std::endl << sigma << std::endl;
      std::cout << "P_reconstructed =" << std::endl
                << P_reconstruct << std::endl;
    }

    BOOST_CHECK_LE(error, small_strain_tol);

    T2_t sigma2, P2;
    T4_t C, K;

    std::tie(sigma2, C) =
        evaluator.evaluate_stress_tangent(eps, Formulation::small_strain);
    std::tie(P2, K) =
        evaluator.evaluate_stress_tangent(F, Formulation::finite_strain);

    error = error_comp(sigma2, sigma);
    BOOST_CHECK_LE(error, tol);
    error = error_comp(P2, P);
    BOOST_CHECK_LE(error, tol);

    error = error_comp(C, K);
    if (not(error <= small_strain_tol)) {
      std::cout << "F =" << std::endl << F << std::endl;
      std::cout << "ε =" << std::endl << eps << std::endl;
      std::cout << "P =" << std::endl << P << std::endl;
      std::cout << "σ =" << std::endl << sigma << std::endl;
      std::cout << "K =" << std::endl << K << std::endl;
      std::cout << "C =" << std::endl << C << std::endl;
    }
    BOOST_CHECK_LE(error, small_strain_tol);

    /*
     * Now, the material already has a pixel, adding more should be rejected
     */
    BOOST_CHECK_THROW(mat.add_pixel({1}), muGrid::FieldCollectionError);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(with_per_pixel_data) {
    using Mat_t = MaterialLinearElastic2<twoD>;

    constexpr Real Young{210e9};
    constexpr Real Poisson{.33};

    auto mat_eval{Mat_t::make_evaluator(Young, Poisson)};
    auto & mat{*std::get<0>(mat_eval)};
    auto & evaluator{std::get<1>(mat_eval)};

    using T2_t = Eigen::Matrix<Real, twoD, twoD>;
    using T4_t = muGrid::T4Mat<Real, twoD>;
    const T2_t F{(T2_t::Random() - (T2_t::Ones() * .5)) * 1e-4 +
                 T2_t::Identity()};
    const T2_t eps{
        .5 * ((F - T2_t::Identity()) + (F - T2_t::Identity()).transpose())};

    T2_t eigen_strain{[](auto x) {
      return 1e-4 * (x + x.transpose());
    }(T2_t::Random() - T2_t::Ones() * .5)};

    mat.add_pixel(0, eigen_strain);

    const T2_t sigma{evaluator.evaluate_stress(eps, Formulation::small_strain)};
    const T2_t P{evaluator.evaluate_stress(F, Formulation::finite_strain)};

    auto J{F.determinant()};
    auto P_reconstruct{J * sigma * F.inverse().transpose()};

    auto error_comp{[](const auto & a, const auto & b) {
      return (a - b).norm() / (a + b).norm();
    }};
    auto error{error_comp(P, P_reconstruct)};

    constexpr Real small_strain_tol{1e-3};
    if (not(error <= small_strain_tol)) {
      std::cout << "F =" << std::endl << F << std::endl;
      std::cout << "ε =" << std::endl << eps << std::endl;
      std::cout << "P =" << std::endl << P << std::endl;
      std::cout << "σ =" << std::endl << sigma << std::endl;
      std::cout << "P_reconstructed =" << std::endl
                << P_reconstruct << std::endl;
    }

    BOOST_CHECK_LE(error, small_strain_tol);

    T2_t sigma2, P2;
    T4_t C, K;

    std::tie(sigma2, C) =
        evaluator.evaluate_stress_tangent(eps, Formulation::small_strain);
    std::tie(P2, K) =
        evaluator.evaluate_stress_tangent(F, Formulation::finite_strain);

    error = error_comp(sigma2, sigma);
    BOOST_CHECK_LE(error, tol);
    error = error_comp(P2, P);
    BOOST_CHECK_LE(error, tol);

    error = error_comp(C, K);
    if (not(error <= small_strain_tol)) {
      std::cout << "F =" << std::endl << F << std::endl;
      std::cout << "ε =" << std::endl << eps << std::endl;
      std::cout << "P =" << std::endl << P << std::endl;
      std::cout << "σ =" << std::endl << sigma << std::endl;
      std::cout << "K =" << std::endl << K << std::endl;
      std::cout << "C =" << std::endl << C << std::endl;
    }
    BOOST_CHECK_LE(error, small_strain_tol);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(tangent_estimation) {
    using Mat_t = MaterialLinearElastic1<twoD>;

    constexpr Real Young{210e9};
    constexpr Real Poisson{.33};

    auto mat_eval = Mat_t::make_evaluator(Young, Poisson);
    auto & mat = *std::get<0>(mat_eval);
    auto & evaluator = std::get<1>(mat_eval);

    using T2_t = Eigen::Matrix<Real, twoD, twoD>;
    using T4_t = muGrid::T4Mat<Real, twoD>;
    const T2_t F{(T2_t::Random() - (T2_t::Ones() * .5)) * 1e-4 +
                 T2_t::Identity()};
    const T2_t eps{
        .5 * ((F - T2_t::Identity()) + (F - T2_t::Identity()).transpose())};

    mat.add_pixel({});

    T2_t sigma, P;
    T4_t C, K;

    std::tie(sigma, C) =
        evaluator.evaluate_stress_tangent(eps, Formulation::small_strain);
    std::tie(P, K) =
        evaluator.evaluate_stress_tangent(F, Formulation::finite_strain);

    constexpr Real linear_step{1.};
    constexpr Real nonlin_step{1.e-6};
    T4_t C_estim{evaluator.estimate_tangent(eps, Formulation::small_strain,
                                            linear_step)};
    T4_t K_estim{
        evaluator.estimate_tangent(F, Formulation::finite_strain, nonlin_step)};

    constexpr Real finite_diff_tol{1e-9};
    Real error{rel_error(K, K_estim)};
    if (not(error <= finite_diff_tol)) {
      std::cout << "K =" << std::endl << K << std::endl;
      std::cout << "K_estim =" << std::endl << K_estim << std::endl;
    }
    BOOST_CHECK_LE(error, finite_diff_tol);

    error = rel_error(C, C_estim);
    if (not(error <= tol)) {
      std::cout << "centred difference:" << std::endl;
      std::cout << "C =" << std::endl << C << std::endl;
      std::cout << "C_estim =" << std::endl << C_estim << std::endl;
    }
    BOOST_CHECK_LE(error, tol);

    C_estim = evaluator.estimate_tangent(eps, Formulation::small_strain,
                                         linear_step, FiniteDiff::forward);
    error = rel_error(C, C_estim);
    if (not(error <= tol)) {
      std::cout << "forward difference:" << std::endl;
      std::cout << "C =" << std::endl << C << std::endl;
      std::cout << "C_estim =" << std::endl << C_estim << std::endl;
    }
    BOOST_CHECK_LE(error, tol);

    C_estim = evaluator.estimate_tangent(eps, Formulation::small_strain,
                                         linear_step, FiniteDiff::backward);
    error = rel_error(C, C_estim);
    if (not(error <= tol)) {
      std::cout << "backward difference:" << std::endl;
      std::cout << "C =" << std::endl << C << std::endl;
      std::cout << "C_estim =" << std::endl << C_estim << std::endl;
    }
    BOOST_CHECK_LE(error, tol);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muSpectre

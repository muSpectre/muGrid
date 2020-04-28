/**
 * @file   test_materials_toolbox.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   05 Nov 2017
 *
 * @brief  Tests for the materials toolbox
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

#include <libmugrid/T4_map_proxy.hh>
#include <materials/materials_toolbox.hh>
#include <materials/stress_transformations_default_case.hh>
#include <materials/stress_transformations_PK1_impl.hh>
#include <materials/stress_transformations_PK2_impl.hh>
#include <materials/stress_transformations.hh>

#include <boost/mpl/list.hpp>
#include <Eigen/Dense>

namespace muSpectre {

  using muGrid::testGoodies::rel_error;

  BOOST_AUTO_TEST_SUITE(materials_toolbox)

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_strain_conversion, Fix,
                                   muGrid::testGoodies::dimlist, Fix) {
    constexpr Dim_t dim{Fix::dim};
    using T2 = Eigen::Matrix<Real, dim, dim>;

    T2 F{(T2::Random() - .5 * T2::Ones()) / 20 + T2::Identity()};

    // checking Green-Lagrange
    T2 Eref = .5 * (F.transpose() * F - T2::Identity());

    T2 E_tb = MatTB::convert_strain<StrainMeasure::Gradient,
                                    StrainMeasure::GreenLagrange>(
        Eigen::Map<Eigen::Matrix<Real, dim, dim>>(F.data()));

    Real error = rel_error(Eref, E_tb);
    BOOST_CHECK_LT(error, tol);

    // checking Left Cauchy-Green
    Eref = F * F.transpose();
    E_tb = MatTB::convert_strain<StrainMeasure::Gradient,
                                 StrainMeasure::LCauchyGreen>(F);

    error = rel_error(Eref, E_tb);
    BOOST_CHECK_LT(error, tol);

    // checking Right Cauchy-Green
    Eref = F.transpose() * F;
    E_tb = MatTB::convert_strain<StrainMeasure::Gradient,
                                 StrainMeasure::RCauchyGreen>(F);

    error = rel_error(Eref, E_tb);
    BOOST_CHECK_LT(error, tol);

    // checking Hencky (logarithmic) strain
    Eref = F.transpose() * F;
    Eigen::SelfAdjointEigenSolver<T2> EigSolv(Eref);
    Eref.setZero();
    for (size_t i{0}; i < dim; ++i) {
      auto && vec = EigSolv.eigenvectors().col(i);
      auto && val = EigSolv.eigenvalues()(i);
      Eref += .5 * std::log(val) * vec * vec.transpose();
    }

    E_tb =
        MatTB::convert_strain<StrainMeasure::Gradient, StrainMeasure::Log>(F);

    error = rel_error(Eref, E_tb);
    BOOST_CHECK_LT(error, tol);

    auto F_tb =
        MatTB::convert_strain<StrainMeasure::Gradient, StrainMeasure::Gradient>(
            F);

    error = rel_error(F, F_tb);
    BOOST_CHECK_LT(error, tol);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(dumb_tensor_mult_test, Fix,
                                   muGrid::testGoodies::dimlist, Fix) {
    constexpr Dim_t dim{Fix::dim};
    using T4 = muGrid::T4Mat<Real, dim>;
    T4 A, B, R1, R2;
    A.setRandom();
    B.setRandom();
    R1 = A * B;
    R2.setZero();
    for (Dim_t i = 0; i < dim; ++i) {
      for (Dim_t j = 0; j < dim; ++j) {
        for (Dim_t a = 0; a < dim; ++a) {
          for (Dim_t b = 0; b < dim; ++b) {
            for (Dim_t k = 0; k < dim; ++k) {
              for (Dim_t l = 0; l < dim; ++l) {
                muGrid::get(R2, i, j, k, l) +=
                    muGrid::get(A, i, j, a, b) * muGrid::get(B, a, b, k, l);
              }
            }
          }
        }
      }
    }
    auto error{rel_error(R1, R2)};
    BOOST_CHECK_LT(error, tol);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_PK1_stress, Fix,
                                   muGrid::testGoodies::dimlist, Fix) {
    using Matrices::Tens2_t;
    using Matrices::Tens4_t;
    using Matrices::tensmult;

    constexpr Dim_t dim{Fix::dim};
    using T2 = Eigen::Matrix<Real, dim, dim>;
    using T4 = muGrid::T4Mat<Real, dim>;
    muGrid::testGoodies::RandRange<Real> rng;
    T2 F = T2::Identity() * 2;
    // F.setRandom();
    T2 E_tb = MatTB::convert_strain<StrainMeasure::Gradient,
                                    StrainMeasure::GreenLagrange>(
        Eigen::Map<Eigen::Matrix<Real, dim, dim>>(F.data()));
    Real lambda = 3;  // rng.randval(1, 2);
    Real mu = 4;      // rng.randval(1,2);
    T4 J = Matrices::Itrac<dim>();
    T2 I = Matrices::I2<dim>();
    T4 I4 = Matrices::Isymm<dim>();
    T4 C = lambda * J + 2 * mu * I4;
    T2 S = Matrices::tensmult(C, E_tb);
    T2 Sref = lambda * E_tb.trace() * I + 2 * mu * E_tb;

    auto error = rel_error(Sref, S);
    BOOST_CHECK_LT(error, tol);

    T4 K = (Matrices::outer_under(I, S) +
            (Matrices::outer_under(F, I) * C *
             Matrices::outer_under(F.transpose(), I)));

    // See Curnier, 2000, "Méthodes numériques en mécanique des solides", p 252
    T4 Kref;
    Real Fkrkr = (F.array() * F.array()).sum();
    T2 Fkmkn = F.transpose() * F;
    T2 Fisjs = F * F.transpose();

    Kref.setZero();
    for (Dim_t i = 0; i < dim; ++i) {
      for (Dim_t j = 0; j < dim; ++j) {
        for (Dim_t m = 0; m < dim; ++m) {
          for (Dim_t n = 0; n < dim; ++n) {
            muGrid::get(Kref, i, m, j, n) =
                (lambda * ((Fkrkr - dim) / 2 * I(i, j) * I(m, n) +
                           F(i, m) * F(j, n)) +
                 mu * (I(i, j) * Fkmkn(m, n) + Fisjs(i, j) * I(m, n) -
                       I(i, j) * I(m, n) + F(i, n) * F(j, m)));
          }
        }
      }
    }

    error = rel_error(Kref, K);
    BOOST_CHECK_LT(error, tol);

    T2 P = MatTB::PK1_stress<StressMeasure::PK2, StrainMeasure::GreenLagrange>(
        F, S);
    T2 Pref = F * S;
    error = rel_error(P, Pref);
    BOOST_CHECK_LT(error, tol);

    T2 S_back =
        MatTB::PK2_stress<StressMeasure::PK1, StrainMeasure::Gradient>(F, P);
    error = rel_error(S_back, S);
    BOOST_CHECK_LT(error, tol);

    error = rel_error(S_back, S_back.transpose());
    BOOST_CHECK_LT(error, tol);

    auto && stress_tgt =
        MatTB::PK1_stress<StressMeasure::PK2, StrainMeasure::GreenLagrange>(
            F, S, C);

    T2 P_t = std::move(std::get<0>(stress_tgt));
    T4 K_t = std::move(std::get<1>(stress_tgt));

    error = rel_error(P_t, Pref);
    BOOST_CHECK_LT(error, tol);

    error = rel_error(K_t, Kref);
    BOOST_CHECK_LT(error, tol);

    auto && stress_tgt_back =
        MatTB::PK2_stress<StressMeasure::PK1, StrainMeasure::Gradient>(F, P,
                                                                       K_t);

    T2 stress_back = std::move(std::get<0>(stress_tgt_back));
    T4 stiffness_back = std::move(std::get<1>(stress_tgt_back));

    error = rel_error(stress_back, S);
    BOOST_CHECK_LT(error, tol);

    error = rel_error(stress_back, stress_back.transpose());
    BOOST_CHECK_LT(error, tol);

    error = rel_error(stiffness_back, C);
    BOOST_CHECK_LT(error, tol);

    error = rel_error(stiffness_back, stiffness_back.transpose());
    BOOST_CHECK_LT(error, tol);

    auto && stress_tgt_trivial =
        MatTB::PK1_stress<StressMeasure::PK1, StrainMeasure::Gradient>(F, P, K);
    T2 P_u = std::move(std::get<0>(stress_tgt_trivial));
    T4 K_u = std::move(std::get<1>(stress_tgt_trivial));

    error = rel_error(P_u, Pref);
    BOOST_CHECK_LT(error, tol);

    error = rel_error(K_u, Kref);
    BOOST_CHECK_LT(error, tol);

    T2 P_g;
    T4 K_g;
    std::tie(P_g, K_g) =
        muGrid::testGoodies::objective_hooke_explicit(lambda, mu, F);

    error = rel_error(P_g, Pref);
    BOOST_CHECK_LT(error, tol);

    error = rel_error(K_g, Kref);
    BOOST_CHECK_LT(error, tol);
  }

  BOOST_AUTO_TEST_CASE(elastic_modulus_conversions) {
    // define original input
    constexpr Real E{123.456};
    constexpr Real nu{.3};

    using MatTB::convert_elastic_modulus;
    // derived values
    constexpr Real K{
        convert_elastic_modulus<ElasticModulus::Bulk, ElasticModulus::Young,
                                ElasticModulus::Poisson>(E, nu)};
    constexpr Real lambda{
        convert_elastic_modulus<ElasticModulus::lambda, ElasticModulus::Young,
                                ElasticModulus::Poisson>(E, nu)};
    constexpr Real mu{
        convert_elastic_modulus<ElasticModulus::Shear, ElasticModulus::Young,
                                ElasticModulus::Poisson>(E, nu)};

    // recover original inputs
    Real comp =
        convert_elastic_modulus<ElasticModulus::Young, ElasticModulus::Bulk,
                                ElasticModulus::Shear>(K, mu);
    Real err = E - comp;
    BOOST_CHECK_LT(err, tol);

    comp =
        convert_elastic_modulus<ElasticModulus::Poisson, ElasticModulus::Bulk,
                                ElasticModulus::Shear>(K, mu);
    err = nu - comp;
    BOOST_CHECK_LT(err, tol);

    comp =
        convert_elastic_modulus<ElasticModulus::Young, ElasticModulus::lambda,
                                ElasticModulus::Shear>(lambda, mu);
    err = E - comp;
    BOOST_CHECK_LT(err, tol);

    // check inversion resistance
    Real compA =
        convert_elastic_modulus<ElasticModulus::Poisson, ElasticModulus::Bulk,
                                ElasticModulus::Shear>(K, mu);
    Real compB =
        convert_elastic_modulus<ElasticModulus::Poisson, ElasticModulus::Shear,
                                ElasticModulus::Bulk>(mu, K);
    BOOST_CHECK_EQUAL(compA, compB);

    // check trivial self-returning
    comp = convert_elastic_modulus<ElasticModulus::Bulk, ElasticModulus::Bulk,
                                   ElasticModulus::Shear>(K, mu);
    BOOST_CHECK_EQUAL(K, comp);

    comp = convert_elastic_modulus<ElasticModulus::Shear, ElasticModulus::Bulk,
                                   ElasticModulus::Shear>(K, mu);
    BOOST_CHECK_EQUAL(mu, comp);

    // check alternative calculation of computed values

    comp = convert_elastic_modulus<ElasticModulus::lambda,
                                   ElasticModulus::K,  // alternative for "Bulk"
                                   ElasticModulus::mu>(
        K, mu);  // alternative for "Shear"
    BOOST_CHECK_LE(std::abs((comp - lambda) / lambda), tol);
  }

  template <FiniteDiff FinDiff>
  struct FiniteDifferencesHolder {
    constexpr static FiniteDiff value{FinDiff};
  };

  using FinDiffList =
      boost::mpl::list<FiniteDifferencesHolder<FiniteDiff::forward>,
                       FiniteDifferencesHolder<FiniteDiff::backward>,
                       FiniteDifferencesHolder<FiniteDiff::centred>>;

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(numerical_tangent_test, Fix, FinDiffList,
                                   Fix) {
    constexpr Dim_t Dim{twoD};
    using T4_t = muGrid::T4Mat<Real, Dim>;
    using T2_t = Eigen::Matrix<Real, Dim, Dim>;

    bool verbose{false};

    T4_t Q{};
    Q << 1., 2., 0., 0., 0., 1.66666667, 0., 0., 0., 0., 2.33333333, 0., 0., 0.,
        0., 3.;
    if (verbose) {
      std::cout << Q << std::endl << std::endl;
    }

    T2_t B{};
    B << 2., 3.33333333, 2.66666667, 4.;
    if (verbose) {
      std::cout << B << std::endl << std::endl;
    }

    using cmap_t = Eigen::Map<const Eigen::Matrix<Real, Dim * Dim, 1>>;
    using map_t = Eigen::Map<Eigen::Matrix<Real, Dim * Dim, 1>>;

    auto fun = [&](const T2_t & x) -> T2_t {
      cmap_t x_vec{x.data()};
      T2_t ret_val{};
      map_t(ret_val.data()) = Q * x_vec + cmap_t(B.data());
      return ret_val;
    };

    T2_t temp_res = fun(T2_t::Ones());
    if (verbose) {
      std::cout << temp_res << std::endl << std::endl;
    }

    T4_t numerical_tangent{MatTB::compute_numerical_tangent<Dim, Fix::value>(
        fun, T2_t::Ones(), 1e-2)};

    if (verbose) {
      std::cout << numerical_tangent << std::endl << std::endl;
    }

    Real error{rel_error(numerical_tangent, Q)};

    BOOST_CHECK_LT(error, tol);
    if (not(error < tol)) {
      switch (Fix::value) {
      case FiniteDiff::backward: {
        std::cout << "backward difference: " << std::endl;
        break;
      }
      case FiniteDiff::forward: {
        std::cout << "forward difference: " << std::endl;
        break;
      }
      case FiniteDiff::centred: {
        std::cout << "centered difference: " << std::endl;
        break;
      }
      }

      std::cout << "error = " << error << std::endl;
      std::cout << "numerical tangent:\n" << numerical_tangent << std::endl;
      std::cout << "reference:\n" << Q << std::endl;
    }
  }

  BOOST_AUTO_TEST_CASE(deviatoric_stress_test) {
    constexpr Dim_t Dim{threeD};
    using T2_t = Eigen::Matrix<Real, Dim, Dim>;

    bool verbose{false};

    T2_t stress(Dim, Dim);
    stress << 1, 0.2, 0, 0, 2, 0, 0, 0, 3;
    if (verbose) {
      std::cout << "stress:\n" << stress << std::endl;
    }
    T2_t sigma_dev{MatTB::compute_deviatoric<Dim>(stress)};
    T2_t sigma_dev_analytic{Eigen::Matrix<Real, Dim, Dim>::Zero()};
    sigma_dev_analytic(0, 0) = -1;
    sigma_dev_analytic(0, 1) = 0.2;
    sigma_dev_analytic(2, 2) = 1;
    if (verbose) {
      std::cout << "deviatoric stress:\n" << sigma_dev << std::endl;
    }
    auto sigma_dev_error{(sigma_dev - sigma_dev_analytic).norm() /
                         sigma_dev.norm()};
    BOOST_CHECK_LT(sigma_dev_error, tol);
  }

  BOOST_AUTO_TEST_CASE(equivalent_von_Mises_stress_test) {
    constexpr Dim_t Dim{threeD};
    using T2_t = Eigen::Matrix<Real, Dim, Dim>;
    using T2Map_t = Eigen::Map<const Eigen::Matrix<Real, Dim, Dim>>;

    bool verbose{false};

    T2_t stress(Dim, Dim);
    stress << 1, 0.2, 0, 0, 2, 0, 0, 0, 3;
    if (verbose) {
      std::cout << "stress:\n" << stress << std::endl;
    }
    const auto * p = &stress(0, 0);  // pointer to stress
    if (verbose) {
      std::cout << "*p: " << p << std::endl;
    }

    T2Map_t stress_map(p, 3, 3);
    auto sigma_eq{MatTB::compute_equivalent_von_Mises_stress<Dim>(stress_map)};
    Real sigma_eq_analytic{1.7492855684535902};  // computed with python
    if (verbose) {
      std::cout << "equivalen von Mises stress:" << sigma_eq << std::endl;
    }
    BOOST_CHECK_CLOSE(sigma_eq, sigma_eq_analytic, 1e-8);
  }
  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muSpectre

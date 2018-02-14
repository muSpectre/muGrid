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

#include <Eigen/Dense>

#include "tests.hh"
#include "materials/materials_toolbox.hh"
#include "common/T4_map_proxy.hh"
#include "common/tensor_algebra.hh"
#include "tests/test_goodies.hh"


namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(materials_toolbox)

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_strain_conversion, Fix,
                                   testGoodies::dimlist, Fix){
    constexpr Dim_t dim{Fix::dim};
    using T2 = Eigen::Matrix<Real, dim, dim>;

    T2 F{(T2::Random() -.5*T2::Ones())/20 + T2::Identity()};

    // checking Green-Lagrange
    T2 Eref = .5*(F.transpose()*F-T2::Identity());

    T2 E_tb = MatTB::convert_strain<StrainMeasure::Gradient,
                                      StrainMeasure::GreenLagrange>
      (Eigen::Map<Eigen::Matrix<Real, dim, dim>>(F.data()));

    Real error = (Eref-E_tb).norm();
    BOOST_CHECK_LT(error, tol);

    // checking Left Cauchy-Green
    Eref = F*F.transpose();
    E_tb = MatTB::convert_strain<StrainMeasure::Gradient,
                                 StrainMeasure::LCauchyGreen>(F);

    error = (Eref-E_tb).norm();
    BOOST_CHECK_LT(error, tol);

    // checking Right Cauchy-Green
    Eref = F.transpose()*F;
    E_tb = MatTB::convert_strain<StrainMeasure::Gradient,
                                 StrainMeasure::RCauchyGreen>(F);

    error = (Eref-E_tb).norm();
    BOOST_CHECK_LT(error, tol);

    // checking Hencky (logarithmic) strain
    Eref = F.transpose()*F;
    Eigen::SelfAdjointEigenSolver<T2> EigSolv(Eref);
    Eref.setZero();
    for (size_t i{0}; i < dim; ++i) {
      auto && vec = EigSolv.eigenvectors().col(i);
      auto && val = EigSolv.eigenvalues()(i);
      Eref += .5*std::log(val) * vec*vec.transpose();
    }

    E_tb = MatTB::convert_strain<StrainMeasure::Gradient,
                                 StrainMeasure::Log>(F);

    error = (Eref-E_tb).norm();
    BOOST_CHECK_LT(error, tol);

    auto F_tb = MatTB::convert_strain<StrainMeasure::Gradient, StrainMeasure::Gradient>(F);

    error = (F-F_tb).norm();
    BOOST_CHECK_LT(error, tol);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(dumb_tensor_mult_test, Fix,
                                   testGoodies::dimlist, Fix) {
    constexpr Dim_t dim{Fix::dim};
    using T4 = T4Mat<Real, dim>;
    T4 A,B, R1, R2;
    A.setRandom();
    B.setRandom();
    R1 = A*B;
    R2.setZero();
    for (Dim_t i = 0; i < dim; ++i) {
      for (Dim_t j = 0; j < dim; ++j) {
        for (Dim_t a = 0; a < dim; ++a) {
          for (Dim_t b = 0; b < dim; ++b) {
            for (Dim_t k = 0; k < dim; ++k) {
              for (Dim_t l = 0; l < dim; ++l) {
                get(R2,i,j,k,l) += get(A, i,j,a,b)*get(B, a,b, k,l);
              }
            }
          }
        }
      }
    }
    auto error{(R1-R2).norm()};
    BOOST_CHECK_LT(error, tol);
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_PK1_stress, Fix,
                                   testGoodies::dimlist, Fix) {
    using namespace Matrices;
    constexpr Dim_t dim{Fix::dim};
    using T2 = Eigen::Matrix<Real, dim, dim>;
    using T4 = T4Mat<Real, dim>;
    testGoodies::RandRange<Real> rng;
    T2 F=T2::Identity()*2 ;
    //F.setRandom();
    T2 E_tb = MatTB::convert_strain<StrainMeasure::Gradient, StrainMeasure::GreenLagrange>
      (Eigen::Map<Eigen::Matrix<Real, dim, dim>>(F.data()));
    Real lambda = 3;//rng.randval(1, 2);
    Real mu = 4;//rng.randval(1,2);
    T4 J = Itrac<dim>();
    T2 I = I2<dim>();
    T4 I4 = Isymm<dim>();
    T4 C = lambda*J + 2*mu*I4;
    T2 S = tensmult(C, E_tb);
    T2 Sref = lambda*E_tb.trace()*I + 2*mu*E_tb;

    auto error{(Sref-S).norm()};
    BOOST_CHECK_LT(error, tol);

    T4 K = outer_under(I,S) + outer_under(F,I)*C*outer_under(F.transpose(),I);

    // See Curnier, 2000, "Méthodes numériques en mécanique des solides", p 252
    T4 Kref;
    Real Fkrkr = (F.array()*F.array()).sum();
    T2 Fkmkn = F.transpose()*F;
    T2 Fisjs = F*F.transpose();
    Kref.setZero();
    for (Dim_t i = 0; i < dim; ++i) {
      for (Dim_t j = 0; j < dim; ++j) {
        for (Dim_t m = 0; m < dim; ++m) {
          for (Dim_t n = 0; n < dim; ++n) {
            get(Kref, i, m, j, n) =
              (lambda*((Fkrkr-dim)/2 * I(i,j)*I(m,n) + F(i,m)*F(j,n)) +
               mu * (I(i,j)*Fkmkn(m,n) + Fisjs(i,j)*I(m,n) -
                     I(i,j) *I(m,n) + F(i,n)*F(j,m)));
          }
        }
      }
    }
    error = (Kref-K).norm();
    BOOST_CHECK_LT(error, tol);

    T2 P = MatTB::PK1_stress<StressMeasure::PK2, StrainMeasure::GreenLagrange>(F, S);
    T2 Pref = F*S;
    error = (P-Pref).norm();
    BOOST_CHECK_LT(error, tol);

    auto && stress_tgt = MatTB::PK1_stress<StressMeasure::PK2, StrainMeasure::GreenLagrange>(F, S, C);
    T2 P_t = std::move(std::get<0>(stress_tgt));
    T4 K_t = std::move(std::get<1>(stress_tgt));
    error = (P_t-Pref).norm();
    BOOST_CHECK_LT(error, tol);

    error = (K_t-Kref).norm();
    BOOST_CHECK_LT(error, tol);

    auto && stress_tgt_trivial =
      MatTB::PK1_stress<StressMeasure::PK1, StrainMeasure::Gradient>(F, P, K);
    T2 P_u = std::move(std::get<0>(stress_tgt_trivial));
    T4 K_u = std::move(std::get<1>(stress_tgt_trivial));

    error = (P_u-Pref).norm();
    BOOST_CHECK_LT(error, tol);

    error = (K_u-Kref).norm();
    BOOST_CHECK_LT(error, tol);

    T2 P_g;
    T4 K_g;
    std::tie(P_g, K_g) = testGoodies::objective_hooke_explicit(lambda, mu, F);

    error = (P_g-Pref).norm();
    BOOST_CHECK_LT(error, tol);

    error = (K_g-Kref).norm();
    BOOST_CHECK_LT(error, tol);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre

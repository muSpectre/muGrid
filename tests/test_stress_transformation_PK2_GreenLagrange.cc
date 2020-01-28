/**
 * @file   test_stress_transformation_PK2_GreenLagrange.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   22 Jan 2020
 *
 * @brief  Testing the stress transformation using an anisotropic cons. law
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

#include "test_stress_transformation.hh"

namespace muSpectre {
  BOOST_AUTO_TEST_SUITE(stress_transformations_GreenLagrange_PK2);

  using mats_PK2_GreenLagrange =
      boost::mpl::list<STMatFixture<twoD, StrainMeasure::GreenLagrange,
                                    StressMeasure::PK2, PureSphericalStrain>,
                       STMatFixture<twoD, StrainMeasure::GreenLagrange,
                                    StressMeasure::PK2, PureShearStrain>,
                       STMatFixture<twoD, StrainMeasure::GreenLagrange,
                                    StressMeasure::PK2, CombinedStrain>,
                       STMatFixture<threeD, StrainMeasure::GreenLagrange,
                                    StressMeasure::PK2, PureSphericalStrain>,
                       STMatFixture<threeD, StrainMeasure::GreenLagrange,
                                    StressMeasure::PK2, PureShearStrain>,
                       STMatFixture<threeD, StrainMeasure::GreenLagrange,
                                    StressMeasure::PK2, CombinedStrain>>;

  // ----------------------------------------------------------------------
  // checking that C_voigt is positive definite and symmetric
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(C_test, Fix, mats_PK2_GreenLagrange, Fix) {
    using V_t = typename Fix::V_t;

    const V_t C_voigt{Fix::get_C_voigt()};
    const V_t C_voigt_T{C_voigt.transpose()};

    constexpr Real tol{1e-10};
    Real err{rel_error(C_voigt, C_voigt_T)};
    BOOST_CHECK_LT(err, tol);

    if (not(err < tol)) {
      std::cout << std::endl
                << "The stiffness is not symmetric and its difference with its "
                   "transpose is:"
                << std::endl
                << C_voigt - C_voigt_T << std::endl;
    }

    // computing the eigen values of the stiffenss matrix:
    Eigen::SelfAdjointEigenSolver<V_t> es(C_voigt);
    Eigen::Matrix<Real, vsize(Fix::mdim()), 1> eigen_values{es.eigenvalues()};

    for (int i{0}; i < eigen_values.rows(); ++i) {
      Real eigen_value{eigen_values(i)};
      BOOST_CHECK_GT(eigen_value, 0);

      if (eigen_value <= 0) {
        std::cout << "C has a non-positive eigen value equal to: "
                  << eigen_value << std::endl;
      }
    }
  }

  // ----------------------------------------------------------------------
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(PK2_GreenLagrange, Fix,
                                   mats_PK2_GreenLagrange, Fix) {
    using T2_t = typename Fix::T2_t;
    using T4_t = typename Fix::T4_t;

    auto & material{*std::get<0>(Fix::mat_eval)};
    auto & evaluator{std::get<1>(Fix::mat_eval)};

    auto && F{Fix::F};

    const T2_t E{.5 * (F * F.transpose() - T2_t::Identity())};

    material.add_pixel(0);
    material.set_F(F);

    auto && nonlin_step{Fix::get_nonlin_step()};
    auto && tol{Fix::get_tol()};
    auto && C{material.get_C()};

    T2_t S{material.evaluate_stress(E)};
    T4_t C_estim{
        evaluator.estimate_tangent(E, Formulation::native, nonlin_step)};

    Real err0{rel_error(C, C_estim)};
    BOOST_CHECK_LT(err0, tol);

    // Using the closed form (in contrary to index expanded form) of the stress
    // transformation gives us compact and pretty convenient form of the tangent
    // transforamtion. However, considering using products of identity these
    // formulations are not the most computationally efficient ways to
    // implement tangent transforamtions and one may need to implement the
    // index expnaded form to maximize efficiency. By the way, the closed form
    // can give us another checkpoint to verfiy the index expanded notion of the
    // tangent transformation
    T4_t S_T4{Matrices::outer_under(T2_t::Identity(), S)};
    T4_t F_T4{Matrices::outer_under(F, T2_t::Identity())};
    T4_t F_T_T4{Matrices::outer_under(F.transpose(), T2_t::Identity())};
    // K = [I _⊗ S] + [F_⊗ I] C [Fᵀ _⊗ I]
    T4_t K_closed{S_T4 + F_T4 * C_estim * F_T_T4};

    // Note: Using evaluator K estimation, we avoid using the transforamtion
    // implemented for stiffness. Instead, merely the transformation functions
    // of stresses is utilized. Considereing that the implementation of the
    // stress transforamtion (not stiffness) is rather straightforward and it is
    // supposedly easy to avoid any mistake in their implemenation. Therefore,
    // we can use this estimation also as a referenece to verify the
    // transforamtio of the tangents.
    T4_t K_estim{
        evaluator.estimate_tangent(F, Formulation::finite_strain, nonlin_step)};

    // Utilizing, explictly, the implementation of the
    // streess_tange_transforamtions implemented in the materials folder. (this
    // is the very thing that we need to verfiy)
    auto && P_K_stress_tangent_convertion{
        MatTB::PK1_stress<StressMeasure::PK2, StrainMeasure::GreenLagrange>(
            F, S, C_estim)};

    T4_t K_stress_tangent_convertion{
        std::get<1>(P_K_stress_tangent_convertion)};

    Real err1{rel_error(K_closed, K_estim)};
    BOOST_CHECK_LT(err1, tol);

    Real err2{rel_error(K_stress_tangent_convertion, K_estim)};
    BOOST_CHECK_LT(err2, tol);

    if (not(err0 < tol) or not(err1 < tol) or not(err2 < tol)) {
      std::cout << Fix::get_strain_state() << std::endl;
      std::cout << "F:" << std::endl << F << std::endl << std::endl;
      std::cout << "E:" << std::endl << E << std::endl << std::endl;
      std::cout << "S:" << std::endl << S << std::endl << std::endl;
      std::cout << "C:E:" << std::endl
                << Matrices::tensmult(C, E) << std::endl
                << std::endl;
      std::cout << "C(∂S/∂E):" << std::endl << C << std::endl << std::endl;
      std::cout << "Estimated C(∂S/∂E):" << std::endl
                << C_estim << std::endl
                << std::endl;
      std::cout
          << "Estiamted K  of STMateriallinearelasticgeneric1 (Refernece):"
          << std::endl
          << K_estim << std::endl
          << std::endl;

      std::cout << "Closed form K (convertion applied on Estimated C ):"
                << std::endl
                << K_closed << std::endl
                << std::endl;

      std::cout
          << "K(Implemetend stress_tangent conversion applied on Estimated C:"
          << std::endl
          << K_stress_tangent_convertion << std::endl
          << std::endl;
    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(PK2_GreenLagrange_small_strain, Fix,
                                   mats_PK2_GreenLagrange, Fix) {
    using T2_t = typename Fix::T2_t;
    using T4_t = typename Fix::T4_t;

    auto & material{*std::get<0>(Fix::mat_eval)};
    auto & evaluator{std::get<1>(Fix::mat_eval)};

    auto && F{Fix::F};

    const T2_t E{.5 * (F * F.transpose() - T2_t::Identity())};

    material.add_pixel(0);
    material.set_F(F);
    constexpr Real linear_step{1.};
    constexpr Real nonlin_step{1.e-8};
    constexpr Real tol{25 * nonlin_step};

    auto && C{material.get_C()};

    T2_t S{material.evaluate_stress(E)};
    T4_t C_estim{
        evaluator.estimate_tangent(E, Formulation::small_strain, linear_step)};

    Real err0{rel_error(C, C_estim)};
    BOOST_CHECK_LT(err0, tol);

    // Using the closed form (in contrary to index expanded form) of the stress
    // transformation gives us compact and pretty convenient form of the tangent
    // transformation. However, considering using products of identity these
    // formulations are not the most computationally efficient ways to
    // implement tangent transformations and one may need to implement the
    // index expanded form to maximize efficiency. By the way, the closed form
    // can give us another checkpoint to verfiy the index expanded notation of
    // the tangent transformation
    T4_t S_T4{Matrices::outer_under(T2_t::Identity(), S)};
    T4_t F_T4{Matrices::outer_under(F, T2_t::Identity())};
    T4_t F_T_T4{Matrices::outer_under(F.transpose(), T2_t::Identity())};
    // K = [I _⊗ S] + [F_⊗ I] C [Fᵀ _⊗ I]
    T4_t K_closed{S_T4 + F_T4 * C_estim * F_T_T4};

    // Note: Using evaluator K estimation, we avoid using the transformation
    // implemented for stiffness. Instead, merely the transformation functions
    // of stresses is utilized. Considereing that the implementation of the
    // stress transformation (not stiffness) is rather straightforward and it is
    // supposedly easy to avoid any mistake in their implementation. Therefore,
    // we can use this estimation also as a referenece to verify the
    // transformation of the tangents.
    T4_t K_estim{
        evaluator.estimate_tangent(F, Formulation::finite_strain, nonlin_step)};

    // Utilizing, explictly, the implementation of the
    // streess_tangent_transformations implemented in the materials folder.
    // (this is the very thing that we need to verfiy)
    auto && P_K_stress_tangent_conversion{
        MatTB::PK1_stress<StressMeasure::PK2, StrainMeasure::GreenLagrange>(
            F, S, C_estim)};

    T4_t K_stress_tangent_conversion{
        std::get<1>(P_K_stress_tangent_conversion)};

    Real err1{rel_error(K_closed, K_estim)};
    BOOST_CHECK_LT(err1, tol);

    Real err2{rel_error(K_stress_tangent_conversion, K_estim)};
    BOOST_CHECK_LT(err2, tol);

    if (not(err0 < tol) or not(err1 < tol) or not(err2 < tol)) {
      std::cout << Fix::get_strain_state() << std::endl;
      std::cout << "F:" << std::endl << F << std::endl << std::endl;
      std::cout << "E:" << std::endl << E << std::endl << std::endl;
      std::cout << "S:" << std::endl << S << std::endl << std::endl;
      std::cout << "C:E:" << std::endl
                << Matrices::tensmult(C, E) << std::endl
                << std::endl;
      std::cout << "C(∂S/∂E):" << std::endl << C << std::endl << std::endl;
      std::cout << "Estimated C(∂S/∂E):" << std::endl
                << C_estim << std::endl
                << std::endl;
      std::cout
          << "Estimated K  of STMateriallinearelasticgeneric1 (Refernece):"
          << std::endl
          << K_estim << std::endl
          << std::endl;

      std::cout << "Closed form K (conversion applied on Estimated C ):"
                << std::endl
                << K_closed << std::endl
                << std::endl;

      std::cout
          << "K(Implemented stress_tangent conversion applied on Estimated C:"
          << std::endl
          << K_stress_tangent_conversion << std::endl
          << std::endl;
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muSpectre

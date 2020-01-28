/**
 * @file   test_stress_transformation_Kirchhoff_Gradient.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   22 Jan 2020
 *
 * @brief  test_stress_transformation_PK2_GreenLagrange.cc
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
  BOOST_AUTO_TEST_SUITE(stress_transformations_Gradient_Kirchhoff);

  using mats_Kirchhoff_Gradient = boost::mpl::list<
      STMatFixture<twoD, StrainMeasure::Gradient, StressMeasure::Kirchhoff,
                   PureSphericalStrain>,
      STMatFixture<twoD, StrainMeasure::Gradient, StressMeasure::Kirchhoff,
                   PureShearStrain>,
      STMatFixture<twoD, StrainMeasure::Gradient, StressMeasure::Kirchhoff,
                   CombinedStrain>,
      STMatFixture<threeD, StrainMeasure::Gradient, StressMeasure::Kirchhoff,
                   PureSphericalStrain>,
      STMatFixture<threeD, StrainMeasure::Gradient, StressMeasure::Kirchhoff,
                   PureShearStrain>,
      STMatFixture<threeD, StrainMeasure::Gradient, StressMeasure::Kirchhoff,
                   CombinedStrain>>;

  // ----------------------------------------------------------------------
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(Kirchhoff_Gradient, Fix,
                                   mats_Kirchhoff_Gradient, Fix) {
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

    T2_t tau{material.evaluate_stress(F)};
    T2_t S{MatTB::PK2_stress<StressMeasure::Kirchhoff, StrainMeasure::Gradient>(
        F, tau)};

    T4_t c_estim{
        evaluator.estimate_tangent(F, Formulation::native, nonlin_step)};

    T2_t F_inv{F.inverse()};
    T4_t I_Finv_T4{
        Matrices::outer_under(T2_t::Identity(), F_inv)};  //! [I _⊗  F⁻¹]
    T4_t tauFinvT_Finv_T4{Matrices::outer_over(tau * F_inv.transpose(),
                                               F_inv)};  //! [τF⁻ᵀ ⁻⊗ F⁻¹]

    // K = [I _⊗  F⁻¹] C - [τF⁻ᵀ ⁻⊗ F⁻¹]
    T4_t K_closed{I_Finv_T4 * c_estim - tauFinvT_Finv_T4};

    auto && P_K_stress_tangent_conversion{
        MatTB::PK1_stress<StressMeasure::Kirchhoff, StrainMeasure::Gradient>(
            F, tau, c_estim)};

    T4_t K_stress_tangent_conversion{
        std::get<1>(P_K_stress_tangent_conversion)};

    // Reference:
    T4_t K_estim{
        evaluator.estimate_tangent(F, Formulation::finite_strain, nonlin_step)};

    // K from C = ∂S/∂E
    T4_t K_from_C{std::get<1>(
        MatTB::PK1_stress<StressMeasure::PK2, StrainMeasure::GreenLagrange>(
            F, S, C))};

    Real err1{rel_error(K_closed, K_estim)};
    BOOST_CHECK_LT(err1, tol);

    Real err2{rel_error(K_stress_tangent_conversion, K_estim)};
    BOOST_CHECK_LT(err2, tol);

    Real err3{rel_error(K_stress_tangent_conversion, K_from_C)};
    BOOST_CHECK_LT(err3, tol);

    if (not(err1 < tol) or not(err2 < tol) or not(err3 < tol)) {
      std::cout << Fix::get_strain_state() << std::endl;
      std::cout << "F:" << std::endl << F << std::endl << std::endl;
      std::cout << "E:" << std::endl << E << std::endl << std::endl;
      std::cout << "τ:" << std::endl << tau << std::endl << std::endl;
      std::cout << "S from τ:" << std::endl << S << std::endl << std::endl;
      std::cout << "C:E:" << std::endl
                << Matrices::tensmult(C, E) << std::endl
                << std::endl;
      std::cout << "C(∂S/∂E):" << std::endl << C << std::endl << std::endl;
      std::cout << "Estimated c(∂τ/∂F):" << std::endl
                << c_estim << std::endl
                << std::endl;

      std::cout << "K from C = (∂S/∂E) (Refernece):" << std::endl
                << K_from_C << std::endl
                << std::endl;

      std::cout << "Estimated K (Refernece):" << std::endl
                << K_estim << std::endl
                << std::endl;

      std::cout << "Closed form K conversion applied on Estimated C :"
                << std::endl
                << K_closed << std::endl
                << std::endl;

      std::cout
          << "K(Implemented stress_tangent conversion applied on Estimated C):"
          << std::endl
          << K_stress_tangent_conversion << std::endl
          << std::endl;
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muSpectre

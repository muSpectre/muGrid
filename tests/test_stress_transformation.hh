/**
 * @file   test_stress_transformation.hh
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   22 Jan 2020
 *
 * @brief  The Fixture used in stress transformation tests
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

#include "common/voigt_conversion.hh"
#include "materials/material_linear_elastic1.hh"
#include "materials/s_t_material_linear_elastic_generic1.hh"
#include "materials/stress_transformations_Kirchhoff.hh"

#include <memory>

#ifndef TESTS_TEST_STRESS_TRANSFORMATION_HH_
#define TESTS_TEST_STRESS_TRANSFORMATION_HH_

namespace muSpectre {

  using muGrid::testGoodies::rel_error;

  constexpr Dim_t PureSphericalStrain{1};
  constexpr Dim_t PureShearStrain{2};
  constexpr Dim_t CombinedStrain{3};

  template <Dim_t Dim, StrainMeasure StrainM, StressMeasure StressM,
            Dim_t StrainState = PureSphericalStrain>
  struct STMatFixture {
    using Mat_t = STMaterialLinearElasticGeneric1<Dim, StrainM, StressM>;

    using T2_t = Eigen::Matrix<Real, Dim, Dim>;
    using T4_t = muGrid::T4Mat<Real, Dim>;
    using V_t = Eigen::Matrix<Real, vsize(Dim), vsize(Dim)>;

    constexpr static Real lambda{2}, mu{1.5};
    constexpr static Real get_lambda() { return lambda; }
    constexpr static Real get_mu() { return mu; }
    constexpr static Real young{mu * (3 * lambda + 2 * mu) / (lambda + mu)};
    constexpr static Real poisson{lambda / (2 * (lambda + mu))};

    constexpr static Dim_t mdim() { return Dim; }
    constexpr static Dim_t NbQuadPts() { return 1; }

    constexpr static Real nonlin_step{1.e-6}, tol{1.e-4};
    constexpr static Real get_nonlin_step() { return nonlin_step; }
    constexpr static Real get_tol() { return tol; }

    std::string get_strain_state() {
      switch (StrainState) {
      case 1: {
        return "Pure Spherical Strain";
      }
      case 2: {
        return "Pure Shear Strain";
      }
      case 3: {
        return "Combined Shear and Spherical Strain";
      }
      default: {
        throw std::runtime_error(
            "The strain state is not defined that allowed strain states "
            "are\n1.pure_spherical_strain\n2.pure_shear_strain\n3.combined_"
            "strain");
      }
      }
    }

    //! constructor
    STMatFixture()
        : C_voigt_holder{std::make_unique<V_t>(make_C_voigt())},
          C_voigt{*this->C_voigt_holder},
          F_holder{std::make_unique<T2_t>(make_F())}, F{*this->F_holder},
          mat_eval(Mat_t::make_evaluator(C_voigt)) {}

    static V_t make_C_voigt() {
      V_t C_voigt{};
      C_voigt.setZero();
      C_voigt.template topLeftCorner<Dim, Dim>().setConstant(get_lambda());
      C_voigt.template topLeftCorner<Dim, Dim>() +=
          2 * get_mu() * T2_t::Identity();
      constexpr Dim_t Rest{vsize(Dim) - Dim};
      using Rest_t = Eigen::Matrix<Real, Rest, Rest>;
      C_voigt.template bottomRightCorner<Rest, Rest>() +=
          get_mu() * Rest_t::Identity();

      for (int i{0}; i < vsize(Dim); ++i) {
        for (int j{0}; j < vsize(Dim); ++j) {
          C_voigt(i, j) *= ((i + j) * 0.5 + 1);
        }
      }
      return C_voigt;
    }

    const V_t & get_C_voigt() const { return this->C_voigt; }

    static T2_t make_F() {
      auto && dim{mdim()};
      T2_t F{T2_t::Zero()};
      T2_t shear_F{T2_t::Zero()};

      Real spherical_strain_coeff{0.5};
      Real shear_strain_coeff{0.3};

      auto make_shear{[&dim](const auto & coeff) {
        auto && shear_value{(sqrt(1.0 + coeff) - sqrt(1.0 - coeff)) / 2.0};
        auto && sphere_value{(sqrt(1.0 + coeff) + sqrt(1.0 - coeff)) / 2.0};
        switch (dim) {
        case 2: {
          T2_t shear_F{
              (T2_t() << sphere_value, shear_value, shear_value, sphere_value)
                  .finished()};
          return shear_F;
          break;
        }
        case 3: {
          T2_t shear_F{(T2_t() << sphere_value, shear_value, 0.0, shear_value,
                        sphere_value, 0.0, 0.0, 0.0, sphere_value)
                           .finished()};
          return shear_F;
          break;
        }
        default:
          throw(std::runtime_error("Invalid dimension"));
          break;
        }
      }};

      switch (StrainState) {
      case 1: {
        F = (1.0 + spherical_strain_coeff) * T2_t::Identity();
        break;
      }
      case 2: {
        F = make_shear(shear_strain_coeff);
        Real J{F.determinant()};
        F = F / (std::pow(J, (1.0 / dim)));
        break;
      }
      case 3: {
        F = (1.0 + spherical_strain_coeff) * T2_t::Identity() +
            make_shear(shear_strain_coeff);
        break;
      }
      default: {
        throw(std::runtime_error(
            "The strain state is not defined that allowed strain states "
            "are\n1.pure_spherical_strain\n2.pure_shear_strain\n3.combined_"
            "strain"));
      }
      }
      return F;
    }

    std::unique_ptr<V_t> C_voigt_holder;
    V_t & C_voigt;

    std::unique_ptr<T2_t> F_holder;
    const T2_t & F;

    std::tuple<std::shared_ptr<Mat_t>, MaterialEvaluator<Dim>> mat_eval;
  };

}  // namespace muSpectre

#endif  // TESTS_TEST_STRESS_TRANSFORMATION_HH_

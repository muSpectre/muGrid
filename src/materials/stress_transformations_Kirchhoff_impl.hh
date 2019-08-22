/**
 * @file   stress_transformations_Kirchhoff_impl.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   29 Oct 2018
 *
 * @brief  Implementation of stress conversions for Kirchhoff stress
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
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#ifndef SRC_MATERIALS_STRESS_TRANSFORMATIONS_KIRCHHOFF_IMPL_HH_
#define SRC_MATERIALS_STRESS_TRANSFORMATIONS_KIRCHHOFF_IMPL_HH_

namespace muSpectre {

  namespace MatTB {

    namespace internal {

     // ----------------------------------------------------------------------
      /**
       * Specialisation for the case where we get Kirchhoff stress (τ)
       */
      template <Dim_t Dim, StrainMeasure StrainM>
      struct PK1_stress<Dim, StressMeasure::Kirchhoff, StrainM>
          : public PK1_stress<Dim, StressMeasure::no_stress_,
                              StrainMeasure::no_strain_> {
        //! returns the converted stress
        template <class Strain_t, class Stress_t>
        inline static decltype(auto) compute(Strain_t && F, Stress_t && tau) {
          return tau * F.inverse().transpose();
        }
      };

     // ----------------------------------------------------------------------
      /**
       * Specialisation for the case where we get Kirchhoff stress (τ) derived
       * with respect to Gradient
       */
      template <Dim_t Dim>
      struct PK1_stress<Dim, StressMeasure::Kirchhoff, StrainMeasure::Gradient>
          : public PK1_stress<Dim, StressMeasure::Kirchhoff,
                              StrainMeasure::no_strain_> {
        //! short-hand
        using Parent = PK1_stress<Dim, StressMeasure::Kirchhoff,
                                  StrainMeasure::no_strain_>;
        using Parent::compute;

        //! returns the converted stress and stiffness
        template <class Strain_t, class Stress_t, class Tangent_t>
        inline static decltype(auto) compute(Strain_t && F, Stress_t && tau,
                                             Tangent_t && C) {
          using muGrid::get;
          using T4_t = muGrid::T4Mat<Real, Dim>;
          using Mat_t = Eigen::Matrix<Real, Dim, Dim>;
          Mat_t F_inv{F.inverse()};
          T4_t increment{T4_t::Zero()};
          for (int i{0}; i < Dim; ++i) {
            const int k{i};
            for (int j{0}; j < Dim; ++j) {
              const int a{j};
              for (int l{0}; l < Dim; ++l) {
                get(increment, i, j, k, l) -= tau(a, l);
              }
            }
          }
          T4_t Ka{C + increment};

          T4_t Kb{T4_t::Zero()};
          for (int i{0}; i < Dim; ++i) {
            for (int j{0}; j < Dim; ++j) {
              for (int k{0}; k < Dim; ++k) {
                for (int l{0}; l < Dim; ++l) {
                  for (int a{0}; a < Dim; ++a) {
                    for (int b{0}; b < Dim; ++b) {
                      get(Kb, j, i, k, l) +=
                          F_inv(i, a) * get(Ka, a, j, k, b) * F_inv(l, b);
                    }
                  }
                }
              }
            }
          }
          Mat_t P = tau * F_inv.transpose();
          return std::make_tuple(std::move(P), std::move(Kb));
        }
      };

    }  // namespace internal

  }  // namespace MatTB

}  // namespace muSpectre

#endif  // SRC_MATERIALS_STRESS_TRANSFORMATIONS_KIRCHHOFF_IMPL_HH_

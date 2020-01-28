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

#ifndef SRC_MATERIALS_STRESS_TRANSFORMATIONS_KIRCHHOFF_IMPL_HH_
#define SRC_MATERIALS_STRESS_TRANSFORMATIONS_KIRCHHOFF_IMPL_HH_
namespace muSpectre {
  namespace MatTB {
    namespace internal {
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
          using T4_t = muGrid::T4Mat<Real, Dim>;
          using T2_t = Eigen::Matrix<Real, Dim, Dim>;
          T2_t F_inv{F.inverse()};
          T4_t K{T4_t::Zero()};

          // K = [I _⊗  F⁻¹] c - [τF⁻ᵀ ⁻⊗ F⁻¹]
          for (Dim_t i{0}; i < Dim; ++i) {
            for (Dim_t j{0}; j < Dim; ++j) {
              for (Dim_t k{0}; k < Dim; ++k) {
                for (Dim_t l{0}; l < Dim; ++l) {
                  for (Dim_t n{0}; n < Dim; ++n) {
                    get(K, i, j, k, l) += F_inv(j, n) * get(C, i, n, k, l);
                  }
                  for (Dim_t a{0}; a < Dim; ++a) {
                    get(K, i, j, k, l) -=
                        (tau(i, a) * F_inv(l, a) * F_inv(j, k));
                  }
                }
              }
            }
          }
          T2_t P{tau * F_inv.transpose()};
          return std::make_tuple(std::move(P), std::move(K));
        }
      };

      /**
       * Specialisation for the case where we get Kirchhoff stress (τ) derived
       * with respect to GreenLagrange
       */
      template <Dim_t Dim>
      struct PK1_stress<Dim, StressMeasure::Kirchhoff,
                        StrainMeasure::GreenLagrange>
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
          using T4_t = muGrid::T4Mat<Real, Dim>;
          using T2_t = Eigen::Matrix<Real, Dim, Dim>;
          T2_t F_inv{F.inverse()};
          T4_t K{T4_t::Zero()};

          // K = [I _⊗  F⁻¹] C [Fᵀ _⊗  I] - [τF⁻ᵀ ⁻⊗ F⁻¹]
          for (Dim_t i{0}; i < Dim; ++i) {
            for (Dim_t j{0}; j < Dim; ++j) {
              for (Dim_t k{0}; k < Dim; ++k) {
                for (Dim_t l{0}; l < Dim; ++l) {
                  for (Dim_t n{0}; n < Dim; ++n) {
                    for (Dim_t s{0}; s < Dim; ++s) {
                      get(K, i, j, k, l) +=
                          F_inv(j, n) * (get(C, i, n, s, l) * F(s, k));
                    }
                  }
                  for (Dim_t a{0}; a < Dim; ++a) {
                    get(K, i, j, k, l) -=
                        (tau(i, a) * F_inv(l, a) * F_inv(j, k));
                  }
                }
              }
            }
          }
          T2_t P{tau * F_inv.transpose()};

          return std::make_tuple(std::move(P), std::move(K));
        }
      };

      /**
       * Specialisation for the case where we get Kirchhoff stress (τ) and we
       * need PK2(S)
       */
      template <Dim_t Dim, StrainMeasure StrainM>
      struct PK2_stress<Dim, StressMeasure::Kirchhoff, StrainM>
          : public PK2_stress<Dim, StressMeasure::no_stress_,
                              StrainMeasure::no_strain_> {
        //! returns the converted stress
        template <class Strain_t, class Stress_t>
        inline static decltype(auto) compute(Strain_t && F, Stress_t && tau) {
          return F.inverse() * tau * F.inverse().transpose();
        }
      };

    }  // namespace internal

  }  // namespace MatTB

}  // namespace muSpectre

#endif  // SRC_MATERIALS_STRESS_TRANSFORMATIONS_KIRCHHOFF_IMPL_HH_

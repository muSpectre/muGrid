/**
 * @file   stress_transformations_PK2_impl.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   29 Oct 2018
 *
 * @brief  Implementation of stress conversions for PK2 stress
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

#ifndef SRC_MATERIALS_STRESS_TRANSFORMATIONS_PK2_IMPL_HH_
#define SRC_MATERIALS_STRESS_TRANSFORMATIONS_PK2_IMPL_HH_

namespace muSpectre {

  namespace MatTB {

    namespace internal {

      /* ----------------------------------------------------------------------
       */
      /**
       * Specialisation for the case where we get material stress
       * (Piola-Kirchhoff-2, PK2)
       */
      template <Dim_t Dim, StrainMeasure StrainM>
      struct PK1_stress<Dim, StressMeasure::PK2, StrainM>
          : public PK1_stress<Dim, StressMeasure::no_stress_,
                              StrainMeasure::no_strain_> {
        //! returns the converted stress
        template <class Strain_t, class Stress_t>
        inline static decltype(auto) compute(Strain_t && F, Stress_t && S) {
          return F * S;
        }
      };

      /* ----------------------------------------------------------------------
       */
      /**
       * Specialisation for the case where we get material stress
       * (Piola-Kirchhoff-2, PK2) derived with respect to
       * Green-Lagrange strain
       */
      template <Dim_t Dim>
      struct PK1_stress<Dim, StressMeasure::PK2, StrainMeasure::GreenLagrange>
          : public PK1_stress<Dim, StressMeasure::PK2,
                              StrainMeasure::no_strain_> {
        //! base class
        using Parent =
            PK1_stress<Dim, StressMeasure::PK2, StrainMeasure::no_strain_>;
        using Parent::compute;

        //! returns the converted stress and stiffness
        template <class Strain_t, class Stress_t, class Tangent_t>
        inline static decltype(auto) compute(Strain_t && F, Stress_t && S,
                                             Tangent_t && C) {
          using T4 = typename std::remove_reference_t<Tangent_t>::PlainObject;
          using Tmap = T4MatMap<Real, Dim>;
          T4 K;
          Tmap Kmap{K.data()};
          K.setZero();

          for (int i = 0; i < Dim; ++i) {
            for (int m = 0; m < Dim; ++m) {
              for (int n = 0; n < Dim; ++n) {
                get(Kmap, i, m, i, n) += S(m, n);
                for (int j = 0; j < Dim; ++j) {
                  for (int r = 0; r < Dim; ++r) {
                    for (int s = 0; s < Dim; ++s) {
                      get(Kmap, i, m, j, n) +=
                          F(i, r) * get(C, r, m, s, n) * (F(j, s));
                    }
                  }
                }
              }
            }
          }
          auto && P =
              compute(std::forward<Strain_t>(F), std::forward<Stress_t>(S));
          return std::make_tuple(std::move(P), std::move(K));
        }
      };

      /* ----------------------------------------------------------------------
       */
      /**
       * Specialisation for the case where we get material stress
       * (Piola-Kirchhoff-2, PK2) derived with respect to
       * the placement Gradient (F)
       */
      template <Dim_t Dim>
      struct PK1_stress<Dim, StressMeasure::PK2, StrainMeasure::Gradient>
          : public PK1_stress<Dim, StressMeasure::PK2,
                              StrainMeasure::no_strain_> {
        //! base class
        using Parent =
            PK1_stress<Dim, StressMeasure::PK2, StrainMeasure::no_strain_>;
        using Parent::compute;

        //! returns the converted stress and stiffness
        template <class Strain_t, class Stress_t, class Tangent_t>
        inline static decltype(auto) compute(Strain_t && F, Stress_t && S,
                                             Tangent_t && C) {
          using T4 = typename std::remove_reference_t<Tangent_t>::PlainObject;
          using Tmap = T4MatMap<Real, Dim>;
          T4 K;
          Tmap Kmap{K.data()};
          K.setZero();

          for (int i = 0; i < Dim; ++i) {
            for (int m = 0; m < Dim; ++m) {
              for (int n = 0; n < Dim; ++n) {
                get(Kmap, i, m, i, n) += S(m, n);
                for (int j = 0; j < Dim; ++j) {
                  for (int r = 0; r < Dim; ++r) {
                    get(Kmap, i, m, j, n) += F(i, r) * get(C, r, m, j, n);
                  }
                }
              }
            }
          }
          auto && P =
              compute(std::forward<Strain_t>(F), std::forward<Stress_t>(S));
          return std::make_tuple(std::move(P), std::move(K));
        }
      };

    }  // namespace internal

  }  // namespace MatTB

}  // namespace muSpectre

#endif  // SRC_MATERIALS_STRESS_TRANSFORMATIONS_PK2_IMPL_HH_

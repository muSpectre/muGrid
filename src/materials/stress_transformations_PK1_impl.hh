/**
 * @file   stress_transformations_PK1_impl.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   29 Oct 2018
 *
 * @brief  implementation of stress conversion for PK1 stress
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

#ifndef SRC_MATERIALS_STRESS_TRANSFORMATIONS_PK1_IMPL_HH_
#define SRC_MATERIALS_STRESS_TRANSFORMATIONS_PK1_IMPL_HH_

#include "common/muSpectre_common.hh"
#include <libmugrid/T4_map_proxy.hh>

namespace muSpectre {

  namespace MatTB {

    namespace internal {

      // ----------------------------------------------------------------------

      /** Specialisation for the transparent case, where we already have
       * Piola-Kirchhoff-1, PK1
       */
      template <Dim_t Dim, StrainMeasure StrainM>
      struct PK1_stress<Dim, StressMeasure::PK1, StrainM>
          : public PK1_stress<Dim, StressMeasure::no_stress_,
                              StrainMeasure::no_strain_> {
        //! returns the converted stress
        template <class Strain_t, class Stress_t>
        inline static decltype(auto) compute(Strain_t && /*dummy*/,
                                             Stress_t && P) {
          return std::forward<Stress_t>(P);
        }
      };

      // ----------------------------------------------------------------------
      /** Specialisation for the transparent case, where we already have PK1
          stress *and* stiffness is given with respect to the transformation
          gradient
       **/
      template <Dim_t Dim>
      struct PK1_stress<Dim, StressMeasure::PK1, StrainMeasure::Gradient>
          : public PK1_stress<Dim, StressMeasure::PK1,
                              StrainMeasure::no_strain_> {
        //! base class
        using Parent =
            PK1_stress<Dim, StressMeasure::PK1, StrainMeasure::no_strain_>;
        using Parent::compute;

        //! returns the converted stress and stiffness
        template <class Strain_t, class Stress_t, class Tangent_t>
        inline static decltype(auto) compute(Strain_t && /*dummy*/,
                                             Stress_t && P, Tangent_t && K) {
          return std::make_tuple(std::forward<Stress_t>(P),
                                 std::forward<Tangent_t>(K));
        }
      };

      // ----------------------------------------------------------------------
      /**
       * Specialisation for the case where we get material stress
       * (Piola-Kirchhoff-1, PK1)
       */
      template <Dim_t Dim, StrainMeasure StrainM>
      struct PK2_stress<Dim, StressMeasure::PK1, StrainM>
          : public PK2_stress<Dim, StressMeasure::no_stress_,
                              StrainMeasure::no_strain_> {
        //! returns the converted stress
        template <class Strain_t, class Stress_t>
        inline static decltype(auto) compute(Strain_t && F, Stress_t && P) {
          return F.inverse() * P;
        }
      };

      // ----------------------------------------------------------------------
      /**
       * Specialisation for the case where we get material stress
       * (Piola-Kirchhoff-1, PK1) derived with respect to
       * the placement Gradient (F)
       */
      template <Dim_t Dim>
      struct PK2_stress<Dim, StressMeasure::PK1, StrainMeasure::Gradient>
          : public PK2_stress<Dim, StressMeasure::PK1,
                              StrainMeasure::no_strain_> {
        //! base class
        using Parent =
            PK2_stress<Dim, StressMeasure::PK1, StrainMeasure::no_strain_>;
        using Parent::compute;

        //! returns the converted stress and stiffness
        template <class Strain_t, class Stress_t, class Tangent_t>
        inline static decltype(auto) compute(Strain_t && F, Stress_t && P,
                                             Tangent_t && K) {
          using T4 = typename std::remove_reference_t<Tangent_t>::PlainObject;
          using T2 = typename std::remove_reference_t<Strain_t>::PlainObject;

          auto && S{
              compute(std::forward<Strain_t>(F), std::forward<Stress_t>(P))};

          auto && S_T4{muGrid::Matrices::outer_under(T2::Identity(), S)};
          auto && K_tmp{K - S_T4};

          T4 F_T4_inv{
              muGrid::Matrices::outer_under(F, T2::Identity()).inverse()};
          T4 F_T4_T_inv{
              (muGrid::Matrices::outer_under(F, T2::Identity()).transpose())
                  .inverse()};
          T4 C{F_T4_inv * K_tmp * F_T4_T_inv};

          return std::make_tuple(std::move(S), std::move(C));
        }
      };
      /* ---------------------------------------------------------------*/
    }  // namespace internal

  }  // namespace MatTB

}  // namespace muSpectre

#endif  // SRC_MATERIALS_STRESS_TRANSFORMATIONS_PK1_IMPL_HH_

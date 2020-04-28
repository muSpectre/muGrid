/**
 * @file   stress_transformations_default_case.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   29 Oct 2018
 *
 * @brief  default structure for stress conversions
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

#ifndef SRC_MATERIALS_STRESS_TRANSFORMATIONS_DEFAULT_CASE_HH_
#define SRC_MATERIALS_STRESS_TRANSFORMATIONS_DEFAULT_CASE_HH_

#include "common/muSpectre_common.hh"

#include <libmugrid/T4_map_proxy.hh>

namespace muSpectre {

  namespace MatTB {

    namespace internal {

      /** Structure for functions returning PK1 stress from other stress
       *measures
       **/
      template <Dim_t Dim, StressMeasure StressM, StrainMeasure StrainM>
      struct PK1_stress {
        //! returns the converted stress
        template <class Strain_t, class Stress_t>
        inline static decltype(auto) compute(Strain_t && /*strain*/,
                                             Stress_t && /*stress*/) {
          // the following test always fails to generate a compile-time error
          static_assert((StressM == StressMeasure::Cauchy) &&
                            (StressM == StressMeasure::PK1),
                        "The requested Stress conversion is not implemented. "
                        "You either made a programming mistake or need to "
                        "implement it as a specialisation of this function. "
                        "See PK2_stress<PK1,T1> for an example.");
        }

        //! returns the converted stress and stiffness
        template <class Strain_t, class Stress_t, class Tangent_t>
        inline static decltype(auto) compute(Strain_t && /*strain*/,
                                             Stress_t && /*stress*/,
                                             Tangent_t && /*stiffness*/) {
          // the following test always fails to generate a compile-time error
          static_assert((StressM == StressMeasure::Cauchy) &&
                            (StressM == StressMeasure::PK1),
                        "The requested Stress conversion is not implemented. "
                        "You either made a programming mistake or need to "
                        "implement it as a specialisation of this function. "
                        "See PK2_stress<PK1,T1, T2> for an example.");
        }
      };

      // ----------------------------------------------------------------------
      /** Structure for functions returning PK2 stress from other stress
       *measures
       **/
      template <Dim_t Dim, StressMeasure StressM, StrainMeasure StrainM>
      struct PK2_stress {
        //! returns the converted stress
        template <class Strain_t, class Stress_t>
        inline static decltype(auto) compute(Strain_t && /*strain*/,
                                             Stress_t && /*stress*/) {
          // the following test always fails to generate a compile-time error
          static_assert((StressM == StressMeasure::Cauchy) &&
                            (StressM == StressMeasure::PK2),
                        "The requested Stress conversion is not implemented. "
                        "You either made a programming mistake or need to "
                        "implement it as a specialisation of this function. "
                        "See PK1_stress<PK2,T1> for an example.");
        }

        //! returns the converted stress and stiffness
        template <class Strain_t, class Stress_t, class Tangent_t>
        inline static decltype(auto) compute(Strain_t && /*strain*/,
                                             Stress_t && /*stress*/,
                                             Tangent_t && /*stiffness*/) {
          // the following test always fails to generate a compile-time error
          static_assert((StressM == StressMeasure::Cauchy) &&
                            (StressM == StressMeasure::PK2),
                        "The requested Stress conversion is not implemented. "
                        "You either made a programming mistake or need to "
                        "implement it as a specialisation of this function. "
                        "See PK1_stress<PK2,T1, T2> for an example.");
        }
      };

      // ----------------------------------------------------------------------
      /** Structure for functions returning Kirchhoff stress from other stress
       *  measures
       **/
      template <Dim_t Dim, StressMeasure StressM, StrainMeasure StrainM>
      struct Kirchhoff_stress {
        //! returns the converted stress
        template <class Strain_t, class Stress_t>
        inline static decltype(auto) compute(Strain_t && /*strain*/,
                                             Stress_t && /*stress*/) {
          // the following test always fails to generate a compile-time error
          static_assert((StressM == StressMeasure::Cauchy) &&
                            (StressM == StressMeasure::Kirchhoff),
                        "The requested Stress conversion is not implemented. "
                        "You either made a programming mistake or need to "
                        "implement it as a specialisation of this function. "
                        "See PK1stress<PK2,T1, T2> for an example.");
        }

        //! returns the converted stress and stiffness
        template <class Strain_t, class Stress_t, class Tangent_t>
        inline static decltype(auto) compute(Strain_t && /*strain*/,
                                             Stress_t && /*stress*/,
                                             Tangent_t && /*stiffness*/) {
          // the following test always fails to generate a compile-time error
          static_assert((StressM == StressMeasure::Cauchy) &&
                            (StressM == StressMeasure::PK2),
                        "The requested Stress conversion is not implemented."
                        "You either made a programming mistake or need to "
                        "implement it as a specialisation of this function. "
                        "See PK1stress<PK2,T1, T2> for an example.");
        }
      };

    }  // namespace internal

  }  // namespace MatTB

}  // namespace muSpectre

#endif  // SRC_MATERIALS_STRESS_TRANSFORMATIONS_DEFAULT_CASE_HH_

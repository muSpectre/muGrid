/**
 * @file   stress_transformations.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   29 Oct 2018
 *
 * @brief  isolation of stress conversions for quicker compilation
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

#ifndef SRC_MATERIALS_STRESS_TRANSFORMATIONS_HH_
#define SRC_MATERIALS_STRESS_TRANSFORMATIONS_HH_

#include "common/muSpectre_common.hh"

#include <libmugrid/eigen_tools.hh>

namespace muSpectre {

  namespace MatTB {

    /* ---------------------------------------------------------------------- */
    //! set of functions returning an expression for PK1 stress based on
    template <StressMeasure StressM, StrainMeasure StrainM, class Stress_t,
              class Strain_t>
    decltype(auto) PK1_stress(Strain_t && strain, Stress_t && stress) {
      constexpr Dim_t dim{muGrid::EigenCheck::tensor_dim<Strain_t>::value};
      static_assert((dim == muGrid::EigenCheck::tensor_dim<Stress_t>::value),
                    "Stress and strain tensors have differing dimensions");
      return internal::PK1_stress<dim, StressM, StrainM>::compute(
          std::forward<Strain_t>(strain), std::forward<Stress_t>(stress));
    }

    /* ---------------------------------------------------------------------- */
    //! set of functions returning an expression for PK1 stress based on
    template <StressMeasure StressM, StrainMeasure StrainM, class Stress_t,
              class Strain_t, class Tangent_t>
    decltype(auto) PK1_stress(Strain_t && strain, Stress_t && stress,
                              Tangent_t && tangent) {
      constexpr Dim_t dim{muGrid::EigenCheck::tensor_dim<Strain_t>::value};
      static_assert((dim == muGrid::EigenCheck::tensor_dim<Stress_t>::value),
                    "Stress and strain tensors have differing dimensions");
      static_assert((dim == muGrid::EigenCheck::tensor_4_dim<Tangent_t>::value),
                    "Stress and tangent tensors have differing dimensions");
      return internal::PK1_stress<dim, StressM, StrainM>::compute(
          std::forward<Strain_t>(strain), std::forward<Stress_t>(stress),
          std::forward<Tangent_t>(tangent));
    }

    /* ---------------------------------------------------------------------- */
    //! set of functions returning an expression for PK2 stress based on
    template <StressMeasure StressM, StrainMeasure StrainM, class Stress_t,
              class Strain_t>
    decltype(auto) PK2_stress(Strain_t && strain, Stress_t && stress) {
      constexpr Dim_t dim{muGrid::EigenCheck::tensor_dim<Strain_t>::value};
      static_assert((dim == muGrid::EigenCheck::tensor_dim<Stress_t>::value),
                    "Stress and strain tensors have differing dimensions");
      return internal::PK2_stress<dim, StressM, StrainM>::compute(
          std::forward<Strain_t>(strain), std::forward<Stress_t>(stress));
    }

    /* ---------------------------------------------------------------------- */
    //! set of functions returning an expression for PK2 stress based on
    template <StressMeasure StressM, StrainMeasure StrainM, class Stress_t,
              class Strain_t, class Tangent_t>
    decltype(auto) PK2_stress(Strain_t && strain, Stress_t && stress,
                              Tangent_t && tangent) {
      constexpr Dim_t dim{muGrid::EigenCheck::tensor_dim<Strain_t>::value};
      static_assert((dim == muGrid::EigenCheck::tensor_dim<Stress_t>::value),
                    "Stress and strain tensors have differing dimensions");
      static_assert((dim == muGrid::EigenCheck::tensor_4_dim<Tangent_t>::value),
                    "Stress and tangent tensors have differing dimensions");

      return internal::PK2_stress<dim, StressM, StrainM>::compute(
          std::forward<Strain_t>(strain), std::forward<Stress_t>(stress),
          std::forward<Tangent_t>(tangent));
    }

    /* ---------------------------------------------------------------------- */
    //! set of functions returning an expression for Kirchhoff stress based on
    template <StressMeasure StressM, StrainMeasure StrainM, class Stress_t,
              class Strain_t>
    decltype(auto) Kirchhoff_stress(Strain_t && strain, Stress_t && stress) {
      constexpr Dim_t dim{muGrid::EigenCheck::tensor_dim<Strain_t>::value};
      static_assert((dim == muGrid::EigenCheck::tensor_dim<Stress_t>::value),
                    "Stress and strain tensors have differing dimensions");
      return internal::Kirchhoff_stress<dim, StressM, StrainM>::compute(
          std::forward<Strain_t>(strain), std::forward<Stress_t>(stress));
    }

  }  // namespace MatTB

}  // namespace muSpectre
#endif  // SRC_MATERIALS_STRESS_TRANSFORMATIONS_HH_

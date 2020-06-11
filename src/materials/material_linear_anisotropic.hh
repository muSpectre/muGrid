/**
 * @file   material_linear_anisotropic.hh
 *
 * @author Ali Falsafi<ali.falsafi@epfl.ch>
 *
 * @date   9 Jul 2018
 *
 * @brief  defenition of general anisotropic linear constitutive model
 *
 * Copyright © 2017 Till Junge
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

#ifndef SRC_MATERIALS_MATERIAL_LINEAR_ANISOTROPIC_HH_
#define SRC_MATERIALS_MATERIAL_LINEAR_ANISOTROPIC_HH_

#include "materials/stress_transformations_PK2.hh"
#include "materials/material_base.hh"
#include "materials/material_muSpectre_base.hh"
#include "materials/materials_toolbox.hh"
#include "common/muSpectre_common.hh"
#include "common/voigt_conversion.hh"

#include "libmugrid/T4_map_proxy.hh"
#include "libmugrid/tensor_algebra.hh"
#include "libmugrid/eigen_tools.hh"
#include "libmugrid/mapped_field.hh"

namespace muSpectre {

  template <Index_t DimM>
  class MaterialLinearAnisotropic;

  // traits for anisotropic material
  template <Index_t DimM>
  struct MaterialMuSpectre_traits<MaterialLinearAnisotropic<DimM>> {
    //! expected map type for strain fields
    using StrainMap_t =
        muGrid::T2FieldMap<Real, Mapping::Const, DimM, IterUnit::SubPt>;
    //! expected map type for stress fields
    using StressMap_t =
        muGrid::T2FieldMap<Real, Mapping::Mut, DimM, IterUnit::SubPt>;
    //! expected map type for tangent stiffness fields
    using TangentMap_t =
        muGrid::T4FieldMap<Real, Mapping::Mut, DimM, IterUnit::SubPt>;

    //! declare what type of strain measure your law takes as input
    constexpr static auto strain_measure{StrainMeasure::GreenLagrange};
    //! declare what type of stress measure your law yields as output
    constexpr static auto stress_measure{StressMeasure::PK2};
  };

  /**
   * Material implementation for anisotropic constitutive law
   */
  template <Index_t DimM>
  class MaterialLinearAnisotropic
      : public MaterialMuSpectre<MaterialLinearAnisotropic<DimM>, DimM> {
   public:
    //! base class
    using Parent = MaterialMuSpectre<MaterialLinearAnisotropic, DimM>;

    using Stiffness_t = muGrid::T4Mat<Real, DimM>;

    //! traits of this material
    using traits = MaterialMuSpectre_traits<MaterialLinearAnisotropic>;

    //! Hooke's law implementation
    using Hooke =
        typename MatTB::Hooke<DimM, typename traits::StrainMap_t::reference,
                              typename traits::TangentMap_t::reference>;

    //! Default constructor
    MaterialLinearAnisotropic() = delete;

    // constructor
    // a std::vector is utilized as the input of the constructor to
    // enable us to check its length so to prevent user mistake
    MaterialLinearAnisotropic(const std::string & name,
                              const Index_t & spatial_dimension,
                              const Index_t & nb_quad_pts,
                              const std::vector<Real> & input_c);

    //! Copy constructor
    MaterialLinearAnisotropic(const MaterialLinearAnisotropic & other) = delete;

    //! Move constructor
    MaterialLinearAnisotropic(MaterialLinearAnisotropic && other) = delete;

    //! Destructor
    virtual ~MaterialLinearAnisotropic() = default;

    template <class s_t>
    inline auto evaluate_stress(s_t && E) -> decltype(auto);

    template <class s_t>
    inline auto evaluate_stress(s_t && E, const size_t & /*pixel_index*/)
        -> decltype(auto);

    /**
     * evaluates both second Piola-Kirchhoff stress and stiffness given
     * the Green-Lagrange strain (or Cauchy stress and stiffness if
     * called with a small strain tensor) and the local stiffness tensor.
     */

    template <class s_t>
    inline auto evaluate_stress_tangent(s_t && E) -> decltype(auto);

    template <class s_t>
    inline auto evaluate_stress_tangent(s_t && E, const size_t &
                                        /*pixel_index*/) -> decltype(auto);

    // takes the elements of the C and makes it:
    static auto c_maker(std::vector<Real> input) -> Stiffness_t;

   protected:
    std::unique_ptr<Stiffness_t> C_holder;  //! memory for stiffness tensor
    Stiffness_t & C;                        //!< stiffness tensor
  };

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class s_t>
  auto MaterialLinearAnisotropic<DimM>::evaluate_stress(s_t && E)
      -> decltype(auto) {
    return Matrices::tensmult(this->C, E);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class s_t>
  auto MaterialLinearAnisotropic<DimM>::evaluate_stress_tangent(s_t && E)
      -> decltype(auto) {
    return std::make_tuple(Hooke::evaluate_stress(this->C, E), this->C);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class s_t>
  auto MaterialLinearAnisotropic<DimM>::evaluate_stress(s_t && E, const size_t &
                                                        /*pixel_index*/)
      -> decltype(auto) {
    return MaterialLinearAnisotropic<DimM>::evaluate_stress(E);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class s_t>
  auto MaterialLinearAnisotropic<DimM>::evaluate_stress_tangent(s_t && E,
                                                                const size_t &
                                                                /*pixel_index*/)
      -> decltype(auto) {
    return MaterialLinearAnisotropic<DimM>::evaluate_stress_tangent(E);
  }

}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_LINEAR_ANISOTROPIC_HH_

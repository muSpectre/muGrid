/**
 * @file   material_linear_orthotropic.hh
 *
 * @author Ali Falsafi<ali.falsafi@epfl.ch>
 *
 * @date   11 Jul 2018
 *
 * @brief  defenition of general orthotropic linear constitutive model
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

#ifndef SRC_MATERIALS_MATERIAL_LINEAR_ORTHOTROPIC_HH_
#define SRC_MATERIALS_MATERIAL_LINEAR_ORTHOTROPIC_HH_
#include "stress_transformations_PK2.hh"
#include "material_base.hh"
#include "material_muSpectre_base.hh"
#include "material_linear_anisotropic.hh"
#include "common/muSpectre_common.hh"
#include "cell/cell.hh"

#include "libmugrid/field_map_static.hh"

namespace muSpectre {

  template <Index_t DimM>
  class MaterialLinearOrthotropic;

  // traits for orthotropic material
  template <Index_t DimM>
  struct MaterialMuSpectre_traits<MaterialLinearOrthotropic<DimM>> {
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
   * Material implementation for orthotropic constitutive law
   */
  template <Index_t DimM>
  class MaterialLinearOrthotropic : public MaterialLinearAnisotropic<DimM> {
   public:
    //! base class
    using Parent = MaterialLinearAnisotropic<DimM>;

    using Stiffness_t = muGrid::T4Mat<Real, DimM>;

    //! traits of this material
    using traits = MaterialMuSpectre_traits<MaterialLinearOrthotropic>;

    //! Default constructor
    MaterialLinearOrthotropic() = delete;
    // consturcutor
    // a std::vector is utilized as the input of the constructor to
    // enable us to check its length so to prevent user mistake
    MaterialLinearOrthotropic(const std::string & name,
                              const Index_t & spatial_dimension,
                              const Index_t & nb_quad_pts,
                              const std::vector<Real> & input);

    //! Copy constructor
    MaterialLinearOrthotropic(const MaterialLinearOrthotropic & other) = delete;

    //! Move constructor
    MaterialLinearOrthotropic(MaterialLinearOrthotropic && other) = delete;

    //! Destructor
    virtual ~MaterialLinearOrthotropic() = default;

    /**
     * make function needs to be overloaded, because this class does not
     * directly inherit from MaterialMuSpectre. If this overload is not made,
     * calls to make for MaterialLinearOrthotropic would call the constructor
     * for MaterialLinearAnisotropic
     */
    static MaterialLinearOrthotropic<DimM> &
    make(Cell & cell, const std::string & name,
         const std::vector<Real> & input);

   protected:
    std::vector<Real> input_c_maker(const std::vector<Real> & input);
    /**
     * these variable are used to determine which elements of the
     * stiffness matrix should be replaced with the inpts for the
     * orthotropic material
     */
    constexpr static std::array<std::size_t, 2> output_size{6, 21};
    static std::array<bool, output_size[DimM - 2]> ret_flag;
  };

}  // namespace muSpectre
#endif  // SRC_MATERIALS_MATERIAL_LINEAR_ORTHOTROPIC_HH_

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
 * along with GNU Emacs; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.yyyy
 */
#ifndef SRC_MATERIALS_MATERIAL_LINEAR_ORTHOTROPIC_HH_
#define SRC_MATERIALS_MATERIAL_LINEAR_ORTHOTROPIC_HH_
#include "stress_transformations_PK2.hh"
#include "material_base.hh"
#include "material_muSpectre_base.hh"
#include "material_linear_anisotropic.hh"
#include "common/muSpectre_common.hh"
#include "libmugrid/T4_map_proxy.hh"
#include "cell/cell_base.hh"

namespace muSpectre {

  // Forward declaration for factory function
  // template <Dim_t DimS, Dim_t DimM>
  // class CellBase;

  template <Dim_t DimS, Dim_t DimM>
  class MaterialLinearOrthotropic;

  // traits for orthotropic material
  template <Dim_t DimS, Dim_t DimM>
  struct MaterialMuSpectre_traits<MaterialLinearOrthotropic<DimS, DimM>> {
    using Parent = MaterialMuSpectre_traits<void>;  //!< base for elasticity

    //! global field collection
    using GFieldCollection_t =
        typename MaterialBase<DimS, DimM>::GFieldCollection_t;

    //! expected map type for strain fields
    using StrainMap_t =
        muGrid::MatrixFieldMap<GFieldCollection_t, Real, DimM, DimM, true>;
    //! expected map type for stress fields
    using StressMap_t =
        muGrid::MatrixFieldMap<GFieldCollection_t, Real, DimM, DimM>;
    //! expected map type for tangent stiffness fields
    using TangentMap_t =
        muGrid::T4MatrixFieldMap<GFieldCollection_t, Real, DimM>;

    //! declare what type of strain measure your law takes as input
    constexpr static auto strain_measure{StrainMeasure::GreenLagrange};
    //! declare what type of stress measure your law yields as output
    constexpr static auto stress_measure{StressMeasure::PK2};
  };
  /**
   * Material implementation for orthotropic constitutive law
   */
  template <Dim_t DimS, Dim_t DimM = DimS>
  class MaterialLinearOrthotropic
      : public MaterialLinearAnisotropic<DimS, DimM> {
   public:
    //! base class
    using Parent = MaterialLinearAnisotropic<DimS, DimM>;

    using Stiffness_t = muGrid::T4Mat<Real, DimM>;

    //! traits of this material
    using traits = MaterialMuSpectre_traits<MaterialLinearOrthotropic>;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    //! global field collection
    using GFieldCollection_t =
        typename MaterialBase<DimS, DimM>::GFieldCollection_t;

    //! expected map type for tangent stiffness fields
    using Tangent_t = muGrid::T4MatrixFieldMap<GFieldCollection_t, Real, DimM>;

    //! Default constructor
    MaterialLinearOrthotropic() = delete;
    // consturcutor
    // a std::vector is utilized as the input of the constructor to
    // enable us to check its length so to prevent user mistake
    MaterialLinearOrthotropic(std::string name, std::vector<Real> input);

    //! Copy constructor
    MaterialLinearOrthotropic(const MaterialLinearOrthotropic & other) = delete;

    //! Move constructor
    MaterialLinearOrthotropic(MaterialLinearOrthotropic && other) = delete;

    //! Destructor
    virtual ~MaterialLinearOrthotropic() = default;

    /* overloaded make function in order to make python binding
       able to make an object of materila orthotropic*/
    static MaterialLinearOrthotropic<DimS, DimM> &
    make(CellBase<DimS, DimM> & cell, std::string name,
         std::vector<Real> input);

    std::vector<Real> input_c_maker(std::vector<Real> input);

   protected:
    /**
     * these variable are used to determine which elements of the
     * stiffness matrix should be replaced with the inpts for the
     * orthotropic material
     */
    constexpr static std::array<std::size_t, 2> output_size{6, 21};
    static std::array<bool, output_size[DimM - 2]> ret_flag;
  };

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  MaterialLinearOrthotropic<DimS, DimM> &
  MaterialLinearOrthotropic<DimS, DimM>::make(CellBase<DimS, DimM> & cell,
                                              std::string name,
                                              std::vector<Real> input) {
    auto mat =
        std::make_unique<MaterialLinearOrthotropic<DimS, DimM>>(name, input);
    auto & mat_ref = *mat;
    cell.add_material(std::move(mat));
    return mat_ref;
  }

}  // namespace muSpectre
#endif  // SRC_MATERIALS_MATERIAL_LINEAR_ORTHOTROPIC_HH_

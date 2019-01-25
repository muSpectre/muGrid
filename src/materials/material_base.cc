/**
 * @file   material_base.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   01 Nov 2017
 *
 * @brief  implementation of materi
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
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 */

#include "materials/material_base.hh"

namespace muSpectre {

  //----------------------------------------------------------------------------//
  template <Dim_t DimS, Dim_t DimM>
  MaterialBase<DimS, DimM>::MaterialBase(std::string name) : name(name) {
    static_assert((DimM == oneD) || (DimM == twoD) || (DimM == threeD),
                  "only 1, 2, or threeD supported");
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  const std::string & MaterialBase<DimS, DimM>::get_name() const {
    return this->name;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void MaterialBase<DimS, DimM>::add_pixel(const Ccoord & ccoord) {
    this->internal_fields.add_pixel(ccoord);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void MaterialBase<DimS, DimM>::compute_stresses(const Field_t & F,
                                                  Field_t & P,
                                                  Formulation form) {
    this->compute_stresses(StrainField_t::check_ref(F),
                           StressField_t::check_ref(P), form);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void MaterialBase<DimS, DimM>::compute_stresses_tangent(const Field_t & F,
                                                          Field_t & P,
                                                          Field_t & K,
                                                          Formulation form) {
    this->compute_stresses_tangent(StrainField_t::check_ref(F),
                                   StressField_t::check_ref(P),
                                   TangentField_t::check_ref(K), form);
  }

  //----------------------------------------------------------------------------//
  template <Dim_t DimS, Dim_t DimM>
  auto MaterialBase<DimS, DimM>::get_real_field(std::string field_name)
      -> EigenMap {
    if (not this->internal_fields.check_field_exists(field_name)) {
      std::stringstream err{};
      err << "Field '" << field_name << "' does not exist in material '"
          << this->name << "'.";
      throw muGrid::FieldCollectionError(err.str());
    }
    auto & field{this->internal_fields[field_name]};
    if (field.get_stored_typeid().hash_code() != typeid(Real).hash_code()) {
      std::stringstream err{};
      err << "Field '" << field_name << "' is not real-valued";
      throw muGrid::FieldCollectionError(err.str());
    }

    return static_cast<muGrid::TypedField<MFieldCollection_t, Real> &>(field).eigen();
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  std::vector<std::string> MaterialBase<DimS, DimM>::list_fields() const {
    return this->internal_fields.list_fields();
  }

  template class MaterialBase<2, 2>;
  template class MaterialBase<2, 3>;
  template class MaterialBase<3, 3>;

}  // namespace muSpectre

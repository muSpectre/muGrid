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
 * Boston, MA 02111-1307, USA.
 */

#include "materials/material_base.hh"

namespace muSpectre {

  //----------------------------------------------------------------------------//
  template <Dim_t DimS, Dim_t DimM>
  MaterialBase<DimS, DimM>::MaterialBase(std::string name)
    :name(name) {
    static_assert((DimM == oneD)||
                  (DimM == twoD)||
                  (DimM == threeD), "only 1, 2, or threeD supported");
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  const std::string & MaterialBase<DimS, DimM>::get_name() const {
    return this->name;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void MaterialBase<DimS, DimM>::add_pixel(const Ccoord &ccoord) {
    this->internal_fields.add_pixel(ccoord);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void MaterialBase<DimS, DimM>::compute_stresses(const Field_t & F,
                                                  Field_t & P,
                                                  Formulation form) {
    this->compute_stresses(StrainField_t::check_ref(F),
                           StressField_t::check_ref(P),
                           form);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void MaterialBase<DimS, DimM>::compute_stresses_tangent(const Field_t & F,
                                                          Field_t & P,
                                                          Field_t & K,
                                                          Formulation form) {
    this->compute_stresses_tangent(StrainField_t::check_ref(F),
                                   StressField_t::check_ref(P),
                                   TangentField_t::check_ref(K),
                                   form);
  }

  template class MaterialBase<2, 2>;
  template class MaterialBase<2, 3>;
  template class MaterialBase<3, 3>;

}  // muSpectre

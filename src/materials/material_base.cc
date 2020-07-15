/**
 * @file   material_base.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   01 Nov 2017
 *
 * @brief  implementation of material
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

#include "materials/material_base.hh"

#include <libmugrid/field.hh>
#include <libmugrid/field_typed.hh>

using muGrid::RuntimeError;

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  MaterialBase::MaterialBase(
      const std::string & name, const Index_t & spatial_dimension,
      const Index_t & material_dimension, const Index_t & nb_quad_pts,
      const std::shared_ptr<muGrid::LocalFieldCollection> &
          parent_field_collection)
      : name(name),
        internal_fields{
            parent_field_collection == nullptr
                ? std::make_shared<muGrid::LocalFieldCollection>(
                      spatial_dimension,
                      // setting the map for nb_sub_pts on the fly here to avoid
                      // having to set the number of quadrature points
                      // conditionally in the constructor function body. This
                      // lambda simply creates a map, fills in the nb_quad_pts
                      // and returns it
                      [&nb_quad_pts]() {
                        muGrid::LocalFieldCollection::SubPtMap_t map{};
                        map[QuadPtTag] = nb_quad_pts;
                        return map;
                      }())
                : parent_field_collection},
        material_dimension{material_dimension},
        prefix{parent_field_collection == nullptr ? "" : name + "::"} {
    if (not((this->material_dimension == oneD) ||
            (this->material_dimension == twoD) ||
            (this->material_dimension == threeD))) {
      throw MaterialError("only 1, 2, or threeD supported");
    }
  }

  /* ---------------------------------------------------------------------- */
  const std::string & MaterialBase::get_name() const { return this->name; }

  /* ---------------------------------------------------------------------- */
  void MaterialBase::add_pixel(const size_t & global_index) {
    this->internal_fields->add_pixel(global_index);
  }

  /* ---------------------------------------------------------------------- */
  void MaterialBase::add_pixel_split(const size_t & global_index,
                                     const Real & ratio) {
    this->add_pixel(global_index);
    this->assigned_ratio->get_field().push_back(ratio);
  }

  /* ---------------------------------------------------------------------- */
  void MaterialBase::compute_stresses(
      const muGrid::Field & F, muGrid::Field & P, const Formulation & form,
      const SplitCell & is_cell_split,
      const StoreNativeStress & store_native_stress) {
    const auto t2_dim{muGrid::ipow(this->material_dimension, 2)};
    const auto & real_F{muGrid::RealField::safe_cast(F, t2_dim, QuadPtTag)};
    auto & real_P{muGrid::RealField::safe_cast(P, t2_dim, QuadPtTag)};
    this->compute_stresses(real_F, real_P, form, is_cell_split,
                           store_native_stress);
  }

  /* ---------------------------------------------------------------------- */
  void MaterialBase::allocate_optional_fields(SplitCell is_cell_split) {
    if (is_cell_split == SplitCell::simple) {
      this->assigned_ratio =
          std::make_unique<muGrid::MappedScalarField<Real, muGrid::Mapping::Mut,
                                                     IterUnit::SubPt>>(
              "ratio", *this->internal_fields, QuadPtTag);
    }
  }
  /* ---------------------------------------------------------------------- */
  void MaterialBase::get_assigned_ratios(
      std::vector<Real> & quad_pt_assigned_ratios) {
    quad_pt_assigned_ratios.reserve(
        this->assigned_ratio->get_field().get_nb_components());
    for (auto && tup : this->assigned_ratio->get_map().enumerate_indices()) {
      const auto & index = std::get<0>(tup);
      const auto & val = std::get<1>(tup);
      quad_pt_assigned_ratios[index] += val;
    }
  }

  /* ---------------------------------------------------------------------- */
  Real MaterialBase::get_assigned_ratio(const size_t & pixel_id) {
    auto id{this->internal_fields->get_global_to_local_index_map()[pixel_id]};
    auto && tmp{this->assigned_ratio->get_map()};
    return tmp[id];
  }

  /* ----------------------------------------------------------------------*/
  muGrid::RealField & MaterialBase::get_assigned_ratio_field() {
    return this->assigned_ratio->get_field();
  }

  /* ---------------------------------------------------------------------- */
  void MaterialBase::compute_stresses_tangent(
      const muGrid::Field & F, muGrid::Field & P, muGrid::Field & K,
      const Formulation & form, const SplitCell & is_cell_split,
      const StoreNativeStress & store_native_stress) {
    const auto t2_dim{muGrid::ipow(this->material_dimension, 2)};
    const auto & real_F{muGrid::RealField::safe_cast(F, t2_dim, QuadPtTag)};
    auto & real_P{muGrid::RealField::safe_cast(P, t2_dim, QuadPtTag)};
    auto & real_K{
        muGrid::RealField::safe_cast(K, muGrid::ipow(t2_dim, 2), QuadPtTag)};
    this->compute_stresses_tangent(real_F, real_P, real_K, form, is_cell_split,
                                   store_native_stress);
  }

  /* ---------------------------------------------------------------------- */
  auto MaterialBase::get_pixel_indices() const ->
      typename muGrid::LocalFieldCollection::PixelIndexIterable {
    return this->internal_fields->get_pixel_indices_fast();
  }

  /* ---------------------------------------------------------------------- */
  auto MaterialBase::get_quad_pt_indices() const ->
      typename muGrid::LocalFieldCollection::IndexIterable {
    return this->internal_fields->get_sub_pt_indices(QuadPtTag);
  }

  /* ---------------------------------------------------------------------- */
  std::vector<std::string> MaterialBase::list_fields() const {
    return this->internal_fields->list_fields();
  }

  /* ---------------------------------------------------------------------- */
  muGrid::LocalFieldCollection & MaterialBase::get_collection() {
    return *this->internal_fields;
  }

  /* ---------------------------------------------------------------------- */
  bool MaterialBase::has_native_stress() const { return false; }

  /* ---------------------------------------------------------------------- */
  muGrid::RealField & MaterialBase::get_native_stress() {
    throw RuntimeError("Not implemented for this material");
  }

  /* ---------------------------------------------------------------------- */
  void MaterialBase::initialise() {
    if (!this->is_initialised) {
      this->internal_fields->initialise();
      this->is_initialised = true;
    } else {
      std::stringstream err_str{};
      err_str << "The material " << this->name
              << " has been already initialised."
              << "Therefore, it cannot be initialised again" << std::endl;
      throw RuntimeError(err_str.str());
    }
  }

  /* ---------------------------------------------------------------------- */
  void MaterialBase::check_small_strain_capability(
      const StrainMeasure & expected_strain_measure) {
    if (not(is_objective(expected_strain_measure))) {
      std::stringstream err_str{};
      err_str
          << "The material expected strain measure is: "
          << expected_strain_measure
          << ", while in small strain the required strain measure should be "
             "objective (in order to be obtainable from infinitesimal strain)."
          << " Accordingly, this material is not meant to be utilized in "
             "small strain formulation"
          << std::endl;
      throw(muGrid::RuntimeError(err_str.str()));
    }
  }

  /* ---------------------------------------------------------------------- */
  bool MaterialBase::was_last_step_nonlinear() const {
    return this->last_step_was_nonlinear;
  }

  /* ---------------------------------------------------------------------- */
  const bool & MaterialBase::get_is_initialised() {
    return this->is_initialised;
  }

}  // namespace muSpectre

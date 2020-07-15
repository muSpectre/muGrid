/**
 * @file   cell.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   05 Oct 2019
 *
 * @brief  implementation for the Cell class
 *
 * Copyright © 2019 Till Junge
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

#include <libmugrid/exception.hh>

#include "cell_adaptor.hh"
#include "cell.hh"

#ifdef WITH_SPLIT
#include "materials/material_laminate.hh"
#include "common/intersection_octree.hh"
#endif

#include <libmugrid/state_field.hh>
#include <libmugrid/field_map.hh>
#include <libmugrid/field_map_static.hh>

#include <set>

using muGrid::RuntimeError;

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  Cell::Cell(Projection_ptr projection, SplitCell is_cell_split)
      : projection{std::move(projection)},
        fields{std::make_unique<muGrid::GlobalFieldCollection>(
            this->get_spatial_dim(),
            this->projection->get_nb_subdomain_grid_pts(),
            this->projection->get_subdomain_locations(),
            muGrid::FieldCollection::SubPtMap_t{
                {QuadPtTag, this->get_nb_quad_pts()},
                {NodalPtTag, this->get_nb_nodal_pts()}})},
        // We request the DOFs for a single quadrature point, since quadrature
        // points are handled by the field.
        strain{this->fields->register_real_field(
            "strain",
            shape_for_formulation(this->get_formulation(),
                                  this->get_material_dim()),
            QuadPtTag)},
        stress{this->fields->register_real_field(
            "stress",
            shape_for_formulation(this->get_formulation(),
                                  this->get_material_dim()),
            QuadPtTag)},
        is_cell_split{is_cell_split} {}

  /* ---------------------------------------------------------------------- */
  bool Cell::is_initialised() const { return this->initialised; }

  /* ---------------------------------------------------------------------- */
  Index_t Cell::get_nb_dof() const {
    const auto & strain_shape{this->get_strain_shape()};
    return this->get_nb_pixels() * this->get_nb_quad_pts() * strain_shape[0] *
           strain_shape[1];
  }

  /* ---------------------------------------------------------------------- */
  size_t Cell::get_nb_pixels() const { return this->fields->get_nb_pixels(); }

  /* ---------------------------------------------------------------------- */
  const muFFT::Communicator & Cell::get_communicator() const {
    return this->projection->get_communicator();
  }

  /* ---------------------------------------------------------------------- */
  const Formulation & Cell::get_formulation() const {
    return this->projection->get_formulation();
  }

  /* ---------------------------------------------------------------------- */
  Index_t Cell::get_material_dim() const { return this->get_spatial_dim(); }

  /* ---------------------------------------------------------------------- */
  void
  Cell::set_uniform_strain(const Eigen::Ref<const Matrix_t> & uniform_strain) {
    if (not this->initialised) {
      this->initialise();
    }
    Index_t && dim{this->get_material_dim()};
    muGrid::FieldMap<Real, Mapping::Mut>(this->strain, dim) = uniform_strain;
  }

  /* ---------------------------------------------------------------------- */
  MaterialBase & Cell::add_material(Material_ptr mat) {
    if (mat->get_material_dimension() != this->get_spatial_dim()) {
      throw RuntimeError(
          "this cell class only accepts materials with the same dimensionality "
          "as the spatial problem.");
    }
    this->materials.push_back(std::move(mat));
    return *this->materials.back();
  }

  /* ---------------------------------------------------------------------- */
  void Cell::complete_material_assignment_simple(MaterialBase & material) {
    if (this->is_initialised()) {
      throw RuntimeError(
          "The cell is already initialised. Therefore, it is not "
          "possible to complete material assignemnt for it");
    } else {
      for (auto && mat : this->materials) {
        if (mat->get_name() != material.get_name()) {
          if (!mat->get_is_initialised()) {
            mat->initialise();
          }
        }
      }
    }

    auto nb_pixels = muGrid::CcoordOps::get_size(
        this->get_projection().get_nb_subdomain_grid_pts());
    std::vector<bool> assignments(nb_pixels, false);
    for (auto & mat : this->materials) {
      for (auto & index : mat->get_pixel_indices()) {
        assignments[index] = true;
      }
    }
    for (auto && tup : akantu::enumerate(assignments)) {
      auto && index{std::get<0>(tup)};
      auto && is_assigned{std::get<1>(tup)};
      if (!is_assigned) {
        material.add_pixel(index);
      }
    }
  }
  /* ---------------------------------------------------------------------- */
  CellAdaptor<Cell> Cell::get_adaptor() { return CellAdaptor<Cell>(*this); }

  /* ---------------------------------------------------------------------- */
  void Cell::save_history_variables() {
    for (auto && mat : this->materials) {
      mat->save_history_variables();
    }
  }

  /* ---------------------------------------------------------------------- */
  Shape_t Cell::get_strain_shape() const {
    switch (this->get_formulation()) {
    case Formulation::finite_strain: {
      return Shape_t{this->get_material_dim(),
                     this->get_material_dim()};
      break;
    }
    case Formulation::small_strain: {
      return Shape_t{this->get_material_dim(),
                     this->get_material_dim()};
      break;
    }
    default:
      throw RuntimeError("Formulation not implemented");
      break;
    }
  }

  /* ---------------------------------------------------------------------- */
  Index_t Cell::get_strain_size() const {
    auto && shape{this->get_strain_shape()};
    return shape[0] * shape[1];
  }

  /* ---------------------------------------------------------------------- */
  const Index_t & Cell::get_spatial_dim() const {
    return this->projection->get_dim();
  }

  /* ---------------------------------------------------------------------- */
  const Index_t & Cell::get_nb_quad_pts() const {
    return this->projection->get_nb_quad_pts();
  }

  /* ---------------------------------------------------------------------- */
  const Index_t & Cell::get_nb_nodal_pts() const {
    // this true for Cells only capable of projection-based solution
    return OneNode;
  }

  /* ---------------------------------------------------------------------- */
  void Cell::check_material_coverage() const {
    auto nb_pixels{muGrid::CcoordOps::get_size(
        this->projection->get_nb_subdomain_grid_pts())};
    std::vector<MaterialBase *> assignments(nb_pixels, nullptr);

    for (auto & mat : this->materials) {
      for (auto & index : mat->get_pixel_indices()) {
        auto & assignment{assignments.at(index)};
        if (assignment != nullptr) {
          std::stringstream err{};
          err << "Pixel " << index << "is already assigned to material '"
              << assignment->get_name()
              << "' and cannot be reassigned to material '" << mat->get_name();
          throw RuntimeError(err.str());
        } else {
          assignments[index] = mat.get();
        }
      }
    }

    // find and identify unassigned pixels
    std::vector<DynCcoord_t> unassigned_pixels;
    for (size_t i = 0; i < assignments.size(); ++i) {
      if (assignments[i] == nullptr) {
        unassigned_pixels.push_back(this->fields->get_ccoord(i));
      }
    }

    if (unassigned_pixels.size() != 0) {
      std::stringstream err{};
      err << "The following pixels have were not assigned a material: ";
      for (auto & pixel : unassigned_pixels) {
        muGrid::operator<<(err, pixel) << ", ";
      }
      err << "and that cannot be handled";
      throw RuntimeError(err.str());
    }
  }

  /* ---------------------------------------------------------------------- */
  void Cell::initialise() {
    // check that all pixels have been assigned exactly one material
    if (this->is_initialised()) {
      throw RuntimeError(
          "The cell is already initialised. Therefore, it is not "
          "possible to complete material assignemnt for it");
    } else {
      for (auto && mat : this->materials) {
        if (!mat->get_is_initialised()) {
          mat->initialise();
        }
      }
    }
    this->check_material_coverage();
    // initialise the projection and compute the fft plan
    this->projection->initialise();
    this->initialised = true;
  }

  /* ---------------------------------------------------------------------- */
  auto Cell::get_pixels() const -> const muGrid::CcoordOps::DynamicPixels & {
    return this->fields->get_pixels();
  }

  /* ---------------------------------------------------------------------- */
  auto Cell::get_quad_pt_indices() const
      -> muGrid::FieldCollection::IndexIterable {
    return this->fields->get_sub_pt_indices(QuadPtTag);
  }

  /* ---------------------------------------------------------------------- */
  auto Cell::get_pixel_indices() const
      -> muGrid::FieldCollection::PixelIndexIterable {
    return this->fields->get_pixel_indices_fast();
  }

  /* ---------------------------------------------------------------------- */
  muGrid::RealField & Cell::get_strain() { return this->strain; }

  /* ---------------------------------------------------------------------- */
  const muGrid::RealField & Cell::get_stress() const { return this->stress; }

  /* ---------------------------------------------------------------------- */
  const muGrid::RealField & Cell::get_tangent(bool do_create) {
    if (not this->tangent) {
      if (do_create) {
        this->tangent = this->fields->register_real_field(
            "Tangent_stiffness",
            t4shape_for_formulation(this->get_formulation(),
                                    this->get_material_dim()),
            QuadPtTag);
      } else {
        throw RuntimeError("Tangent has not been created");
      }
    }
    return this->tangent.value();
  }

  /* ---------------------------------------------------------------------- */
  const muGrid::RealField & Cell::evaluate_stress() {
    if (this->initialised == false) {
      this->initialise();
    }
    for (auto & mat : this->materials) {
      mat->compute_stresses(this->strain, this->stress,
                            this->get_formulation());
    }
    return this->stress;
  }

  /* ---------------------------------------------------------------------- */
  auto Cell::evaluate_stress_eigen() -> Eigen_cmap {
    return this->evaluate_stress().eigen_vec();
  }

  /* ---------------------------------------------------------------------- */
  std::tuple<const muGrid::RealField &, const muGrid::RealField &>
  Cell::evaluate_stress_tangent() {
    if (not this->initialised) {
      this->initialise();
    }

    constexpr bool create_tangent{true};
    this->get_tangent(create_tangent);

    for (auto & mat : this->materials) {
      mat->compute_stresses_tangent(this->strain, this->stress,
                                    this->tangent.value(),
                                    this->get_formulation());
    }
    return std::tie(this->stress, this->tangent.value());
  }

  /* ---------------------------------------------------------------------- */
  auto Cell::evaluate_stress_tangent_eigen()
      -> std::tuple<const Eigen_cmap, const Eigen_cmap> {
    auto && fields{this->evaluate_stress_tangent()};
    return std::tuple<const Eigen_cmap, const Eigen_cmap>(
        std::get<0>(fields).eigen_vec(), std::get<1>(fields).eigen_vec());
  }

  /* ---------------------------------------------------------------------- */
  muGrid::RealField &
  Cell::globalise_real_internal_field(const std::string & unique_name) {
    return this->template globalise_internal_field<Real>(unique_name);
  }
  /* ---------------------------------------------------------------------- */
  muGrid::IntField &
  Cell::globalise_int_internal_field(const std::string & unique_name) {
    return this->template globalise_internal_field<Int>(unique_name);
  }
  /* ---------------------------------------------------------------------- */
  muGrid::UintField &
  Cell::globalise_uint_internal_field(const std::string & unique_name) {
    return this->template globalise_internal_field<Uint>(unique_name);
  }
  /* ---------------------------------------------------------------------- */
  muGrid::ComplexField &
  Cell::globalise_complex_internal_field(const std::string & unique_name) {
    return this->template globalise_internal_field<Complex>(unique_name);
  }

  /* ---------------------------------------------------------------------- */
  muGrid::RealField &
  Cell::globalise_real_current_field(const std::string & unique_name) {
    return this->template globalise_current_field<Real>(unique_name);
  }
  /* ---------------------------------------------------------------------- */
  muGrid::IntField &
  Cell::globalise_int_current_field(const std::string & unique_name) {
    return this->template globalise_current_field<Int>(unique_name);
  }
  /* ---------------------------------------------------------------------- */
  muGrid::UintField &
  Cell::globalise_uint_current_field(const std::string & unique_name) {
    return this->template globalise_current_field<Uint>(unique_name);
  }
  /* ---------------------------------------------------------------------- */
  muGrid::ComplexField &
  Cell::globalise_complex_current_field(const std::string & unique_name) {
    return this->template globalise_current_field<Complex>(unique_name);
  }

  /* ---------------------------------------------------------------------- */
  muGrid::RealField &
  Cell::globalise_real_old_field(const std::string & unique_name,
                                 const size_t & nb_steps_ago) {
    return this->template globalise_old_field<Real>(unique_name, nb_steps_ago);
  }
  /* ---------------------------------------------------------------------- */
  muGrid::IntField &
  Cell::globalise_int_old_field(const std::string & unique_name,
                                const size_t & nb_steps_ago) {
    return this->template globalise_old_field<Int>(unique_name, nb_steps_ago);
  }
  /* ---------------------------------------------------------------------- */
  muGrid::UintField &
  Cell::globalise_uint_old_field(const std::string & unique_name,
                                 const size_t & nb_steps_ago) {
    return this->template globalise_old_field<Uint>(unique_name, nb_steps_ago);
  }
  /* ---------------------------------------------------------------------- */
  muGrid::ComplexField &
  Cell::globalise_complex_old_field(const std::string & unique_name,
                                    const size_t & nb_steps_ago) {
    return this->template globalise_old_field<Complex>(unique_name,
                                                       nb_steps_ago);
  }

  /* ---------------------------------------------------------------------- */
  muGrid::GlobalFieldCollection & Cell::get_fields() { return *this->fields; }

  using muGrid::operator<<;
  /* ---------------------------------------------------------------------- */
  void Cell::apply_projection(muGrid::TypedFieldBase<Real> & field) {
    this->projection->apply_projection(field);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void Cell::apply_directional_stiffness(
      const muGrid::TypedFieldBase<Real> & delta_strain,
      const muGrid::TypedFieldBase<Real> & tangent,
      muGrid::TypedFieldBase<Real> & delta_stress) {
    muGrid::T2FieldMap<Real, muGrid::Mapping::Const, DimM, IterUnit::SubPt>
        strain_map{delta_strain};
    muGrid::T4FieldMap<Real, muGrid::Mapping::Const, DimM, IterUnit::SubPt>
        tangent_map{tangent};
    muGrid::T2FieldMap<Real, muGrid::Mapping::Mut, DimM, IterUnit::SubPt>
        stress_map{delta_stress};
    for (auto && tup : akantu::zip(strain_map, tangent_map, stress_map)) {
      auto & df = std::get<0>(tup);
      auto & k = std::get<1>(tup);
      auto & dp = std::get<2>(tup);
      dp = Matrices::tensmult(k, df);
    }
  }

  /* ---------------------------------------------------------------------- */
  void Cell::evaluate_projected_directional_stiffness(
      const muGrid::TypedFieldBase<Real> & delta_strain,
      muGrid::TypedFieldBase<Real> & del_stress) {
    if (not this->tangent) {
      throw RuntimeError("evaluate_projected_directional_stiffness "
                         "requires the tangent moduli");
    }
    if (delta_strain.get_nb_components() != this->get_strain_size()) {
      std::stringstream err{};
      err << "The input field should have " << this->get_strain_size()
          << " components per quadrature point, but has "
          << delta_strain.get_nb_components() << " components.";
      throw RuntimeError(err.str());
    }
    if (delta_strain.get_nb_sub_pts() != this->get_strain().get_nb_sub_pts()) {
      std::stringstream err{};
      err << "The input field should have "
          << this->get_strain().get_nb_sub_pts()
          << " quadrature point per pixel, but has "
          << delta_strain.get_nb_sub_pts() << " points.";
      throw RuntimeError(err.str());
    }
    if (delta_strain.get_collection().get_nb_pixels() !=
        this->get_strain().get_collection().get_nb_pixels()) {
      std::stringstream err{};
      err << "The input field should have "
          << this->get_strain().get_collection().get_nb_pixels()
          << " pixels, but has "
          << delta_strain.get_collection().get_nb_pixels() << " pixels.";
      throw RuntimeError(err.str());
    }
    switch (this->get_material_dim()) {
    case twoD: {
      this->template apply_directional_stiffness<twoD>(
          delta_strain, this->tangent.value(), del_stress);
      break;
    }
    case threeD: {
      this->template apply_directional_stiffness<threeD>(
          delta_strain, this->tangent.value(), del_stress);
      break;
    }
    default:
      std::stringstream err{};
      err << "unknown dimension " << this->get_material_dim() << std::endl;
      throw RuntimeError(err.str());
      break;
    }
    this->apply_projection(del_stress);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  void Cell::add_projected_directional_stiffness_helper(
      const muGrid::TypedFieldBase<Real> & delta_strain,
      const muGrid::TypedFieldBase<Real> & tangent, const Real & alpha,
      muGrid::TypedFieldBase<Real> & delta_stress) {
    muGrid::T2FieldMap<Real, muGrid::Mapping::Const, DimM, IterUnit::SubPt>
        strain_map{delta_strain};
    muGrid::T4FieldMap<Real, muGrid::Mapping::Const, DimM, IterUnit::SubPt>
        tangent_map{tangent};
    muGrid::T2FieldMap<Real, muGrid::Mapping::Mut, DimM, IterUnit::SubPt>
        stress_map{delta_stress};
    for (auto && tup : akantu::zip(strain_map, tangent_map, stress_map)) {
      auto & df = std::get<0>(tup);
      auto & k = std::get<1>(tup);
      auto & dp = std::get<2>(tup);
      dp += alpha * Matrices::tensmult(k, df);
    }
  }
  /* ---------------------------------------------------------------------- */
  void Cell::add_projected_directional_stiffness(EigenCVec_t delta_strain,
                                                 const Real & alpha,
                                                 EigenVec_t del_stress) {
    auto delta_strain_field_ptr{muGrid::WrappedField<Real>::make_const(
        "delta_strain", *this->fields, this->get_strain_size(), delta_strain,
        QuadPtTag)};
    muGrid::WrappedField<Real> del_stress_field{"delta_stress", *this->fields,
                                                this->get_strain_size(),
                                                del_stress, QuadPtTag};
    switch (this->get_material_dim()) {
    case twoD: {
      this->template add_projected_directional_stiffness_helper<twoD>(
          *delta_strain_field_ptr, this->tangent.value(), alpha,
          del_stress_field);
      break;
    }
    case threeD: {
      this->template add_projected_directional_stiffness_helper<threeD>(
          *delta_strain_field_ptr, this->tangent.value(), alpha,
          del_stress_field);
      break;
    }
    default:
      std::stringstream err{};
      err << "unknown dimension " << this->get_material_dim() << std::endl;
      throw RuntimeError(err.str());
      break;
    }
    this->apply_projection(del_stress_field);
  }

  /* ---------------------------------------------------------------------- */
  auto Cell::get_projection() const -> const ProjectionBase & {
    return *this->projection;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  muGrid::TypedField<T> &
  Cell::globalise_internal_field(const std::string & unique_name) {
    // start by checking that the field exists at least once, and that
    // it always has th same number of components, and the same subdivision tag
    std::set<Index_t> nb_component_categories{};
    std::set<std::string> tag_categories{};
    std::vector<std::reference_wrapper<const muGrid::Field>> local_fields;

    for (auto & mat : this->materials) {
      auto && collection{mat->get_collection()};
      if (collection.field_exists(unique_name)) {
        auto && field{muGrid::TypedField<T>::safe_cast(
            collection.get_field(unique_name))};
        local_fields.push_back(field);
        nb_component_categories.insert(field.get_nb_components());
        tag_categories.insert(field.get_sub_division_tag());
      }
    }

    // reject if the field appears with differing numbers of components
    if (nb_component_categories.size() != 1) {
      const auto & nb_match{nb_component_categories.size()};
      std::stringstream err_str{};
      if (nb_match > 1) {
        err_str
            << "The fields named '" << unique_name << "' do not have the "
            << "same number of components in every material, which is a "
            << "requirement for globalising them! The following values were "
            << "found by material:" << std::endl;
        for (auto & mat : this->materials) {
          auto & coll = mat->get_collection();
          if (coll.field_exists(unique_name)) {
            auto & field{coll.get_field(unique_name)};
            err_str << field.get_nb_components()
                    << " components in material '" << mat->get_name() << "'"
                    << std::endl;
          }
        }
      } else {
        err_str << "The field named '" << unique_name << "' does not exist in "
                << "any of the materials and can therefore not be globalised!";
      }
      throw RuntimeError(err_str.str());
    }

    const Index_t nb_components{*nb_component_categories.begin()};

    // reject if the field appears with differing subdivision tags
    if (tag_categories.size() != 1) {
      const auto & nb_match{tag_categories.size()};
      std::stringstream err_str{};
      if (nb_match > 1) {
        err_str
            << "The fields named '" << unique_name << "' do not have the "
            << "same sub-division in every material, which is a "
            << "requirement for globalising them! The following values were "
            << "found by material:" << std::endl;
        for (auto & mat : this->materials) {
          auto & coll = mat->get_collection();
          if (coll.field_exists(unique_name)) {
            auto & field{coll.get_field(unique_name)};
            err_str << "tag '" << field.get_sub_division_tag()
                    << "' in material '" << mat->get_name() << "'" << std::endl;
          }
        }
      } else {
        err_str << "The field named '" << unique_name << "' does not exist in "
                << "any of the materials and can therefore not be globalised!";
      }
      throw RuntimeError(err_str.str());
    }

    const std::string tag{*tag_categories.begin()};

    // get and prepare the field
    muGrid::TypedField<T> & global_field{
        this->fields->field_exists(unique_name)
            ? dynamic_cast<muGrid::TypedField<T> &>(
                  this->fields->get_field(unique_name))
            : this->fields->template register_field<T>(unique_name,
                                                       nb_components, tag)};
    global_field.set_zero();

    auto global_map{global_field.get_pixel_map()};

    // fill it with local internal values
    for (auto & local_field : local_fields) {
      auto pixel_map{
          muGrid::TypedField<T>::safe_cast(local_field).get_pixel_map()};
      for (auto && pixel_id__value : pixel_map.enumerate_pixel_indices_fast()) {
        const auto & pixel_id{std::get<0>(pixel_id__value)};
        const auto & value{std::get<1>(pixel_id__value)};
        global_map[pixel_id] = value;
      }
    }
    return global_field;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  muGrid::TypedField<T> &
  Cell::globalise_old_field(const std::string & unique_prefix,
                            const size_t & nb_steps_ago) {
    // start by checking that the field exists at least once, and that
    // it always has th same number of components
    std::set<Index_t> nb_component_categories{};
    std::set<std::string> tag_categories{};
    std::vector<std::reference_wrapper<const muGrid::Field>> local_fields_old;

    for (auto & mat : this->materials) {
      auto && collection{mat->get_collection()};
      if (collection.state_field_exists(unique_prefix)) {
        auto && state_field{collection.get_state_field(unique_prefix)};
        auto && field_old{
            muGrid::TypedField<T>::safe_cast(state_field.old(nb_steps_ago))};
        local_fields_old.push_back(field_old);
        nb_component_categories.insert(field_old.get_nb_components());
        tag_categories.insert(field_old.get_sub_division_tag());
      }
    }

    // reject if the field appears with differing numbers of components
    if (nb_component_categories.size() != 1) {
      const auto & nb_match{nb_component_categories.size()};
      std::stringstream err_str{};
      if (nb_match > 1) {
        err_str
            << "The state fields named '" << unique_prefix
            << "' do not have the "
            << "same number of components in every material, which is a "
            << "requirement for globalising them! The following values were "
            << "found by material:" << std::endl;
        for (auto & mat : this->materials) {
          auto && coll{mat->get_collection()};
          if (coll.state_field_exists(unique_prefix)) {
            auto && field{
                coll.get_state_field(unique_prefix).old(nb_steps_ago)};
            err_str << field.get_nb_components()
                    << " components in material '" << mat->get_name() << "'"
                    << std::endl;
          }
        }
      } else {
        err_str << "The state field named '" << unique_prefix
                << " does not exist in "
                << "any of the materials and can therefore not be globalised!";
      }
      throw RuntimeError(err_str.str());
    }

    const Index_t nb_components{*nb_component_categories.begin()};

    // reject if the field appears with differing subdivision tags
    if (tag_categories.size() != 1) {
      const auto & nb_match{tag_categories.size()};
      std::stringstream err_str{};
      if (nb_match > 1) {
        err_str
            << "The fields named '" << unique_prefix << "' do not have the "
            << "same sub-division in every material, which is a "
            << "requirement for globalising them! The following values were "
            << "found by material:" << std::endl;
        for (auto & mat : this->materials) {
          auto & coll = mat->get_collection();
          if (coll.field_exists(unique_prefix)) {
            auto & field{coll.get_field(unique_prefix)};
            err_str << "tag '" << field.get_sub_division_tag()
                    << "' in material '" << mat->get_name() << "'" << std::endl;
          }
        }
      } else {
        err_str << "The field named '" << unique_prefix
                << "' does not exist in "
                << "any of the materials and can therefore not be globalised!";
      }
      throw RuntimeError(err_str.str());
    }

    const std::string tag{*tag_categories.begin()};

    muGrid::TypedField<T> & global_field{
        this->fields->field_exists(unique_prefix)
            ? dynamic_cast<muGrid::TypedField<T> &>(
                  this->fields->get_field(unique_prefix))
            : this->fields->template register_field<T>(unique_prefix,
                                                       nb_components, tag)};
    global_field.set_zero();

    auto global_map{global_field.get_pixel_map()};

    // fill it with local old state of internal values
    for (auto & local_field : local_fields_old) {
      auto pixel_map{
          muGrid::TypedField<T>::safe_cast(local_field).get_pixel_map()};
      for (auto && pixel_id__value : pixel_map.enumerate_pixel_indices_fast()) {
        auto && pixel_id{std::get<0>(pixel_id__value)};
        auto && value{std::get<1>(pixel_id__value)};
        global_map[pixel_id] = value;
      }
    }
    return global_field;
  }

  /* ---------------------------------------------------------------------- */
  template <typename T>
  muGrid::TypedField<T> &
  Cell::globalise_current_field(const std::string & unique_prefix) {
    // start by checking that the field exists at least once, and that
    // it always has th same number of components
    std::set<Index_t> nb_component_categories{};
    std::set<std::string> tag_categories{};

    std::vector<std::reference_wrapper<const muGrid::Field>>
        local_fields_current;
    for (auto & mat : this->materials) {
      auto && collection{mat->get_collection()};
      if (collection.state_field_exists(unique_prefix)) {
        auto && state_field{collection.get_state_field(unique_prefix)};
        auto && field_current(
            muGrid::TypedField<T>::safe_cast(state_field.current()));
        local_fields_current.push_back(field_current);
        nb_component_categories.insert(field_current.get_nb_components());
        tag_categories.insert(field_current.get_sub_division_tag());
      }
    }

    // reject if the field appears with differing numbers of components
    if (nb_component_categories.size() != 1) {
      const auto & nb_match{nb_component_categories.size()};
      std::stringstream err_str{};
      if (nb_match > 1) {
        err_str
            << "The state fields named '" << unique_prefix
            << "' do not have the "
            << "same number of components in every material, which is a "
            << "requirement for globalising them! The following values were "
            << "found by material:" << std::endl;
        for (auto & mat : this->materials) {
          auto && coll{mat->get_collection()};
          if (coll.state_field_exists(unique_prefix)) {
            auto && field{coll.get_state_field(unique_prefix).current()};
            err_str << field.get_nb_components()
                    << " components in material '" << mat->get_name() << "'"
                    << std::endl;
          }
        }
      } else {
        err_str << "The state field named '" << unique_prefix
                << "' does not exist in "
                << "any of the materials and can therefore not be globalised!";
      }
      throw RuntimeError(err_str.str());
    }

    const Index_t nb_components{*nb_component_categories.begin()};

    // reject if the field appears with differing subdivision tags
    if (tag_categories.size() != 1) {
      const auto & nb_match{tag_categories.size()};
      std::stringstream err_str{};
      if (nb_match > 1) {
        err_str
            << "The fields named '" << unique_prefix << "' do not have the "
            << "same sub-division in every material, which is a "
            << "requirement for globalising them! The following values were "
            << "found by material:" << std::endl;
        for (auto & mat : this->materials) {
          auto & coll = mat->get_collection();
          if (coll.field_exists(unique_prefix)) {
            auto & field{coll.get_field(unique_prefix)};
            err_str << "tag '" << field.get_sub_division_tag()
                    << "' in material '" << mat->get_name() << "'" << std::endl;
          }
        }
      } else {
        err_str << "The field named '" << unique_prefix
                << "' does not exist in "
                << "any of the materials and can therefore not be globalised!";
      }
      throw RuntimeError(err_str.str());
    }
    const std::string tag{*tag_categories.begin()};

    muGrid::TypedField<T> & global_field{
        this->fields->field_exists(unique_prefix)
            ? dynamic_cast<muGrid::TypedField<T> &>(
                  this->fields->get_field(unique_prefix))
            : this->fields->template register_field<T>(unique_prefix,
                                                       nb_components, tag)};
    global_field.set_zero();

    auto global_map{global_field.get_pixel_map()};

    // fill it with local current state of internal values
    for (auto & local_field : local_fields_current) {
      auto pixel_map{
          muGrid::TypedField<T>::safe_cast(local_field).get_pixel_map()};
      for (auto && pixel_id__value : pixel_map.enumerate_pixel_indices_fast()) {
        auto && pixel_id{std::get<0>(pixel_id__value)};
        auto && value{std::get<1>(pixel_id__value)};
        global_map[pixel_id] = value;
      }
    }
    return global_field;
  }

  /* ---------------------------------------------------------------------- */
  bool Cell::is_point_inside(const DynRcoord_t & point) const {
    auto length_pixels = this->get_projection().get_domain_lengths();
    Index_t counter = 0;

    for (int i = 0; i < this->get_spatial_dim(); i++) {
      if (point[i] <= length_pixels[i]) {
        counter++;
      }
    }
    return counter == this->get_spatial_dim();
  }

  /* ---------------------------------------------------------------------- */
  bool Cell::is_pixel_inside(const DynCcoord_t & pixel) const {
    auto nb_pixels = this->get_projection().get_nb_domain_grid_pts();
    Index_t counter = 0;
    for (int i = 0; i < this->get_spatial_dim(); i++) {
      if (pixel[i] < nb_pixels[i]) {
        counter++;
      }
    }
    return counter == this->get_spatial_dim();
  }

  /* ---------------------------------------------------------------------- */
  bool Cell::was_last_eval_non_linear() const {
    if (this->get_formulation() == Formulation::finite_strain) {
      return true;
    }
    for (auto & mat : this->materials) {
      if (mat->was_last_step_nonlinear()) {
        return true;
      }
    }
    return false;
  }

  /* ---------------------------------------------------------------------- */
#ifdef WITH_SPLIT
  void Cell::make_pixels_precipitate_for_laminate_material(
      const std::vector<DynRcoord_t> & precipitate_vertices,
      MaterialBase & mat_laminate, MaterialBase & mat_precipitate_cell,
      Material_sptr mat_precipitate, Material_sptr mat_matrix) {
    switch (this->get_formulation()) {
    case Formulation::small_strain: {
      switch (this->get_spatial_dim()) {
      case twoD: {
        this->make_pixels_precipitate_for_laminate_material_helper<
            twoD, Formulation::small_strain>(precipitate_vertices, mat_laminate,
                                             mat_precipitate_cell,
                                             mat_precipitate, mat_matrix);
        break;
      }
      case threeD: {
        this->make_pixels_precipitate_for_laminate_material_helper<
            threeD, Formulation::small_strain>(
            precipitate_vertices, mat_laminate, mat_precipitate_cell,
            mat_precipitate, mat_matrix);
        break;
      }
      default:
        throw RuntimeError("Invalid dimension");
        break;
      }
      break;
    }
    case Formulation::finite_strain: {
      switch (this->get_spatial_dim()) {
      case twoD: {
        this->make_pixels_precipitate_for_laminate_material_helper<
            twoD, Formulation::finite_strain>(
            precipitate_vertices, mat_laminate, mat_precipitate_cell,
            mat_precipitate, mat_matrix);
        break;
      }
      case threeD: {
        this->make_pixels_precipitate_for_laminate_material_helper<
            threeD, Formulation::finite_strain>(
            precipitate_vertices, mat_laminate, mat_precipitate_cell,
            mat_precipitate, mat_matrix);
        break;
      }
      default:
        throw RuntimeError("Invalid dimension");
        break;
      }
      break;
    }
    default:
      throw RuntimeError("Invalid formulation");
      break;
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t Dim, Formulation Form>
  void Cell::make_pixels_precipitate_for_laminate_material_helper(
      const std::vector<DynRcoord_t> & precipitate_vertices,
      MaterialBase & mat_laminate, MaterialBase & mat_precipitate_cell,
      Material_sptr mat_precipitate, Material_sptr mat_matrix) {
    if (not(Form == Formulation::finite_strain or
            Form == Formulation::small_strain)) {
      std::stringstream err_str{};
      err_str << "Material laminate is not defined for " << Form << std::endl;
      throw RuntimeError(err_str.str());
    }
    using MaterialLaminate_t =
        std::conditional_t<Form == Formulation::small_strain,
                           MaterialLaminate<Dim, Formulation::small_strain>,
                           MaterialLaminate<Dim, Formulation::finite_strain>>;

    auto && mat_lam_cast{static_cast<MaterialLaminate_t &>(mat_laminate)};

    RootNode<SplitCell::laminate> precipitate(*this, precipitate_vertices);
    auto && precipitate_intersects{precipitate.get_intersected_pixels()};
    auto && precipitate_intersects_id{precipitate.get_intersected_pixels_id()};
    auto && precipitate_intersection_ratios{
        precipitate.get_intersection_ratios()};
    auto && precipitate_intersection_normals{
        precipitate.get_intersection_normals()};
    auto && precipitate_intersection_states{
        precipitate.get_intersection_status()};
    bool if_print{false};

    for (auto && tup : akantu::enumerate(precipitate_intersects_id)) {
      auto && counter{std::get<0>(tup)};
      auto && pix_id{std::get<1>(tup)};
      auto && pix{precipitate_intersects[counter]};
      auto && state{precipitate_intersection_states[counter]};
      auto && normal{precipitate_intersection_normals[counter]};
      auto && ratio{precipitate_intersection_ratios[counter]};

      // these outputs are used for debugging:
      if (if_print) {
        if (state != corkpp::IntersectionState::enclosing) {
          std::cout
              << pix[0] << ", " << pix[1] << ", "
              << "s: "
              << static_cast<
                     std::underlying_type<corkpp::IntersectionState>::type>(
                     state)
              << ",r: " << ratio << ",n: " << normal(0) << ", " << normal(1)
              << std::endl;
        }
      }

      if (state == corkpp::IntersectionState::enclosing) {
        mat_precipitate_cell.add_pixel(pix_id);
      } else if (state == corkpp::IntersectionState::intersecting) {
        mat_lam_cast.add_pixel(pix_id, mat_precipitate, mat_matrix, ratio,
                               normal);
      }
    }
  }

/* ---------------------------------------------------------------------- */
#endif

}  // namespace muSpectre

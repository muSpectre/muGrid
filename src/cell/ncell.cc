/**
 * @file   ncell.cc
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
 * General Public License for more details.
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
 */

#include "cell_adaptor.hh"
#include "ncell.hh"

#include <libmugrid/nfield_map.hh>
#include <libmugrid/nfield_map_static.hh>

#include <set>

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  NCell::NCell(Projection_ptr projection)
      : projection{std::move(projection)},
        fields{std::make_unique<muGrid::GlobalNFieldCollection>(
            this->get_spatial_dim(), this->get_nb_quad())},
        strain{this->fields->register_real_field(
            "strain", dof_for_formulation(this->get_formulation(),
                                          this->get_material_dim()))},
        stress{this->fields->register_real_field(
            "stress", dof_for_formulation(this->get_formulation(),
                                          this->get_material_dim()))} {
    this->fields->initialise(this->projection->get_nb_subdomain_grid_pts(),
                             this->projection->get_subdomain_locations());
  }

  /* ---------------------------------------------------------------------- */
  bool NCell::is_initialised() const { return this->initialised; }

  /* ---------------------------------------------------------------------- */
  Dim_t NCell::get_nb_dof() const {
    const auto & strain_shape{this->get_strain_shape()};
    return this->get_nb_pixels() * this->get_nb_quad() * strain_shape[0] *
           strain_shape[1];
  }

  /* ---------------------------------------------------------------------- */
  size_t NCell::get_nb_pixels() const { return this->fields->get_nb_pixels(); }

  /* ---------------------------------------------------------------------- */
  const muFFT::Communicator & NCell::get_communicator() const {
    return this->projection->get_communicator();
  }

  /* ---------------------------------------------------------------------- */
  const Formulation & NCell::get_formulation() const {
    return this->projection->get_formulation();
  }

  /* ---------------------------------------------------------------------- */
  Dim_t NCell::get_material_dim() const { return this->get_spatial_dim(); }

  /* ---------------------------------------------------------------------- */
  void
  NCell::set_uniform_strain(const Eigen::Ref<const Matrix_t> & uniform_strain) {
    if (not this->initialised) {
      this->initialise();
    }
    Dim_t && dim{this->get_material_dim()};
    muGrid::NFieldMap<Real, Mapping::Mut>(this->strain, dim) = uniform_strain;
  }

  /* ---------------------------------------------------------------------- */
  MaterialBase & NCell::add_material(Material_ptr mat) {
    if (mat->get_material_dimension() != this->get_spatial_dim()) {
      throw std::runtime_error(
          "this cell class only accepts materials with the same dimensionality "
          "as the spatial problem.");
    }
    this->materials.push_back(std::move(mat));
    return *this->materials.back();
  }

  /* ---------------------------------------------------------------------- */
  CellAdaptor<NCell> NCell::get_adaptor() { return CellAdaptor<NCell>(*this); }

  /* ---------------------------------------------------------------------- */
  void NCell::save_history_variables() {
    for (auto && mat : this->materials) {
      mat->save_history_variables();
    }
  }

  /* ---------------------------------------------------------------------- */
  std::array<Dim_t, 2> NCell::get_strain_shape() const {
    switch (this->get_formulation()) {
    case Formulation::finite_strain: {
      return std::array<Dim_t, 2>{this->get_material_dim(),
                                  this->get_material_dim()};
      break;
    }
    case Formulation::small_strain: {
      return std::array<Dim_t, 2>{this->get_material_dim(),
                                  this->get_material_dim()};
      break;
    }
    default:
      throw std::runtime_error("Formulation not implemented");
      break;
    }
  }

  /* ---------------------------------------------------------------------- */
  Dim_t NCell::get_strain_size() const {
    auto && shape{this->get_strain_shape()};
    return shape[0] * shape[1];
  }

  /* ---------------------------------------------------------------------- */
  const Dim_t & NCell::get_spatial_dim() const {
    return this->projection->get_dim();
  }

  /* ---------------------------------------------------------------------- */
  const Dim_t & NCell::get_nb_quad() const {
    return this->projection->get_nb_quad();
  }

  /* ---------------------------------------------------------------------- */
  void NCell::check_material_coverage() const {
    auto nb_pixels{muGrid::CcoordOps::get_size(
        this->projection->get_nb_subdomain_grid_pts())};
    std::vector<MaterialBase *> assignments(nb_pixels, nullptr);

    for (auto & mat : this->materials) {
      mat->initialise();
      for (auto & index : mat->get_pixel_indices()) {
        auto & assignment{assignments.at(index)};
        if (assignment != nullptr) {
          std::stringstream err{};
          err << "Pixel " << index << "is already assigned to material '"
              << assignment->get_name()
              << "' and cannot be reassigned to material '" << mat->get_name();
          throw std::runtime_error(err.str());
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
      throw std::runtime_error(err.str());
    }
  }

  /* ---------------------------------------------------------------------- */
  void NCell::initialise(muFFT::FFT_PlanFlags flags) {
    // check that all pixels have been assigned exactly one material
    this->check_material_coverage();
    for (auto && mat : this->materials) {
      mat->initialise();
    }
    // initialise the projection and compute the fft plan
    this->projection->initialise(flags);
    this->initialised = true;
  }

  /* ---------------------------------------------------------------------- */
  auto NCell::get_pixels() const -> const muGrid::CcoordOps::DynamicPixels & {
    return this->fields->get_pixels();
  }

  /* ---------------------------------------------------------------------- */
  auto NCell::get_quad_pt_indices() const
      -> muGrid::NFieldCollection::IndexIterable {
    return this->fields->get_quad_pt_indices();
  }

  /* ---------------------------------------------------------------------- */
  auto NCell::get_pixel_indices() const
      -> muGrid::NFieldCollection::PixelIndexIterable {
    return this->fields->get_pixel_indices_fast();
  }

  /* ---------------------------------------------------------------------- */
  muGrid::RealNField & NCell::get_strain() { return this->strain; }

  /* ---------------------------------------------------------------------- */
  const muGrid::RealNField & NCell::get_stress() const { return this->stress; }

  /* ---------------------------------------------------------------------- */
  const muGrid::RealNField & NCell::get_tangent(bool do_create) {
    if (not this->tangent) {
      if (do_create) {
        this->tangent = this->fields->register_real_field(
            "Tangent_stiffness",
            muGrid::ipow(dof_for_formulation(this->get_formulation(),
                                             this->get_material_dim()),
                         2));
      } else {
        throw std::runtime_error("Tangent has not been created");
      }
    }
    return this->tangent.value();
  }

  /* ---------------------------------------------------------------------- */
  const muGrid::RealNField & NCell::evaluate_stress() {
    if (this->initialised == false) {
      this->initialise();
    }
    for (auto & mat : this->materials) {
      mat->compute_stresses(this->strain, this->stress,
                            this->get_formulation());
    }
    return this->stress;
  }

  //----------------------------------------------------------------------------//
  auto NCell::evaluate_stress_eigen() -> Eigen_cmap {
    return this->evaluate_stress().eigen_vec();
  }

  /* ---------------------------------------------------------------------- */
  std::tuple<const muGrid::RealNField &, const muGrid::RealNField &>
  NCell::evaluate_stress_tangent() {
    if (this->initialised == false) {
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

  //----------------------------------------------------------------------------//
  auto NCell::evaluate_stress_tangent_eigen()
      -> std::tuple<const Eigen_cmap, const Eigen_cmap> {
    auto && fields{this->evaluate_stress_tangent()};
    return std::tuple<const Eigen_cmap, const Eigen_cmap>(
        std::get<0>(fields).eigen_vec(), std::get<1>(fields).eigen_vec());
  }

  /* ---------------------------------------------------------------------- */
  muGrid::RealNField &
  NCell::globalise_real_internal_field(const std::string & unique_name) {
    return this->template globalise_internal_field<Real>(unique_name);
  }
  /* ---------------------------------------------------------------------- */
  muGrid::IntNField &
  NCell::globalise_int_internal_field(const std::string & unique_name) {
    return this->template globalise_internal_field<Int>(unique_name);
  }
  /* ---------------------------------------------------------------------- */
  muGrid::UintNField &
  NCell::globalise_uint_internal_field(const std::string & unique_name) {
    return this->template globalise_internal_field<Uint>(unique_name);
  }
  /* ---------------------------------------------------------------------- */
  muGrid::ComplexNField &
  NCell::globalise_complex_internal_field(const std::string & unique_name) {
    return this->template globalise_internal_field<Complex>(unique_name);
  }

  /* ---------------------------------------------------------------------- */
  muGrid::GlobalNFieldCollection & NCell::get_fields() { return *this->fields; }

  /* ---------------------------------------------------------------------- */
  void NCell::apply_projection(muGrid::TypedNFieldBase<Real> & field) {
    this->projection->apply_projection(field);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimM>
  void
  NCell::apply_directional_stiffness(const muGrid::RealNField & delta_strain,
                                     const muGrid::RealNField & tangent,
                                     muGrid::RealNField & delta_stress) {
    muGrid::T2NFieldMap<Real, muGrid::Mapping::Const, DimM> strain_map{
        delta_strain};
    muGrid::T4NFieldMap<Real, muGrid::Mapping::Const, DimM> tangent_map{
        tangent};
    muGrid::T2NFieldMap<Real, muGrid::Mapping::Mut, DimM> stress_map{
        delta_stress};
    for (auto && tup : akantu::zip(strain_map, tangent_map, stress_map)) {
      auto & df = std::get<0>(tup);
      auto & k = std::get<1>(tup);
      auto & dp = std::get<2>(tup);
      dp = -Matrices::tensmult(k, df);
    }
  }

  /* ---------------------------------------------------------------------- */
  void NCell::evaluate_projected_directional_stiffness(
      const muGrid::RealNField & delta_strain,
      muGrid::RealNField & del_stress) {
    if (not this->tangent) {
      throw std::runtime_error("evaluate_projected_directional_stiffness "
                               "requires the tangent moduli");
    }
    if (delta_strain.get_nb_components() != this->get_strain_size()) {
      std::stringstream err{};
      err << "The input field should have " << this->get_strain_size()
          << " components per quadrature point, but has "
          << delta_strain.get_nb_components() << " components.";
      throw std::runtime_error(err.str());
    }
    if (delta_strain.get_collection().get_nb_quad() !=
        this->get_strain().get_collection().get_nb_quad()) {
      std::stringstream err{};
      err << "The input field should have "
          << this->get_strain().get_collection().get_nb_quad()
          << " quadrature point per pixel, but has "
          << delta_strain.get_collection().get_nb_quad() << " points.";
      throw std::runtime_error(err.str());
    }
    if (delta_strain.get_collection().get_nb_pixels() !=
        this->get_strain().get_collection().get_nb_pixels()) {
      std::stringstream err{};
      err << "The input field should have "
          << this->get_strain().get_collection().get_nb_pixels()
          << " pixels, but has "
          << delta_strain.get_collection().get_nb_pixels() << " pixels.";
      throw std::runtime_error(err.str());
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
      throw std::runtime_error(err.str());
      break;
    }
    this->apply_projection(del_stress);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimM>
  void NCell::add_projected_directional_stiffness_helper(
      const muGrid::TypedNFieldBase<Real> & delta_strain,
      const muGrid::TypedNFieldBase<Real> & tangent, const Real & alpha,
      muGrid::TypedNFieldBase<Real> & delta_stress) {
    muGrid::T2NFieldMap<Real, muGrid::Mapping::Const, DimM> strain_map{
        delta_strain};
    muGrid::T4NFieldMap<Real, muGrid::Mapping::Const, DimM> tangent_map{
        tangent};
    muGrid::T2NFieldMap<Real, muGrid::Mapping::Mut, DimM> stress_map{
        delta_stress};
    for (auto && tup : akantu::zip(strain_map, tangent_map, stress_map)) {
      auto & df = std::get<0>(tup);
      auto & k = std::get<1>(tup);
      auto & dp = std::get<2>(tup);
      dp += alpha * Matrices::tensmult(k, df);
    }
  }
  /* ---------------------------------------------------------------------- */
  void NCell::add_projected_directional_stiffness(EigenCVec_t delta_strain,
                                                  const Real & alpha,
                                                  EigenVec_t del_stress) {
    auto delta_strain_field_ptr{muGrid::WrappedNField<Real>::make_const(
        "delta_strain", *this->fields, this->get_strain_size(), delta_strain)};
    muGrid::WrappedNField<Real> del_stress_field{
        "delta_stress", *this->fields, this->get_strain_size(), del_stress};
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
      throw std::runtime_error(err.str());
      break;
    }
    this->apply_projection(del_stress_field);
  }
  /* ---------------------------------------------------------------------- */
  template <typename T>
  muGrid::TypedNField<T> &
  NCell::globalise_internal_field(const std::string & unique_name) {
    // start by checking that the field exists at least once, and that
    // it always has th same number of components
    std::set<Dim_t> nb_component_categories{};
    std::vector<std::reference_wrapper<muGrid::NField>> local_fields;

    for (auto & mat : this->materials) {
      auto & collection{mat->get_collection()};
      if (collection.field_exists(unique_name)) {
        auto & field{muGrid::TypedNField<T>::safe_cast(
            collection.get_field(unique_name))};
        local_fields.push_back(field);
        nb_component_categories.insert(field.get_nb_components());
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
            err_str << field.get_nb_components() << " components in material '"
                    << mat->get_name() << "'" << std::endl;
          }
        }
      } else {
        err_str << "The field named '" << unique_name << "' does not exist in "
                << "any of the materials and can therefore not be globalised!";
      }
      throw std::runtime_error(err_str.str());
    }

    const Dim_t nb_components{*nb_component_categories.begin()};

    // get and prepare the field
    auto & global_field{
        this->fields->template register_field<T>(unique_name, nb_components)};
    global_field.set_zero();

    auto global_map{global_field.get_pixel_map()};

    // fill it with local internal values
    for (auto & local_field : local_fields) {
      auto pixel_map{muGrid::TypedNField<T>::safe_cast(local_field)
                                .get_pixel_map()};
      for (auto && pixel_id__value : pixel_map.enumerate_pixel_indices_fast()) {
        const auto & pixel_id{std::get<0>(pixel_id__value)};
        const auto & value{std::get<1>(pixel_id__value)};
        global_map[pixel_id] = value;
      }
    }
    return global_field;
  }

}  // namespace muSpectre

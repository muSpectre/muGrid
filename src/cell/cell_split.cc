/**
 * @file   cell_split.cc
 *
 * @author Ali Falsafi <ali.faslafi@epfl.ch>
 *
 * @date   10 Dec 2019
 *
 * @brief  Implementation for cell base class
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

#include <libmugrid/exception.hh>

#include "cell/cell_split.hh"

using muGrid::RuntimeError;

namespace muSpectre {

  /* ---------------------------------------------------------------------- */

  CellSplit::CellSplit(Projection_ptr projection)
      : Parent(std::move(projection), SplitCell::simple) {}

  /* ---------------------------------------------------------------------- */

  std::vector<Real> CellSplit::get_assigned_ratios() {
    auto && nb_pixels{muGrid::CcoordOps::get_size(
        this->get_projection().get_nb_subdomain_grid_pts())};
    std::vector<Real> pixel_assigned_ratios(nb_pixels, 0.0);
    for (auto && mat : this->materials) {
      mat->get_assigned_ratios(pixel_assigned_ratios);
    }
    return pixel_assigned_ratios;
  }
  /* ---------------------------------------------------------------------- */

  std::vector<Real> CellSplit::get_unassigned_ratios_incomplete_pixels() const {
    auto nb_pixels{muGrid::CcoordOps::get_size(
        this->get_projection().get_nb_subdomain_grid_pts())};
    std::vector<Real> pixel_assigned_ratios(nb_pixels, 0.0);
    std::vector<Real> pixel_assigned_ratios_incomplete_pixels;
    for (auto && mat : this->materials) {
      mat->get_assigned_ratios(pixel_assigned_ratios);
    }
    for (auto && ratio : pixel_assigned_ratios) {
      if (ratio < 1) {
        pixel_assigned_ratios_incomplete_pixels.push_back(1.0 - ratio);
      }
    }
    return pixel_assigned_ratios_incomplete_pixels;
  }

  /* ---------------------------------------------------------------------- */

  auto CellSplit::make_incomplete_pixels() -> IncompletePixels {
    return IncompletePixels(*this);
  }

  /* ---------------------------------------------------------------------- */

  std::vector<Index_t> CellSplit::get_index_incomplete_pixels() const {
    auto nb_pixels{muGrid::CcoordOps::get_size(
        this->get_projection().get_nb_subdomain_grid_pts())};
    std::vector<Real> pixel_assigned_ratios(nb_pixels, 0.0);
    std::vector<Index_t> index_unassigned_pixels;
    for (auto && mat : this->materials) {
      mat->get_assigned_ratios(pixel_assigned_ratios);
    }
    for (auto && tup : this->get_pixels().enumerate()) {
      auto && i{std::get<0>(tup)};
      if (pixel_assigned_ratios[i] < 1) {
        index_unassigned_pixels.push_back(i);
      }
    }
    return index_unassigned_pixels;
  }

  /* ---------------------------------------------------------------------- */

  std::vector<DynCcoord_t> CellSplit::get_unassigned_pixels() {
    auto nb_pixels{muGrid::CcoordOps::get_size(
        this->get_projection().get_nb_subdomain_grid_pts())};
    std::vector<Real> pixel_assigned_ratios(nb_pixels, 0.0);
    std::vector<DynCcoord_t> unassigned_pixels{};
    for (auto & mat : this->materials) {
      mat->get_assigned_ratios(pixel_assigned_ratios);
    }
    for (auto && tup : this->get_pixels().enumerate()) {
      auto && index{std::get<0>(tup)};
      auto && pix{std::get<1>(tup)};
      if (pixel_assigned_ratios[index] < 1) {
        unassigned_pixels.push_back(pix);
      }
    }
    return unassigned_pixels;
  }

  /* ---------------------------------------------------------------------- */

  void CellSplit::check_material_coverage() const {
    auto nb_pixels{muGrid::CcoordOps::get_size(
        this->get_projection().get_nb_subdomain_grid_pts())};
    std::vector<Real> pixel_assigned_ratios(nb_pixels, 0.0);
    for (auto & mat : this->materials) {
      mat->get_assigned_ratios(pixel_assigned_ratios);
    }
    std::vector<DynCcoord_t> over_assigned_pixels{};
    std::vector<DynCcoord_t> under_assigned_pixels;
    for (size_t i{0}; i < nb_pixels; ++i) {
      if (pixel_assigned_ratios[i] > 1.0) {
        over_assigned_pixels.emplace_back(muGrid::CcoordOps::get_ccoord<2>(
            this->get_projection().get_nb_subdomain_grid_pts(),
            this->get_projection().get_subdomain_locations(), i));
      } else if (pixel_assigned_ratios[i] < 1.0) {
        under_assigned_pixels.emplace_back(muGrid::CcoordOps::get_ccoord<2>(
            this->get_projection().get_nb_subdomain_grid_pts(),
            this->get_projection().get_subdomain_locations(), i));
      }
    }
    if (over_assigned_pixels.size() != 0) {
      std::stringstream err{};
      err << "Execesive material is assigned to the following pixels: ";
      for (auto & pixel : over_assigned_pixels) {
        muGrid::operator<<(err, pixel);
      }
      err << "and that cannot be handled";
      throw RuntimeError(err.str());
    }
    if (under_assigned_pixels.size() != 0) {
      std::stringstream err{};
      err << "Insufficient material is assigned to the following pixels: ";
      for (auto & pixel : under_assigned_pixels) {
        muGrid::operator<<(err, pixel);
      }
      err << "and that cannot be handled";
      throw RuntimeError(err.str());
    }
  }

  /* ---------------------------------------------------------------------- */

  // this function handles the evaluation of stress
  // in case the cells have materials in which pixels are partially composed of
  // diffferent materials.
  const muGrid::RealField & CellSplit::evaluate_stress() {
    if (not this->initialised) {
      this->initialise();
    }
    // Here we should first set P equal to zeros first because we
    // want to add up contribution
    // of the partial influence of different materials assigend to each pixel.
    // Therefore, this values should be
    // initialised as zero filled tensors
    this->stress.set_zero();
    for (auto & mat : this->materials) {
      mat->compute_stresses(this->strain, this->stress,
                            this->get_formulation());
    }
    return this->stress;
  }

  /* ---------------------------------------------------------------------- */
  // this function handles the evaluation of stress and tangent matrix
  // in case the cells have materials in which pixels are partially composed of
  // diffferent materials.
  std::tuple<const muGrid::RealField &, const muGrid::RealField &>
  CellSplit::evaluate_stress_tangent() {
    if (not this->initialised) {
      this->initialise();
    }
    //! High level compatibility checks
    if (strain.get_nb_entries() == muGrid::Unknown or
        strain.get_nb_entries() != this->strain.get_nb_entries()) {
      throw RuntimeError("Size mismatch");
    }
    constexpr bool create_tangent{true};
    this->get_tangent(create_tangent);

    // Here we should first set P and K matrixes equal to zeros first because
    // we want to add up contribution of the partial influence of different
    // materials assigend to each pixel. Therefore, this values should be
    // initiialised as zero filled tensors
    this->set_p_k_zero();

    // full response is composed of the stresses and tangent matrix is
    // retruned by this function
    for (auto & mat : this->materials) {
      mat->compute_stresses_tangent(strain, this->stress, this->tangent.value(),
                                    this->get_formulation(), SplitCell::simple);
    }
    return std::tie(this->stress, this->tangent.value());
  }

  /* ----------------------------------------------------------------------- */
  void CellSplit::make_automatic_precipitate_split_pixels(
      const std::vector<DynRcoord_t> & precipitate_vertices,
      MaterialBase & material) {
    RootNode<SplitCell::simple> precipitate(*this, precipitate_vertices);
    auto && precipitate_intersects{precipitate.get_intersected_pixels()};

    auto && precipitate_intersects_id{precipitate.get_intersected_pixels_id()};

    auto && precipitate_intersection_ratios{
        precipitate.get_intersection_ratios()};

    for (auto tup :
         akantu::zip(precipitate_intersects_id, precipitate_intersects,
                     precipitate_intersection_ratios)) {
      auto pix_id{std::get<0>(tup)};
      // auto pix { std::get<1>(tup)};
      auto ratio{std::get<2>(tup)};
      material.add_pixel_split(pix_id, ratio);
    }
  }

  /* ----------------------------------------------------------------------
   */
  MaterialBase & CellSplit::add_material(Material_ptr mat) {
    if (mat->get_material_dimension() != this->get_spatial_dim()) {
      throw RuntimeError(
          "this cell class only accepts materials with the same "
          "dimensionality "
          "as the spatial problem.");
    }
    mat->allocate_optional_fields();
    this->materials.push_back(std::move(mat));
    return *this->materials.back();
  }

  /* ----------------------------------------------------------------------*/

  void CellSplit::complete_material_assignment(MaterialBase & material) {
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
    std::vector<Real> pixel_assigned_ratio(this->get_assigned_ratios());
    for (auto && tup : akantu::enumerate(this->get_pixel_indices())) {
      auto && pixel{std::get<1>(tup)};
      auto iterator{std::get<0>(tup)};
      if (pixel_assigned_ratio[iterator] < 1.0) {
        material.add_pixel_split(pixel, 1.0 - pixel_assigned_ratio[iterator]);
      }
    }
  }

  /* ----------------------------------------------------------------------*/
  void CellSplit::set_p_k_zero() {
    this->stress.set_zero();
    this->tangent.value().get().set_zero();
  }

  /* ----------------------------------------------------------------------*/
  CellSplit::IncompletePixels::IncompletePixels(const CellSplit & cell)
      : cell(cell), incomplete_assigned_ratios(
                        cell.get_unassigned_ratios_incomplete_pixels()),
        index_incomplete_pixels(cell.get_index_incomplete_pixels()) {}

  /* ----------------------------------------------------------------------*/
  CellSplit::IncompletePixels::iterator::iterator(
      const IncompletePixels & pixels, Index_t dim, bool begin)
      : incomplete_pixels(pixels), dim{dim},
        index{begin ? 0
                    : this->incomplete_pixels.index_incomplete_pixels.size()} {}

  /* ----------------------------------------------------------------------*/
  auto CellSplit::IncompletePixels::iterator::operator++() -> iterator & {
    this->index++;
    return *this;
  }

  /* ---------------------------------------------------------------------*/
  bool
  CellSplit::IncompletePixels::iterator::operator!=(const iterator & other) {
    return (this->index != other.index);
  }

  /* ---------------------------------------------------------------------*/
  auto CellSplit::IncompletePixels::iterator::operator*() const -> value_type {
    switch (this->dim) {
    case twoD: {
      return this->template deref_helper<twoD>();
      break;
    }
    case threeD: {
      return this->template deref_helper<threeD>();
      break;
    }
    default: {
      std::stringstream err;
      err << "Input dimesnion is not correct. Valid dimnensions are only twoD "
             "or threeD ";
      throw(RuntimeError(err.str()));
      break;
    }
    }
  }

  /* ---------------------------------------------------------------------*/
  template <Index_t DimS>
  auto CellSplit::IncompletePixels::iterator::deref_helper() const
      -> value_type {
    DynCcoord_t ccoord{muGrid::CcoordOps::get_ccoord<DimS>(
        this->incomplete_pixels.cell.get_projection().get_nb_domain_grid_pts(),
        this->incomplete_pixels.cell.get_projection().get_subdomain_locations(),
        this->incomplete_pixels.index_incomplete_pixels[index])};
    auto ratio{this->incomplete_pixels.incomplete_assigned_ratios[index]};
    return std::make_tuple(ccoord, ratio);
  }

  /* ---------------------------------------------------------------------*/

}  // namespace muSpectre

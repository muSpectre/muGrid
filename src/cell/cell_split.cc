/**
 * @file   cell_split.cc
 *
 * @author Ali Falsafi <ali.faslafi@epfl.ch>
 *
 * @date   19 Apr 2018
 *
 * @brief  Implementation for cell base class
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
 * General Public License for more details.\
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Emacs; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */
#include "common/muSpectre_common.hh"
#include "libmugrid/ccoord_operations.hh"
#include "libmugrid/field.hh"
// #include "common/utilities.hh"
#include "materials/material_base.hh"
#include "projection/projection_base.hh"
#include "cell/cell_traits.hh"
#include "cell/cell_base.hh"
#include "cell/cell_split.hh"

#include <vector>
#include <memory>
#include <tuple>
#include <sstream>
#include <algorithm>

namespace muSpectre {

  /* ---------------------------------------------------------------------- */

  template <Dim_t DimS, Dim_t DimM>
  CellSplit<DimS, DimM>::CellSplit(Projection_ptr projection)
      : Parent(std::move(projection), SplitCell::simple) {}

  /* ---------------------------------------------------------------------- */

  template <Dim_t DimS, Dim_t DimM>
  std::vector<Real> CellSplit<DimS, DimM>::get_assigned_ratios() {
    auto nb_pixels = muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts);
    std::vector<Real> pixel_assigned_ratios(nb_pixels, 0.0);
    for (auto & mat : this->materials) {
      mat->get_assigned_ratios(pixel_assigned_ratios,
                               this->nb_subdomain_grid_pts,
                               this->subdomain_locations);
    }
    return pixel_assigned_ratios;
  }
  /* ---------------------------------------------------------------------- */

  template <Dim_t DimS, Dim_t DimM>
  std::vector<Real>
  CellSplit<DimS, DimM>::get_unassigned_ratios_incomplete_pixels() {
    auto nb_pixels = muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts);
    std::vector<Real> pixel_assigned_ratios(nb_pixels, 0.0);
    std::vector<Real> pixel_assigned_ratios_incomplete_pixels;
    for (auto & mat : this->materials) {
      mat->get_assigned_ratios(pixel_assigned_ratios,
                               this->nb_subdomain_grid_pts,
                               this->subdomain_locations);
    }
    for (auto && ratio : pixel_assigned_ratios) {
      if (ratio < 1) {
        pixel_assigned_ratios_incomplete_pixels.push_back(1.0 - ratio);
      }
    }
    return pixel_assigned_ratios_incomplete_pixels;
  }
  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  auto CellSplit<DimS, DimM>::make_incomplete_pixels() -> IncompletePixels {
    return IncompletePixels(*this);
  }
  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  std::vector<int> CellSplit<DimS, DimM>::get_index_incomplete_pixels() {
    auto nb_pixels = muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts);
    std::vector<Real> pixel_assigned_ratios(nb_pixels, 0.0);
    std::vector<int> index_unassigned_pixels;
    for (auto & mat : this->materials) {
      mat->get_assigned_ratios(pixel_assigned_ratios,
                               this->nb_subdomain_grid_pts,
                               this->subdomain_locations);
    }
    for (auto && tup : akantu::enumerate(this->pixels)) {
      auto && i{std::get<0>(tup)};
      if (pixel_assigned_ratios[i] < 1) {
        index_unassigned_pixels.push_back(i);
      }
    }
    return index_unassigned_pixels;
  }
  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  std::vector<Ccoord_t<DimS>> CellSplit<DimS, DimM>::get_unassigned_pixels() {
    auto nb_pixels = muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts);
    std::vector<Real> pixel_assigned_ratios(nb_pixels, 0.0);
    std::vector<Ccoord_t<DimS>> unassigned_pixels;
    for (auto & mat : this->materials) {
      mat->get_assigned_ratios(pixel_assigned_ratios,
                               this->nb_subdomain_grid_pts,
                               this->subdomain_locations);
    }
    for (auto && tup : akantu::enumerate(this->pixels)) {
      auto && index{std::get<0>(tup)};
      auto && pix{std::get<1>(tup)};
      if (pixel_assigned_ratios[index] < 1) {
        unassigned_pixels.push_back(pix);
      }
    }
    return unassigned_pixels;
  }

  /* ---------------------------------------------------------------------- */

  template <Dim_t DimS, Dim_t DimM>
  void CellSplit<DimS, DimM>::check_material_coverage() {
    auto nb_pixels = muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts);
    std::vector<Real> pixel_assigned_ratios(nb_pixels, 0.0);
    for (auto & mat : this->materials) {
      mat->get_assigned_ratios(pixel_assigned_ratios,
                               this->nb_subdomain_grid_pts,
                               this->subdomain_locations);
    }
    std::vector<Ccoord_t<DimM>> over_assigned_pixels;
    std::vector<Ccoord_t<DimM>> under_assigned_pixels;
    for (size_t i = 0; i < nb_pixels; ++i) {
      if (pixel_assigned_ratios[i] > 1.0) {
        over_assigned_pixels.push_back(muGrid::CcoordOps::get_ccoord(
            this->nb_subdomain_grid_pts, this->subdomain_locations, i));
      } else if (pixel_assigned_ratios[i] < 1.0) {
        under_assigned_pixels.push_back(muGrid::CcoordOps::get_ccoord(
            this->nb_subdomain_grid_pts, this->subdomain_locations, i));
      }
    }
    if (over_assigned_pixels.size() != 0) {
      std::stringstream err{};
      err << "Execesive material is assigned to the following pixels: ";
      for (auto & pixel : over_assigned_pixels) {
        muGrid::operator<<(err, pixel);
      }
      err << "and that cannot be handled";
      throw std::runtime_error(err.str());
    }
    if (under_assigned_pixels.size() != 0) {
      std::stringstream err{};
      err << "Insufficient material is assigned to the following pixels: ";
      for (auto & pixel : under_assigned_pixels) {
        muGrid::operator<<(err, pixel);
      }
      err << "and that cannot be handled";
      throw std::runtime_error(err.str());
    }
  }
  /* ---------------------------------------------------------------------- */
  // this piece of code handles the evaluation of stress an dtangent matrix
  // in case the cells have materials in which pixels are partially composed of
  // diffferent materials.

  template <Dim_t DimS, Dim_t DimM>
  typename CellSplit<DimS, DimM>::FullResponse_t
  CellSplit<DimS, DimM>::evaluate_stress_tangent(StrainField_t & F) {
    if (this->initialised == false) {
      this->initialise();
    }
    //! High level compatibility checks
    if (F.size() != this->F.size()) {
      throw std::runtime_error("Size mismatch");
    }
    constexpr bool create_tangent{true};
    this->get_tangent(create_tangent);

    // Here we should first set P and K matrixes equal to zeros first because we
    // want to add up contribution
    // of the partial influence of different materials assigend to each pixel.
    // Therefore, this values should be
    // initiialised as zero filled tensors
    this->set_p_k_zero();

    // full response is composed of the stresses and tangent matrix is retruned
    // by this function
    for (auto & mat : this->materials) {
      mat->compute_stresses_tangent(F, this->P, this->K.value(),
                                    this->get_formulation(),
                                    this->is_cell_split);
    }
    return std::tie(this->P, this->K.value());
  }
  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  auto CellSplit<DimS, DimM>::evaluate_stress() -> ConstVector_ref {
    if (not this->initialised) {
      this->initialise();
    }
    this->P.set_zero();
    for (auto & mat : this->materials) {
      mat->compute_stresses(this->F, this->P, this->get_formulation());
    }

    return this->P.const_eigenvec();
  }
  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  auto CellSplit<DimS, DimM>::evaluate_stress_tangent()
      -> std::array<ConstVector_ref, 2> {
    if (not this->initialised) {
      this->initialise();
    }

    constexpr bool create_tangent{true};
    this->get_tangent(create_tangent);

    // Here we should first set P and K matrixes equal to zeros first because we
    // want to add up contribution
    // of the partial influence of different materials assigend to each pixel.
    // Therefore, this values should be
    // initiialised as zero filled tensors
    this->set_p_k_zero();
    for (auto & mat : this->materials) {
      mat->compute_stresses_tangent(this->F, this->P, this->K.value(),
                                    this->get_formulation(),
                                    this->is_cell_split);
    }
    const TangentField_t & k = this->K.value();
    return std::array<ConstVector_ref, 2>{this->P.const_eigenvec(),
                                          k.const_eigenvec()};
  }
  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void CellSplit<DimS, DimM>::make_automatic_precipitate_split_pixels(
      std::vector<Rcoord_t<DimS>> precipitate_vertices,
      MaterialBase<DimS, DimM> & material) {
    RootNode<DimS, SplitCell::simple> precipitate(*this, precipitate_vertices);
    auto && precipitate_intersects = precipitate.get_intersected_pixels();

    auto && precipitate_intersection_ratios =
        precipitate.get_intersection_ratios();

    for (auto tup :
         akantu::zip(precipitate_intersects, precipitate_intersection_ratios)) {
      auto pix = std::get<0>(tup);
      auto ratio = std::get<1>(tup);
      material.add_pixel_split(pix, ratio);
    }
  }
  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void CellSplit<DimS, DimM>::complete_material_assignment(
      MaterialBase<DimS, DimM> & material) {
    std::vector<Real> pixel_assigned_ratio(this->get_assigned_ratios());
    for (auto && tup : akantu::enumerate(*this)) {
      auto && pixel = std::get<1>(tup);
      auto iterator = std::get<0>(tup);
      if (pixel_assigned_ratio[iterator] < 1.0) {
        material.add_pixel_split(pixel, 1.0 - pixel_assigned_ratio[iterator]);
      }
    }
  }
  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void CellSplit<DimS, DimM>::set_p_k_zero() {
    this->P.set_zero();
    this->K.value().get().set_zero();
  }
  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  CellSplit<DimS, DimM>::IncompletePixels::IncompletePixels(
      CellSplit<DimS, DimM> & cell)
      : cell(cell), incomplete_assigned_ratios(
                        cell.get_unassigned_ratios_incomplete_pixels()),
        index_incomplete_pixels(cell.get_index_incomplete_pixels()) {}

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  CellSplit<DimS, DimM>::IncompletePixels::iterator::iterator(
      const IncompletePixels & pixels, bool begin)
      : incomplete_pixels(pixels),
        index{begin ? 0
                    : this->incomplete_pixels.index_incomplete_pixels.size()} {}
  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  auto CellSplit<DimS, DimM>::IncompletePixels::iterator::operator++()
      -> iterator & {
    this->index++;
    return *this;
  }
  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  auto CellSplit<DimS, DimM>::IncompletePixels::iterator::
  operator!=(const iterator & other) -> bool {
    //        return (this->incomplete_pixels.index_incomplete_pixels[index] !=
    //                other.incomplete_pixels.index_incomplete_pixels[index]);
    return (this->index != other.index);
  }
  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  auto CellSplit<DimS, DimM>::IncompletePixels::iterator::operator*()
      -> value_type const {
    auto ccoord = muGrid::CcoordOps::get_ccoord(
        this->incomplete_pixels.cell.get_nb_domain_grid_pts(),
        this->incomplete_pixels.cell.get_subdomain_locations(),
        this->incomplete_pixels.index_incomplete_pixels[index]);
    auto ratio = this->incomplete_pixels.incomplete_assigned_ratios[index];
    return std::make_tuple(ccoord, ratio);
  }
  /* ---------------------------------------------------------------------- */
  template class CellSplit<twoD, twoD>;
  template class CellSplit<threeD, threeD>;
  /* ---------------------------------------------------------------------- */

}  // namespace muSpectre

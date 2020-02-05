/**
 * @file   material_laminate.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date    04 Jun 2018
 *
 * @brief material that uses laminae homogenisation
 *
 * Copyright © 2018 Ali Falsafi
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
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#include "material_laminate.hh"

namespace muSpectre {
  template <Dim_t DimM>
  MaterialLaminate<DimM>::MaterialLaminate(const std::string & name,
                                           const Dim_t & spatial_dimension,
                                           const Dim_t & nb_quad_pts)
      : Parent(name, spatial_dimension, DimM, nb_quad_pts),
        normal_vector_field{"normal vector", this->internal_fields},
        volume_ratio_field{"volume ratio", this->internal_fields} {}
  /* ----------------------------------------------------------------------
   */
  template <Dim_t DimM>
  void MaterialLaminate<DimM>::add_pixel(const size_t & /*pixel_id*/) {
    throw muGrid::RuntimeError("This material needs two material "
                             "(shared)pointers making the layers of "
                             "lamiante, "
                             "their volume fraction, "
                             "and normal vector for adding pixel");
  }
  /* ----------------------------------------------------------------------
   */
  template <Dim_t DimM>
  void MaterialLaminate<DimM>::add_pixel(
      const size_t & pixel_id, MatPtr_t mat1, MatPtr_t mat2, const Real & ratio,
      const Eigen::Ref<const Eigen::Matrix<Real, Eigen::Dynamic, 1>> &
          normal_vector) {
    this->internal_fields.add_pixel(pixel_id);

    this->material_left_vector.push_back(mat1);
    this->material_right_vector.push_back(mat2);

    this->volume_ratio_field.get_field().push_back(ratio);
    this->normal_vector_field.get_field().push_back(normal_vector);
  }
  /* ----------------------------------------------------------------------
   */
  template <Dim_t DimM>
  void MaterialLaminate<DimM>::add_pixels_precipitate(
      const std::vector<Ccoord_t<DimM>> & intersected_pixels,
      const std::vector<Dim_t> & intersected_pixels_id,
      const std::vector<Real> & intersection_ratios,
      const std::vector<Eigen::Matrix<Real, DimM, 1>> & intersection_normals,
      MatPtr_t mat1, MatPtr_t mat2) {
    for (auto && tup : akantu::zip(intersected_pixels, intersected_pixels_id,
                                   intersection_ratios, intersection_normals)) {
      // auto pix { std::get<0>(tup)};
      auto pix_id{std::get<1>(tup)};
      auto ratio{std::get<2>(tup)};
      auto normal{std::get<3>(tup)};
      this->add_pixel(pix_id, mat1, mat2, ratio, normal);
    }
  }

  /* ----------------------------------------------------------------------*/
  template <Dim_t DimM>
  MaterialLaminate<DimM> &
  MaterialLaminate<DimM>::make(Cell & cell, const std::string & name) {
    auto mat{std::make_unique<MaterialLaminate<DimM>>(
        name, cell.get_spatial_dim(), cell.get_nb_quad_pts())};
    auto & mat_ref{*mat};
    auto is_cell_split{cell.get_splitness()};
    mat_ref.allocate_optional_fields(is_cell_split);
    cell.add_material(std::move(mat));
    return mat_ref;
  }
  /* ----------------------------------------------------------------------
   */
  template <Dim_t DimM>
  void MaterialLaminate<DimM>::compute_stresses(
      const RealField & F, RealField & P, const Formulation & form,
      const SplitCell & is_cell_split,
      const StoreNativeStress & store_native_stress) {
    if (store_native_stress != StoreNativeStress::no) {
      throw muGrid::RuntimeError(
          "native stress is not defined for laminate materials");
    }
    switch (form) {
    case Formulation::finite_strain: {
      switch (is_cell_split) {
      case (SplitCell::no):
      case (SplitCell::laminate): {
        this->compute_stresses_worker<Formulation::finite_strain,
                                      SplitCell::no>(F, P);
        break;
      }
      case (SplitCell::simple): {
        this->compute_stresses_worker<Formulation::finite_strain,
                                      SplitCell::simple>(F, P);
        break;
      }
      default:
        throw muGrid::RuntimeError("Unknown Splitness status");
      }
      break;
    }
    case Formulation::small_strain: {
      switch (is_cell_split) {
      case (SplitCell::no):
      case (SplitCell::laminate): {
        this->compute_stresses_worker<Formulation::small_strain, SplitCell::no>(
            F, P);
        break;
      }
      case (SplitCell::simple): {
        this->compute_stresses_worker<Formulation::small_strain,
                                      SplitCell::simple>(F, P);
        break;
      }
      default:
        throw muGrid::RuntimeError("Unknown Splitness status");
      }
      break;
    }
    default:
      throw muGrid::RuntimeError("Unknown formulation");
      break;
    }
  }

  /* ----------------------------------------------------------------------
   */
  template <Dim_t DimM>
  void MaterialLaminate<DimM>::compute_stresses_tangent(
      const RealField & F, RealField & P, RealField & K,
      const Formulation & form, const SplitCell & is_cell_split,
      const StoreNativeStress & store_native_stress) {
    if (store_native_stress != StoreNativeStress::no) {
      throw muGrid::RuntimeError(
          "native stress is not defined for laminate materials");
    }
    switch (form) {
    case Formulation::finite_strain: {
      switch (is_cell_split) {
      case (SplitCell::no):
      case (SplitCell::laminate): {
        this->compute_stresses_worker<Formulation::finite_strain,
                                      SplitCell::no>(F, P, K);
        break;
      }
      case (SplitCell::simple): {
        this->compute_stresses_worker<Formulation::finite_strain,
                                      SplitCell::simple>(F, P, K);
        break;
      }
      default:
        throw muGrid::RuntimeError("Unknown Splitness status");
      }
      break;
    }
    case Formulation::small_strain: {
      switch (is_cell_split) {
      case (SplitCell::no):
      case (SplitCell::laminate): {
        this->compute_stresses_worker<Formulation::small_strain, SplitCell::no>(
            F, P, K);
        break;
      }
      case (SplitCell::simple): {
        this->compute_stresses_worker<Formulation::small_strain,
                                      SplitCell::simple>(F, P, K);
        break;
      }
      default:
        throw muGrid::RuntimeError("Unknown Splitness status");
      }
      break;
    }
    default:
      throw muGrid::RuntimeError("Unknown formulation");
      break;
    }
  }
  /* ----------------------------------------------------------------------*/
  template class MaterialLaminate<twoD>;
  template class MaterialLaminate<threeD>;

}  // namespace muSpectre

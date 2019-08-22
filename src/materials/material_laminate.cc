/**
 * @file   material_laminate.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   14 Jun 2019
 *
 * @brief  Implementation of MaterialLamiante which is a lamiante approximation
 * constitutive law for two underlting materials with arbitrary constutive law
 *
 * Copyright © 2017 Till Jungex
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

#include "material_laminate.hh"

namespace muSpectre {
  template <Dim_t DimS, Dim_t DimM>
  MaterialLaminate<DimS, DimM>::MaterialLaminate(std::string name)
      : Parent(name), normal_vector_field{muGrid::make_field<VectorField_t>(
                          "Normal Vector", this->internal_fields)},
        normal_vector_map{normal_vector_field.get_map()},
        volume_ratio_field{muGrid::make_field<ScalarField_t>(
            "Volume Ratio", this->internal_fields)},
        volume_ratio_map{volume_ratio_field.get_map()} {}
  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void
  MaterialLaminate<DimS, DimM>::add_pixel(const Ccoord_t<DimS> & /*pixel*/) {
    throw std::runtime_error("This material needs two material "
                             "(shared)pointers making the layers of lamiante, "
                             "their volume fraction, "
                             "and normal vector for adding pixel");
  }
  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void MaterialLaminate<DimS, DimM>::add_pixel(
      const Ccoord_t<DimS> & pixel, MatPtr_t mat1, MatPtr_t mat2, Real ratio,
      const Eigen::Ref<const Eigen::Matrix<Real, DimM, 1>> & normal_vector) {
    this->internal_fields.add_pixel(pixel);

    this->material_left_vector.push_back(mat1);
    this->material_right_vector.push_back(mat2);

    this->volume_ratio_field.push_back(ratio);
    this->normal_vector_field.push_back(normal_vector);
  }
  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void MaterialLaminate<DimS, DimM>::add_pixels_precipitate(
      std::vector<Ccoord_t<DimS>> intersected_pixels,
      std::vector<Real> intersection_ratios,
      std::vector<Eigen::Matrix<Real, DimM, 1>> intersection_normals,
      MatPtr_t mat1, MatPtr_t mat2) {
    for (auto && tup : akantu::zip(intersected_pixels, intersection_ratios,
                                   intersection_normals)) {
      auto pix = std::get<0>(tup);
      auto ratio = std::get<1>(tup);
      auto normal = std::get<2>(tup);
      this->add_pixel(pix, mat1, mat2, ratio, normal);
    }
  }

  /* ----------------------------------------------------------------------*/
  template <Dim_t DimS, Dim_t DimM>
  MaterialLaminate<DimS, DimM> &
  MaterialLaminate<DimS, DimM>::make(CellBase<DimS, DimM> & cell,
                                     std::string name) {
    auto mat = std::make_unique<MaterialLaminate<DimS, DimM>>(name);
    auto & mat_ref = *mat;
    auto is_cell_split{cell.get_splitness()};
    mat_ref.allocate_optional_fields(is_cell_split);
    cell.add_material(std::move(mat));
    return mat_ref;
  }
  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void MaterialLaminate<DimS, DimM>::compute_stresses(const StrainField_t & F,
                                                      StressField_t & P,
                                                      Formulation form,
                                                      SplitCell is_cell_split) {
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
        throw std::runtime_error("Unknown Splitness status");
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
        throw std::runtime_error("Unknown Splitness status");
      }
      break;
    }
    default:
      throw std::runtime_error("Unknown formulation");
      break;
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  void MaterialLaminate<DimS, DimM>::compute_stresses_tangent(
      const StrainField_t & F, StressField_t & P, TangentField_t & K,
      Formulation form, SplitCell is_cell_split) {
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
        throw std::runtime_error("Unknown Splitness status");
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
        throw std::runtime_error("Unknown Splitness status");
      }
      break;
    }
    default:
      throw std::runtime_error("Unknown formulation");
      break;
    }
  }
  /* ----------------------------------------------------------------------*/
  template class MaterialLaminate<twoD, twoD>;
  template class MaterialLaminate<threeD, threeD>;

}  // namespace muSpectre

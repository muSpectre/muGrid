/**
 * @file   material_laminate.cc
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   18 May 2020
 *
 * @brief  implementation of MaterialLaminate class
 *
 * Copyright © 2020 Ali Falsafi
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

#include "materials/material_laminate.hh"
#include "materials/materials_toolbox.hh"
#include "materials/material_evaluator.hh"
#include "materials/laminate_homogenisation.hh"
#include "materials/stress_transformations_PK2.hh"
#include "materials/stress_transformations_PK1.hh"

#include "common/intersection_octree.hh"

namespace muSpectre {
  template <Index_t DimM, Formulation Form>
  MaterialLaminate<DimM, Form>::MaterialLaminate(
      const std::string & name, const Index_t & spatial_dimension,
      const Index_t & nb_quad_pts,
      std::shared_ptr<muGrid::LocalFieldCollection> parent_field)
      : Parent(name, spatial_dimension, nb_quad_pts, parent_field),
        normal_vector_field{this->get_prefix() + "normal vector",
                            *this->internal_fields, QuadPtTag},
        volume_ratio_field{this->get_prefix() + "volume ratio",
                           *this->internal_fields, QuadPtTag} {}

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM, Formulation Form>
  void MaterialLaminate<DimM, Form>::add_pixel(
      const size_t & /*pixel_id*/) {
    throw muGrid::RuntimeError("This material needs two material "
                               "(shared) pointers for making the layers of "
                               "a laminate pixel, in addition to  their volume"
                               " fraction, and normal vector at their interface"
                               " for adding pixel");
  }

  /* ---------------------------------------------------------------------*/
  template <Index_t DimM, Formulation Form>
  void MaterialLaminate<DimM, Form>::add_pixel(
      const size_t & pixel_id, MatPtr_t mat1, MatPtr_t mat2, const Real & ratio,
      const Eigen::Ref<const Eigen::Matrix<Real, DimM, 1>> & normal_vector) {
    this->internal_fields->add_pixel(pixel_id);

    this->material_left_vector.push_back(mat1);
    this->material_right_vector.push_back(mat2);

    this->volume_ratio_field.get_field().push_back(ratio);
    this->normal_vector_field.get_field().push_back(normal_vector);
  }

  /* --------------------------------------------------------------------*/
  template <Index_t DimM, Formulation Form>
  void MaterialLaminate<DimM, Form>::add_pixels_precipitate(
      const std::vector<Ccoord_t<DimM>> & intersected_pixels,
      const std::vector<Index_t> & intersected_pixels_id,
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
  template <Index_t DimM, Formulation Form>
  template <class Strain>
  auto MaterialLaminate<DimM, Form>::evaluate_stress(
      const Eigen::MatrixBase<Strain> & E, const size_t & pixel_index) -> T2_t {
    using Output_t = std::tuple<T2_t, T4_t>;
    using Function_t = std::function<Output_t(const Eigen::Ref<const T2_t> &)>;
    auto && mat_l{material_left_vector[pixel_index]};
    auto && mat_r{material_right_vector[pixel_index]};

    T2_t E_eval(E);

    const Function_t mat_l_evaluate_stress_tangent_func{
        [&mat_l, &pixel_index](const Eigen::Ref<const T2_t> & E) {
          return mat_l->constitutive_law_dynamic(std::move(E), pixel_index,
                                                 Form);
        }};

    const Function_t mat_r_evaluate_stress_tangent_func{
        [&mat_r, &pixel_index](const Eigen::Ref<const T2_t> & E) {
          return mat_r->constitutive_law_dynamic(std::move(E), pixel_index,
                                                 Form);
        }};

    auto && ratio{this->volume_ratio_field[pixel_index]};
    auto && normal_vec{this->normal_vector_field[pixel_index]};

    return LamHomogen<DimM, Form>::evaluate_stress(
        E_eval, mat_l_evaluate_stress_tangent_func,
        mat_r_evaluate_stress_tangent_func, ratio, normal_vec);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM, Formulation Form>
  template <class Strain>
  auto MaterialLaminate<DimM, Form>::evaluate_stress_tangent(
      const Eigen ::MatrixBase<Strain> & E, const size_t & pixel_index)
      -> std::tuple<T2_t, T4_t> {
    using Output_t = std::tuple<T2_t, T4_t>;
    using Function_t = std::function<Output_t(const Eigen::Ref<const T2_t> &)>;
    auto && mat_l{material_left_vector[pixel_index]};
    auto && mat_r{material_right_vector[pixel_index]};
    T2_t E_eval(E);

    Function_t mat_l_evaluate_stress_tangent_func{
        [&mat_l, &pixel_index](const Eigen::Ref<const T2_t> & E) {
          return mat_l->constitutive_law_dynamic(std::move(E), pixel_index,
                                                 Form);
        }};

    Function_t mat_r_evaluate_stress_tangent_func{
        [&mat_r, &pixel_index](const Eigen::Ref<const T2_t> & E) {
          return mat_r->constitutive_law_dynamic(std::move(E), pixel_index,
                                                 Form);
        }};

    std::tuple<T2_t, T4_t> ret_stress_stiffness{};
    auto && ratio{this->volume_ratio_field[pixel_index]};
    auto && normal_vec{this->normal_vector_field[pixel_index]};

    return LamHomogen<DimM, Form>::evaluate_stress_tangent(
        E_eval, mat_l_evaluate_stress_tangent_func,
        mat_r_evaluate_stress_tangent_func, ratio, normal_vec);
  }

  /* ----------------------------------------------------------------------*/
  template class MaterialLaminate<twoD, Formulation::finite_strain>;
  template class MaterialLaminate<threeD, Formulation::finite_strain>;
  template class MaterialLaminate<twoD, Formulation::small_strain>;
  template class MaterialLaminate<threeD, Formulation::small_strain>;

}  // namespace muSpectre

/**
 * @file   material_linear_elastic2.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   03 Feb 2018
 *
 * @brief linear elastic material with imposed eigenstrain and its
 *        type traits. Uses the MaterialMuSpectre facilities to keep it
 *        simple
 *
 * Copyright © 2018 Till Junge
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

#ifndef SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC2_HH_
#define SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC2_HH_

#include "materials/material_linear_elastic1.hh"

#include <libmugrid/mapped_field.hh>

#include <Eigen/Dense>

namespace muSpectre {

  template <Index_t DimM>
  class MaterialLinearElastic2;

  /**
   * traits for objective linear elasticity with per pixel strength
   */
  template <Index_t DimM>
  struct MaterialMuSpectre_traits<MaterialLinearElastic2<DimM>> {
    //! expected map type for strain fields
    using StrainMap_t =
        muGrid::T2FieldMap<double, Mapping::Const, DimM, IterUnit::SubPt>;
    //! expected map type for stress fields
    using StressMap_t =
        muGrid::T2FieldMap<double, Mapping::Mut, DimM, IterUnit::SubPt>;
    //! expected map type for tangent stiffness fields
    using TangentMap_t =
        muGrid::T4FieldMap<double, Mapping::Mut, DimM, IterUnit::SubPt>;

    //! declare what type of strain measure your law takes as input
    constexpr static auto strain_measure{StrainMeasure::GreenLagrange};
    //! declare what type of stress measure your law yields as output
    constexpr static auto stress_measure{StressMeasure::PK2};
  };

  /**
   * implements objective linear elasticity with damage an  per pixel strength
   */
  template <Index_t DimM>
  class MaterialLinearElastic2
      : public MaterialMuSpectre<MaterialLinearElastic2<DimM>, DimM> {
   public:
    //! base class
    using Parent = MaterialMuSpectre<MaterialLinearElastic2, DimM>;

    //! traits of this material
    using traits = MaterialMuSpectre_traits<MaterialLinearElastic2>;

    //! reference to any type that casts to a matrix
    using StrainTensor = Eigen::Ref<const Eigen::Matrix<Real, DimM, DimM>>;

    //! Default constructor
    MaterialLinearElastic2() = delete;

    //! Construct by name, Young's modulus and Poisson's ratio
    MaterialLinearElastic2(const std::string & name,
                           const Index_t & spatial_dimension,
                           const Index_t & nb_quad_pts, Real young,
                           Real poisson);

    //! Copy constructor
    MaterialLinearElastic2(const MaterialLinearElastic2 & other) = delete;

    //! Move constructor
    MaterialLinearElastic2(MaterialLinearElastic2 && other) = delete;

    //! Destructor
    virtual ~MaterialLinearElastic2() = default;

    //! Copy assignment operator
    MaterialLinearElastic2 &
    operator=(const MaterialLinearElastic2 & other) = delete;

    //! Move assignment operator
    MaterialLinearElastic2 &
    operator=(MaterialLinearElastic2 && other) = delete;

    /**
     * evaluates second Piola-Kirchhoff stress given the Green-Lagrange
     * strain (or Cauchy stress if called with a small strain tensor)
     */
    template <class s_t>
    inline decltype(auto) evaluate_stress(s_t && E,
                                          const size_t & quad_pt_index);

    /**
     * evaluates both second Piola-Kirchhoff stress and stiffness given
     * the Green-Lagrange strain (or Cauchy stress and stiffness if
     * called with a small strain tensor)
     */
    template <class s_t>
    inline decltype(auto) evaluate_stress_tangent(s_t && E,
                                                  const size_t & quad_pt_index);

    /**
     * overload add_pixel to write into eigenstrain
     */
    void add_pixel(const size_t & pixel_index) final;

    /**
     * overload add_pixel to write into eigenstrain
     */
    void add_pixel(const size_t & pixel_index, const StrainTensor & E_eig);

   protected:
    //! linear material without eigenstrain used to compute response
    MaterialLinearElastic1<DimM> material;
    //! storage for eigenstrain
    muGrid::MappedT2Field<Real, Mapping::Const, DimM, IterUnit::SubPt>
        eigen_strains;
  };

  /* ----------------------------------------------------------------------*/
  template <Index_t DimM>
  template <class s_t>
  auto MaterialLinearElastic2<DimM>::evaluate_stress(
      s_t && E, const size_t & quad_pt_index) -> decltype(auto) {
    auto && E_eig = this->eigen_strains[quad_pt_index];
    return this->material.evaluate_stress(E - E_eig, quad_pt_index);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class s_t>
  auto MaterialLinearElastic2<DimM>::evaluate_stress_tangent(
      s_t && E, const size_t & quad_pt_index) -> decltype(auto) {
    auto && E_eig{this->eigen_strains[quad_pt_index]};
    return this->material.evaluate_stress_tangent(E - E_eig, quad_pt_index);
  }

}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC2_HH_

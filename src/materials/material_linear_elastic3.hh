/**
 * @file   material_linear_elastic3.hh
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *
 * @date   20 Feb 2018
 *
 * @brief linear elastic material with distribution of stiffness properties.
 *        Uses the MaterialMuSpectre facilities to keep it simple.
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

#ifndef SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC3_HH_
#define SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC3_HH_

#include "materials/material_linear_elastic1.hh"

#include <libmugrid/mapped_field.hh>

#include <Eigen/Dense>

namespace muSpectre {

  template <Index_t DimM>
  class MaterialLinearElastic3;

  /**
   * traits for objective linear elasticity with eigenstrain
   */
  template <Index_t DimM>
  struct MaterialMuSpectre_traits<MaterialLinearElastic3<DimM>> {
    //! expected map type for strain fields
    using StrainMap_t =
        muGrid::T2FieldMap<Real, Mapping::Const, DimM, IterUnit::SubPt>;
    //! expected map type for stress fields
    using StressMap_t =
        muGrid::T2FieldMap<Real, Mapping::Mut, DimM, IterUnit::SubPt>;

    //! expected map type for tangent stiffness fields
    using TangentMap_t =
        muGrid::T4FieldMap<Real, Mapping::Mut, DimM, IterUnit::SubPt>;

    //! declare what type of strain measure your law takes as input
    constexpr static auto strain_measure{StrainMeasure::GreenLagrange};
    //! declare what type of stress measure your law yields as output
    constexpr static auto stress_measure{StressMeasure::PK2};
  };

  /**
   * implements objective linear elasticity with an eigenstrain per pixel
   */
  template <Index_t DimM>
  class MaterialLinearElastic3
      : public MaterialMuSpectre<MaterialLinearElastic3<DimM>, DimM> {
   public:
    //! base class
    using Parent = MaterialMuSpectre<MaterialLinearElastic3, DimM>;
    /**
     * type used to determine whether the
     * `muSpectre::MaterialMuSpectre::iterable_proxy` evaluate only
     * stresses or also tangent stiffnesses
     */
    using NeedTangent = typename Parent::NeedTangent;
    //! global field collection

    //! traits of this material
    using traits = MaterialMuSpectre_traits<MaterialLinearElastic3>;

    //! Hooke's law implementation
    using Hooke =
        typename MatTB::Hooke<DimM, typename traits::StrainMap_t::reference,
                              typename traits::TangentMap_t::reference>;

    //! short hand for storage type of elastic tensors
    using StiffnessField_t =
        muGrid::MappedT4Field<Real, Mapping::Const, DimM, IterUnit::SubPt>;

    //! Default constructor
    MaterialLinearElastic3() = delete;

    //! Construct by name
    MaterialLinearElastic3(const std::string & name,
                           const Index_t & spatial_dimension,
                           const Index_t & nb_quad_pts);

    //! Copy constructor
    MaterialLinearElastic3(const MaterialLinearElastic3 & other) = delete;

    //! Move constructor
    MaterialLinearElastic3(MaterialLinearElastic3 && other) = delete;

    //! Destructor
    virtual ~MaterialLinearElastic3() = default;

    //! Copy assignment operator
    MaterialLinearElastic3 &
    operator=(const MaterialLinearElastic3 & other) = delete;

    //! Move assignment operator
    MaterialLinearElastic3 &
    operator=(MaterialLinearElastic3 && other) = delete;

    /**
     * evaluates second Piola-Kirchhoff stress given the Green-Lagrange
     * strain (or Cauchy stress if called with a small strain tensor)
     * and the local stiffness tensor.
     */
    template <class Derived>
    inline decltype(auto)
    evaluate_stress(const Eigen::MatrixBase<Derived> & E,
                    const typename StiffnessField_t::Return_t & C);

    /**
     * evaluates second Piola-Kirchhoff stress given the Green-Lagrange
     * strain (or Cauchy stress if called with a small strain tensor)
     * and the local pixel id.
     */
    template <class Derived>
    inline decltype(auto) evaluate_stress(const Eigen::MatrixBase<Derived> & E,
                                          const size_t & quad_pt_index) {
      auto && C{this->C_field[quad_pt_index]};
      return this->evaluate_stress(E, C);
    }

    /**
     * evaluates both second Piola-Kirchhoff stress and tangent moduli given
     * the Green-Lagrange strain (or Cauchy stress and tangent moduli if
     * called with a small strain tensor) and the local tangent moduli tensor.
     */
    template <class Derived>
    inline decltype(auto)
    evaluate_stress_tangent(const Eigen::MatrixBase<Derived> & E,
                            const typename StiffnessField_t::Return_t & C);

    /**
     * evaluates both second Piola-Kirchhoff stress and tangent moduli given
     * the Green-Lagrange strain (or Cauchy stress and tangent moduli if
     * called with a small strain tensor) and the local pixel id.
     */
    template <class Derived>
    inline decltype(auto)
    evaluate_stress_tangent(const Eigen::MatrixBase<Derived> & E,
                            const size_t & quad_pt_index) {
      auto && C{this->C_field[quad_pt_index]};
      return this->evaluate_stress_tangent(E, C);
    }

    /**
     * overload add_pixel to write into loacal stiffness tensor
     */
    void add_pixel(const size_t & pixel_index) final;

    /**
     * overload add_pixel to write into local stiffness tensor
     */
    void add_pixel(const size_t & pixel_index, const Real & Young,
                   const Real & PoissonRatio);

   protected:
    //! storage for stiffness tensor
    StiffnessField_t C_field;
  };

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class Derived>
  auto MaterialLinearElastic3<DimM>::evaluate_stress(
      const Eigen::MatrixBase<Derived> & E,
      const typename StiffnessField_t::Return_t & C) -> decltype(auto) {
    return Matrices::tensmult(C, E);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class Derived>
  auto MaterialLinearElastic3<DimM>::evaluate_stress_tangent(
      const Eigen::MatrixBase<Derived> & E,
      const typename StiffnessField_t::Return_t & C) -> decltype(auto) {
    return std::make_tuple(evaluate_stress(E, C), C);
  }

}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC3_HH_

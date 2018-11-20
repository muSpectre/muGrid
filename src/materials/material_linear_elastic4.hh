/**
 * @file   material_linear_elastic4.hh
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *
 * @date   15 March 2018
 *
 * @brief linear elastic material with distribution of stiffness properties.
 *        In difference to material_linear_elastic3 two Lame constants are
 *        stored per pixel instead of the whole elastic matrix C.
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

#ifndef MATERIAL_LINEAR_ELASTIC_RANDOM_STIFFNESS_2_H
#define MATERIAL_LINEAR_ELASTIC_RANDOM_STIFFNESS_2_H

#include "materials/material_linear_elastic1.hh"
#include "common/field.hh"
#include "common/tensor_algebra.hh"

#include <Eigen/Dense>

namespace muSpectre {

  template <Dim_t DimS, Dim_t DimM>
  class MaterialLinearElastic4;

  /**
   * traits for objective linear elasticity with eigenstrain
   */
  template <Dim_t DimS, Dim_t DimM>
  struct MaterialMuSpectre_traits<MaterialLinearElastic4<DimS, DimM>>
  {
    //! global field collection
    using GFieldCollection_t = typename
      MaterialBase<DimS, DimM>::GFieldCollection_t;

    //! expected map type for strain fields
    using StrainMap_t = MatrixFieldMap<GFieldCollection_t, Real, DimM, DimM, true>;
    //! expected map type for stress fields
    using StressMap_t = MatrixFieldMap<GFieldCollection_t, Real, DimM, DimM>;
    //! expected map type for tangent stiffness fields
    using TangentMap_t = T4MatrixFieldMap<GFieldCollection_t, Real, DimM>;

    //! declare what type of strain measure your law takes as input
    constexpr static auto strain_measure{StrainMeasure::GreenLagrange};
    //! declare what type of stress measure your law yields as output
    constexpr static auto stress_measure{StressMeasure::PK2};

    //! local field_collections used for internals
    using LFieldColl_t = LocalFieldCollection<DimS>;
    //! local Lame constant type
    using LLameConstantMap_t = ScalarFieldMap<LFieldColl_t, Real, true>;
    //! elasticity without internal variables
    using InternalVariables = std::tuple<LLameConstantMap_t,
                                         LLameConstantMap_t>;

  };

  /**
   * implements objective linear elasticity with an eigenstrain per pixel
   */
  template <Dim_t DimS, Dim_t DimM>
  class MaterialLinearElastic4:
    public MaterialMuSpectre<MaterialLinearElastic4<DimS, DimM>, DimS, DimM>
  {
  public:

    //! base class
    using Parent = MaterialMuSpectre<MaterialLinearElastic4, DimS, DimM>;
    /**
     * type used to determine whether the
     * `muSpectre::MaterialMuSpectre::iterable_proxy` evaluate only
     * stresses or also tangent stiffnesses
     */
    using NeedTangent = typename Parent::NeedTangent;
    //! global field collection

    using Stiffness_t = Eigen::TensorFixedSize
      <Real, Eigen::Sizes<DimM, DimM, DimM, DimM>>;

    //! traits of this material
    using traits = MaterialMuSpectre_traits<MaterialLinearElastic4>;

    //! Type of container used for storing eigenstrain
    using InternalVariables = typename traits::InternalVariables;

    //! Hooke's law implementation
    using Hooke = typename
      MatTB::Hooke<DimM,
                   typename traits::StrainMap_t::reference,
                   typename traits::TangentMap_t::reference>;

    //! Default constructor
    MaterialLinearElastic4() = delete;

    //! Construct by name
    MaterialLinearElastic4(std::string name);


    //! Copy constructor
    MaterialLinearElastic4(const MaterialLinearElastic4 &other) = delete;

    //! Move constructor
    MaterialLinearElastic4(MaterialLinearElastic4 &&other) = delete;

    //! Destructor
    virtual ~MaterialLinearElastic4() = default;

    //! Copy assignment operator
    MaterialLinearElastic4& operator=(const MaterialLinearElastic4 &other) = delete;

    //! Move assignment operator
    MaterialLinearElastic4& operator=(MaterialLinearElastic4 &&other) = delete;

    /**
     * evaluates second Piola-Kirchhoff stress given the Green-Lagrange
     * strain (or Cauchy stress if called with a small strain tensor), the first
     * Lame constant (lambda) and the second Lame constant (shear modulus/mu).
     */
    template <class s_t>
    inline decltype(auto) evaluate_stress(s_t && E,
                                          const Real & lambda,
                                          const Real & mu);

    /**
     * evaluates both second Piola-Kirchhoff stress and stiffness given
     * the Green-Lagrange strain (or Cauchy stress and stiffness if
     * called with a small strain tensor), the first Lame constant (lambda) and
     * the second Lame constant (shear modulus/mu).
     */
    template <class s_t>
    inline decltype(auto) evaluate_stress_tangent(s_t &&  E,
                                                  const Real & lambda,
                                                  const Real & mu);

    /**
     * return the empty internals tuple
     */
    InternalVariables & get_internals() {
      return this->internal_variables;};

    /**
     * overload add_pixel to write into loacal stiffness tensor
     */
    void add_pixel(const Ccoord_t<DimS> & pixel) override final;

    /**
     * overload add_pixel to write into local stiffness tensor
     */
    void add_pixel(const Ccoord_t<DimS> & pixel,
                   const Real & Poisson_ratio, const Real & Youngs_modulus);

  protected:
    //! storage for first Lame constant 'lambda'
    //! and second Lame constant(shear modulus) 'mu'
    using Field_t = MatrixField<LocalFieldCollection<DimS>, Real, oneD, oneD>;
    Field_t & lambda_field;
    Field_t & mu_field;
    //! tuple for iterable eigen_field
    InternalVariables internal_variables;
  private:
  };

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  template <class s_t>
  auto
  MaterialLinearElastic4<DimS, DimM>::
  evaluate_stress(s_t && E, const Real & lambda, const Real & mu)
    -> decltype(auto) {
    auto C = Hooke::compute_C_T4(lambda, mu);
    return Matrices::tensmult(C, E);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  template <class s_t>
  auto
  MaterialLinearElastic4<DimS, DimM>::
  evaluate_stress_tangent(s_t && E, const Real & lambda,
                          const Real & mu) -> decltype(auto)
  {
    auto C = Hooke::compute_C_T4(lambda, mu);
    return std::make_tuple(Matrices::tensmult(C, E), C);
  }


}  // muSpectre

#endif /* MATERIAL_LINEAR_ELASTIC_RANDOM_STIFFNESS_2_H */

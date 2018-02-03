/**
 * @file   material_linear_elastic1.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   13 Nov 2017
 *
 * @brief  Implementation for linear elastic reference material like in de Geus
 *         2017. This follows the simplest and likely not most efficient
 *         implementation (with exception of the Python law)
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
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Emacs; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */


#ifndef MATERIAL_LINEAR_ELASTIC1_H
#define MATERIAL_LINEAR_ELASTIC1_H

#include "common/common.hh"
#include "materials/material_muSpectre_base.hh"

namespace muSpectre {
  template<Dim_t DimS, Dim_t DimM>
  class MaterialLinearElastic1;

  /**
   * traits for objective linear elasticity
   */
  template <Dim_t DimS, Dim_t DimM>
  struct MaterialMuSpectre_traits<MaterialLinearElastic1<DimS, DimM>>:
    public MaterialMuSpectre_traits<void> {
    using Parent = MaterialMuSpectre_traits<void>;//!< base for elasticity
    //! elasticity without internal variables
    using InternalVariables = typename Parent::DefaultInternalVariables;
  };

  //! DimS spatial dimension (dimension of problem
  //! DimM material_dimension (dimension of constitutive law)
  /**
   * implements objective linear elasticity
   */
  template<Dim_t DimS, Dim_t DimM>
  class MaterialLinearElastic1:
    public MaterialMuSpectre<MaterialLinearElastic1<DimS, DimM>, DimS, DimM>
  {
  public:
    //! base class
    using Parent = MaterialMuSpectre<MaterialLinearElastic1, DimS, DimM>;
    /**
     * type used to determine whether the
     * `muSpectre::MaterialMuSpectre::iterable_proxy` evaluate only
     * stresses or also tangent stiffnesses
     */
    using NeedTangent = typename Parent::NeedTangent;
    //! global field collection
    using GFieldCollection_t = typename Parent::GFieldCollection_t;
    //! declare what type of strain measure your law takes as input
    constexpr static auto strain_measure{StrainMeasure::GreenLagrange};
    //! declare what type of stress measure your law yields as output
    constexpr static auto stress_measure{StressMeasure::PK2};
    //! declare whether the derivative of stress with respect to strain is uniform
    constexpr static bool uniform_stiffness = true;

    //! expected map type for strain fields
    using StrainMap_t = MatrixFieldMap<GFieldCollection_t, Real, DimM, DimM, true>;
    //! expected map type for stress fields
    using StressMap_t = MatrixFieldMap<GFieldCollection_t, Real, DimM, DimM>;
    //! expected map type for tangent stiffness fields
    using TangentMap_t = T4MatrixFieldMap<GFieldCollection_t, Real, DimM>;
    //! expected type for strain values
    using Strain_t = typename StrainMap_t::const_reference;
    //! expected type for stressvalues
    using Stress_t = typename StressMap_t::reference;
    //! expected type for tangent stiffness values
    using Tangent_t = typename TangentMap_t::reference;
    //! type for the stiffness (actual value, not a map or reference)
    using Stiffness_t = Eigen::TensorFixedSize
      <Real, Eigen::Sizes<DimM, DimM, DimM, DimM>>;

    //! this law does not have any internal variables
    using InternalVariables = typename Parent::DefaultInternalVariables;

    //! Default constructor
    MaterialLinearElastic1() = delete;

    //! Copy constructor
    MaterialLinearElastic1(const MaterialLinearElastic1 &other) = delete;

    //! Construct by name, Young's modulus and Poisson's ratio
    MaterialLinearElastic1(std::string name, Real young, Real poisson);


    //! Move constructor
    MaterialLinearElastic1(MaterialLinearElastic1 &&other) = delete;

    //! Destructor
    virtual ~MaterialLinearElastic1() = default;

    //! Copy assignment operator
    MaterialLinearElastic1& operator=(const MaterialLinearElastic1 &other) = delete;

    //! Move assignment operator
    MaterialLinearElastic1& operator=(MaterialLinearElastic1 &&other) = delete;

    /**
     * evaluates second Piola-Kirchhoff stress given the Green-Lagrange
     * strain (or Cauchy stress if called with a small strain tensor)
     */
    template <class s_t>
    inline decltype(auto) evaluate_stress(s_t && E);

    /**
     * evaluates both second Piola-Kirchhoff stress and stiffness given
     * the Green-Lagrange strain (or Cauchy stress and stiffness if
     * called with a small strain tensor)
     */
    template <class s_t>
    inline decltype(auto) evaluate_stress_tangent(s_t &&  E);

  protected:
    const Real young;  //!< Young's modulus
    const Real poisson;//!< Poisson's ratio
    const Real lambda; //!< first Lamé constant
    const Real mu;     //!< second Lamé constant (shear modulus)
    const Stiffness_t C; //!< stiffness tensor
  private:
  };


  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  template <class s_t>
  decltype(auto)
  MaterialLinearElastic1<DimS, DimM>::evaluate_stress(s_t && E) {
    return E.trace()*lambda * Strain_t::Identity() + 2*mu*E;
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  template <class s_t>
  decltype(auto)
  MaterialLinearElastic1<DimS, DimM>::evaluate_stress_tangent(s_t && E) {
    return std::make_tuple(std::move(this->evaluate_stress(std::move(E))),
                           std::move(Tangent_t(const_cast<double*>(this->C.data()))));
  }

}  // muSpectre

#endif /* MATERIAL_LINEAR_ELASTIC1_H */

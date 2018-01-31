/**
 * file   material_linear_elastic1.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   13 Nov 2017
 *
 * @brief  Implementation for linear elastic reference material like in de Geus
 *         2017. This follows the simplest and likely not most efficient
 *         implementation (with exception of the Python law)
 *
 * @section LICENSE
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

  template <Dim_t DimS, Dim_t DimM>
  struct MaterialMuSpectre_traits<MaterialLinearElastic1<DimS, DimM>>:
    public MaterialMuSpectre_traits<void> {
    using Parent = MaterialMuSpectre_traits<void>;
    using InternalVariables = typename Parent::DefaultInternalVariables;
  };

  //! DimS spatial dimension (dimension of problem
  //! DimM material_dimension (dimension of constitutive law)
  template<Dim_t DimS, Dim_t DimM>
  class MaterialLinearElastic1:
    public MaterialMuSpectre<MaterialLinearElastic1<DimS, DimM>, DimS, DimM>
  {
  public:
    using Parent = MaterialMuSpectre<MaterialLinearElastic1, DimS, DimM>;
    using NeedTangent = typename Parent::NeedTangent;
    using GFieldCollection_t = typename Parent::GFieldCollection_t;
    // declare what type of strain measure your law takes as input
    constexpr static auto strain_measure{StrainMeasure::GreenLagrange};
    // declare what type of stress measure your law yields as output
    constexpr static auto stress_measure{StressMeasure::PK2};
    // declare whether the derivative of stress with respect to strain is uniform
    constexpr static bool uniform_stiffness = true;
    // declare the type in which you wish to receive your strain measure
    using StrainMap_t = MatrixFieldMap<GFieldCollection_t, Real, DimM, DimM, true>;
    using StressMap_t = MatrixFieldMap<GFieldCollection_t, Real, DimM, DimM>;
    using TangentMap_t = T4MatrixFieldMap<GFieldCollection_t, Real, DimM>;
    using Strain_t = typename StrainMap_t::const_reference;
    using Stress_t = typename StressMap_t::reference;
    using Tangent_t = typename TangentMap_t::reference;
    using Stiffness_t = Eigen::TensorFixedSize
      <Real, Eigen::Sizes<DimM, DimM, DimM, DimM>>;

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

    template <class s_t>
    inline decltype(auto) evaluate_stress(s_t && E);
    template <class s_t>
    inline decltype(auto) evaluate_stress_tangent(s_t &&  E);

    const Tangent_t & get_stiffness() const;

  protected:
    const Real young, poisson, lambda, mu;
    const Stiffness_t C;
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

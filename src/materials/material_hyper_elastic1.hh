/**
 * file   material_hyper_elastic1.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   13 Nov 2017
 *
 * @brief  Implementation for hyperelastic reference material like in de Geus
 *         2017. This follows the simplest and likely not most efficient
 *         implementation (with exception of the Python law)
 *
 * @section LICENCE
 *
 * Copyright (C) 2017 Till Junge
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


#include "materials/material_muSpectre_base.hh"

#ifndef MATERIAL_HYPER_ELASTIC1_H
#define MATERIAL_HYPER_ELASTIC1_H

namespace muSpectre {

  //! DimS spatial dimension (dimension of problem
  //! DimM material_dimension (dimension of constitutive law)
  template<Dim_t DimS, Dim_t DimM>
  class MaterialHyperElastic1: public MaterialMuSpectre<MaterialHyperElastic1<DimS, DimM>, DimS, DimM>
  {
  public:
    using Parent = MaterialMuSpectre<MaterialHyperElastic1, DimS, DimM>;
    using NeedTangent = typename Parent::NeedTangent;
    using GFieldCollection_t = typename Parent::GFieldCollection_t;
    // declare what type of strain measure your law takes as input
    constexpr static auto strain_measure{StrainMeasure::GreenLagrange};
    // declare what type of stress measure your law yields as output
    constexpr static auto stress_measure{StressMeasure::PK2};
    // declare whether the derivative of stress with respect to strain is uniform
    constexpr static bool uniform_stiffness = true;
    // declare the type in which you wish to receive your strain measure
    using StrainMap_t = MatrixFieldMap<GFieldCollection_t, Real, DimM, DimM>;
    using StressMap_t = StrainMap_t;
    using TangentMap_t = T4MatrixFieldMap<GFieldCollection_t, Real, DimM>;
    using Strain_t = typename StrainMap_t::const_reference;
    using Stress_t = typename StressMap_t::reference;
    using Tangent_t = typename TangentMap_t::reference::ConstType;
    using Stiffness_t = Eigen::TensorFixedSize
      <Real, Eigen::Sizes<DimM, DimM, DimM, DimM>>;

    using InternalVariables = typename Parent::DefaultInternalVariables;

    //! Default constructor
    MaterialHyperElastic1() = delete;

    //! Copy constructor
    MaterialHyperElastic1(const MaterialHyperElastic1 &other) = delete;

    //! Construct by name, Young's modulus and Poisson's ratio
    MaterialHyperElastic1(std::string name, Real young, Real poisson);


    //! Move constructor
    MaterialHyperElastic1(MaterialHyperElastic1 &&other) noexcept = delete;

    //! Destructor
    virtual ~MaterialHyperElastic1() noexcept = default;

    //! Copy assignment operator
    MaterialHyperElastic1& operator=(const MaterialHyperElastic1 &other) = delete;

    //! Move assignment operator
    MaterialHyperElastic1& operator=(MaterialHyperElastic1 &&other) noexcept = delete;

    template <class s_t>
    decltype(auto) evaluate_stress(s_t && E);
    template <class s_t>
    decltype(auto) evaluate_stress_tangent(s_t &&  E);

    const Tangent_t & get_stiffness() const;

  protected:
    const Real young, poisson, lambda, mu;
    const Stiffness_t C;
  private:
  };

}  // muSpectre

#endif /* MATERIAL_HYPER_ELASTIC1_H */

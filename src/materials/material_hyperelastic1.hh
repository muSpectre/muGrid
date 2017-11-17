/**
 * file   material_hyperelastic1.hh
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

#ifndef MATERIAL_HYPERELASTIC1_H
#define MATERIAL_HYPERELASTIC1_H

namespace muSpectre {

  //! DimS spatial dimension (dimension of problem
  //! DimM material_dimension (dimension of constitutive law)
  template<Dim_t DimS, Dim_t DimM>
  class MaterialHyperElastic1: public MaterialMuSpectre<MaterialHyperElastic1>
  {
  public:
    using Parent = public MaterialMuSpectre<MaterialHyperElastic1>;
    // declare what type of strain measure your law takes as input
    constexpr static auto strain_measure{MatTB::StrainMeasure::GreenLagrange};
    // declare what type of stress measure your law yields as output
    constexpr static auto strain_measure{MatTB::StressMeasure::PK2};
    // declare whether the derivative of stress with respect to strain is uniform
    constexpr static bool uniform_stiffness = true;
    // declare the type in which you wish to receive your strain measure
    using Strain_t = Eigen::Matrix<Real, DimM, DimM>;
    using Stress_t = Strain_t;
    using Stiffness_t = Eigen::TensorFixedSize
      <Real, Eigen::Sizes<DimM, DimM, DimM, Dim>, Eigen::RowMajor>;

    //! Default constructor
    MaterialHyperElastic1() = delete;

    //! Copy constructor
    MaterialHyperElastic1(const MaterialHyperElastic1 &other) = delete;

    //! Construct by name, Young's modulus and Poisson's ratio
    MaterialMuSpectre(std::string name, Real young, real Poisson);


    //! Move constructor
    MaterialHyperElastic1(MaterialHyperElastic1 &&other) noexcept = delete;

    //! Destructor
    virtual ~MaterialHyperElastic1() noexcept = default;

    //! Copy assignment operator
    MaterialHyperElastic1& operator=(const MaterialHyperElastic1 &other) = delete;

    //! Move assignment operator
    MaterialHyperElastic1& operator=(MaterialHyperElastic1 &&other) noexcept = delete;

    decltype(auto) evaluate_stress(const Strain_t & E);

    const Stiffness_t & get_stiffness() const;


  protected:
    const Real young, poisson, lambda, mu;
    const Stiffness_t C;
  private:
  };

}  // muSpectre

#endif /* MATERIAL_HYPERELASTIC1_H */

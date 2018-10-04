/**
 * @file   material_linear_elastic_generic.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   21 Sep 2018
 *
 * @brief Implementation fo a generic linear elastic material that
 *        stores the full elastic stiffness tensor. Convenient but not the
 *        most efficient
 *
 * Copyright © 2018 Till Junge
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

#ifndef MATERIAL_LINEAR_ELASTIC_GENERIC_H
#define MATERIAL_LINEAR_ELASTIC_GENERIC_H

#include "common/common.hh"
#include "common/T4_map_proxy.hh"
#include "materials/material_muSpectre_base.hh"
#include "common/tensor_algebra.hh"

namespace muSpectre {

  /**
   * forward declaration
   */
  template <Dim_t DimS, Dim_t DimM>
  class MaterialLinearElasticGeneric;

  /**
   * traits for use by MaterialMuSpectre for crtp
   */
  template <Dim_t DimS, Dim_t DimM>
  struct MaterialMuSpectre_traits<MaterialLinearElasticGeneric<DimS, DimM>> {

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

    //! elasticity without internal variables
    using InternalVariables = std::tuple<>;
  };


  /**
   * Linear elastic law defined by a full stiffness tensor. Very
   * generic, but not most efficient
   */
  template <Dim_t DimS, Dim_t DimM>
  class MaterialLinearElasticGeneric:
    public MaterialMuSpectre<
    MaterialLinearElasticGeneric<DimS, DimM>, DimS, DimM>
  {
  public:
    //! parent type
    using Parent = MaterialMuSpectre<
    MaterialLinearElasticGeneric<DimS, DimM>, DimS, DimM>;
    //! generic input tolerant to python input
    using CInput_t = Eigen::Ref
      <Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>, 0,
       Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
    //! Default constructor
    MaterialLinearElasticGeneric() = delete;

    /**
     * Constructor by name and stiffness tensor.
     *
     * @param name unique material name
     * @param C_voigt elastic tensor in Voigt notation
     */
    MaterialLinearElasticGeneric(const std::string & name,
                                 const CInput_t& C_voigt);

    //! Copy constructor
    MaterialLinearElasticGeneric(const MaterialLinearElasticGeneric &other) = delete;

    //! Move constructor
    MaterialLinearElasticGeneric(MaterialLinearElasticGeneric &&other) = delete;

    //! Destructor
    virtual ~MaterialLinearElasticGeneric() = default;

    //! Copy assignment operator
    MaterialLinearElasticGeneric&
    operator=(const MaterialLinearElasticGeneric &other) = delete;

    //! Move assignment operator
    MaterialLinearElasticGeneric&
    operator=(MaterialLinearElasticGeneric &&other) = delete;

    //! see http://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        /**
     * evaluates second Piola-Kirchhoff stress given the Green-Lagrange
     * strain (or Cauchy stress if called with a small strain tensor)
     */
    template <class Derived>
    inline decltype(auto)
    evaluate_stress(const Eigen::MatrixBase<Derived> & E);

    /**
     * evaluates both second Piola-Kirchhoff stress and stiffness given
     * the Green-Lagrange strain (or Cauchy stress and stiffness if
     * called with a small strain tensor)
     */
    template <class s_t>
    inline decltype(auto) evaluate_stress_tangent(s_t &&  E);

    /**
     * return the empty internals tuple
     */
    std::tuple<> & get_internals() {
      return this->internal_variables;};

    /**
     * return a reference to teh stiffness tensor
     */
    const T4Mat<Real, DimM>& get_C() const {return this->C;}

  protected:
    T4Mat<Real, DimM> C{};
    //! empty tuple
    std::tuple<> internal_variables{};
  private:
  };

   /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  template <class Derived>
  auto
  MaterialLinearElasticGeneric<DimS, DimM>::
  evaluate_stress(const Eigen::MatrixBase<Derived> & E)
    -> decltype(auto) {
    static_assert(Derived::ColsAtCompileTime == DimM,
                  "wrong input size");
    static_assert(Derived::RowsAtCompileTime == DimM,
                  "wrong input size");
    return Matrices::tensmult(this->C, E);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  template <class s_t>
  auto
  MaterialLinearElasticGeneric<DimS, DimM>::evaluate_stress_tangent(s_t && E)
    -> decltype(auto) {
    using Stress_t = decltype(this->evaluate_stress(std::forward<s_t>(E)));
    using Stiffness_t = Eigen::Map<T4Mat<Real, DimM>>;
    using Ret_t = std::tuple<Stress_t, Stiffness_t>;
    return Ret_t{this->evaluate_stress(std::forward<s_t>(E)),
        Stiffness_t(this->C.data())};
  }
}  // muSpectre

#endif /* MATERIAL_LINEAR_ELASTIC_GENERIC_H */

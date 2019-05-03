/**
 * @file   material_linear_elastic_generic1.hh
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

#ifndef SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC_GENERIC1_HH_
#define SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC_GENERIC1_HH_

#include "common/muSpectre_common.hh"
#include "materials/stress_transformations_PK2.hh"
#include "materials/material_muSpectre_base.hh"

#include <libmugrid/T4_map_proxy.hh>

namespace muSpectre {

  /**
   * forward declaration
   */
  template <Dim_t DimS, Dim_t DimM>
  class MaterialLinearElasticGeneric1;

  /**
   * traits for use by MaterialMuSpectre for crtp
   */
  template <Dim_t DimS, Dim_t DimM>
  struct MaterialMuSpectre_traits<MaterialLinearElasticGeneric1<DimS, DimM>> {
    //! global field collection
    using GFieldCollection_t =
        typename MaterialBase<DimS, DimM>::GFieldCollection_t;

    //! expected map type for strain fields
    using StrainMap_t =
        muGrid::MatrixFieldMap<GFieldCollection_t, Real, DimM, DimM, true>;
    //! expected map type for stress fields
    using StressMap_t =
        muGrid::MatrixFieldMap<GFieldCollection_t, Real, DimM, DimM>;
    //! expected map type for tangent stiffness fields
    using TangentMap_t =
        muGrid::T4MatrixFieldMap<GFieldCollection_t, Real, DimM>;

    //! declare what type of strain measure your law takes as input
    constexpr static auto strain_measure{StrainMeasure::GreenLagrange};
    //! declare what type of stress measure your law yields as output
    constexpr static auto stress_measure{StressMeasure::PK2};
  };

  /**
   * Linear elastic law defined by a full stiffness tensor. Very
   * generic, but not most efficient
   */
  template <Dim_t DimS, Dim_t DimM>
  class MaterialLinearElasticGeneric1
      : public MaterialMuSpectre<MaterialLinearElasticGeneric1<DimS, DimM>,
                                 DimS, DimM> {
   public:
    //! parent type
    using Parent = MaterialMuSpectre<MaterialLinearElasticGeneric1<DimS, DimM>,
                                     DimS, DimM>;
    //! generic input tolerant to python input
    using CInput_t =
        Eigen::Ref<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>, 0,
                   Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
    //! Default constructor
    MaterialLinearElasticGeneric1() = delete;

    /**
     * Constructor by name and stiffness tensor.
     *
     * @param name unique material name
     * @param C_voigt elastic tensor in Voigt notation
     */
    MaterialLinearElasticGeneric1(const std::string & name,
                                  const CInput_t & C_voigt);

    //! Copy constructor
    MaterialLinearElasticGeneric1(const MaterialLinearElasticGeneric1 & other) =
        delete;

    //! Move constructor
    MaterialLinearElasticGeneric1(MaterialLinearElasticGeneric1 && other) =
        delete;

    //! Destructor
    virtual ~MaterialLinearElasticGeneric1() = default;

    //! Copy assignment operator
    MaterialLinearElasticGeneric1 &
    operator=(const MaterialLinearElasticGeneric1 & other) = delete;

    //! Move assignment operator
    MaterialLinearElasticGeneric1 &
    operator=(MaterialLinearElasticGeneric1 && other) = delete;

    //! see
    //! http://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /**
     * evaluates second Piola-Kirchhoff stress given the Green-Lagrange
     * strain (or Cauchy stress if called with a small strain tensor). Note: the
     * pixel index is ignored.
     */
    template <class Derived>
    inline decltype(auto) evaluate_stress(const Eigen::MatrixBase<Derived> & E,
                                          const size_t & pixel_index = 0);

    /**
     * evaluates both second Piola-Kirchhoff stress and stiffness given
     * the Green-Lagrange strain (or Cauchy stress and stiffness if
     * called with a small strain tensor). Note: the pixel index is ignored.
     */
    template <class Derived>
    inline decltype(auto)
    evaluate_stress_tangent(const Eigen::MatrixBase<Derived> & E,
                            const size_t & pixel_index = 0);

    /**
     * return the empty internals tuple
     */
    std::tuple<> & get_internals() { return this->internal_variables; }

    /**
     * return a reference to the stiffness tensor
     */
    const muGrid::T4Mat<Real, DimM> & get_C() const { return this->C; }

   protected:
    muGrid::T4Mat<Real, DimM> C{};  //! stiffness tensor
    //! empty tuple
    std::tuple<> internal_variables{};
  };

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  template <class Derived>
  auto MaterialLinearElasticGeneric1<DimS, DimM>::evaluate_stress(
      const Eigen::MatrixBase<Derived> & E, const size_t & /*pixel_index*/)
      -> decltype(auto) {
    static_assert(Derived::ColsAtCompileTime == DimM, "wrong input size");
    static_assert(Derived::RowsAtCompileTime == DimM, "wrong input size");
    return Matrices::tensmult(this->C, E);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  template <class Derived>
  auto MaterialLinearElasticGeneric1<DimS, DimM>::evaluate_stress_tangent(
      const Eigen::MatrixBase<Derived> & E, const size_t & /*pixel_index*/)
      -> decltype(auto) {
    using Stress_t = decltype(this->evaluate_stress(E));
    using Stiffness_t = Eigen::Map<muGrid::T4Mat<Real, DimM>>;
    using Ret_t = std::tuple<Stress_t, Stiffness_t>;
    return Ret_t{this->evaluate_stress(E),
                 Stiffness_t(this->C.data())};
  }
}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC_GENERIC1_HH_

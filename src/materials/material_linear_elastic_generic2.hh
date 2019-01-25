/**
 * @file   material_linear_elastic_generic2.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   20 Dec 2018
 *
 * @brief  implementation of a generic linear elastic law with eigenstrains
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
 * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 */

#ifndef SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC_GENERIC2_HH_
#define SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC_GENERIC2_HH_

#include "material_linear_elastic_generic1.hh"

namespace muSpectre {

  /**
   * forward declaration
   */
  template <Dim_t DimS, Dim_t DimM>
  class MaterialLinearElasticGeneric2;

  /**
   * traits for use by MaterialMuSpectre for crtp
   */

  template <Dim_t DimS, Dim_t DimM>
  struct MaterialMuSpectre_traits<MaterialLinearElasticGeneric2<DimS, DimM>> {
    //! global field collection
    using GFieldCollection_t =
        typename MaterialBase<DimS, DimM>::GFieldCollection_t;

    //! expected map type for strain fields
    using StrainMap_t =
        muGrid::MatrixFieldMap<GFieldCollection_t, Real, DimM, DimM, true>;
    //! expected map type for stress fields
    using StressMap_t = muGrid::MatrixFieldMap<GFieldCollection_t, Real, DimM, DimM>;
    //! expected map type for tangent stiffness fields
    using TangentMap_t = muGrid::T4MatrixFieldMap<GFieldCollection_t, Real, DimM>;

    //! declare what type of strain measure your law takes as input
    constexpr static auto strain_measure{StrainMeasure::GreenLagrange};
    //! declare what type of stress measure your law yields as output
    constexpr static auto stress_measure{StressMeasure::PK2};

    //! local field_collections used for internals
    using LFieldColl_t = muGrid::LocalFieldCollection<DimS>;
    //! local strain type
    using LStrainMap_t = muGrid::MatrixFieldMap<LFieldColl_t, Real, DimM, DimM, true>;
    //! elasticity with eigenstrain
    using InternalVariables = std::tuple<LStrainMap_t>;
  };

  /**
   * Implementation proper of the class
   */
  template <Dim_t DimS, Dim_t DimM>
  class MaterialLinearElasticGeneric2
      : public MaterialMuSpectre<MaterialLinearElasticGeneric2<DimS, DimM>,
                                 DimS, DimM> {
    //! parent type
    using Parent = MaterialMuSpectre<MaterialLinearElasticGeneric2<DimS, DimM>,
                                     DimS, DimM>;
    //! underlying worker class
    using Law_t = MaterialLinearElasticGeneric1<DimS, DimM>;

    //! generic input tolerant to python input
    using CInput_t = typename Law_t::CInput_t;

    //! reference to any type that casts to a matrix
    using StrainTensor = Eigen::Ref<Eigen::Matrix<Real, DimM, DimM>>;

    //! traits of this material
    using traits = MaterialMuSpectre_traits<MaterialLinearElasticGeneric2>;

    //! Type of container used for storing eigenstrain
    using InternalVariables_t = typename traits::InternalVariables;

   public:
    //! Default constructor
    MaterialLinearElasticGeneric2() = delete;

    //! Construct by name and elastic stiffness tensor
    MaterialLinearElasticGeneric2(const std::string & name,
                                  const CInput_t & C_voigt);

    //! Copy constructor
    MaterialLinearElasticGeneric2(const MaterialLinearElasticGeneric2 & other) =
        delete;

    //! Move constructor
    MaterialLinearElasticGeneric2(MaterialLinearElasticGeneric2 && other) =
        default;

    //! Destructor
    virtual ~MaterialLinearElasticGeneric2() = default;

    //! Copy assignment operator
    MaterialLinearElasticGeneric2 &
    operator=(const MaterialLinearElasticGeneric2 & other) = delete;

    //! Move assignment operator
    MaterialLinearElasticGeneric2 &
    operator=(MaterialLinearElasticGeneric2 && other) = default;

    //! see
    //! http://eigen.tuxfamily.org/dox/group__TopicStructHavingEigenMembers.html
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    /**
     * evaluates second Piola-Kirchhoff stress given the Green-Lagrange
     * strain (or Cauchy stress if called with a small strain tensor)
     */
    template <class DerivedA, class DerivedB>
    inline decltype(auto)
    evaluate_stress(const Eigen::MatrixBase<DerivedA> & E,
                    const Eigen::MatrixBase<DerivedB> & E_eig);
    /**
     * evaluates both second Piola-Kirchhoff stress and stiffness given
     * the Green-Lagrange strain (or Cauchy stress and stiffness if
     * called with a small strain tensor)
     */
    template <class DerivedA, class DerivedB>
    inline decltype(auto)
    evaluate_stress_tangent(const Eigen::MatrixBase<DerivedA> & E,
                            const Eigen::MatrixBase<DerivedB> & E_eig);

    /**
     * returns tuple with only the eigenstrain field
     */
    InternalVariables_t & get_internals() { return this->internal_variables; }

    /**
     * return a reference to the stiffness tensor
     */
    const muGrid::T4Mat<Real, DimM> & get_C() const { return this->worker.get_C(); }

    /**
     * overload add_pixel to write into eigenstrain
     */
    void add_pixel(const Ccoord_t<DimS> & pixel) final;

    /**
     * overload add_pixel to write into eigenstrain
     */
    void add_pixel(const Ccoord_t<DimS> & pixel, const StrainTensor & E_eig);

   protected:
    Law_t worker;  //! underlying law to be evaluated
    //! storage for eigenstrain
    using Field_t =
        muGrid::TensorField<muGrid::LocalFieldCollection<DimS>, Real, secondOrder, DimM>;
    Field_t & eigen_field;  //!< field holding the eigen strain per pixel
    InternalVariables_t internal_variables;
  };

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  template <class DerivedA, class DerivedB>
  auto MaterialLinearElasticGeneric2<DimS, DimM>::evaluate_stress(
      const Eigen::MatrixBase<DerivedA> & E,
      const Eigen::MatrixBase<DerivedB> & E_eig) -> decltype(auto) {
    return this->worker.evaluate_stress(E - E_eig);
  }

  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  template <class DerivedA, class DerivedB>
  auto MaterialLinearElasticGeneric2<DimS, DimM>::evaluate_stress_tangent(
      const Eigen::MatrixBase<DerivedA> & E,
      const Eigen::MatrixBase<DerivedB> & E_eig) -> decltype(auto) {
    return this->worker.evaluate_stress_tangent(E - E_eig);
  }

}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC_GENERIC2_HH_

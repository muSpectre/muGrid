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
 * Lesser General Public License for more details.
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
 *
 */

#ifndef SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC_GENERIC2_HH_
#define SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC_GENERIC2_HH_

#include "material_linear_elastic_generic1.hh"
#include "libmugrid/mapped_field.hh"

namespace muSpectre {

  /**
   * forward declaration
   */
  template <Index_t DimM>
  class MaterialLinearElasticGeneric2;

  /**
   * traits for use by MaterialMuSpectre for crtp
   */

  template <Index_t DimM>
  struct MaterialMuSpectre_traits<MaterialLinearElasticGeneric2<DimM>> {
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
   * Implementation proper of the class
   */
  template <Index_t DimM>
  class MaterialLinearElasticGeneric2
      : public MaterialMuSpectre<MaterialLinearElasticGeneric2<DimM>, DimM> {
    //! parent type
    using Parent = MaterialMuSpectre<MaterialLinearElasticGeneric2<DimM>, DimM>;
    //! underlying worker class
    using Law_t = MaterialLinearElasticGeneric1<DimM>;

    //! generic input tolerant to python input
    using CInput_t = typename Law_t::CInput_t;

    //! reference to any type that casts to a matrix
    using StrainTensor = Eigen::Ref<Eigen::Matrix<Real, DimM, DimM>>;

    //! traits of this material
    using traits = MaterialMuSpectre_traits<MaterialLinearElasticGeneric2>;

   public:
    //! Default constructor
    MaterialLinearElasticGeneric2() = delete;

    //! Construct by name and elastic stiffness tensor
    MaterialLinearElasticGeneric2(const std::string & name,
                                  const Index_t & spatial_dimension,
                                  const Index_t & nb_quad_pts,
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

    /**
     * evaluates second Piola-Kirchhoff stress given the Green-Lagrange
     * strain (or Cauchy stress if called with a small strain tensor)
     */
    template <class Derived>
    inline decltype(auto) evaluate_stress(
        const Eigen::MatrixBase<Derived> & E,
        const Eigen::Map<const Eigen::Matrix<Real, DimM, DimM>> & E_eig);

    /**
     * evaluates second Piola-Kirchhoff stress given the Green-Lagrange
     * strain (or Cauchy stress if called with a small strain tensor) and the
     * local pixel id
     */
    template <class Derived>
    inline decltype(auto) evaluate_stress(const Eigen::MatrixBase<Derived> & E,
                                          const size_t & quad_pt_index) {
      auto && E_eig{this->eigen_field[quad_pt_index]};
      return this->evaluate_stress(E, E_eig);
    }

    /**
     * evaluates both second Piola-Kirchhoff stress and stiffness given
     * the Green-Lagrange strain (or Cauchy stress and stiffness if
     * called with a small strain tensor)
     */
    template <class Derived>
    inline decltype(auto) evaluate_stress_tangent(
        const Eigen::MatrixBase<Derived> & E,
        const Eigen::Map<const Eigen::Matrix<Real, DimM, DimM>> & E_eig);

    /**
     * evaluates both second Piola-Kirchhoff stress and tangent moduli given
     * the Green-Lagrange strain (or Cauchy stress and stiffness if
     * called with a small strain tensor) and the local pixel id
     */
    template <class Derived>
    inline decltype(auto)
    evaluate_stress_tangent(const Eigen::MatrixBase<Derived> & E,
                            const size_t & quad_pt_index) {
      auto && E_eig{this->eigen_field[quad_pt_index]};
      return this->evaluate_stress_tangent(E, E_eig);
    }

    /**
     * return a reference to the stiffness tensor
     */
    const muGrid::T4Mat<Real, DimM> & get_C() const {
      return this->worker.get_C();
    }

    /**
     * overload add_pixel to write into eigenstrain
     */
    void add_pixel(const size_t & pixel_index) final;

    /**
     * overload add_pixel to write into eigenstrain
     */
    void add_pixel(const size_t & pixel_index, const StrainTensor & E_eig);

   protected:
    //! elastic law without eigenstrain used as worker
    Law_t worker;  //! underlying law to be evaluated
    //! storage for eigenstrain
    muGrid::MappedT2Field<Real, Mapping::Const, DimM, IterUnit::SubPt>
        eigen_field;
  };

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class Derived>
  auto MaterialLinearElasticGeneric2<DimM>::evaluate_stress(
      const Eigen::MatrixBase<Derived> & E,
      const Eigen::Map<const Eigen::Matrix<Real, DimM, DimM>> & E_eig)
      -> decltype(auto) {
    return this->worker.evaluate_stress(E - E_eig);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class Derived>
  auto MaterialLinearElasticGeneric2<DimM>::evaluate_stress_tangent(
      const Eigen::MatrixBase<Derived> & E,
      const Eigen::Map<const Eigen::Matrix<Real, DimM, DimM>> & E_eig)
      -> decltype(auto) {
    return this->worker.evaluate_stress_tangent(E - E_eig);
  }

}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC_GENERIC2_HH_

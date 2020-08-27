/**
 * @file   material_linear_elastic_damage1.hh
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   04 May 2020
 *
 * @brief  The linear elastic material with damage
 *
 * Copyright © 2020 Ali Falsafi
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

#ifndef SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC_DAMAGE1_HH_
#define SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC_DAMAGE1_HH_

#include "materials/material_linear_elastic1.hh"
#include "materials/material_muSpectre.hh"
#include "materials/materials_toolbox.hh"
#include "materials/stress_transformations_PK2.hh"

#include <libmugrid/mapped_state_field.hh>

namespace muSpectre {
  template <Index_t DimM>
  class MaterialLinearElasticDamage1;

  /**
   * traits for objective linear visco_elasticity
   */
  template <Index_t DimM>
  struct MaterialMuSpectre_traits<MaterialLinearElasticDamage1<DimM>>
      : public DefaultMechanics_traits<DimM, StrainMeasure::GreenLagrange,
                                       StressMeasure::PK2> {};

  /**
   * implements objective linear material with damage
   * DimM material_dimension (dimension of constitutive law)
   */

  template <Index_t DimM>
  class MaterialLinearElasticDamage1
      : public MaterialMuSpectreMechanics<MaterialLinearElasticDamage1<DimM>,
                                          DimM> {
   public:
    //! base class
    using Parent =
        MaterialMuSpectreMechanics<MaterialLinearElasticDamage1, DimM>;

    //! short-hand for second-rank tensors
    using T2_t = Eigen::Matrix<Real, DimM, DimM>;

    //! short-hand for fourth-rank tensors
    using T4_t = muGrid::T4Mat<Real, DimM>;

    //! traits of this material
    using traits = MaterialMuSpectre_traits<MaterialLinearElasticDamage1>;

    //! Hooke's law implementation
    using Hooke =
        typename MatTB::Hooke<DimM, typename traits::StrainMap_t::reference,
                              typename traits::TangentMap_t::reference>;

    //! type in which the previous strain state is referenced
    using T2StRef_t =
        typename muGrid::MappedT2StateField<Real, Mapping::Mut, DimM,
                                            IterUnit::SubPt>::Return_t;

    using ScalarStRef_t =
        typename muGrid::MappedScalarStateField<Real, Mapping::Mut,
                                                IterUnit::SubPt>::Return_t;

    using ScalarRef_t =
        typename muGrid::MappedScalarField<Real, Mapping::Mut,
                                           IterUnit::SubPt>::Return_t;

    //! Type for the child material
    using MatChild_t = MaterialLinearElastic1<DimM>;

    //! Default constructor
    MaterialLinearElasticDamage1() = delete;

    //! Copy constructor
    MaterialLinearElasticDamage1(const MaterialLinearElasticDamage1 & other) =
        delete;

    //! Construct by name, Young's modulus and Poisson's ratio
    MaterialLinearElasticDamage1(
        const std::string & name, const Index_t & spatial_dimension,
        const Index_t & nb_quad_pts, const Real & young, const Real & poisson,
        const Real & kappa_init, const Real & alpha, const Real & beta,
        const std::shared_ptr<muGrid::LocalFieldCollection> &
            parent_field_collection = nullptr);

    //! Move constructor
    MaterialLinearElasticDamage1(MaterialLinearElasticDamage1 && other) =
        delete;

    //! Destructor
    virtual ~MaterialLinearElasticDamage1() = default;

    //! Copy assignment operator
    MaterialLinearElasticDamage1 &
    operator=(const MaterialLinearElasticDamage1 & other) = delete;

    //! Move assignment operator
    MaterialLinearElasticDamage1 &
    operator=(MaterialLinearElasticDamage1 && other) = delete;

    /**
     * evaluates Kirchhoff stress given the current placement gradient
     * Fₜ,
     */
    T2_t evaluate_stress(const Eigen::Ref<const T2_t> & E, ScalarStRef_t kappa);

    /**
     * evaluates Kirchhoff stress given the local placement gradient and pixel
     * id.
     */
    template <class Derived>
    inline decltype(auto) evaluate_stress(const Eigen::MatrixBase<Derived> & E,
                                          const size_t & quad_pt_index);

    /**
     * update the damage_measure varaible before proceeding with stress
     * evaluation
     */
    void update_damage_measure(const Eigen::Ref<const T2_t> & E,
                               ScalarStRef_t kappa);

    /**
     * evaluates Kirchhoff stress and tangent moduli given the ...
     */
    std::tuple<T2_t, T4_t>
    evaluate_stress_tangent(const Eigen::Ref<const T2_t> & E,
                            ScalarStRef_t kappa);

    /**
     * evaluates Kirchhoff stress and tangent moduli given the local
     * placement gradient and pixel id.
     */
    template <class Derived>
    inline decltype(auto)
    evaluate_stress_tangent(const Eigen::MatrixBase<Derived> & E,
                            const size_t & quad_pt_index);

    /**
     * @brief      computes the damage_measure driving measure (it can be the
     * norm of strain or elastic strain energy or etc.)
     *
     * @param      Strain (T2_t)
     *
     * @return     A measure for damage parameter update (Real)
     */
    template <class Derived>
    inline Real compute_strain_measure(const Eigen::MatrixBase<Derived> & E);

    /**
     * @brief      Updates the damage measure given the strain measure
     *
     * @param      strain_measure (Real)
     *
     * @return     damage measure (Real)
     */
    Real compute_damage_measure(const Real & Kappa);

    /**
     * The statefields need to be cycled at the end of each load increment
     */
    void save_history_variables() final;

    /**
     * set the previous gradients to identity
     */
    void initialise() final;

    //! getter for internal variable field of strain measure
    inline muGrid::MappedScalarStateField<Real, Mapping::Mut, IterUnit::SubPt> &
    get_kappa_field() {
      return this->kappa_field;
    }

    //! getter for kappa_init
    inline Real get_kappa_init() { return this->kappa_init; }

   protected:
    // Child material (used as a worker for evaluating stress and tangent)
    MatChild_t material_child;

    //! storage for damage variable
    muGrid::MappedScalarStateField<Real, Mapping::Mut, IterUnit::SubPt>
        kappa_field;

    // damage evolution parameters:
    const Real kappa_init;  //!< threshold of damage (strength)
    const Real alpha;       //! damage evaluation parameter([0, ∞])
    const Real beta;        //! age evaluation parameter([0, 1])
  };

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class Derived>
  decltype(auto) MaterialLinearElasticDamage1<DimM>::evaluate_stress(
      const Eigen::MatrixBase<Derived> & E, const size_t & quad_pt_index) {
    auto && kappa{this->get_kappa_field()[quad_pt_index]};
    return this->evaluate_stress(std::move(E), kappa);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class Derived>
  decltype(auto) MaterialLinearElasticDamage1<DimM>::evaluate_stress_tangent(
      const Eigen::MatrixBase<Derived> & E, const size_t & quad_pt_index) {
    auto && kappa{this->get_kappa_field()[quad_pt_index]};
    return this->evaluate_stress_tangent(std::move(E), kappa);
  }

  /* ---------------------------------------------------------------------- */
}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC_DAMAGE1_HH_

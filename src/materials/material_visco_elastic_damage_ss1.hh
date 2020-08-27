/**
 * @file material_visco_elastic_damage_ss1.hh
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   20 Dec 2019
 *
 * @brief this material constitutive law is inspired by the standard linear
 * viscos solid described in Chapter 10 of "Computational inelasticity"
 * by J. Simo et al. Besides, the damage part in taken from
 * "ON A FULLY THREE-DIMENSIONAL FINITE-STRAIN VISCOELASTIC DAMAGE MODEL:
 * FORMULATION AND COMPUTATIONAL ASPECTS" by J. Simo.
 * Note: it is assumed that the viscous effect merely exists on the deviatoric
 * contribution of the response by performing a stress multiplicative split.
 * The schematic of the rheological model is:
                  E∞
         ------|\/\/\|-------
        |                    |
     ---|                    |---
        |                    |
         ----|\/\/\|---[]----
                Eᵥ     ηᵥ
 *
 * Copyright © 2019 Ali Falsafi
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

#ifndef SRC_MATERIALS_MATERIAL_VISCO_ELASTIC_DAMAGE_SS1_HH_
#define SRC_MATERIALS_MATERIAL_VISCO_ELASTIC_DAMAGE_SS1_HH_

#include "materials/material_visco_elastic_ss.hh"
#include "materials/material_muSpectre_mechanics.hh"
#include "materials/materials_toolbox.hh"
#include "materials/stress_transformations_PK2.hh"

#include <libmugrid/mapped_state_field.hh>

namespace muSpectre {
  template <Index_t DimM>
  class MaterialViscoElasticDamageSS1;

  /**
   * traits for objective linear visco_elasticity with damage
   */
  template <Index_t DimM>
  struct MaterialMuSpectre_traits<MaterialViscoElasticDamageSS1<DimM>>
      : public DefaultMechanics_traits<DimM, StrainMeasure::GreenLagrange,
                                       StressMeasure::PK2> {};

  /**
   * implements objective linear visco_elasticity with damage
   * DimM material_dimension (dimension of constitutive law)
   */
  template <Index_t DimM>
  class MaterialViscoElasticDamageSS1
      : public MaterialMuSpectreMechanics<MaterialViscoElasticDamageSS1<DimM>,
                                          DimM> {
   public:
    //! base class
    using Parent =
        MaterialMuSpectreMechanics<MaterialViscoElasticDamageSS1, DimM>;

    //! short-hand for second-rank tensors
    using T2_t = Eigen::Matrix<Real, DimM, DimM>;

    //! short-hand for fourth-rank tensors
    using T4_t = muGrid::T4Mat<Real, DimM>;

    //! traits of this material
    using traits = MaterialMuSpectre_traits<MaterialViscoElasticDamageSS1>;

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

    //! Default constructor
    MaterialViscoElasticDamageSS1() = delete;

    //! Copy constructor
    MaterialViscoElasticDamageSS1(const MaterialViscoElasticDamageSS1 & other) =
        delete;

    //! Construct by name, Young's modulus and Poisson's ratio
    MaterialViscoElasticDamageSS1(
        const std::string & name, const Index_t & spatial_dimension,
        const Index_t & nb_quad_pts, const Real & young_inf,
        const Real & young_v, const Real & eta_v, const Real & poisson,
        const Real & kappa_init, const Real & alpha, const Real & beta,
        const Real & dt,
        const std::shared_ptr<muGrid::LocalFieldCollection> &
            parent_field_collection = nullptr);

    //! Move constructor
    MaterialViscoElasticDamageSS1(MaterialViscoElasticDamageSS1 && other) =
        delete;

    //! Destructor
    virtual ~MaterialViscoElasticDamageSS1() = default;

    //! Copy assignment operator
    MaterialViscoElasticDamageSS1 &
    operator=(const MaterialViscoElasticDamageSS1 & other) = delete;

    //! Move assignment operator
    MaterialViscoElasticDamageSS1 &
    operator=(MaterialViscoElasticDamageSS1 && other) = delete;

    /**
     * evaluates Kirchhoff stress given the current placement gradient
     * Fₜ,
     */
    T2_t evaluate_stress(const Eigen::Ref<const T2_t> & E, T2StRef_t h_prev,
                         T2StRef_t s_null_prev, ScalarStRef_t kappa);

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
    evaluate_stress_tangent(const Eigen::Ref<const T2_t> & E, T2StRef_t h_prev,
                            T2StRef_t s_null_prev, ScalarStRef_t kappa);

    /**
     * evaluates Kirchhoff stressstiffness and tangent moduli given the local
     * placement gradient and pixel id.
     */
    template <class Derived>
    inline decltype(auto)
    evaluate_stress_tangent(const Eigen::MatrixBase<Derived> & E,
                            const size_t & quad_pt_index);

    /**
     * @brief      computes the damage_measure driving measure(it can be the
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

    //! getter for internal variable field History Integral
    muGrid::MappedT2StateField<Real, Mapping::Mut, DimM, IterUnit::SubPt> &
    get_history_integral();

    //! getter for internal variable field of Elastic stress
    muGrid::MappedT2StateField<Real, Mapping::Mut, DimM, IterUnit::SubPt> &
    get_s_null_prev_field();

    //! getter for internal variable field of Elastic stress
    muGrid::MappedScalarStateField<Real, Mapping::Mut, IterUnit::SubPt> &
    get_kappa_field();

    // //!getter for kappa_init
    const Real & get_kappa_init();

   protected:
    //! Child material (used as a worker for evaluating stress and tangent)
    MaterialViscoElasticSS<DimM> material_child;

    //! storage for damage variable
    muGrid::MappedScalarStateField<Real, Mapping::Mut, IterUnit::SubPt>
        kappa_field;

    // damage evolution parameters:
    const Real kappa_init;  //!< threshold of damage (strength)
    const Real alpha;       //!< damage evaluation parameter([0, ∞])
    const Real beta;        //!< another damage evaluation parameter([0, 1])
  };

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class Derived>
  decltype(auto) MaterialViscoElasticDamageSS1<DimM>::evaluate_stress(
      const Eigen::MatrixBase<Derived> & E, const size_t & quad_pt_index) {
    auto && h_prev{this->get_history_integral()[quad_pt_index]};
    auto && s_null_prev{this->get_s_null_prev_field()[quad_pt_index]};
    auto && kappa{this->get_kappa_field()[quad_pt_index]};
    return this->evaluate_stress(E, h_prev, s_null_prev, kappa);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class Derived>
  decltype(auto) MaterialViscoElasticDamageSS1<DimM>::evaluate_stress_tangent(
      const Eigen::MatrixBase<Derived> & E, const size_t & quad_pt_index) {
    auto && h_prev{this->get_history_integral()[quad_pt_index]};
    auto && s_null_prev{this->get_s_null_prev_field()[quad_pt_index]};
    auto && kappa{this->get_kappa_field()[quad_pt_index]};
    return this->evaluate_stress_tangent(E, h_prev, s_null_prev, kappa);
  }

  /* ---------------------------------------------------------------------- */
}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_VISCO_ELASTIC_DAMAGE_SS1_HH_

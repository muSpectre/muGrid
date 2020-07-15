/**
 * @file   material_visco_elastic_ss.hh
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   20 Dec 2019
 *
 * @brief  this material is the material which uses standard solid constitutive
 * law for reproducing linear viscoelastic behaviour the freomulation is
 * obtained from Chapter 10 of Simo JC, Hughes TJ. Computational inelasticity.
 * Springer Science & Business Media; 2006 May 7.
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

#ifndef SRC_MATERIALS_MATERIAL_VISCO_ELASTIC_SS_HH_
#define SRC_MATERIALS_MATERIAL_VISCO_ELASTIC_SS_HH_

#include "materials/material_muSpectre_base.hh"
#include "materials/materials_toolbox.hh"
#include "materials/stress_transformations_PK2.hh"

#include <libmugrid/mapped_state_field.hh>

namespace muSpectre {
  template <Index_t DimM>
  class MaterialViscoElasticSS;

  /**
   * traits for objective linear visco_elasticity
   */
  template <Index_t DimM>
  struct MaterialMuSpectre_traits<MaterialViscoElasticSS<DimM>> {
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
   * implements objective linear visco_elasticity
   * DimM material_dimension (dimension of constitutive law)
   */

  template <Index_t DimM>
  class MaterialViscoElasticSS
      : public MaterialMuSpectre<MaterialViscoElasticSS<DimM>, DimM> {
   public:
    //! base class
    using Parent = MaterialMuSpectre<MaterialViscoElasticSS, DimM>;

    //! short-hand for second-rank tensors
    using T2_t = Eigen::Matrix<Real, DimM, DimM>;

    //! short-hand for fourth-rank tensors
    using T4_t = muGrid::T4Mat<Real, DimM>;

    //! traits of this material
    using traits = MaterialMuSpectre_traits<MaterialViscoElasticSS>;

    //! Hooke's law implementation
    using Hooke =
        typename MatTB::Hooke<DimM, typename traits::StrainMap_t::reference,
                              typename traits::TangentMap_t::reference>;

    //! type in which the previous strain state is referenced
    using T2StRef_t =
        typename muGrid::MappedT2StateField<Real, Mapping::Mut, DimM,
                                            IterUnit::SubPt>::Return_t;

    //! Default constructor
    MaterialViscoElasticSS() = delete;

    //! Copy constructor
    MaterialViscoElasticSS(const MaterialViscoElasticSS & other) = delete;

    //! Construct by name, Young's modulus and Poisson's ratio
    MaterialViscoElasticSS(const std::string & name,
                           const Index_t & spatial_dimension,
                           const Index_t & nb_quad_pts, const Real & young_inf,
                           const Real & young_v, const Real & eta_v,
                           const Real & poisson, const Real & dt,
                           const std::shared_ptr<muGrid::LocalFieldCollection> &
                               parent_field_collection = nullptr);

    //! Move constructor
    MaterialViscoElasticSS(MaterialViscoElasticSS && other) = delete;

    //! Destructor
    virtual ~MaterialViscoElasticSS() = default;

    //! Copy assignment operator
    MaterialViscoElasticSS &
    operator=(const MaterialViscoElasticSS & other) = delete;

    //! Move assignment operator
    MaterialViscoElasticSS &
    operator=(MaterialViscoElasticSS && other) = delete;

    /**
     * evaluates Kirchhoff stress given the current placement gradient
     * Fₜ,
     */
    T2_t evaluate_stress(const Eigen::Ref<const T2_t> & F, T2StRef_t h_prev,
                         T2StRef_t s_null_prev);

    /**
     *
     */
    using Worker_t = typename std::tuple<T2_t, T2_t, T2_t, T2_t, T2_t, T2_t,
                                         Real, Real, Real, Real>;
    Worker_t evaluate_stress_worker(const Eigen::Ref<const T2_t> & E,
                                    T2StRef_t h_prev, T2StRef_t s_null_prev);
    /**
     * evaluates Kirchhoff stress given the local placement gradient and pixel
     * id.
     */
    template <class Derived>
    T2_t evaluate_stress(const Eigen::MatrixBase<Derived> & E,
                         const size_t & quad_pt_index) {
      auto && h_prev{this->h_prev_field[quad_pt_index]};
      auto && s_null_prev{this->s_null_prev_field[quad_pt_index]};
      return this->evaluate_stress(E, h_prev, s_null_prev);
    }

    /**
     * evaluates the volumetric part of the elastic stress
     */
    template <class Derived>

    inline T2_t evaluate_elastic_stress(const Eigen::MatrixBase<Derived> & E);

    /**
     * evaluates the volumetric part of the elastic stress
     */
    template <class Derived>

    inline T2_t
    evaluate_elastic_volumetric_stress(const Eigen::MatrixBase<Derived> & E);

    /**
     * evaluates the deviatoric part of the elastic stress
     */
    template <class Derived>
    inline T2_t
    evaluate_elastic_deviatoric_stress(const Eigen::MatrixBase<Derived> & e);
    /**
     * evaluates Kirchhoff stress and tangent moduli given the current placement
     * gradient Fₜ, the previous Gradient Fₜ₋₁ and the cumulated plastic flow εₚ
     */
    std::tuple<T2_t, T4_t>
    evaluate_stress_tangent(const Eigen::Ref<const T2_t> & E, T2StRef_t h_prev,
                            T2StRef_t s_null_prev);
    /**
     * evaluates Kirchhoff stress  and tangent moduli given the local
     * placement gradient and pixel id.
     */
    template <class Derived>
    std::tuple<T2_t, T4_t>
    evaluate_stress_tangent(const Eigen::MatrixBase<Derived> & E,
                            const size_t & quad_pt_index) {
      auto && h_prev{this->h_prev_field[quad_pt_index]};
      auto && s_null_prev{this->s_null_prev_field[quad_pt_index]};
      return this->evaluate_stress_tangent(E, h_prev, s_null_prev);
    }

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

    //! getter for returning the λₜₒₜ:
    Real get_lambda_tot() { return this->lambda_tot; }

    //! getter for returning the μₜₒₜ:
    Real get_mu_tot() { return this->mu_tot; }

   protected:
    //! storage for previous history intgral()
    muGrid::MappedT2StateField<Real, Mapping::Mut, DimM, IterUnit::SubPt>
        s_null_prev_field;
    //! storage for previous history intgral()
    muGrid::MappedT2StateField<Real, Mapping::Mut, DimM, IterUnit::SubPt>
        h_prev_field;
    const Real young_inf;  //!< Young's modulus (E∞)
    const Real young_v;    //!< Young's modulus (Eᵥ)
    const Real eta_v;      //!< viscosity of the linear dashpot(ηᵥ)
    const Real poisson;    //!< Poisson's ratio
    //! The non-viscous branch
    const Real lambda_inf;  //!< first Lamé constant
    const Real mu_inf;      //!< second Lamé constant (shear modulus)
    const Real K_inf;       //!< Bulk Modulus
    //! The viscous branch:
    const Real lambda_v;  //!< first Lamé constant
    const Real mu_v;      //!< second Lamé constant (shear modulus)
    const Real K_v;       //!< Bulk Modulus
    const Real tau_v;     //!< Time constant (τ = ηᵥ/Eᵥ)
    //! Sum of the branches:
    const Real young_tot;   //!< ∑E
    const Real K_tot;       //!< Bulk Modulus
    const Real mu_tot;      //!< shear Modulus
    const Real lambda_tot;  //!< first Lamé constant
    //!
    const Real gamma_inf;  //!< γᵥ = (Eᵥ/Eₜₒₜ);
    const Real gamma_v;    //!< γ∞ = (E∞/Eₜₒₜ);
    const Real
        dt;  //!< The time step of solution (Incremental load application)
  };

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class Derived>
  auto MaterialViscoElasticSS<DimM>::evaluate_elastic_volumetric_stress(
      const Eigen::MatrixBase<Derived> & E) -> T2_t {
    return (E.trace() * (this->lambda_tot + (2 * this->mu_tot / DimM))) *
           T2_t::Identity();
    // return E.trace() * (K_tot)*T2_t::Identity();
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class Derived>
  auto MaterialViscoElasticSS<DimM>::evaluate_elastic_deviatoric_stress(
      const Eigen::MatrixBase<Derived> & e) -> T2_t {
    return 2 * this->mu_tot * e;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class Derived>
  auto MaterialViscoElasticSS<DimM>::evaluate_elastic_stress(
      const Eigen::MatrixBase<Derived> & E) -> T2_t {
    auto && e{MatTB::compute_deviatoric<DimM>(E)};
    return 2 * this->mu_tot * e +
           (E.trace() * (this->lambda_tot + (2 * this->mu_tot / DimM))) *
               T2_t::Identity();
  }
  /* ---------------------------------------------------------------------- */

}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_VISCO_ELASTIC_SS_HH_

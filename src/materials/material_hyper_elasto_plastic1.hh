/**
 * @file   material_hyper_elasto_plastic1.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   20 Feb 2018
 *
 * @brief  Material for logarithmic hyperelasto-plasticity, as defined in de
 *         Geus 2017 (https://doi.org/10.1016/j.cma.2016.12.032) and further
 *         explained in Geers 2003 (https://doi.org/10.1016/j.cma.2003.07.014)
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



#ifndef MATERIAL_HYPER_ELASTO_PLASTIC1_H
#define MATERIAL_HYPER_ELASTO_PLASTIC1_H

#include "materials/material_muSpectre_base.hh"
#include "materials/materials_toolbox.hh"
#include "common/eigen_tools.hh"
#include "common/statefield.hh"

#include <algorithm>

namespace muSpectre {

  template <Dim_t DimS, Dim_t DimM>
  class MaterialHyperElastoPlastic1;

  /**
   * traits for hyper-elastoplastic material
   */
  template <Dim_t DimS, Dim_t DimM>
  struct MaterialMuSpectre_traits<MaterialHyperElastoPlastic1<DimS, DimM>> {
    //! global field collection
    using GFieldCollection_t = typename
      MaterialBase<DimS, DimM>::GFieldCollection_t;

    //! expected map type for strain fields
    using StrainMap_t =
      MatrixFieldMap<GFieldCollection_t, Real, DimM, DimM, true>;
    //! expected map type for stress fields
    using StressMap_t = MatrixFieldMap<GFieldCollection_t, Real, DimM, DimM>;
    //! expected map type for tangent stiffness fields
    using TangentMap_t = T4MatrixFieldMap<GFieldCollection_t, Real, DimM>;

    //! declare what type of strain measure your law takes as input
    constexpr static auto strain_measure{StrainMeasure::Gradient};
    //! declare what type of stress measure your law yields as output
    constexpr static auto stress_measure{StressMeasure::Kirchhoff};

    //! local field collection used for internals
    using LFieldColl_t = LocalFieldCollection<DimS>;

    //! storage type for plastic flow measure (εₚ in the papers)
    using LScalarMap_t = StateFieldMap<ScalarFieldMap<LFieldColl_t, Real>>;
    /**
     * storage type for for previous gradient Fᵗ and elastic left
     * Cauchy-Green deformation tensor bₑᵗ
     */
    using LStrainMap_t = StateFieldMap<
      MatrixFieldMap<LFieldColl_t, Real, DimM, DimM, false>>;
    /**
     * format in which to receive internals (previous gradient Fᵗ,
     * previous elastic lef Cauchy-Green deformation tensor bₑᵗ, and
     * the plastic flow measure εₚ
     */
    using InternalVariables = std::tuple<LStrainMap_t, LStrainMap_t,
                                         LScalarMap_t>;

  };


  /**
   * Material implementation for hyper-elastoplastic constitutive law
   */
  template <Dim_t DimS, Dim_t DimM=DimS>
  class MaterialHyperElastoPlastic1: public
    MaterialMuSpectre<MaterialHyperElastoPlastic1<DimS, DimM>, DimS, DimM>
  {
  public:

    //! base class
    using Parent = MaterialMuSpectre
      <MaterialHyperElastoPlastic1<DimS, DimM>, DimS, DimM>;

    /**
     * type used to determine whether the
     * `muSpectre::MaterialMuSpectre::iterable_proxy` evaluate only
     * stresses or also tangent stiffnesses
     */
    using NeedTangent = typename Parent::NeedTangent;

    //! shortcut to traits
    using traits = MaterialMuSpectre_traits<MaterialHyperElastoPlastic1>;

    //! Hooke's law implementation
    using Hooke = typename
      MatTB::Hooke<DimM,
                   typename traits::StrainMap_t::reference,
                   typename traits::TangentMap_t::reference>;

    //! type in which the previous strain state is referenced
    using StrainStRef_t = typename traits::LStrainMap_t::reference;
    //! type in which the previous plastic flow is referenced
    using FlowStRef_t = typename traits::LScalarMap_t::reference;

    //! Default constructor
    MaterialHyperElastoPlastic1() = delete;

    //! Constructor with name and material properties
    MaterialHyperElastoPlastic1(std::string name, Real young, Real poisson,
                                Real tau_y0, Real H);

    //! Copy constructor
    MaterialHyperElastoPlastic1(const MaterialHyperElastoPlastic1 &other) = delete;

    //! Move constructor
    MaterialHyperElastoPlastic1(MaterialHyperElastoPlastic1 &&other) = delete;

    //! Destructor
    virtual ~MaterialHyperElastoPlastic1() = default;

    //! Copy assignment operator
    MaterialHyperElastoPlastic1& operator=(const MaterialHyperElastoPlastic1 &other) = delete;

    //! Move assignment operator
    MaterialHyperElastoPlastic1& operator=(MaterialHyperElastoPlastic1 &&other) = delete;

    /**
     * evaluates Kirchhoff stress given the current placement gradient
     * Fₜ, the previous Gradient Fₜ₋₁ and the cumulated plastic flow
     * εₚ
     */
    template <class grad_t>
    inline decltype(auto) evaluate_stress(grad_t && F, StrainStRef_t F_prev,
                                          StrainStRef_t be_prev,
                                          FlowStRef_t plast_flow);

    /**
     * evaluates Kirchhoff stress and stiffness given the current placement gradient
     * Fₜ, the previous Gradient Fₜ₋₁ and the cumulated plastic flow
     * εₚ
     */
    template <class grad_t>
    inline decltype(auto) evaluate_stress_tangent(grad_t && F, StrainStRef_t F_prev,
                                                  StrainStRef_t be_prev,
                                                  FlowStRef_t plast_flow);

    /**
     * The statefields need to be cycled at the end of each load increment
     */
    virtual void save_history_variables() override;

    /**
     * set the previous gradients to identity
     */
    virtual void initialise() override final;

    /**
     * return the internals tuple
     */
    typename traits::InternalVariables & get_internals() {
      return this->internal_variables;};


  protected:

    /**
     * worker function computing stresses and internal variables
     */
    template <class grad_t>
    inline decltype(auto) stress_n_internals_worker(grad_t && F,
                                                    StrainStRef_t& F_prev,
                                                    StrainStRef_t& be_prev,
                                                    FlowStRef_t& plast_flow);
    //! Local FieldCollection type for field storage
    using LColl_t = LocalFieldCollection<DimS>;
    //! storage for cumulated plastic flow εₚ
    StateField<ScalarField<LColl_t, Real>>  plast_flow_field;

    //! storage for previous gradient Fᵗ
    StateField<TensorField<LColl_t, Real, secondOrder, DimM>> F_prev_field;

    //! storage for elastic left Cauchy-Green deformation tensor bₑᵗ
    StateField<TensorField<LColl_t, Real, secondOrder, DimM>> be_prev_field;

    // material properties
    const Real young;          //!< Young's modulus
    const Real poisson;        //!< Poisson's ratio
    const Real lambda;         //!< first Lamé constant
    const Real mu;             //!< second Lamé constant (shear modulus)
    const Real K;              //!< Bulk modulus
    const Real tau_y0;         //!< initial yield stress
    const Real H;              //!< hardening modulus
    const T4Mat<Real, DimM> C; //!< stiffness tensor

    //! Field maps and state field maps over internal fields
    typename traits::InternalVariables internal_variables;
  private:
  };

  //----------------------------------------------------------------------------//
  template <Dim_t DimS, Dim_t DimM>
  template <class grad_t>
  decltype(auto)
  MaterialHyperElastoPlastic1<DimS, DimM>::
  stress_n_internals_worker(grad_t && F, StrainStRef_t& F_prev,
                            StrainStRef_t& be_prev, FlowStRef_t& eps_p)  {

    // the notation in this function follows Geers 2003
    // (https://doi.org/10.1016/j.cma.2003.07.014).

    // computation of trial state
    using Mat_t = Eigen::Matrix<Real, DimM, DimM>;
    auto && f{F*F_prev.old().inverse()};
    Mat_t be_star{f*be_prev.old()*f.transpose()};
    Mat_t ln_be_star{logm(std::move(be_star))};
    Mat_t tau_star{.5*Hooke::evaluate_stress(this->lambda, this->mu, ln_be_star)};
    // deviatoric part of Kirchhoff stress
    Mat_t tau_d_star{tau_star - tau_star.trace()/DimM*tau_star.Identity()};
    auto && tau_eq_star{std::sqrt(3*.5*(tau_d_star.array()*
                                     tau_d_star.transpose().array()).sum())};
    Mat_t N_star{3*.5*tau_d_star/tau_eq_star};
    // this is eq (27), and the std::min enforces the Kuhn-Tucker relation (16)
    Real phi_star{std::max(tau_eq_star - this->tau_y0 - this->H * eps_p.old(), 0.)};

    // return mapping
    Real Del_gamma{phi_star/(this->H + 3 * this->mu)};
    auto && tau{tau_star - 2*Del_gamma*this->mu*N_star};
    /////auto && tau_eq{tau_eq_star - 3*this->mu*Del_gamma};

    // update the previous values to the new ones
    F_prev.current() = F;
    ln_be_star -= 2*Del_gamma*N_star;
    be_prev.current() = expm(std::move(ln_be_star));
    eps_p.current() += Del_gamma;


    // transmit info whether this is a plastic step or not
    bool is_plastic{phi_star > 0};
    return std::tuple<Mat_t, Real, Real, Mat_t, bool>
      (std::move(tau), std::move(tau_eq_star),
       std::move(Del_gamma), std::move(N_star),
       std::move(is_plastic));
  }
  //----------------------------------------------------------------------------//
  template <Dim_t DimS, Dim_t DimM>
  template <class grad_t>
  decltype(auto)
  MaterialHyperElastoPlastic1<DimS, DimM>::
  evaluate_stress(grad_t && F, StrainStRef_t F_prev, StrainStRef_t be_prev,
                  FlowStRef_t eps_p)  {
    auto retval(std::move(std::get<0>(this->stress_n_internals_worker
                                                                 (std::forward<grad_t>(F), F_prev, be_prev, eps_p))));
    return retval;
  }
  //----------------------------------------------------------------------------//
  template <Dim_t DimS, Dim_t DimM>
  template <class grad_t>
  decltype(auto)
  MaterialHyperElastoPlastic1<DimS, DimM>::
  evaluate_stress_tangent(grad_t && F, StrainStRef_t F_prev, StrainStRef_t be_prev,
                          FlowStRef_t eps_p)  {
    //! after the stress computation, all internals are up to date
    auto && vals{this->stress_n_internals_worker
        (std::forward<grad_t>(F), F_prev, be_prev, eps_p)};
    auto && tau        {std::get<0>(vals)};
    auto && tau_eq_star{std::get<1>(vals)};
    auto && Del_gamma  {std::get<2>(vals)};
    auto && N_star     {std::get<3>(vals)};
    auto && is_plastic {std::get<4>(vals)};

    if (is_plastic) {
      auto && a0 = Del_gamma*this->mu/tau_eq_star;
      auto && a1 = this->mu/(this->H + 3*this->mu);
      return std::make_tuple(std::move(tau), T4Mat<Real, DimM>{
        ((this->K/2. - this->mu/3 + a0*this->mu)*Matrices::Itrac<DimM>() +
         (1 - 3*a0) * this->mu*Matrices::Isymm<DimM>() +
         2*this->mu * (a0 - a1)*Matrices::outer(N_star, N_star))});
    } else {
      return std::make_tuple(std::move(tau), T4Mat<Real, DimM>{this->C});
    }
  }



}  // muSpectre

#endif /* MATERIAL_HYPER_ELASTO_PLASTIC1_H */

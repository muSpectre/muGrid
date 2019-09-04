/**
 * @file   material_hyper_elasto_plastic2.hh
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *
 * @date   08 Jul 2019
 *
 * @brief  copy of material_hyper_elasto_plastic1 with Young, Poisson, yield
 *         criterion and  hardening modulus per pixel. As defined in de Geus
 *         2017 (https://doi.org/10.1016/j.cma.2016.12.032) and further
 *         explained in Geers 2003 (https://doi.org/10.1016/j.cma.2003.07.014).
 *
 * Copyright © 2019 Till Junge
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


#ifndef SRC_MATERIALS_MATERIAL_HYPER_ELASTO_PLASTIC2_HH_
#define SRC_MATERIALS_MATERIAL_HYPER_ELASTO_PLASTIC2_HH_

#include "materials/material_muSpectre_base.hh"
#include "materials/materials_toolbox.hh"

#include <libmugrid/eigen_tools.hh>
#include <libmugrid/mapped_field.hh>

#include <algorithm>

namespace muSpectre {

  template <Dim_t DimS, Dim_t DimM>
  class MaterialHyperElastoPlastic2;

  /**
   * traits for hyper-elastoplastic material
   */
  template <Dim_t DimS, Dim_t DimM>
  struct MaterialMuSpectre_traits<MaterialHyperElastoPlastic2<DimS, DimM>> {
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
    constexpr static auto strain_measure{StrainMeasure::Gradient};
    //! declare what type of stress measure your law yields as output
    constexpr static auto stress_measure{StressMeasure::Kirchhoff};

    //! local field collection used for internals
    using LFieldColl_t = muGrid::LocalFieldCollection<DimS>;

    //! storage type for plastic flow measure (εₚ in the papers)
    using LScalarMap_t =
        muGrid::StateFieldMap<muGrid::ScalarFieldMap<LFieldColl_t, Real>>;
    /**
     * storage type for for previous gradient Fᵗ and elastic left
     * Cauchy-Green deformation tensor bₑᵗ
     */
    using LStrainMap_t = muGrid::StateFieldMap<
        muGrid::MatrixFieldMap<LFieldColl_t, Real, DimM, DimM, false>>;

    //! local Young modulus, Poisson ratio, yield threshold, hardening parameter
    // using ScalarMap_t = muGrid::ScalarFieldMap<LFieldColl_t, Real, true>;

    // //! internal variables of hyper elasto plastic 2:(young, poisson, tau_y0,
    //     H)
    // using InternalVariables =
    //    std::tuple<ScalarMap_t, ScalarMap_t, ScalarMap_t, ScalarMap_t>
  };

  /**
   * material implementation for hyper-elastoplastic constitutive law.
   */
  template <Dim_t DimS, Dim_t DimM = DimS>
  class MaterialHyperElastoPlastic2
      : public MaterialMuSpectre<MaterialHyperElastoPlastic2<DimS, DimM>, DimS,
                                 DimM> {
   public:
    //! base class
    using Parent =
        MaterialMuSpectre<MaterialHyperElastoPlastic2<DimS, DimM>, DimS, DimM>;
    using T2_t = Eigen::Matrix<Real, DimM, DimM>;
    using T4_t = muGrid::T4Mat<Real, DimM>;

    /**
     * type used to determine whether the
     * `muSpectre::MaterialMuSpectre::iterable_proxy` evaluate only
     * stresses or also tangent stiffnesses
     */
    using NeedTangent = typename Parent::NeedTangent;

    //! shortcut to traits
    using traits = MaterialMuSpectre_traits<MaterialHyperElastoPlastic2>;

    //! storage type for scalar material constant fields
    using Field_t = muGrid::MappedScalarField<Real, DimS, true>;

    //! Hooke's law implementation
    using Hooke =
        typename MatTB::Hooke<DimM, typename traits::StrainMap_t::reference,
                              typename traits::TangentMap_t::reference>;

    //! type in which the previous strain state is referenced
    using StrainStRef_t = typename traits::LStrainMap_t::reference;
    //! type in which the previous plastic flow is referenced
    using FlowStRef_t = typename traits::LScalarMap_t::reference;

    //! Local FieldCollection type for field storage
    using LColl_t = muGrid::LocalFieldCollection<DimS>;

    //! Default constructor
    MaterialHyperElastoPlastic2() = delete;

    //! Constructor with name
    explicit MaterialHyperElastoPlastic2(std::string name);

    //! Copy constructor
    MaterialHyperElastoPlastic2(const MaterialHyperElastoPlastic2 & other) =
        delete;

    //! Move constructor
    MaterialHyperElastoPlastic2(MaterialHyperElastoPlastic2 && other) = delete;

    //! Destructor
    virtual ~MaterialHyperElastoPlastic2() = default;

    //! Copy assignment operator
    MaterialHyperElastoPlastic2 &
    operator=(const MaterialHyperElastoPlastic2 & other) = delete;

    //! Move assignment operator
    MaterialHyperElastoPlastic2 &
    operator=(MaterialHyperElastoPlastic2 && other) = delete;

    /**
     * evaluates Kirchhoff stress given the current placement gradient
     * Fₜ, the previous Gradient Fₜ₋₁ and the cumulated plastic flow
     * εₚ
     */
    T2_t evaluate_stress(const T2_t & F, StrainStRef_t F_prev,
                         StrainStRef_t be_prev, FlowStRef_t plast_flow,
                         const Real lambda, const Real mu, const Real tau_y0,
                         const Real H);
    /**
     * evaluates Kirchhoff stress given the local placement gradient and pixel
     * id.
     */
    T2_t evaluate_stress(const T2_t & F, const size_t & pixel_index) {
      auto && F_prev{this->F_prev_field[pixel_index]};
      auto && be_prev{this->be_prev_field[pixel_index]};
      auto && plast_flow{this->plast_flow_field[pixel_index]};
      auto && lambda{this->lambda_field[pixel_index]};
      auto && mu{this->mu_field[pixel_index]};
      auto && tau_y0{this->tau_y0_field[pixel_index]};
      auto && H{this->H_field[pixel_index]};
      return this->evaluate_stress(F, F_prev, be_prev, plast_flow, lambda, mu,
                                   tau_y0, H);
    }
    /**
     * evaluates Kirchhoff stress and tangent moduli given the current placement
     * gradient Fₜ, the previous Gradient Fₜ₋₁ and the cumulated plastic flow εₚ
     */
    std::tuple<T2_t, T4_t>
    evaluate_stress_tangent(const T2_t & F, StrainStRef_t F_prev,
                            StrainStRef_t be_prev, FlowStRef_t plast_flow,
                            const Real lambda, const Real mu, const Real tau_y0,
                            const Real H, const Real K);
    /**
     * evaluates Kirchhoff stressstiffness and tangent moduli given the local
     * placement gradient and pixel id.
     */
    std::tuple<T2_t, T4_t> evaluate_stress_tangent(const T2_t & F,
                                                   const size_t & pixel_index) {
      auto && F_prev{this->F_prev_field[pixel_index]};
      auto && be_prev{this->be_prev_field[pixel_index]};
      auto && plast_flow{this->plast_flow_field[pixel_index]};
      auto && lambda{this->lambda_field[pixel_index]};
      auto && mu{this->mu_field[pixel_index]};
      auto && tau_y0{this->tau_y0_field[pixel_index]};
      auto && H{this->H_field[pixel_index]};
      auto && K{this->K_field[pixel_index]};
      return this->evaluate_stress_tangent(F, F_prev, be_prev, plast_flow,
                                           lambda, mu, tau_y0, H, K);
    }

    /**
     * The statefields need to be cycled at the end of each load increment
     */
    void save_history_variables() override;

    /**
     * set the previous gradients to identity
     */
    void initialise() final;

    /**
     * overload add_pixel to write into loacal stiffness tensor
     */
    void add_pixel(const Ccoord_t<DimS> & pixel) final;

    /**
     * overload add_pixel to write into local stiffness tensor
     */
    void add_pixel(const Ccoord_t<DimS> & pixel, const Real & Youngs_modulus,
                   const Real & Poisson_ratio, const Real & tau_y0,
                   const Real & H);

    //! getter for internal variable field εₚ
    muGrid::StateField<muGrid::ScalarField<LColl_t, Real>> &
    get_plast_flow_field() {
      return this->plast_flow_field.get_field();
    }

    //! getter for previous gradient field Fᵗ
    muGrid::StateField<muGrid::TensorField<LColl_t, Real, secondOrder, DimM>> &
    get_F_prev_field() {
      return this->F_prev_field.get_field();
    }

    //! getterfor elastic left Cauchy-Green deformation tensor bₑᵗ
    muGrid::StateField<muGrid::TensorField<LColl_t, Real, secondOrder, DimM>> &
    get_be_prev_field() {
      return this->be_prev_field.get_field();
    }

    /**
     * needed to accomodate the static-sized member variable C, see
     * http://eigen.tuxfamily.org/dox-devel/group__TopicStructHavingEigenMembers.html
     */
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

   protected:
    /**
     * worker function computing stresses and internal variables
     */
    using Worker_t =
        std::tuple<T2_t, Real, Real, T2_t, bool, muGrid::Decomp_t<DimM>>;
    Worker_t stress_n_internals_worker(const T2_t & F, StrainStRef_t & F_prev,
                                       StrainStRef_t & be_prev,
                                       FlowStRef_t & plast_flow,
                                       const Real lambda, const Real mu,
                                       const Real tau_y0, const Real H);
    //! storage for cumulated plastic flow εₚ
    muGrid::MappedScalarStateField<Real, DimS> plast_flow_field;
    //! storage for previous gradient Fᵗ
    muGrid::MappedT2StateField<Real, DimS, DimM> F_prev_field;

    //! storage for elastic left Cauchy-Green deformation tensor bₑᵗ
    muGrid::MappedT2StateField<Real, DimS, DimM> be_prev_field;

    //! storage for first Lamé constant λ
    Field_t lambda_field;
    //! storage for second Lamé constant (shear modulus) μ
    Field_t mu_field;
    //! storage for initial yield stress
    Field_t tau_y0_field;
    //! storage for hardening modulus
    Field_t H_field;
    //! storage for Bulk modulus
    Field_t K_field;
  };

}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_HYPER_ELASTO_PLASTIC2_HH_

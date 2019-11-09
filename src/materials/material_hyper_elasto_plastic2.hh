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


#ifndef SRC_MATERIALS_MATERIAL_HYPER_ELASTO_PLASTIC2_HH_
#define SRC_MATERIALS_MATERIAL_HYPER_ELASTO_PLASTIC2_HH_

#include "materials/material_muSpectre_base.hh"
#include "materials/materials_toolbox.hh"

#include <libmugrid/eigen_tools.hh>
#include <libmugrid/mapped_state_nfield.hh>

#include <algorithm>

namespace muSpectre {

  template <Dim_t DimM>
  class MaterialHyperElastoPlastic2;

  /**
   * traits for hyper-elastoplastic material
   */
  template <Dim_t DimM>
  struct MaterialMuSpectre_traits<MaterialHyperElastoPlastic2<DimM>> {
    //! expected map type for strain fields
    using StrainMap_t = muGrid::T2NFieldMap<Real, Mapping::Const, DimM>;
    //! expected map type for stress fields
    using StressMap_t = muGrid::T2NFieldMap<Real, Mapping::Mut, DimM>;
    //! expected map type for tangent stiffness fields
    using TangentMap_t = muGrid::T4NFieldMap<Real, Mapping::Mut, DimM>;

    //! declare what type of strain measure your law takes as input
    constexpr static auto strain_measure{StrainMeasure::Gradient};
    //! declare what type of stress measure your law yields as output
    constexpr static auto stress_measure{StressMeasure::Kirchhoff};
  };

  /**
   * material implementation for hyper-elastoplastic constitutive law.
   */
  template <Dim_t DimM>
  class MaterialHyperElastoPlastic2
      : public MaterialMuSpectre<MaterialHyperElastoPlastic2<DimM>, DimM> {
   public:
    //! base class
    using Parent =
        MaterialMuSpectre<MaterialHyperElastoPlastic2<DimM>, DimM>;
    using T2_t = Eigen::Matrix<Real, DimM, DimM>;
    using T4_t = muGrid::T4Mat<Real, DimM>;


    //! shortcut to traits
    using traits = MaterialMuSpectre_traits<MaterialHyperElastoPlastic2>;

    //! storage type for scalar material constant fields
    using Field_t = muGrid::MappedScalarNField<Real, Mapping::Const>;

    //! Hooke's law implementation
    using Hooke =
        typename MatTB::Hooke<DimM, typename traits::StrainMap_t::reference,
                              typename traits::TangentMap_t::reference>;

    using FlowField_t = muGrid::MappedScalarStateNField<Real, Mapping::Mut>;
    using FlowField_ref = typename FlowField_t::Return_t;

    using PrevStrain_t = muGrid::MappedT2StateNField<Real, Mapping::Mut, DimM>;
    using PrevStrain_ref = typename PrevStrain_t::Return_t;

    //! Default constructor
    MaterialHyperElastoPlastic2() = delete;

    //! Constructor with name
    MaterialHyperElastoPlastic2(const std::string & name,
                                const Dim_t & spatial_dimension,
                                const Dim_t & nb_quad_pts);

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
    T2_t evaluate_stress(const T2_t & F, PrevStrain_ref  F_prev,
                         PrevStrain_ref  be_prev, FlowField_ref  plast_flow,
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
    // TODO(junge): Switch to PrevStrain_ref & (requires the iterator to hold a
    // dereferenced iterate
    std::tuple<T2_t, T4_t>
    evaluate_stress_tangent(const T2_t & F, PrevStrain_ref F_prev,
                            PrevStrain_ref be_prev, FlowField_ref plast_flow,
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
    void add_pixel(const size_t & pixel_id) final;

    /**
     * overload add_pixel to write into local stiffness tensor
     */
    void add_pixel(const size_t & pixel_id, const Real & Youngs_modulus,
                   const Real & Poisson_ratio, const Real & tau_y0,
                   const Real & H);

    //! getter for internal variable field εₚ
    muGrid::MappedScalarStateNField<Real, Mapping::Mut> &
    get_plast_flow_field() {
      return this->plast_flow_field;
    }

    //! getter for previous gradient field Fᵗ
    muGrid::MappedT2StateNField<Real, Mapping::Mut, DimM> &
    get_F_prev_field() {
      return this->F_prev_field;
    }

    //! getterfor elastic left Cauchy-Green deformation tensor bₑᵗ
    muGrid::MappedT2StateNField<Real, Mapping::Mut, DimM> &
    get_be_prev_field() {
      return this->be_prev_field;
    }

   protected:
    /**
     * worker function computing stresses and internal variables
     */
    using Worker_t =
        std::tuple<T2_t, Real, Real, T2_t, bool, muGrid::Decomp_t<DimM>>;
    Worker_t stress_n_internals_worker(const T2_t & F, PrevStrain_ref & F_prev,
                                       PrevStrain_ref & be_prev,
                                       FlowField_ref & plast_flow,
                                       const Real lambda, const Real mu,
                                       const Real tau_y0, const Real H);
    //! storage for cumulated plastic flow εₚ
    FlowField_t plast_flow_field;
    //! storage for previous gradient Fᵗ
    PrevStrain_t F_prev_field;

    //! storage for elastic left Cauchy-Green deformation tensor bₑᵗ
    PrevStrain_t be_prev_field;

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

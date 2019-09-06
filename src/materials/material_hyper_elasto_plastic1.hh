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
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#ifndef SRC_MATERIALS_MATERIAL_HYPER_ELASTO_PLASTIC1_HH_
#define SRC_MATERIALS_MATERIAL_HYPER_ELASTO_PLASTIC1_HH_

#include "materials/material_muSpectre_base.hh"
#include "materials/materials_toolbox.hh"

#include <libmugrid/eigen_tools.hh>
#include <libmugrid/mapped_nfield.hh>
#include <libmugrid/mapped_state_nfield.hh>

#include <algorithm>

namespace muSpectre {

  template <Dim_t DimM>
  class MaterialHyperElastoPlastic1;

  /**
   * traits for hyper-elastoplastic material
   */
  template <Dim_t DimM>
  struct MaterialMuSpectre_traits<MaterialHyperElastoPlastic1<DimM>> {
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
   * material implementation for hyper-elastoplastic constitutive law. Note for
   * developpers: this law is tested against a reference python implementation
   * in `py_comparison_test_material_hyper_elasto_plastic1.py`
   */
  template <Dim_t DimM>
  class MaterialHyperElastoPlastic1
      : public MaterialMuSpectre<MaterialHyperElastoPlastic1<DimM>, DimM> {
   public:
    //! base class
    using Parent = MaterialMuSpectre<MaterialHyperElastoPlastic1<DimM>, DimM>;
    using T2_t = Eigen::Matrix<Real, DimM, DimM>;
    using T4_t = muGrid::T4Mat<Real, DimM>;

    //! shortcut to traits
    using traits = MaterialMuSpectre_traits<MaterialHyperElastoPlastic1>;

    //! Hooke's law implementation
    using Hooke =
        typename MatTB::Hooke<DimM, typename traits::StrainMap_t::reference,
                              typename traits::TangentMap_t::reference>;

    //! type in which the previous strain state is referenced
    using T2StRef_t = typename muGrid::MappedT2StateNField<Real, Mapping::Mut,
                                                           DimM>::Return_t;
    //! type in which the previous plastic flow is referenced
    using ScalarStRef_t =
        typename muGrid::MappedScalarStateNField<Real, Mapping::Mut>::Return_t;

    //! Default constructor
    MaterialHyperElastoPlastic1() = delete;

    //! Constructor with name and material properties
    MaterialHyperElastoPlastic1(const std::string & name,
                                const Dim_t & spatial_dimension,
                                const Dim_t & nb_quad_pts, const Real & young,
                                const Real & poisson, const Real & tau_y0,
                                const Real & H);

    //! Copy constructor
    MaterialHyperElastoPlastic1(const MaterialHyperElastoPlastic1 & other) =
        delete;

    //! Move constructor
    MaterialHyperElastoPlastic1(MaterialHyperElastoPlastic1 && other) = delete;

    //! Destructor
    virtual ~MaterialHyperElastoPlastic1() = default;

    //! Copy assignment operator
    MaterialHyperElastoPlastic1 &
    operator=(const MaterialHyperElastoPlastic1 & other) = delete;

    //! Move assignment operator
    MaterialHyperElastoPlastic1 &
    operator=(MaterialHyperElastoPlastic1 && other) = delete;

    /**
     * evaluates Kirchhoff stress given the current placement gradient
     * Fₜ, the previous Gradient Fₜ₋₁ and the cumulated plastic flow
     * εₚ
     */
    T2_t evaluate_stress(const T2_t & F, T2StRef_t F_prev, T2StRef_t be_prev,
                         ScalarStRef_t plast_flow);
    /**
     * evaluates Kirchhoff stress given the local placement gradient and pixel
     * id.
     */
    T2_t evaluate_stress(const T2_t & F, const size_t & quad_pt_index) {
      auto && F_prev{this->F_prev_field[quad_pt_index]};
      auto && be_prev{this->be_prev_field[quad_pt_index]};
      auto && plast_flow{this->plast_flow_field[quad_pt_index]};
      return this->evaluate_stress(F, F_prev, be_prev, plast_flow);
    }
    /**
     * evaluates Kirchhoff stress and tangent moduli given the current placement
     * gradient Fₜ, the previous Gradient Fₜ₋₁ and the cumulated plastic flow εₚ
     */
    std::tuple<T2_t, T4_t> evaluate_stress_tangent(const T2_t & F,
                                                   T2StRef_t F_prev,
                                                   T2StRef_t be_prev,
                                                   ScalarStRef_t plast_flow);
    /**
     * evaluates Kirchhoff stressstiffness and tangent moduli given the local
     * placement gradient and pixel id.
     */
    std::tuple<T2_t, T4_t>
    evaluate_stress_tangent(const T2_t & F, const size_t & quad_pt_index) {
      auto && F_prev{this->F_prev_field[quad_pt_index]};
      auto && be_prev{this->be_prev_field[quad_pt_index]};
      auto && plast_flow{this->plast_flow_field[quad_pt_index]};
      return this->evaluate_stress_tangent(F, F_prev, be_prev, plast_flow);
    }

    /**
     * The statefields need to be cycled at the end of each load increment
     */
    void save_history_variables() override;

    /**
     * set the previous gradients to identity
     */
    void initialise() final;

    //! getter for internal variable field εₚ
    muGrid::TypedStateNField<Real> & get_plast_flow_field() {
      return this->plast_flow_field.get_state_field();
    }

    //! getter for previous gradient field Fᵗ
    muGrid::TypedStateNField<Real> & get_F_prev_field() {
      return this->F_prev_field.get_state_field();
    }

    //! getterfor elastic left Cauchy-Green deformation tensor bₑᵗ
    muGrid::TypedStateNField<Real> & get_be_prev_field() {
      return this->be_prev_field.get_state_field();
    }

   protected:
    /**
     * worker function computing stresses and internal variables
     */
    using Worker_t =
        std::tuple<T2_t, Real, Real, T2_t, bool, muGrid::Decomp_t<DimM>>;
    Worker_t stress_n_internals_worker(const T2_t & F, T2StRef_t & F_prev,
                                       T2StRef_t & be_prev,
                                       ScalarStRef_t & plast_flow);
    //! storage for cumulated plastic flow εₚ
    muGrid::MappedScalarStateNField<Real, Mapping::Mut> plast_flow_field;
    //! storage for previous gradient Fᵗ
    muGrid::MappedT2StateNField<Real, Mapping::Mut, DimM> F_prev_field;

    //! storage for elastic left Cauchy-Green deformation tensor bₑᵗ
    muGrid::MappedT2StateNField<Real, Mapping::Mut, DimM> be_prev_field;

    // material properties
    const Real young;    //!< Young's modulus
    const Real poisson;  //!< Poisson's ratio
    const Real lambda;   //!< first Lamé constant
    const Real mu;       //!< second Lamé constant (shear modulus)
    const Real K;        //!< Bulk modulus
    const Real tau_y0;   //!< initial yield stress
    const Real H;        //!< hardening modulus
    std::unique_ptr<const muGrid::T4Mat<Real, DimM>>
        C_holder;  //!< stiffness tensor
    const muGrid::T4Mat<Real, DimM> & C;
  };

}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_HYPER_ELASTO_PLASTIC1_HH_

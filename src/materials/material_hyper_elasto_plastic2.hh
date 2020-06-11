/**
 * @file   material_hyper_elasto_plastic2.hh
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de>
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   08 Apr 2020
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
#include "materials/material_hyper_elasto_plastic1.hh"

#include <libmugrid/eigen_tools.hh>
#include <libmugrid/mapped_state_field.hh>

#include <algorithm>

namespace muSpectre {

  template <Index_t DimM>
  class MaterialHyperElastoPlastic2;

  /**
   * traits for hyper-elastoplastic material
   */
  template <Index_t DimM>
  struct MaterialMuSpectre_traits<MaterialHyperElastoPlastic2<DimM>> {
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
    constexpr static auto strain_measure{StrainMeasure::Gradient};
    //! declare what type of stress measure your law yields as output
    constexpr static auto stress_measure{StressMeasure::Kirchhoff};
  };

  /**
   * material implementation for hyper-elastoplastic constitutive law.
   */
  template <Index_t DimM>
  class MaterialHyperElastoPlastic2
      : public MaterialMuSpectre<MaterialHyperElastoPlastic2<DimM>, DimM> {
   public:
    //! base class
    using Parent = MaterialMuSpectre<MaterialHyperElastoPlastic2<DimM>, DimM>;
    using T2_t = Eigen::Matrix<Real, DimM, DimM>;
    using T4_t = muGrid::T4Mat<Real, DimM>;

    //! shortcut to traits
    using traits = MaterialMuSpectre_traits<MaterialHyperElastoPlastic2>;

    //! storage type for scalar material constant fields
    using Field_t =
        muGrid::MappedScalarField<Real, Mapping::Const, IterUnit::SubPt>;

    //! Hooke's law implementation
    using Hooke =
        typename MatTB::Hooke<DimM, typename traits::StrainMap_t::reference,
                              typename traits::TangentMap_t::reference>;

    using FlowField_t =
        muGrid::MappedScalarStateField<Real, Mapping::Mut, IterUnit::SubPt>;
    using FlowField_ref = typename FlowField_t::Return_t;

    using PrevStrain_t = muGrid::MappedT2StateField<Real, Mapping::Mut, DimM,
                                                    IterUnit::SubPt>;
    using PrevStrain_ref = typename PrevStrain_t::Return_t;

    //! Default constructor
    MaterialHyperElastoPlastic2() = delete;

    //! Constructor with name
    MaterialHyperElastoPlastic2(const std::string & name,
                                const Index_t & spatial_dimension,
                                const Index_t & nb_quad_pts);

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
     * evaluates Kirchhoff stress given the local placement gradient and pixel
     * id.
     */

    T2_t evaluate_stress(const T2_t & F, PrevStrain_ref F_prev,
                         PrevStrain_ref be_prev, FlowField_ref eps_p,
                         const Real lambda, const Real mu, const Real tau_y0,
                         const Real H);

    T2_t evaluate_stress(const T2_t & F, const size_t & pixel_index) {
      auto && F_prev{this->get_F_prev_field()[pixel_index]};
      auto && be_prev{this->get_be_prev_field()[pixel_index]};
      auto && plast_flow{this->get_plast_flow_field()[pixel_index]};
      auto && lambda{this->lambda_field[pixel_index]};
      auto && mu{this->mu_field[pixel_index]};
      auto && tau_y0{this->tau_y0_field[pixel_index]};
      auto && H{this->H_field[pixel_index]};
      return this->evaluate_stress(F, F_prev, be_prev, plast_flow, lambda, mu,
                                   tau_y0, H);
    }

    /**
     * evaluates Kirchhoff stressstiffness and tangent moduli given the local
     * placement gradient and pixel id.
     */
    std::tuple<T2_t, T4_t>
    evaluate_stress_tangent(const T2_t & F, PrevStrain_ref F_prev,
                            PrevStrain_ref be_prev, FlowField_ref eps_p,
                            const Real lambda, const Real mu, const Real tau_y0,
                            const Real H, const Real K);

    std::tuple<T2_t, T4_t> evaluate_stress_tangent(const T2_t & F,
                                                   const size_t & pixel_index) {
      auto && F_prev{this->get_F_prev_field()[pixel_index]};
      auto && be_prev{this->get_be_prev_field()[pixel_index]};
      auto && plast_flow{this->get_plast_flow_field()[pixel_index]};
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
    FlowField_t & get_plast_flow_field() {
      return this->material_child.get_plast_flow_field();
    }

    //! getter for previous gradient field Fᵗ
    PrevStrain_t & get_F_prev_field() {
      return this->material_child.get_F_prev_field();
    }

    //! getter for elastic left Cauchy-Green deformation tensor bₑᵗ
    PrevStrain_t & get_be_prev_field() {
      return this->material_child.get_be_prev_field();
    }

    //! getter for the child material
    MaterialHyperElastoPlastic1<DimM> & get_material_child() {
      return this->material_child;
    }

   protected:
    // Childern material (used as a worker for evaluating stress and tangent)
    MaterialHyperElastoPlastic1<DimM> material_child;

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

/**
 * @file   material_dunant_max.hh
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   13 Jul 2020
 *
 * @brief  material used by dunant as a the damage model
 * The failure criteria is based on the maximum eigen value of the strain
 * tensor. The stress-strain curve is meant to be  and the material experiences
 * strain softening if its maximum eigen value (for strain tensor) exceeds a
 * certain threshold:
 **********************************
 *     σ                          *
 *     ^        ε_y               *
 * σ_y |- - - - /\                *
 *     |       /  \               *
 *     |      /    \              *
 *     |     /      \             *
 *     |    /        \            *
 *     |   /          \           *
 *     |  /            \          *
 *     | /              \         *
 *     |/                \        *
 *     ________________________>ε *
 *                       ε_f      *
 **********************************
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

#ifndef SRC_MATERIALS_MATERIAL_DUNANT_MAX_HH_
#define SRC_MATERIALS_MATERIAL_DUNANT_MAX_HH_

#include "materials/material_linear_elastic1.hh"
#include "materials/material_muSpectre_mechanics.hh"
#include "materials/materials_toolbox.hh"
#include "materials/stress_transformations_PK2.hh"

#include <libmugrid/mapped_state_field.hh>

namespace muSpectre {

  template <Index_t DimM>
  class MaterialDunantMax;

  /**
   * traits for objective linear visco_elasticity
   */
  template <Index_t DimM>
  struct MaterialMuSpectre_traits<MaterialDunantMax<DimM>>
      : public DefaultMechanics_traits<DimM, StrainMeasure::GreenLagrange,
                                       StressMeasure::PK2> {};

  template <Index_t DimM>
  class MaterialDunantMax
      : public MaterialMuSpectreMechanics<MaterialDunantMax<DimM>, DimM> {
   public:
    //! base class
    using Parent = MaterialMuSpectreMechanics<MaterialDunantMax, DimM>;

    //! short-hand for Vector type
    using Vec_t = Eigen::Matrix<Real, DimM, 1>;

    //! short-hand for second-rank tensors
    using T2_t = Eigen::Matrix<Real, DimM, DimM>;

    //! short-hand for fourth-rank tensors
    using T4_t = muGrid::T4Mat<Real, DimM>;

    //! traits of this material
    using traits = MaterialMuSpectre_traits<MaterialDunantMax>;

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

    using IntStRef_t =
        typename muGrid::MappedScalarStateField<muGrid::Int, Mapping::Mut,
                                                IterUnit::SubPt>::Return_t;

    using ScalarRef_t =
        typename muGrid::MappedScalarField<Real, Mapping::Mut,
                                           IterUnit::SubPt>::Return_t;

    using IntRef_t =
        typename muGrid::MappedScalarField<muGrid::Int, Mapping::Mut,
                                           IterUnit::SubPt>::Return_t;

    //! Type for the child material
    using MatChild_t = MaterialLinearElastic1<DimM>;

    //! Default constructor
    MaterialDunantMax() = delete;

    //! Copy constructor
    MaterialDunantMax(const MaterialDunantMax & other) = delete;

    //! Construct by name, Young's modulus and Poisson's ratio
    MaterialDunantMax(const std::string & name,
                      const Index_t & spatial_dimension,
                      const Index_t & nb_quad_pts, const Real & young,
                      const Real & poisson, const Real & kappa_init,
                      const Real & alpha,
                      const std::shared_ptr<muGrid::LocalFieldCollection> &
                          parent_field_collection = nullptr);

    //! Move constructor
    MaterialDunantMax(MaterialDunantMax && other) = delete;

    //! Destructor
    virtual ~MaterialDunantMax() = default;

    //! Copy assignment operator
    MaterialDunantMax & operator=(const MaterialDunantMax & other) = delete;

    //! Move assignment operator
    MaterialDunantMax & operator=(MaterialDunantMax && other) = delete;

    /**
     * evaluates Kirchhoff stress given the current placement gradient
     * Fₜ,
     */
    T2_t evaluate_stress(const T2_t & E, ScalarStRef_t kappa,
                         const Real & kappa_init);
    /**
     * evaluates Kirchhoff stress given the local placement gradient and pixel
     * id.
     */
    T2_t evaluate_stress(const T2_t & E, const size_t & quad_pt_index);

    /**
     * update the damage_measure varaible before proceeding with stress
     * evaluation
     */
    StepState update_damage_measure(const T2_t & E, ScalarStRef_t kappa);

    /**
     * evaluates Kirchhoff stress and tangent moduli given the ...
     */
    std::tuple<T2_t, T4_t> evaluate_stress_tangent(const T2_t & E,
                                                   ScalarStRef_t kappa,
                                                   const Real & kappa_init);
    /**
     * evaluates Kirchhoff stress stiffness and tangent moduli given the local
     * placement gradient and pixel id.
     */
    std::tuple<T2_t, T4_t>
    evaluate_stress_tangent(const T2_t & E, const size_t & quad_pt_index);

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
     * @param      initial strain_measure (Real)
     *
     * @return     damage measure (Real)
     */
    Real compute_reduction(const Real & kappa, const Real & kappa_init_f);

    /**
     * The statefields need to be cycled at the end of each load increment
     */
    void save_history_variables() final;

    /**
     * set the previous gradients to identity
     */
    void initialise() final;

    /**
     * overload add_pixel
     */
    void add_pixel(const size_t & pixel_index) final;

    /**
     * overload add_pixel to write into eigenstrain
     */
    void add_pixel(const size_t & pixel_index, const Real & kappa_variation);

    //! getter for internal variable field of strain measure
    inline muGrid::MappedScalarStateField<Real, Mapping::Mut, IterUnit::SubPt> &
    get_kappa_field() {
      return this->kappa_field;
    }

    //! getter for internal variable field of strain measure
    inline muGrid::MappedScalarField<Real, Mapping::Mut, IterUnit::SubPt> &
    get_kappa_init_field() {
      return this->kappa_init_field;
    }

    //! getter for kappa_init
    inline Real get_kappa_init() { return this->kappa_init; }

    //! restarting the last step nonlinear bool
    void clear_last_step_nonlinear() final;

   protected:
    // Child material (used as a worker for evaluating stress and tangent)
    MatChild_t material_child;

    //! storage for initial kappa field
    muGrid::MappedScalarField<Real, Mapping::Mut, IterUnit::SubPt>
        kappa_init_field;

    //! storage for damage variable
    muGrid::MappedScalarStateField<Real, Mapping::Mut, IterUnit::SubPt>
        kappa_field;

    // damage evolution parameters:
    const Real kappa_init;  //!< threshold of damage (strength)
    const Real kappa_fin;   //!< threshold of damage (strength)
    const Real
        alpha;  //! damage evaluation parameter( recommended to be in [0, 1])
  };

  /* ---------------------------------------------------------------------- */
}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_DUNANT_MAX_HH_

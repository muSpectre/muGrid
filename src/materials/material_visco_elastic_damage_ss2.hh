/**
 * @file   material_visco_elastic_damage_ss2.hh
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   29 Apr 2020
 *
 * @brief  material with viscosity and damage able to handle pixel-wise random
 * field as the damage threshold
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

#ifndef SRC_MATERIALS_MATERIAL_VISCO_ELASTIC_DAMAGE_SS2_HH_
#define SRC_MATERIALS_MATERIAL_VISCO_ELASTIC_DAMAGE_SS2_HH_

#include "materials/material_visco_elastic_damage_ss1.hh"

#include <libmugrid/mapped_field.hh>

#include <Eigen/Dense>

namespace muSpectre {

  template <Index_t DimM>
  class MaterialViscoElasticDamageSS2;

  /**
   * traits for objective linear visco-elasticity with damage and per-pixel
   * strength
   */
  template <Index_t DimM>
  struct MaterialMuSpectre_traits<MaterialViscoElasticDamageSS2<DimM>>
      : public DefaultMechanics_traits<DimM, StrainMeasure::GreenLagrange,
                                       StressMeasure::PK2> {};

  /**
   * objective linear visco-elasticity with damage and per-pixel  strength
   * DimM material_dimension (dimension of constitutive law)
   */
  template <Index_t DimM>
  class MaterialViscoElasticDamageSS2
      : public MaterialMuSpectreMechanics<MaterialViscoElasticDamageSS2<DimM>,
                                          DimM> {
   public:
    //! base class
    using Parent =
        MaterialMuSpectreMechanics<MaterialViscoElasticDamageSS2, DimM>;

    //! traits of this material
    using traits = MaterialMuSpectre_traits<MaterialViscoElasticDamageSS2>;

    //! reference to any type that casts to a matrix
    using StrainTensor = Eigen::Ref<const Eigen::Matrix<Real, DimM, DimM>>;

    //! Default constructor
    MaterialViscoElasticDamageSS2() = delete;

    //! Construct by name, Young's modulus and Poisson's ratio
    MaterialViscoElasticDamageSS2(const std::string & name,
                                  const Index_t & spatial_dimension,
                                  const Index_t & nb_quad_pts,
                                  const Real & young_inf, const Real & young_v,
                                  const Real & eta_v, const Real & poisson,
                                  const Real & kappa_init, const Real & alpha,
                                  const Real & beta, const Real & dt);

    //! Copy constructor
    MaterialViscoElasticDamageSS2(const MaterialViscoElasticDamageSS2 & other) =
        delete;

    //! Move constructor
    MaterialViscoElasticDamageSS2(MaterialViscoElasticDamageSS2 && other) =
        delete;

    //! Destructor
    virtual ~MaterialViscoElasticDamageSS2() = default;

    //! Copy assignment operator
    MaterialViscoElasticDamageSS2 &
    operator=(const MaterialViscoElasticDamageSS2 & other) = delete;

    //! Move assignment operator
    MaterialViscoElasticDamageSS2 &
    operator=(MaterialViscoElasticDamageSS2 && other) = delete;

    /**
     * The statefields need to be cycled at the end of each load increment
     */
    void save_history_variables() final;

    /**
     * set the previous gradients to identity
     */
    void initialise() final;

    /**
     * evaluates second Piola-Kirchhoff stress given the Green-Lagrange
     * strain (or Cauchy stress if called with a small strain tensor)
     */
    template <class Derived>
    inline decltype(auto) evaluate_stress(const Eigen::MatrixBase<Derived> & E,
                                          const size_t & quad_pt_index);

    /**
     * evaluates both second Piola-Kirchhoff stress and stiffness given
     * the Green-Lagrange strain (or Cauchy stress and stiffness if
     * called with a small strain tensor)
     */
    template <class Derived>
    inline decltype(auto)
    evaluate_stress_tangent(const Eigen::MatrixBase<Derived> & E,
                            const size_t & quad_pt_index);

    /**
     * overload add_pixel to write into eigenstrain
     */
    void add_pixel(const size_t & pixel_index) final;

    /**
     * overload add_pixel to write into random field of strengths
     */
    void add_pixel(const size_t & pixel_index,
                   const Real & kappa_variation = 0);

    //! getter for internal variable field History Integral
    inline muGrid::MappedT2StateField<Real, Mapping::Mut, DimM,
                                      IterUnit::SubPt> &
    get_history_integral() {
      return this->material_child.get_history_integral();
    }

    //! getter for internal variable field of static stress
    inline muGrid::MappedT2StateField<Real, Mapping::Mut, DimM,
                                      IterUnit::SubPt> &
    get_s_null_prev_field() {
      return this->material_child.get_s_null_prev_field();
    }

    //! getter for internal variable field  strain threshold
    inline muGrid::MappedScalarStateField<Real, Mapping::Mut, IterUnit::SubPt> &
    get_kappa_field() {
      return this->material_child.get_kappa_field();
    }

   protected:
    //! viscoelastic damage material  used to compute response
    MaterialViscoElasticDamageSS1<DimM> material_child;
  };

  /* ----------------------------------------------------------------------*/
  template <Index_t DimM>
  template <class Derived>
  auto MaterialViscoElasticDamageSS2<DimM>::evaluate_stress(
      const Eigen::MatrixBase<Derived> & E, const size_t & quad_pt_index)
      -> decltype(auto) {
    return this->material_child.evaluate_stress(E, quad_pt_index);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class Derived>
  auto MaterialViscoElasticDamageSS2<DimM>::evaluate_stress_tangent(
      const Eigen::MatrixBase<Derived> & E, const size_t & quad_pt_index)
      -> decltype(auto) {
    return this->material_child.evaluate_stress_tangent(E, quad_pt_index);
  }

}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_VISCO_ELASTIC_DAMAGE_SS2_HH_

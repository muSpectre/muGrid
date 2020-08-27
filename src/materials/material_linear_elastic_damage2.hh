/**
 * @file   material_linear_elastic_damage2.hh
 *
 * @author Ali Falsafi <ali.falsafi@epfl.ch>
 *
 * @date   04 May 2020
 *
 * @brief  MaterialLinearElasticDamage1 with  the ability
 * to handle damage strength parameter as a field (for attributing randomness)
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

#ifndef SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC_DAMAGE2_HH_
#define SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC_DAMAGE2_HH_

#include "materials/material_linear_elastic_damage1.hh"

#include <libmugrid/mapped_field.hh>

#include <Eigen/Dense>

namespace muSpectre {

  template <Index_t DimM>
  class MaterialLinearElasticDamage2;

  /**
   * traits for objective linear elastic damage material with variable strength
   */
  template <Index_t DimM>
  struct MaterialMuSpectre_traits<MaterialLinearElasticDamage2<DimM>>
      : public DefaultMechanics_traits<DimM, StrainMeasure::GreenLagrange,
                                       StressMeasure::PK2> {};

  /**
   * implements objective linear elasticity with strength measure per pixel
   */
  template <Index_t DimM>
  class MaterialLinearElasticDamage2
      : public MaterialMuSpectreMechanics<MaterialLinearElasticDamage2<DimM>,
                                          DimM> {
   public:
    //! base class
    using Parent =
        MaterialMuSpectreMechanics<MaterialLinearElasticDamage2, DimM>;

    //! traits of this material
    using traits = MaterialMuSpectre_traits<MaterialLinearElasticDamage2>;

    //! reference to any type that casts to a matrix
    using StrainTensor = Eigen::Ref<const Eigen::Matrix<Real, DimM, DimM>>;

    //! Default constructor
    MaterialLinearElasticDamage2() = delete;

    //! Construct by name, Young's modulus and Poisson's ratio
    MaterialLinearElasticDamage2(const std::string & name,
                                 const Index_t & spatial_dimension,
                                 const Index_t & nb_quad_pts,
                                 const Real & young, const Real & poisson,
                                 const Real & kappa_init, const Real & alpha,
                                 const Real & beta);

    //! Copy constructor
    MaterialLinearElasticDamage2(const MaterialLinearElasticDamage2 & other) =
        delete;

    //! Move constructor
    MaterialLinearElasticDamage2(MaterialLinearElasticDamage2 && other) =
        delete;

    //! Destructor
    virtual ~MaterialLinearElasticDamage2() = default;

    //! Copy assignment operator
    MaterialLinearElasticDamage2 &
    operator=(const MaterialLinearElasticDamage2 & other) = delete;

    //! Move assignment operator
    MaterialLinearElasticDamage2 &
    operator=(MaterialLinearElasticDamage2 && other) = delete;

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
     * overload add_pixel to write into eigenstrain
     */
    void

    add_pixel(const size_t & pixel_index, const Real & kappa_variation);

    //! getter for internal variable field  strain threshold
    inline muGrid::MappedScalarStateField<Real, Mapping::Mut, IterUnit::SubPt> &
    get_kappa_field() {
      return this->material_child.get_kappa_field();
    }

   protected:
    //! damage material without eigen strain: used to compute
    //! response
    MaterialLinearElasticDamage1<DimM> material_child;
  };

  /* ----------------------------------------------------------------------*/
  template <Index_t DimM>
  template <class Derived>
  auto MaterialLinearElasticDamage2<DimM>::evaluate_stress(
      const Eigen::MatrixBase<Derived> & E, const size_t & quad_pt_index)
      -> decltype(auto) {
    return this->material_child.evaluate_stress(E, quad_pt_index);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class Derived>
  auto MaterialLinearElasticDamage2<DimM>::evaluate_stress_tangent(
      const Eigen::MatrixBase<Derived> & E, const size_t & quad_pt_index)
      -> decltype(auto) {
    return this->material_child.evaluate_stress_tangent(E, quad_pt_index);
  }

}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC_DAMAGE2_HH_

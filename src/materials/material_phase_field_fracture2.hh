/**
 * @file   material_phase_field_fracture2.hh
 *
 * @author W. Beck Andrews <william.beck.andrews@imtek.uni-freiburg.de>
 *
 * @date   02 Dec 2021
 *
 * @brief Material for solving the elasticity subproblem of a phase field
 *        fracture model using a decomposition of the elastic energy density
 *        into volumetric and deviatoric parts based on components of the
 *        (small-strain) strain tensor.  The phase field phi couples to the
 *        deviatoric part in all cases, and to the volumetric part when its
 *        sign is positive.  This decomposition was proposed in Amor et al.,
 *        2009.
 *
 * Copyright © 2021 W. Beck Andrews
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

#ifndef SRC_MATERIALS_MATERIAL_PHASE_FIELD_FRACTURE2_HH_
#define SRC_MATERIALS_MATERIAL_PHASE_FIELD_FRACTURE2_HH_

#include "materials/material_linear_elastic1.hh"
#include "libmugrid/mapped_field.hh"
#include "libmugrid/eigen_tools.hh"

#include <Eigen/Dense>

namespace muSpectre {

  template <Index_t DimM>
  class MaterialPhaseFieldFracture2;

  /**
   * traits for objective linear elasticity with eigenstrain
   */
  template <Index_t DimM>
  struct MaterialMuSpectre_traits<MaterialPhaseFieldFracture2<DimM>>
      : public DefaultMechanics_traits<DimM, StrainMeasure::GreenLagrange,
                                       StressMeasure::PK2> {};

  /**
   * implements objective linear elasticity with an eigenstrain per pixel
   */
  template <Index_t DimM>
  class MaterialPhaseFieldFracture2 : public
      MaterialMuSpectreMechanics<MaterialPhaseFieldFracture2<DimM>, DimM> {
   public:
    //! base class
    using Parent = MaterialMuSpectreMechanics<MaterialPhaseFieldFracture2,
        DimM>;
    //! global field collection

    using Stiffness_t =
        Eigen::TensorFixedSize<Real, Eigen::Sizes<DimM, DimM, DimM, DimM>>;

    //! short-hand for second-rank tensors
    using T2_t = Eigen::Matrix<Real, DimM, DimM>;

    //! short-hand for fourth-rank tensors
    using T4_t = muGrid::T4Mat<Real, DimM>;

    //! traits of this material
    using traits = MaterialMuSpectre_traits<MaterialPhaseFieldFracture2>;

    //! storage type for Lamé constants
    using Field_t =
        muGrid::MappedScalarField<Real, Mapping::Mut, IterUnit::SubPt>;

    //! Hooke's law implementation
    using Hooke =
        typename MatTB::Hooke<DimM, typename traits::StrainMap_t::reference,
                              typename traits::TangentMap_t::reference>;

    //! Default constructor
    MaterialPhaseFieldFracture2() = delete;

    //! Construct by name
    explicit MaterialPhaseFieldFracture2(const std::string & name,
                                    const Index_t & spatial_dimension,
                                    const Index_t & nb_quad_pts,
                                    const Real & ksmall);

    //! Copy constructor
    MaterialPhaseFieldFracture2(const MaterialPhaseFieldFracture2 & other)
        = delete;

    //! Move constructor
    MaterialPhaseFieldFracture2(MaterialPhaseFieldFracture2 && other) = delete;

    //! Destructor
    virtual ~MaterialPhaseFieldFracture2() = default;

    //! Copy assignment operator
    MaterialPhaseFieldFracture2 &
    operator=(const MaterialPhaseFieldFracture2 & other) = delete;

    //! Move assignment operator
    MaterialPhaseFieldFracture2 &
    operator=(MaterialPhaseFieldFracture2 && other) = delete;

    /**
     * Evaluates the Cauchy stress given the Green-Lagrange strain, Lamé
     * constants, phase field, and a small parameter ksmall that is the
     * minimum stiffness of the fully fractured material.
     */
    T2_t evaluate_stress(const Eigen::Ref<const T2_t> & E,
                        const Real & lambda, const Real & mu,
                        const Real & phi, const Real & ksmall);

    //! Wrapper for stress evaluation on quad point.
    template <class Derived>
    inline decltype(auto) evaluate_stress(const Eigen::MatrixBase<Derived> & E,
                                          const size_t & quad_pt_index) {
      auto && lambda{this->lambda_field[quad_pt_index]};
      auto && mu{this->mu_field[quad_pt_index]};
      auto && phi{this->phase_field[quad_pt_index]};
      return this->evaluate_stress(std::move(E), lambda, mu, phi, this->ksmall);
    }

    /**
     * Evaluates the Cauchy stress and stress tangent given the Green-Lagrange
     * strain, Lamé constants, phase field, and a small parameter ksmall that
     * is the minimum stiffness of the fully fractured material.  System is
     * highly non-linear due to anisotropic response between tension and
     * compression.
     */
    std::tuple<T2_t, T4_t>
    evaluate_stress_tangent(const Eigen::Ref<const T2_t> & E,
                            const Real & lambda, const Real & mu,
                            const Real & phi, const Real & ksmall);

    //! Wrapper for stress/stress tangent evaluation on quad point.
    template <class Derived>
    inline decltype(auto)
    evaluate_stress_tangent(const Eigen::MatrixBase<Derived> & E,
                            const size_t & quad_pt_index) {
      auto && lambda{this->lambda_field[quad_pt_index]};
      auto && mu{this->mu_field[quad_pt_index]};
      auto && phi{this->phase_field[quad_pt_index]};
      return this->evaluate_stress_tangent(std::move(E), lambda, mu, phi,
          this->ksmall);
    }

    /**
     * overload add_pixel to write into local stiffness tensor
     */
    void add_pixel(const size_t & pixel_index) final;

    /**
     * overload add_pixel to write into local stiffness tensor
     */
    void add_pixel(const size_t & pixel_index, const Real & Youngs_modulus,
                   const Real & Poisson_ratio, const Real & phase_field);

    /**
     * (re)set the Youngs modulus on a quad_point with quad_point_id
     * The internal stored first and second Lame constants are updated due to
     * the update of the Youngs modulus.
     */
    void set_youngs_modulus(const size_t & quad_pt_id,
                            const Real & Youngs_modulus);

    /**
     * (re)set the Poisson ratio on a quad_point with quad_point_id
     * The internal stored first and second Lame constants are updated due to
     * the update of the Poisson's ratio.
     */
    void set_poisson_ratio(const size_t & quad_pt_id,
                           const Real & Poisson_ratio);

    /**
     * (re)set the phase field on a quad_point with quad_point_id
     */
    void set_phase_field(const size_t & quad_pt_id,
                            const Real & phase_field);

    /**
     * get the Youngs modulus on a quad_point with quad_point_id
     * Youngs modulus is computed from the internal stored first and second
     * lame constant.
     */
    Real get_youngs_modulus(const size_t & quad_pt_id);

    /**
     * get the Poisson ratio on a quad_point with quad_point_id
     * Poissons ratio is computed from the internal stored first and second
     * lame constant.
     */
    Real get_poisson_ratio(const size_t & quad_pt_id);

    /**
     * get the phase field on a quad_point with quad_point_id
     */
    Real get_phase_field(const size_t & quad_pt_id);


   protected:
    //! storage for first Lamé constant λ
    Field_t lambda_field;
    //! storage for second Lamé constant (shear modulus) μ
    Field_t mu_field;
    //! storage for phase field
    Field_t phase_field;

    const Real ksmall;    //!< minimum value for damage interpolation function
  };



}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_PHASE_FIELD_FRACTURE2_HH_

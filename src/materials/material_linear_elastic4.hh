/**
 * @file   material_linear_elastic4.hh
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *
 * @date   15 March 2018
 *
 * @brief linear elastic material with distribution of stiffness properties.
 *        In difference to material_linear_elastic3 two Lame constants are
 *        stored per pixel instead of the whole elastic matrix C.
 *        Uses the MaterialMuSpectre facilities to keep it simple.
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

#ifndef SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC4_HH_
#define SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC4_HH_

#include "materials/material_linear_elastic1.hh"
#include "libmugrid/mapped_field.hh"

#include <Eigen/Dense>

namespace muSpectre {

  template <Index_t DimM>
  class MaterialLinearElastic4;

  /**
   * traits for objective linear elasticity with eigenstrain
   */
  template <Index_t DimM>
  struct MaterialMuSpectre_traits<MaterialLinearElastic4<DimM>> {
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
   * implements objective linear elasticity with an eigenstrain per pixel
   */
  template <Index_t DimM>
  class MaterialLinearElastic4
      : public MaterialMuSpectre<MaterialLinearElastic4<DimM>, DimM> {
   public:
    //! base class
    using Parent = MaterialMuSpectre<MaterialLinearElastic4, DimM>;
    /**
     * type used to determine whether the
     * `muSpectre::MaterialMuSpectre::iterable_proxy` evaluate only
     * stresses or also tangent stiffnesses
     */
    using NeedTangent = typename Parent::NeedTangent;
    //! global field collection

    using Stiffness_t =
        Eigen::TensorFixedSize<Real, Eigen::Sizes<DimM, DimM, DimM, DimM>>;

    //! traits of this material
    using traits = MaterialMuSpectre_traits<MaterialLinearElastic4>;

    //! storage type for Lamé constants
    using Field_t =
        muGrid::MappedScalarField<Real, Mapping::Const, IterUnit::SubPt>;

    //! Hooke's law implementation
    using Hooke =
        typename MatTB::Hooke<DimM, typename traits::StrainMap_t::reference,
                              typename traits::TangentMap_t::reference>;

    //! Default constructor
    MaterialLinearElastic4() = delete;

    //! Construct by name
    explicit MaterialLinearElastic4(const std::string & name,
                                    const Index_t & spatial_dimension,
                                    const Index_t & nb_quad_pts);

    //! Copy constructor
    MaterialLinearElastic4(const MaterialLinearElastic4 & other) = delete;

    //! Move constructor
    MaterialLinearElastic4(MaterialLinearElastic4 && other) = delete;

    //! Destructor
    virtual ~MaterialLinearElastic4() = default;

    //! Copy assignment operator
    MaterialLinearElastic4 &
    operator=(const MaterialLinearElastic4 & other) = delete;

    //! Move assignment operator
    MaterialLinearElastic4 &
    operator=(MaterialLinearElastic4 && other) = delete;

    /**
     * evaluates second Piola-Kirchhoff stress given the Green-Lagrange
     * strain (or Cauchy stress if called with a small strain tensor), the first
     * Lame constant (lambda) and the second Lame constant (shear modulus/mu).
     */
    template <class Derived>
    inline decltype(auto) evaluate_stress(const Eigen::MatrixBase<Derived> & E,
                                          const Real & lambda, const Real & mu);
    /**
     * evaluates second Piola-Kirchhoff stress given the Green-Lagrange
     * strain (or Cauchy stress if called with a small strain tensor), and the
     * local pixel id.
     */
    template <class Derived>
    inline decltype(auto) evaluate_stress(const Eigen::MatrixBase<Derived> & E,
                                          const size_t & quad_pt_index) {
      auto && lambda{this->lambda_field[quad_pt_index]};
      auto && mu{this->mu_field[quad_pt_index]};
      return this->evaluate_stress(E, lambda, mu);
    }

    /**
     * evaluates both second Piola-Kirchhoff stress and stiffness given
     * the Green-Lagrange strain (or Cauchy stress and stiffness if
     * called with a small strain tensor), the first Lame constant (lambda) and
     * the second Lame constant (shear modulus/mu).
     */
    template <class Derived>
    inline decltype(auto)
    evaluate_stress_tangent(const Eigen::MatrixBase<Derived> & E,
                            const Real & lambda, const Real & mu);

    /**
     * evaluates both second Piola-Kirchhoff stress and stiffness given
     * the Green-Lagrange strain (or Cauchy stress and tangent moduli if
     * called with a small strain tensor), and the local pixel id.
     */

    template <class Derived>
    inline decltype(auto)
    evaluate_stress_tangent(const Eigen::MatrixBase<Derived> & E,
                            const size_t & quad_pt_index) {
      auto && lambda{this->lambda_field[quad_pt_index]};
      auto && mu{this->mu_field[quad_pt_index]};
      return this->evaluate_stress_tangent(E, lambda, mu);
    }

    /**
     * overload add_pixel to write into loacal stiffness tensor
     */
    void add_pixel(const size_t & pixel_index) final;

    /**
     * overload add_pixel to write into local stiffness tensor
     */
    void add_pixel(const size_t & pixel_index, const Real & Youngs_modulus,
                   const Real & Poisson_ratio);

   protected:
    //! storage for first Lamé constant λ
    Field_t lambda_field;
    //! storage for second Lamé constant (shear modulus) μ
    Field_t mu_field;
  };

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class Derived>
  auto MaterialLinearElastic4<DimM>::evaluate_stress(
      const Eigen::MatrixBase<Derived> & E, const Real & lambda,
      const Real & mu) -> decltype(auto) {
    auto C = Hooke::compute_C_T4(lambda, mu);
    return Matrices::tensmult(C, E);
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  template <class Derived>
  auto MaterialLinearElastic4<DimM>::evaluate_stress_tangent(
      const Eigen::MatrixBase<Derived> & E, const Real & lambda,
      const Real & mu) -> decltype(auto) {
    muGrid::T4Mat<Real, DimM> C = Hooke::compute_C_T4(lambda, mu);
    return std::make_tuple(this->evaluate_stress(E, lambda, mu), C);
  }

}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_LINEAR_ELASTIC4_HH_

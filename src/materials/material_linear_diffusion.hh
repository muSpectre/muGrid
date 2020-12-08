/**
 * @file   material_linear_diffusion.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   15 Jun 2020
 *
 * @brief  standard linear diffusion equation, e.g., for heat flow problems
 *
 * Copyright © 2020 Till Junge
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

#ifndef SRC_MATERIALS_MATERIAL_LINEAR_DIFFUSION_HH_
#define SRC_MATERIALS_MATERIAL_LINEAR_DIFFUSION_HH_

#include "common/muSpectre_common.hh"
#include "material_muSpectre.hh"

#include <Eigen/Dense>

#include <memory>

namespace muSpectre {

  // forward declaration
  template <Index_t DimM>
  class MaterialLinearDiffusion;

  /**
   * traits for linear diffusion law
   */
  template <Index_t DimM>
  struct MaterialMuSpectre_traits<MaterialLinearDiffusion<DimM>>
      : public DefaultScalar_traits<DimM> {};

  /**
   * implements linear diffusion
   * @tparam DimM dimension of constitutive law
   */
  template <Index_t DimM>
  class MaterialLinearDiffusion
      : public MaterialMuSpectre<MaterialLinearDiffusion<DimM>, DimM> {
   public:
    //! base class
    using Parent = MaterialMuSpectre<MaterialLinearDiffusion, DimM>;
    using traits = MaterialMuSpectre_traits<MaterialLinearDiffusion>;
    using Tangent_t = typename traits::Tangent_t;

    //! Default constructor
    MaterialLinearDiffusion() = delete;

    //! constructor for isotropic material
    MaterialLinearDiffusion(
        const std::string & name, const Index_t & spatial_dimension,
        const Index_t & nb_quad_pts, const Real & diffusion_coeff,
        const muGrid::PhysicsDomain & domain = muGrid::PhysicsDomain::heat());

    //! constructor for anisotropic material
    MaterialLinearDiffusion(
        const std::string & name, const Index_t & spatial_dimension,
        const Index_t & nb_quad_pts,
        const Eigen::Ref<const Tangent_t> & diffusion_coeff,
        const muGrid::PhysicsDomain & domain = muGrid::PhysicsDomain::heat());

    //! Copy constructor
    MaterialLinearDiffusion(const MaterialLinearDiffusion & other) = delete;

    //! Move constructor
    MaterialLinearDiffusion(MaterialLinearDiffusion && other) = default;

    //! Destructor
    virtual ~MaterialLinearDiffusion() = default;

    //! Copy assignment operator
    MaterialLinearDiffusion &
    operator=(const MaterialLinearDiffusion & other) = delete;

    //! Move assignment operator
    MaterialLinearDiffusion &
    operator=(MaterialLinearDiffusion && other) = delete;

    //! evaluate the flux
    template <class Derived, Dim_t test_dim = Derived::SizeAtCompileTime>
    decltype(auto) evaluate_stress(const Eigen::MatrixBase<Derived> & grad,
                                   const size_t & /*quad_pt_index*/) {
      static_assert(Derived::SizeAtCompileTime == DimM, "wrong size for grad");
      return this->A * grad;
    }

    //! evaluate the flux and return both flux and tangent moduli

    template <class Derived>
    decltype(auto)
    evaluate_stress_tangent(const Eigen::MatrixBase<Derived> & grad,
                            const size_t & quad_pt_index) {
      return std::make_tuple(this->evaluate_stress(grad, quad_pt_index),
                             this->A);
    }

    muGrid::PhysicsDomain get_physics_domain() const final;

    //! returns reference to the tangent moduli
    const Tangent_t & get_diffusion_coeff() const;

   protected:
    //! unique ptr to the conductivity/diffusitivy matrix
    std::unique_ptr<const Tangent_t> A_holder;
    const Tangent_t & A;
    muGrid::PhysicsDomain physics_domain;
  };

}  // namespace muSpectre

#endif  // SRC_MATERIALS_MATERIAL_LINEAR_DIFFUSION_HH_

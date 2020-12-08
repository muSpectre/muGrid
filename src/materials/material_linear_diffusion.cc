/**
 * @file   material_linear_diffusion.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   15 Jun 2020
 *
 * @brief  Implementation of diffusion law
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

#include "material_linear_diffusion.hh"

#include <Eigen/Eigenvalues>

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  MaterialLinearDiffusion<DimM>::MaterialLinearDiffusion(
      const std::string & name, const Index_t & spatial_dimension,
      const Index_t & nb_quad_pts, const Real & diffusion_coeff,
      const muGrid::PhysicsDomain & domain)
      : Parent{name, spatial_dimension, nb_quad_pts},
        A_holder{std::make_unique<Tangent_t>(diffusion_coeff *
                                             Tangent_t::Identity())},
        A{*this->A_holder}, physics_domain{domain} {
    this->last_step_was_nonlinear = false;

    if (diffusion_coeff < 0.) {
      std::stringstream error_message{};
      error_message
          << "The diffusion coefficient has to be positive, you provided "
          << diffusion_coeff << ".";
      throw MaterialError{error_message.str()};
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  MaterialLinearDiffusion<DimM>::MaterialLinearDiffusion(
      const std::string & name, const Index_t & spatial_dimension,
      const Index_t & nb_quad_pts,
      const Eigen::Ref<const Tangent_t> & diffusion_coeff,
      const muGrid::PhysicsDomain & domain)
      : Parent{name, spatial_dimension, nb_quad_pts},
        A_holder{std::make_unique<Tangent_t>(diffusion_coeff)},
        A{*this->A_holder}, physics_domain{domain} {
    this->last_step_was_nonlinear = false;

    Eigen::EigenSolver<Tangent_t> eigen_value_solver{
        this->get_diffusion_coeff()};
    auto && eigen_values{eigen_value_solver.eigenvalues()};

    for (Dim_t dim{0}; dim < DimM; ++dim) {
      const Complex & eigen_value{eigen_values(dim)};
      if (eigen_value.imag() != 0. or eigen_value.real() < 0.) {
        std::stringstream error_message{};
        error_message << "The diffusion coefficient matrix has to be positive "
                         "definite (i.e., only positive and real eigenvalues). "
                         "The matrix you've provided has the eigenvalues "
                      << eigen_values.transpose()
                      << ", the offending eigen value is #" << dim + 1 << ": "
                      << eigen_value.real() << " + " << eigen_value.imag()
                      << "j. The matrix is" << std::endl
                      << this->get_diffusion_coeff();
        throw MaterialError{error_message.str()};
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  muGrid::PhysicsDomain
  MaterialLinearDiffusion<DimM>::get_physics_domain() const {
    return this->physics_domain;
  }

  /* ---------------------------------------------------------------------- */
  template <Index_t DimM>
  auto MaterialLinearDiffusion<DimM>::get_diffusion_coeff() const
      -> const Tangent_t & {
    return this->A;
  }

  /* ---------------------------------------------------------------------- */
  template class MaterialLinearDiffusion<oneD>;
  template class MaterialLinearDiffusion<twoD>;
  template class MaterialLinearDiffusion<threeD>;

}  // namespace muSpectre

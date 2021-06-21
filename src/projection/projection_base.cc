/**
 * @file   projection_base.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   06 Dec 2017
 *
 * @brief  implementation of base class for projections
 *
 * Copyright © 2017 Till Junge
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

#include <sstream>

#include "projection/projection_base.hh"

namespace muSpectre {

  /* ---------------------------------------------------------------------- */
  ProjectionBase::ProjectionBase(muFFT::FFTEngine_ptr engine,
                                 const DynRcoord_t & domain_lengths,
                                 const Index_t & nb_quad_pts,
                                 const Index_t & nb_components,
                                 const Gradient_t & gradient,
                                 const Formulation & form,
                                 const MeanControl & mean_control)
      : fft_engine{std::move(engine)}, domain_lengths{domain_lengths},
        nb_quad_pts{nb_quad_pts},
        nb_components{nb_components}, gradient{gradient}, form{form},
        work_space{this->fft_engine->register_fourier_space_field(
            "work_space", this->nb_components * this->nb_quad_pts)},
        mean_control{mean_control} {
    if (nb_quad_pts <= 0) {
      throw std::runtime_error("Number of quadrature points must be larger "
                               "than zero.");
    }
    auto nb_dim{this->get_dim()};
    for (auto tup :
         akantu::enumerate(this->fft_engine->get_nb_domain_grid_pts())) {
      auto & dim{std::get<0>(tup)};
      auto & res{std::get<1>(tup)};
      if (res % 2 == 0) {
        for (Dim_t quad{0}; quad < nb_quad_pts; ++quad) {
          Eigen::VectorXd v(nb_dim);
          v.setZero();
          v(dim) = 0.5;
          // Get Fourier derivative at both boundaries of the Brillouin zone
          auto p{this->gradient[dim + quad * nb_dim]->fourier(v)};
          auto m{this->gradient[dim + quad * nb_dim]->fourier(-v)};
          // This test checks if the Fourier derivative at the Brillouin zon
          // boundary is zero (e.g. for central difference) or if the Fourier
          // derivative at left and right boundary are not the same (e.g. for
          // the Fourier derivative). In both cases, the derivative is
          // ambiguous and the calculation cannot continue. The division by res
          // simply adjusts the threshold for what it means to be zero.
          if (std::abs(p) < 1e-6 / res or std::abs(p - m) > 1e-6 / res) {
            throw ProjectionError(
                "Only an odd number of grid points is supported by this "
                "stencil");
          }
        }
      }
    }
    this->fft_engine->get_fourier_field_collection().set_nb_sub_pts(
        QuadPtTag, nb_quad_pts);
    if (this->domain_lengths.get_dim() != this->fft_engine->get_spatial_dim()) {
      std::stringstream error{};
      error << "The domain lengths supplied are "
            << this->domain_lengths.get_dim()
            << "-dimensional, while the FFT engine is "
            << this->fft_engine->get_spatial_dim() << "-dimensional";
      throw muGrid::RuntimeError(error.str());
    }
  }

  /* ---------------------------------------------------------------------- */
  void ProjectionBase::initialise() { this->initialised = true; }

  /* ---------------------------------------------------------------------- */
  const Index_t & ProjectionBase::get_dim() const {
    return this->fft_engine->get_spatial_dim();
  }

  /* ---------------------------------------------------------------------- */
  const Index_t & ProjectionBase::get_nb_quad_pts() const {
    return this->nb_quad_pts;
  }

  /* ---------------------------------------------------------------------- */
  muFFT::FFTEngineBase & ProjectionBase::get_fft_engine() {
    return *this->fft_engine;
  }

  /* ---------------------------------------------------------------------- */
  const muFFT::FFTEngineBase & ProjectionBase::get_fft_engine() const {
    return *this->fft_engine;
  }

  /* ---------------------------------------------------------------------- */
  const DynCcoord_t & ProjectionBase::get_nb_domain_grid_pts() const {
    return this->fft_engine->get_nb_domain_grid_pts();
  }
  /* ---------------------------------------------------------------------- */

  const DynCcoord_t & ProjectionBase::get_nb_subdomain_grid_pts() const {
    return this->fft_engine->get_nb_subdomain_grid_pts();
  }
  /* ---------------------------------------------------------------------- */

  const DynRcoord_t ProjectionBase::get_pixel_lengths() const {
    auto nb_pixels{this->get_nb_domain_grid_pts()};
    auto length_pixels{this->get_domain_lengths()};
    DynRcoord_t ret_val;
    for (int i{0}; i < this->get_dim(); i++) {
      ret_val[i] = length_pixels[i] / nb_pixels[i];
    }
    return ret_val;
  }

  /* ---------------------------------------------------------------------- */
  const muFFT::Gradient_t & ProjectionBase::get_gradient() const {
    return this->gradient;
  }

}  // namespace muSpectre

/**
 * @file   fft_engine_base.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   03 Dec 2017
 *
 * @brief  implementation for FFT engine base class
 *
 * Copyright © 2017 Till Junge
 *
 * µFFT is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µFFT is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µFFT; see the file COPYING. If not, write to the
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

#include "fft_engine_base.hh"

#include "fft_utils.hh"

namespace muFFT {

  /* ---------------------------------------------------------------------- */
  FFTEngineBase::FFTEngineBase(DynCcoord_t nb_grid_pts, Dim_t nb_dof_per_pixel,
                               Communicator comm)
      : spatial_dimension{nb_grid_pts.get_dim()}, comm{comm},
        work_space_container{this->spatial_dimension, OneQuadPt},
        nb_subdomain_grid_pts{nb_grid_pts},
        subdomain_locations(spatial_dimension),
        nb_fourier_grid_pts{get_nb_hermitian_grid_pts(nb_grid_pts)},
        fourier_locations(spatial_dimension), nb_domain_grid_pts{nb_grid_pts},
        work{work_space_container.register_complex_field("work space",
                                                         nb_dof_per_pixel)},
        norm_factor{1. / muGrid::CcoordOps::get_size(nb_domain_grid_pts)},
        nb_dof_per_pixel{nb_dof_per_pixel} {}

  /* ---------------------------------------------------------------------- */
  void FFTEngineBase::initialise(FFT_PlanFlags /*plan_flags*/) {
    this->work_space_container.initialise(this->nb_fourier_grid_pts);
  }

  /* ---------------------------------------------------------------------- */
  size_t FFTEngineBase::size() const {
    return muGrid::CcoordOps::get_size(this->nb_subdomain_grid_pts);
  }

  /* ---------------------------------------------------------------------- */
  size_t FFTEngineBase::fourier_size() const {
    return muGrid::CcoordOps::get_size(this->nb_fourier_grid_pts);
  }

  /* ---------------------------------------------------------------------- */
  size_t FFTEngineBase::workspace_size() const {
    return this->work_space_container.get_nb_entries();
  }

  /* ---------------------------------------------------------------------- */
  const typename FFTEngineBase::Pixels & FFTEngineBase::get_pixels() const {
    return this->work_space_container.get_pixels();
  }

  /* ---------------------------------------------------------------------- */
  const Dim_t & FFTEngineBase::get_nb_dof_per_pixel() const {
    return this->nb_dof_per_pixel;
  }

  /* ---------------------------------------------------------------------- */
  const Dim_t & FFTEngineBase::get_spatial_dim() const {
    return this->spatial_dimension;
  }

  /* ---------------------------------------------------------------------- */
  const Dim_t & FFTEngineBase::get_nb_quad_pts() const {
    return this->work_space_container.get_nb_quad_pts();
  }

}  // namespace muFFT

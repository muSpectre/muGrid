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

#include <libmugrid/ccoord_operations.hh>

using muGrid::CcoordOps::get_default_strides;

namespace muFFT {

  /* ---------------------------------------------------------------------- */
  FFTEngineBase::FFTEngineBase(const DynCcoord_t & nb_grid_pts,
                               Communicator comm)
      : spatial_dimension{nb_grid_pts.get_dim()}, comm{comm},
        fourier_field_collection{this->spatial_dimension},
        nb_domain_grid_pts{nb_grid_pts}, nb_subdomain_grid_pts{nb_grid_pts},
        subdomain_locations(spatial_dimension),
        subdomain_strides{get_default_strides(nb_grid_pts)},
        nb_fourier_grid_pts{get_nb_hermitian_grid_pts(nb_grid_pts)},
        fourier_locations(spatial_dimension),
        fourier_strides{get_default_strides(nb_fourier_grid_pts)},
        norm_factor{1. / muGrid::CcoordOps::get_size(nb_grid_pts)} {
          fourier_field_collection.set_nb_sub_pts(PixelTag, 1);
        }

  /* ---------------------------------------------------------------------- */
  auto
  FFTEngineBase::register_fourier_space_field(const std::string & unique_name,
                                              const Index_t & nb_dof_per_pixel)
      -> FourierField_t & {
    return this->fourier_field_collection.register_complex_field(
        unique_name, nb_dof_per_pixel, PixelTag);
  }

  /* ---------------------------------------------------------------------- */
  auto FFTEngineBase::fetch_or_register_fourier_space_field(
      const std::string & unique_name, const Index_t & nb_dof_per_pixel)
      -> FourierField_t & {
    if (this->fourier_field_collection.field_exists(unique_name)) {
      auto & field{dynamic_cast<FourierField_t &>(
          this->fourier_field_collection.get_field(unique_name))};
      if (field.get_nb_dof_per_pixel() != nb_dof_per_pixel) {
        std::stringstream message{};
        message << "Field '" << unique_name << "' exists, but it has "
                << field.get_nb_dof_per_pixel()
                << " degrees of freedom per pixel instead of the requested "
                << nb_dof_per_pixel;
        throw muGrid::FieldCollectionError{message.str()};
      }
      return field;
    }
    return this->register_fourier_space_field(unique_name, nb_dof_per_pixel);
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
    return this->fourier_field_collection.get_nb_pixels();
  }

  /* ---------------------------------------------------------------------- */
  const typename FFTEngineBase::Pixels_t & FFTEngineBase::get_pixels() const {
    return this->fourier_field_collection.get_pixels();
  }

  /* ---------------------------------------------------------------------- */
  const Index_t & FFTEngineBase::get_spatial_dim() const {
    return this->spatial_dimension;
  }

  /* ---------------------------------------------------------------------- */
  bool FFTEngineBase::has_plan_for(const Index_t & nb_dof_per_pixel) const {
    return static_cast<bool>(this->planned_nb_dofs.count(nb_dof_per_pixel));
  }

  /* ---------------------------------------------------------------------- */
  Index_t FFTEngineBase::get_required_pad_size(
      const Index_t & /*nb_dof_per_pixel*/) const {
    return 0;
  }

}  // namespace muFFT

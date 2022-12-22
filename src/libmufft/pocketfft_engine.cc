/**
 * @file   pocketfft_engine.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   20 Nov 2022
 *
 * @brief  implements the PocketFFT engine
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

#include <iostream>
#include <sstream>

/*
 * Disable multithreading as we want no surprises respective CPU usage when
 * running multiple instances of muFFT on the same server. We support parallel
 * calculations solely through MPI and the respective engines that support it.
 */

#define POCKETFFT_NO_MULTITHREADING
#include "../../external/pocketfft/pocketfft_hdronly.h"

#include <libmugrid/ccoord_operations.hh>

#include "pocketfft_engine.hh"

namespace muFFT {

  PocketFFTEngine::PocketFFTEngine(const DynCcoord_t & nb_grid_pts,
                                   Communicator comm,
                                   const FFT_PlanFlags & plan_flags,
                                   bool allow_temporary_buffer,
                                   bool allow_destroy_input)
      : Parent{nb_grid_pts, comm, plan_flags, allow_temporary_buffer,
               allow_destroy_input, false} {
    this->initialise_field_collections();
  }

  /* ---------------------------------------------------------------------- */
  PocketFFTEngine::PocketFFTEngine(const DynCcoord_t & nb_grid_pts,
                                   const FFT_PlanFlags & plan_flags,
                                   bool allow_temporary_buffer,
                                   bool allow_destroy_input)
      : PocketFFTEngine{nb_grid_pts, Communicator(), plan_flags,
                        allow_temporary_buffer, allow_destroy_input} {}

  /* ---------------------------------------------------------------------- */
  void PocketFFTEngine::create_plan(const Index_t & nb_dof_per_pixel) {
    if (this->has_plan_for(nb_dof_per_pixel)) {
      // plan already exists, we can bail
      return;
    }
    if (this->comm.size() > 1) {
      std::stringstream error;
      error << "PocketFFT engine does not support MPI parallel execution, but "
            << "a communicator of size " << this->comm.size() << " was passed "
            << "during construction";
      throw FFTEngineError(error.str());
    }

    this->planned_nb_dofs.insert(nb_dof_per_pixel);
  }

  /* ---------------------------------------------------------------------- */
  PocketFFTEngine::~PocketFFTEngine() noexcept {}

  /* ---------------------------------------------------------------------- */
  Index_t _get_offset(Index_t index, Shape_t shape, Shape_t strides) {
    if (index == 0) {
      // This catches corners-cases where shape and stride have different sizes.
      // This happens where the exactly one DOF, in which case `get_strides`
      // strips the component/sub-point dimensions from the stride array.
      return 0;
    }
    assert(shape.size() == strides.size());
    const auto nb_dim{shape.size()};
    Index_t offset{0};
    for (size_t i{0}; i < nb_dim; ++i) {
      auto d{std::div(index, shape[i])};
      offset += d.rem * strides[i];
      index = d.quot;
    }
    return offset;
  }

  /* ---------------------------------------------------------------------- */
  void PocketFFTEngine::compute_fft(const RealField_t & input_field,
                                    FourierField_t & output_field) {
    const size_t nb_dim{static_cast<size_t>(this->get_spatial_dim())};
    const Index_t nb_dof{input_field.get_nb_dof_per_pixel()};
    auto sub_pt_input_shape{
        input_field.get_sub_pt_shape(muGrid::IterUnit::SubPt)};
    auto sub_pt_output_shape{
        output_field.get_sub_pt_shape(muGrid::IterUnit::SubPt)};

    if (nb_dof != output_field.get_nb_dof_per_pixel()) {
      std::stringstream error;
      error << "Input field has " << nb_dof << " DOFs while output field has "
            << output_field.get_nb_dof_per_pixel() << " DOFs";
      throw FFTEngineError(error.str());
    }

    // Copy grid dimensions into PocketFFT-friendly data type.
    pocketfft::shape_t real_shape(nb_dim);
    std::copy(this->nb_subdomain_grid_pts.begin(),
              this->nb_subdomain_grid_pts.end(), real_shape.begin());

    // Note: By muFFT default, the first index is the "hermitian" one with
    // nx/2 + 1 grid points in Fourier space. This means we have to carry out
    // the real to complex transform on the (first) x-axis.
    pocketfft::shape_t fourier_shape(nb_dim);
    std::copy(this->nb_fourier_grid_pts.begin(),
              this->nb_fourier_grid_pts.end(), fourier_shape.begin());

    // Temporary buffer for all strides
    pocketfft::stride_t sub_pt_input_strides{
        input_field.get_strides(muGrid::IterUnit::SubPt)};
    pocketfft::stride_t sub_pt_output_strides{
        output_field.get_strides(muGrid::IterUnit::SubPt)};

    // Copy last part of buffer to pixels strides
    pocketfft::stride_t pixel_input_strides(nb_dim),
        pixel_output_strides(nb_dim);
    const size_t nb_input_dof_dim{sub_pt_input_strides.size() - nb_dim},
        nb_output_dof_dim{sub_pt_output_strides.size() - nb_dim};
    for (size_t dim{0}; dim < nb_dim; ++dim) {
      pixel_input_strides[dim] =
          sub_pt_input_strides[nb_input_dof_dim + dim] * sizeof(Real);
      pixel_output_strides[dim] =
          sub_pt_output_strides[nb_output_dof_dim + dim] * sizeof(Complex);
    }

    // Resize sub-point strides
    sub_pt_input_strides.resize(nb_input_dof_dim);
    sub_pt_output_strides.resize(nb_output_dof_dim);

    if (nb_dim == 1) {
      // One dimensional transforms can be carried out directly
      for (Index_t i{0}; i < nb_dof; ++i) {
        pocketfft::r2c(real_shape,
                       pixel_input_strides,
                       pixel_output_strides,
                       0,
                       pocketfft::FORWARD,  // forward transform
                       input_field.data() +
                           _get_offset(i, sub_pt_input_shape,
                                       sub_pt_input_strides),
                       output_field.data() +
                           _get_offset(i, sub_pt_output_shape,
                                       sub_pt_output_strides),
                       1.0);  // additional multiplicative factor
      }
    } else {
      // For n-dimensional transforms we need to carry out n-transforms in the
      // respective directions
      FourierField_t & tmp_field{
          this->fetch_or_register_fourier_space_field("pocketfft_tmp", 1)};
      pocketfft::stride_t tmp_strides{
          tmp_field.get_strides(muGrid::IterUnit::Pixel, sizeof(Complex))};

      // Prepare axes array -> 1, 2, 3, ..., nb_dim-1
      pocketfft::shape_t axes(nb_dim - 1);
      std::iota(axes.begin(), axes.end(), 1);

      // Loop over all components and sub-points and carry out transform
      for (Index_t i{0}; i < nb_dof; ++i) {
        pocketfft::r2c(real_shape,
                       pixel_input_strides,
                       tmp_strides,
                       0,  // see comment above on Hermitian index
                       pocketfft::FORWARD,  // forward transform
                       input_field.data() +
                           _get_offset(i, sub_pt_input_shape,
                                       sub_pt_input_strides),
                       tmp_field.data(),
                       1.0);  // additional multiplicative factor
        pocketfft::c2c(fourier_shape,
                       tmp_strides,
                       pixel_output_strides, axes,
                       pocketfft::FORWARD,  // forward transform
                       tmp_field.data(),
                       output_field.data() +
                           _get_offset(i, sub_pt_output_shape,
                                       sub_pt_output_strides),
                       1.0);  // additional multiplicative factor
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  void PocketFFTEngine::compute_ifft(const FourierField_t & input_field,
                                     RealField_t & output_field) {
    const size_t nb_dim{static_cast<size_t>(this->get_spatial_dim())};
    const Index_t nb_dof{input_field.get_nb_dof_per_pixel()};
    auto sub_pt_input_shape{
        input_field.get_sub_pt_shape(muGrid::IterUnit::SubPt)};
    auto sub_pt_output_shape{
        output_field.get_sub_pt_shape(muGrid::IterUnit::SubPt)};

    if (nb_dof != output_field.get_nb_dof_per_pixel()) {
      std::stringstream error;
      error << "Input field has " << nb_dof << " DOFs while output field has "
            << output_field.get_nb_dof_per_pixel() << " DOFs";
      throw FFTEngineError(error.str());
    }

    // Copy grid dimensions into PocketFFT-friendly data type.
    pocketfft::shape_t real_shape(nb_dim);
    std::copy(this->nb_subdomain_grid_pts.begin(),
              this->nb_subdomain_grid_pts.end(), real_shape.begin());

    // Note: By muFFT default, the first index is the "hermitian" one with
    // nx/2 + 1 grid points in Fourier space. This means we have to carry out
    // the real to complex transform on the (first) x-axis.
    pocketfft::shape_t fourier_shape(nb_dim);
    std::copy(this->nb_fourier_grid_pts.begin(),
              this->nb_fourier_grid_pts.end(), fourier_shape.begin());

    // Temporary buffer for all strides
    pocketfft::stride_t sub_pt_input_strides{
        input_field.get_strides(muGrid::IterUnit::SubPt)};
    pocketfft::stride_t sub_pt_output_strides{
        output_field.get_strides(muGrid::IterUnit::SubPt)};

    // Copy last part of buffer to pixels strides
    pocketfft::stride_t pixel_input_strides(nb_dim),
        pixel_output_strides(nb_dim);
    const size_t nb_input_dof_dim{sub_pt_input_strides.size() - nb_dim},
        nb_output_dof_dim{sub_pt_output_strides.size() - nb_dim};
    for (size_t dim{0}; dim < nb_dim; ++dim) {
      pixel_input_strides[dim] =
          sub_pt_input_strides[nb_input_dof_dim + dim] * sizeof(Complex);
      pixel_output_strides[dim] =
          sub_pt_output_strides[nb_output_dof_dim + dim] * sizeof(Real);
    }

    // Resize sub-point strides
    sub_pt_input_strides.resize(nb_input_dof_dim);
    sub_pt_output_strides.resize(nb_output_dof_dim);

    if (nb_dim == 1) {
      // One dimensional transforms can be carried out directly
      for (Index_t i{0}; i < nb_dof; ++i) {
        pocketfft::c2r(real_shape,
                       pixel_input_strides,
                       pixel_output_strides, 0,
                       pocketfft::BACKWARD,  // backward transform
                       input_field.data() +
                           _get_offset(i, sub_pt_input_shape,
                                       sub_pt_input_strides),
                       output_field.data() +
                           _get_offset(i, sub_pt_output_shape,
                                       sub_pt_output_strides),
                       1.0);  // additional multiplicative factor
      }
    } else {
      // For n-dimensional transforms we need to carry out n-transforms in the
      // respective directions
      FourierField_t & tmp_field{
          this->fetch_or_register_fourier_space_field("pocketfft_tmp", 1)};
      pocketfft::stride_t tmp_strides{
          tmp_field.get_strides(muGrid::IterUnit::Pixel, sizeof(Complex))};

      // Prepare axes array -> 1, 2, 3, ..., nb_dim-1
      pocketfft::shape_t axes(nb_dim - 1);
      auto n{nb_dim - 1};
      std::generate(axes.begin(), axes.end(), [&n]{ return n--;});

      // Loop over all components and sub-points and carry out transform
      for (Index_t i{0}; i < nb_dof; ++i) {
        pocketfft::c2c(fourier_shape,
                       pixel_input_strides,
                       tmp_strides,
                       axes,
                       pocketfft::BACKWARD,  // backward transform
                       input_field.data() +
                           _get_offset(i, sub_pt_input_shape,
                                       sub_pt_input_strides),
                       tmp_field.data(),
                       1.0);  // additional multiplicative factor
        pocketfft::c2r(real_shape,
                       tmp_strides,
                       pixel_output_strides,
                       0,  // see comment above on Hermitian index
                       pocketfft::BACKWARD,  // backward transform
                       tmp_field.data(),
                       output_field.data() +
                           _get_offset(i, sub_pt_output_shape,
                                       sub_pt_output_strides),
                       1.0);  // additional multiplicative factor
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  std::unique_ptr<FFTEngineBase> PocketFFTEngine::clone() const {
    return std::make_unique<PocketFFTEngine>(
        this->get_nb_domain_grid_pts(), this->get_communicator(),
        this->plan_flags, this->allow_temporary_buffer,
        this->allow_destroy_input);
  }

}  // namespace muFFT

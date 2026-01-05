/**
 * @file   fft/fft_engine.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   18 Dec 2024
 *
 * @brief  Distributed FFT engine with pencil decomposition
 *
 * Copyright (c) 2024 Lars Pastewka
 *
 * muGrid is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * muGrid is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with muGrid; see the file COPYING. If not, write to the
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

#ifndef SRC_LIBMUGRID_FFT_FFT_ENGINE_HH_
#define SRC_LIBMUGRID_FFT_FFT_ENGINE_HH_

#include "fft_engine_base.hh"
#include "fft_backend_traits.hh"
#include "memory/array.hh"
#include "field/field_typed.hh"
#include "core/exception.hh"

namespace muGrid {

/**
 * Distributed FFT engine using pencil (2D) decomposition.
 *
 * This class provides distributed FFT operations on structured grids with
 * MPI parallelization. It uses pencil decomposition which allows efficient
 * scaling to large numbers of ranks.
 *
 * Key features:
 * - Supports 2D and 3D grids
 * - Handles arbitrary ghost buffer configurations in real space
 * - No ghosts in Fourier space (hard assumption)
 * - Compile-time memory space selection (Host, CUDA, HIP)
 * - Unnormalized transforms (like FFTW)
 *
 * The engine owns field collections for both real and Fourier space, and
 * work buffers for intermediate results during the distributed FFT.
 *
 * @tparam MemorySpace The memory space for work buffers (HostSpace, CUDASpace,
 * ROCmSpace)
 */
template <typename MemorySpace>
class FFTEngine : public FFTEngineBase {
 public:
  using Parent_t = FFTEngineBase;
  using WorkBuffer = Array<Complex, MemorySpace>;

  /**
   * Construct an FFT engine with pencil decomposition.
   *
   * @param nb_domain_grid_pts  Global grid dimensions [Nx, Ny] or [Nx, Ny, Nz]
   * @param comm                MPI communicator (default: serial)
   * @param nb_ghosts_left      Ghost cells on low-index side of each dimension
   * @param nb_ghosts_right     Ghost cells on high-index side of each dimension
   * @param nb_sub_pts          Number of sub-points per pixel (optional)
   * @param device              Device for field memory allocation (optional,
   *                            default uses memory space's default device)
   */
  FFTEngine(const DynGridIndex & nb_domain_grid_pts,
            const Communicator & comm = Communicator(),
            const DynGridIndex & nb_ghosts_left = DynGridIndex{},
            const DynGridIndex & nb_ghosts_right = DynGridIndex{},
            const SubPtMap_t & nb_sub_pts = {},
            Device device = memory_space_to_device<MemorySpace>())
      : Parent_t{nb_domain_grid_pts, comm, nb_ghosts_left, nb_ghosts_right,
                 nb_sub_pts, device},
        backend{create_fft_backend<MemorySpace>()} {}

  FFTEngine() = delete;
  FFTEngine(const FFTEngine &) = delete;
  FFTEngine(FFTEngine &&) = delete;
  ~FFTEngine() override = default;

  FFTEngine & operator=(const FFTEngine &) = delete;
  FFTEngine & operator=(FFTEngine &&) = delete;

  // === Transform operations ===

  void fft(const Field & input, Field & output) override {
    // Verify fields belong to correct collections
    if (&input.get_collection() != &this->get_collection()) {
      throw RuntimeError(
          "Input field must belong to the real-space collection");
    }
    if (&output.get_collection() != &this->get_fourier_space_collection()) {
      throw RuntimeError(
          "Output field must belong to the Fourier-space collection");
    }

    // Verify field is in correct memory space
    bool input_on_device = input.is_on_device();
    bool output_on_device = output.is_on_device();
    if (input_on_device != output_on_device) {
      throw RuntimeError("Input and output must be in the same memory space");
    }
    if constexpr (is_device_space_v<MemorySpace>) {
      if (!input_on_device) {
        throw RuntimeError(
            "FFTEngine<DeviceSpace> requires fields on device memory");
      }
    } else {
      if (input_on_device) {
        throw RuntimeError(
            "FFTEngine<HostSpace> requires fields on host memory");
      }
    }

    if (this->spatial_dim == 2) {
      fft_2d(input, output);
    } else {
      fft_3d(input, output);
    }
  }

  void ifft(const Field & input, Field & output) override {
    // Verify fields belong to correct collections
    if (&input.get_collection() != &this->get_fourier_space_collection()) {
      throw RuntimeError(
          "Input field must belong to the Fourier-space collection");
    }
    if (&output.get_collection() != &this->get_collection()) {
      throw RuntimeError(
          "Output field must belong to the real-space collection");
    }

    // Verify field is in correct memory space
    bool input_on_device = input.is_on_device();
    bool output_on_device = output.is_on_device();
    if (input_on_device != output_on_device) {
      throw RuntimeError("Input and output must be in the same memory space");
    }
    if constexpr (is_device_space_v<MemorySpace>) {
      if (!input_on_device) {
        throw RuntimeError(
            "FFTEngine<DeviceSpace> requires fields on device memory");
      }
    } else {
      if (input_on_device) {
        throw RuntimeError(
            "FFTEngine<HostSpace> requires fields on host memory");
      }
    }

    if (this->spatial_dim == 2) {
      ifft_2d(input, output);
    } else {
      ifft_3d(input, output);
    }
  }

  const char * get_backend_name() const override {
    return fft_backend_name<MemorySpace>();
  }

 protected:
  void fft_2d(const Field & input, Field & output) {
    const DynGridIndex & nb_grid_pts = this->get_nb_domain_grid_pts();
    DynGridIndex local_real = this->get_nb_subdomain_grid_pts_without_ghosts();
    const DynGridIndex & ghosts_left = this->get_nb_ghosts_left();
    const DynGridIndex & local_with_ghosts =
        this->get_nb_subdomain_grid_pts_with_ghosts();

    Index_t nb_components = input.get_nb_components();
    if (output.get_nb_components() != nb_components) {
      throw RuntimeError(
          "Input and output fields must have the same number of components");
    }

    constexpr bool is_device = is_device_space_v<MemorySpace>;
    const Real * input_ptr =
        static_cast<const Real *>(input.get_void_data_ptr(!is_device));
    Complex * output_ptr =
        static_cast<Complex *>(output.get_void_data_ptr(!is_device));

    Index_t Nx = nb_grid_pts[0];
    Index_t Fx = Nx / 2 + 1;
    Index_t Ny = nb_grid_pts[1];

    // Storage order: SoA on GPU, AoS on CPU
    // For SoA: components are in separate blocks, stride between X elements = 1
    // For AoS: components are interleaved, stride between X elements = nb_components
    StorageOrder storage_order = input.get_storage_order();
    bool is_soa = (storage_order == StorageOrder::StructureOfArrays);

    Index_t nb_buffer_pixels = local_with_ghosts[0] * local_with_ghosts[1];
    Index_t ghost_pixel_offset =
        ghosts_left[0] + ghosts_left[1] * local_with_ghosts[0];

    Index_t nb_work_pixels = Fx * local_real[1];
    Index_t nb_fourier_pixels = Fx * local_real[1];

    // Compute strides based on storage order
    // For SoA: consecutive X elements have stride 1, component offset = comp * nb_pixels
    // For AoS: consecutive X elements have stride nb_components, component offset = comp
    auto get_soa_strides = [&](Index_t nb_pixels, Index_t row_width) {
      // Returns: {comp_offset_factor, x_stride, row_dist}
      // comp_offset = comp * comp_offset_factor
      // For SoA: x_stride=1, row_dist=row_width
      return std::make_tuple(nb_pixels, Index_t{1}, row_width);
    };
    auto get_aos_strides = [&](Index_t /*nb_pixels*/, Index_t row_width) {
      // For AoS: x_stride=nb_components, row_dist=row_width*nb_components
      return std::make_tuple(Index_t{1}, nb_components, row_width * nb_components);
    };

    // Input field strides
    auto [in_comp_factor, in_x_stride, in_row_dist] =
        is_soa ? get_soa_strides(nb_buffer_pixels, local_with_ghosts[0])
               : get_aos_strides(nb_buffer_pixels, local_with_ghosts[0]);
    Index_t in_base_offset = is_soa ? ghost_pixel_offset : ghost_pixel_offset * nb_components;

    // Work buffer strides (same storage order as input/output)
    auto [work_comp_factor, work_x_stride, work_row_dist] =
        is_soa ? get_soa_strides(nb_work_pixels, Fx)
               : get_aos_strides(nb_work_pixels, Fx);

    Index_t work_size = nb_work_pixels * nb_components;
    WorkBuffer work_buffer(work_size);
    Complex * work_ptr = work_buffer.data();

    // Step 1: r2c FFT along X for each component
    for (Index_t comp = 0; comp < nb_components; ++comp) {
      Index_t in_comp_offset = comp * in_comp_factor;
      Index_t work_comp_offset = comp * work_comp_factor;
      backend->r2c(Nx, local_real[1],
                   input_ptr + in_base_offset + in_comp_offset,
                   in_x_stride, in_row_dist,
                   work_ptr + work_comp_offset,
                   work_x_stride, work_row_dist);
    }

    // Check output storage order (should match work buffer storage order)
    StorageOrder out_storage_order = output.get_storage_order();
    bool out_is_soa = (out_storage_order == StorageOrder::StructureOfArrays);

    // Output field strides
    auto [out_comp_factor, out_x_stride, out_row_dist] =
        out_is_soa ? get_soa_strides(nb_fourier_pixels, Fx)
                   : get_aos_strides(nb_fourier_pixels, Fx);

    // Step 2 & 3: Transpose/copy and c2c FFT along Y
    Transpose * transpose = this->get_transpose_xz(nb_components);
    if (transpose != nullptr) {
      // MPI path: transpose to output, then c2c on output
      // TODO: Transpose needs to handle storage order conversion
      transpose->forward(work_ptr, output_ptr);

      Index_t local_fx = this->nb_fourier_subdomain_grid_pts[0];
      Index_t local_fourier_pixels = local_fx * Ny;

      // Recalculate output strides for MPI path (different shape)
      auto [mpi_out_comp_factor, mpi_out_x_stride, mpi_out_row_dist] =
          out_is_soa ? get_soa_strides(local_fourier_pixels, local_fx)
                     : get_aos_strides(local_fourier_pixels, local_fx);

      // For c2c along Y: stride is between Y elements (row distance)
      // batch dimension is X
      for (Index_t comp = 0; comp < nb_components; ++comp) {
        Index_t comp_offset = comp * mpi_out_comp_factor;
        backend->c2c_forward(Ny, local_fx,
                             output_ptr + comp_offset, mpi_out_row_dist, mpi_out_x_stride,
                             output_ptr + comp_offset, mpi_out_row_dist, mpi_out_x_stride);
      }
    } else {
      // Serial path: c2c on work buffer, then copy to output
      Index_t local_fy = local_real[1];

      // Step 2: c2c FFT along Y on work buffer
      // For c2c along Y: stride is between Y elements, batch is X
      for (Index_t comp = 0; comp < nb_components; ++comp) {
        Index_t work_comp_offset = comp * work_comp_factor;
        backend->c2c_forward(local_fy, Fx,
                             work_ptr + work_comp_offset, work_row_dist, work_x_stride,
                             work_ptr + work_comp_offset, work_row_dist, work_x_stride);
      }

      // Step 3: Copy from work to output (same storage order, direct copy)
      deep_copy<Complex, MemorySpace>(output_ptr, work_ptr, work_size);
    }
  }

  void fft_3d(const Field & input, Field & output) {
    const DynGridIndex & nb_grid_pts = this->get_nb_domain_grid_pts();
    DynGridIndex local_real = this->get_nb_subdomain_grid_pts_without_ghosts();
    const DynGridIndex & ghosts_left = this->get_nb_ghosts_left();
    const DynGridIndex & local_with_ghosts =
        this->get_nb_subdomain_grid_pts_with_ghosts();

    Index_t nb_components = input.get_nb_components();
    if (output.get_nb_components() != nb_components) {
      throw RuntimeError(
          "Input and output fields must have the same number of components");
    }

    Index_t Nx = nb_grid_pts[0];
    Index_t Ny = nb_grid_pts[1];
    Index_t Nz = nb_grid_pts[2];
    Index_t Fx = Nx / 2 + 1;

    Transpose * transpose_xz = this->get_transpose_xz(nb_components);
    Transpose * transpose_yz_fwd =
        this->get_transpose_yz_forward(nb_components);
    Transpose * transpose_yz_bwd =
        this->get_transpose_yz_backward(nb_components);

    bool need_mpi_path =
        (transpose_xz != nullptr || transpose_yz_fwd != nullptr);

    constexpr bool is_device = is_device_space_v<MemorySpace>;
    const Real * input_ptr =
        static_cast<const Real *>(input.get_void_data_ptr(!is_device));
    Complex * output_ptr =
        static_cast<Complex *>(output.get_void_data_ptr(!is_device));

    // Storage order: SoA on GPU, AoS on CPU
    StorageOrder storage_order = input.get_storage_order();
    bool is_soa = (storage_order == StorageOrder::StructureOfArrays);

    Index_t nb_buffer_pixels =
        local_with_ghosts[0] * local_with_ghosts[1] * local_with_ghosts[2];
    Index_t ghost_pixel_offset =
        ghosts_left[0] + ghosts_left[1] * local_with_ghosts[0] +
        ghosts_left[2] * local_with_ghosts[0] * local_with_ghosts[1];

    // Stride helper functions for 3D
    // Returns: {comp_offset_factor, x_stride, y_dist, z_dist}
    auto get_soa_strides_3d = [&](Index_t nb_pixels, Index_t row_x, Index_t rows_y) {
      // SoA: x_stride=1, y_dist=row_x, z_dist=row_x*rows_y
      return std::make_tuple(nb_pixels, Index_t{1}, row_x, row_x * rows_y);
    };
    auto get_aos_strides_3d = [&](Index_t /*nb_pixels*/, Index_t row_x, Index_t rows_y) {
      // AoS: x_stride=nb_comp, y_dist=row_x*nb_comp, z_dist=row_x*rows_y*nb_comp
      return std::make_tuple(Index_t{1}, nb_components, row_x * nb_components,
                             row_x * rows_y * nb_components);
    };

    // Input field strides
    auto [in_comp_factor, in_x_stride, in_y_dist, in_z_dist] =
        is_soa ? get_soa_strides_3d(nb_buffer_pixels, local_with_ghosts[0],
                                    local_with_ghosts[1])
               : get_aos_strides_3d(nb_buffer_pixels, local_with_ghosts[0],
                                    local_with_ghosts[1]);
    Index_t in_base_offset = is_soa ? ghost_pixel_offset : ghost_pixel_offset * nb_components;

    if (need_mpi_path) {
      // MPI path with transposes
      Index_t nb_zpencil_pixels = Fx * local_real[1] * local_real[2];
      Index_t zpencil_size = nb_zpencil_pixels * nb_components;

      // Z-pencil work buffer strides
      auto [work_z_comp_factor, work_z_x_stride, work_z_y_dist, work_z_z_dist] =
          is_soa ? get_soa_strides_3d(nb_zpencil_pixels, Fx, local_real[1])
                 : get_aos_strides_3d(nb_zpencil_pixels, Fx, local_real[1]);

      WorkBuffer work_z(zpencil_size);
      Complex * work_z_ptr = work_z.data();

      // Step 1: r2c FFT along X for each component
      for (Index_t comp = 0; comp < nb_components; ++comp) {
        Index_t in_comp_offset = comp * in_comp_factor;
        Index_t work_comp_offset = comp * work_z_comp_factor;

        for (Index_t iz = 0; iz < local_real[2]; ++iz) {
          Index_t in_idx = in_base_offset + in_comp_offset + iz * in_z_dist;
          Index_t out_idx = work_comp_offset + iz * work_z_z_dist;
          backend->r2c(Nx, local_real[1], input_ptr + in_idx,
                       in_x_stride, in_y_dist,
                       work_z_ptr + out_idx,
                       work_z_x_stride, work_z_y_dist);
        }
      }

      // Step 2a: Transpose Y<->Z
      const DynGridIndex & ypencil_shape =
          this->work_ypencil->get_nb_subdomain_grid_pts_with_ghosts();
      Index_t nb_ypencil_pixels = Fx * Ny * ypencil_shape[2];
      Index_t ypencil_size = nb_ypencil_pixels * nb_components;

      // Y-pencil work buffer strides
      auto [work_y_comp_factor, work_y_x_stride, work_y_y_dist, work_y_z_dist] =
          is_soa ? get_soa_strides_3d(nb_ypencil_pixels, Fx, Ny)
                 : get_aos_strides_3d(nb_ypencil_pixels, Fx, Ny);

      WorkBuffer work_y(ypencil_size);
      Complex * work_y_ptr = work_y.data();

      if (transpose_yz_fwd != nullptr) {
        // TODO: Transpose needs to handle storage order
        transpose_yz_fwd->forward(work_z_ptr, work_y_ptr);
      }

      // Step 2b: c2c FFT along Y for each component
      // Batch Fx transforms per Z plane
      for (Index_t comp = 0; comp < nb_components; ++comp) {
        Index_t comp_offset = comp * work_y_comp_factor;
        for (Index_t iz = 0; iz < ypencil_shape[2]; ++iz) {
          Index_t idx = comp_offset + iz * work_y_z_dist;
          backend->c2c_forward(Ny, Fx, work_y_ptr + idx,
                               work_y_y_dist, work_y_x_stride,
                               work_y_ptr + idx,
                               work_y_y_dist, work_y_x_stride);
        }
      }

      // Step 2c: Transpose Z<->Y
      if (transpose_yz_bwd != nullptr) {
        // TODO: Transpose needs to handle storage order
        transpose_yz_bwd->forward(work_y_ptr, work_z_ptr);
      }

      // Step 3: Transpose X<->Z (or copy if no transpose needed)
      const DynGridIndex & fourier_local = this->nb_fourier_subdomain_grid_pts;
      Index_t nb_fourier_pixels = fourier_local[0] * fourier_local[1] * fourier_local[2];
      Index_t fourier_size = nb_fourier_pixels * nb_components;

      // Output field strides
      StorageOrder out_storage_order = output.get_storage_order();
      bool out_is_soa = (out_storage_order == StorageOrder::StructureOfArrays);
      auto [out_comp_factor, out_x_stride, out_y_dist, out_z_dist] =
          out_is_soa ? get_soa_strides_3d(nb_fourier_pixels, fourier_local[0],
                                          fourier_local[1])
                     : get_aos_strides_3d(nb_fourier_pixels, fourier_local[0],
                                          fourier_local[1]);

      if (transpose_xz != nullptr) {
        // TODO: Transpose needs to handle storage order
        transpose_xz->forward(work_z_ptr, output_ptr);
      } else {
        deep_copy<Complex, MemorySpace>(output_ptr, work_z_ptr, fourier_size);
      }

      // Step 4: c2c FFT along Z for each component
      // For SoA: Batch all local_fx * local_fy transforms together
      // For AoS: Batch local_fx transforms per Y row
      if (out_is_soa) {
        Index_t batch_z = fourier_local[0] * fourier_local[1];
        for (Index_t comp = 0; comp < nb_components; ++comp) {
          Index_t comp_offset = comp * out_comp_factor;
          backend->c2c_forward(Nz, batch_z, output_ptr + comp_offset,
                               out_z_dist, Index_t{1},
                               output_ptr + comp_offset,
                               out_z_dist, Index_t{1});
        }
      } else {
        // AoS: X elements are separated by out_x_stride
        for (Index_t comp = 0; comp < nb_components; ++comp) {
          Index_t comp_offset = comp * out_comp_factor;
          for (Index_t iy = 0; iy < fourier_local[1]; ++iy) {
            Index_t idx = comp_offset + iy * out_y_dist;
            backend->c2c_forward(Nz, fourier_local[0], output_ptr + idx,
                                 out_z_dist, out_x_stride,
                                 output_ptr + idx,
                                 out_z_dist, out_x_stride);
          }
        }
      }
    } else {
      // Serial path: all dimensions are local
      Index_t nb_fourier_pixels = Fx * Ny * Nz;
      Index_t work_size = nb_fourier_pixels * nb_components;

      // Work/output buffer strides (same storage order as input)
      auto [work_comp_factor, work_x_stride, work_y_dist, work_z_dist] =
          is_soa ? get_soa_strides_3d(nb_fourier_pixels, Fx, Ny)
                 : get_aos_strides_3d(nb_fourier_pixels, Fx, Ny);

      WorkBuffer work_buffer(work_size);
      Complex * work_ptr = work_buffer.data();

      // Step 1: r2c FFT along X for each component
      for (Index_t comp = 0; comp < nb_components; ++comp) {
        Index_t in_comp_offset = comp * in_comp_factor;
        Index_t work_comp_offset = comp * work_comp_factor;

        for (Index_t iz = 0; iz < Nz; ++iz) {
          Index_t in_idx = in_base_offset + in_comp_offset + iz * in_z_dist;
          Index_t work_idx = work_comp_offset + iz * work_z_dist;
          backend->r2c(Nx, Ny, input_ptr + in_idx,
                       in_x_stride, in_y_dist,
                       work_ptr + work_idx,
                       work_x_stride, work_y_dist);
        }
      }

      // Step 2: c2c FFT along Y for each component
      // Batch Fx transforms for each Z plane
      // n=Ny, batch=Fx, stride=work_y_dist (between Y elements), dist=work_x_stride (between X batches)
      for (Index_t comp = 0; comp < nb_components; ++comp) {
        Index_t comp_offset = comp * work_comp_factor;
        for (Index_t iz = 0; iz < Nz; ++iz) {
          Index_t idx = comp_offset + iz * work_z_dist;
          backend->c2c_forward(Ny, Fx, work_ptr + idx, work_y_dist, work_x_stride,
                               work_ptr + idx, work_y_dist, work_x_stride);
        }
      }

      // Step 3: c2c FFT along Z for each component
      // For SoA: Batch all Fx*Ny transforms together with dist=1
      // For AoS: Batch Fx transforms per Y row with dist=work_x_stride
      if (is_soa) {
        // SoA: consecutive XY elements are separated by 1
        for (Index_t comp = 0; comp < nb_components; ++comp) {
          Index_t comp_offset = comp * work_comp_factor;
          backend->c2c_forward(Nz, Fx * Ny, work_ptr + comp_offset,
                               work_z_dist, Index_t{1},
                               work_ptr + comp_offset,
                               work_z_dist, Index_t{1});
        }
      } else {
        // AoS: X elements are separated by work_x_stride (nb_components)
        // Batch Fx transforms per Y row
        for (Index_t comp = 0; comp < nb_components; ++comp) {
          Index_t comp_offset = comp * work_comp_factor;
          for (Index_t iy = 0; iy < Ny; ++iy) {
            Index_t idx = comp_offset + iy * work_y_dist;
            backend->c2c_forward(Nz, Fx, work_ptr + idx,
                                 work_z_dist, work_x_stride,
                                 work_ptr + idx,
                                 work_z_dist, work_x_stride);
          }
        }
      }

      // Copy from work to output (same storage order)
      deep_copy<Complex, MemorySpace>(output_ptr, work_ptr, work_size);
    }
  }

  void ifft_2d(const Field & input, Field & output) {
    const DynGridIndex & nb_grid_pts = this->get_nb_domain_grid_pts();
    DynGridIndex local_real = this->get_nb_subdomain_grid_pts_without_ghosts();
    const DynGridIndex & ghosts_left = this->get_nb_ghosts_left();
    const DynGridIndex & local_with_ghosts =
        this->get_nb_subdomain_grid_pts_with_ghosts();

    Index_t nb_components = input.get_nb_components();
    if (output.get_nb_components() != nb_components) {
      throw RuntimeError(
          "Input and output fields must have the same number of components");
    }

    constexpr bool is_device = is_device_space_v<MemorySpace>;
    const Complex * input_ptr =
        static_cast<const Complex *>(input.get_void_data_ptr(!is_device));
    Real * output_ptr =
        static_cast<Real *>(output.get_void_data_ptr(!is_device));

    Index_t Nx = nb_grid_pts[0];
    Index_t Fx = Nx / 2 + 1;
    Index_t Ny = nb_grid_pts[1];

    // Storage order: SoA on GPU, AoS on CPU
    StorageOrder storage_order = output.get_storage_order();
    bool is_soa = (storage_order == StorageOrder::StructureOfArrays);

    Index_t nb_buffer_pixels = local_with_ghosts[0] * local_with_ghosts[1];
    Index_t ghost_pixel_offset =
        ghosts_left[0] + ghosts_left[1] * local_with_ghosts[0];

    // Stride helper functions
    auto get_soa_strides = [&](Index_t nb_pixels, Index_t row_width) {
      return std::make_tuple(nb_pixels, Index_t{1}, row_width);
    };
    auto get_aos_strides = [&](Index_t /*nb_pixels*/, Index_t row_width) {
      return std::make_tuple(Index_t{1}, nb_components, row_width * nb_components);
    };

    // Output field strides
    auto [out_comp_factor, out_x_stride, out_row_dist] =
        is_soa ? get_soa_strides(nb_buffer_pixels, local_with_ghosts[0])
               : get_aos_strides(nb_buffer_pixels, local_with_ghosts[0]);
    Index_t out_base_offset = is_soa ? ghost_pixel_offset : ghost_pixel_offset * nb_components;

    // Input (Fourier) field storage order should match
    StorageOrder in_storage_order = input.get_storage_order();
    bool in_is_soa = (in_storage_order == StorageOrder::StructureOfArrays);

    Transpose * transpose = this->get_transpose_xz(nb_components);
    if (transpose != nullptr) {
      Index_t local_fx = this->nb_fourier_subdomain_grid_pts[0];
      Index_t local_fourier_pixels = local_fx * Ny;
      Index_t local_fourier_size = local_fourier_pixels * nb_components;

      // Input strides for MPI path
      auto [in_comp_factor, in_x_stride, in_row_dist] =
          in_is_soa ? get_soa_strides(local_fourier_pixels, local_fx)
                    : get_aos_strides(local_fourier_pixels, local_fx);

      WorkBuffer temp(local_fourier_size);
      deep_copy<Complex, MemorySpace>(temp.data(), input_ptr, local_fourier_size);

      // Step 1: c2c IFFT along Y for each component
      for (Index_t comp = 0; comp < nb_components; ++comp) {
        Index_t comp_offset = comp * in_comp_factor;
        backend->c2c_backward(Ny, local_fx,
                              temp.data() + comp_offset, in_row_dist, in_x_stride,
                              temp.data() + comp_offset, in_row_dist, in_x_stride);
      }

      // Step 2: Transpose X<->Y backward
      Index_t nb_work_pixels = Fx * local_real[1];
      Index_t work_size = nb_work_pixels * nb_components;
      WorkBuffer work_buffer(work_size);

      // TODO: Transpose needs to handle storage order
      transpose->backward(temp.data(), work_buffer.data());

      // Work buffer strides (same storage order)
      auto [work_comp_factor, work_x_stride, work_row_dist] =
          is_soa ? get_soa_strides(nb_work_pixels, Fx)
                 : get_aos_strides(nb_work_pixels, Fx);

      // Step 3: c2r IFFT along X for each component
      for (Index_t comp = 0; comp < nb_components; ++comp) {
        Index_t work_comp_offset = comp * work_comp_factor;
        Index_t out_comp_offset = comp * out_comp_factor;
        backend->c2r(Nx, local_real[1],
                     work_buffer.data() + work_comp_offset,
                     work_x_stride, work_row_dist,
                     output_ptr + out_base_offset + out_comp_offset,
                     out_x_stride, out_row_dist);
      }
    } else {
      // Serial path
      Index_t local_fy = local_real[1];
      Index_t nb_fourier_pixels = Fx * local_fy;
      Index_t fourier_size = nb_fourier_pixels * nb_components;

      // Input strides
      auto [in_comp_factor, in_x_stride, in_row_dist] =
          in_is_soa ? get_soa_strides(nb_fourier_pixels, Fx)
                    : get_aos_strides(nb_fourier_pixels, Fx);

      // Work buffer uses same storage order
      WorkBuffer temp(fourier_size);
      deep_copy<Complex, MemorySpace>(temp.data(), input_ptr, fourier_size);

      // Step 1: c2c IFFT along Y for each component
      for (Index_t comp = 0; comp < nb_components; ++comp) {
        Index_t comp_offset = comp * in_comp_factor;
        backend->c2c_backward(local_fy, Fx,
                              temp.data() + comp_offset, in_row_dist, in_x_stride,
                              temp.data() + comp_offset, in_row_dist, in_x_stride);
      }

      // Step 2: c2r IFFT along X for each component
      for (Index_t comp = 0; comp < nb_components; ++comp) {
        Index_t in_comp_offset = comp * in_comp_factor;
        Index_t out_comp_offset = comp * out_comp_factor;
        backend->c2r(Nx, local_fy,
                     temp.data() + in_comp_offset,
                     in_x_stride, in_row_dist,
                     output_ptr + out_base_offset + out_comp_offset,
                     out_x_stride, out_row_dist);
      }
    }
  }

  void ifft_3d(const Field & input, Field & output) {
    const DynGridIndex & nb_grid_pts = this->get_nb_domain_grid_pts();
    DynGridIndex local_real = this->get_nb_subdomain_grid_pts_without_ghosts();
    const DynGridIndex & ghosts_left = this->get_nb_ghosts_left();
    const DynGridIndex & local_with_ghosts =
        this->get_nb_subdomain_grid_pts_with_ghosts();

    Index_t nb_components = input.get_nb_components();
    if (output.get_nb_components() != nb_components) {
      throw RuntimeError(
          "Input and output fields must have the same number of components");
    }

    Index_t Nx = nb_grid_pts[0];
    Index_t Ny = nb_grid_pts[1];
    Index_t Nz = nb_grid_pts[2];
    Index_t Fx = Nx / 2 + 1;

    Transpose * transpose_xz = this->get_transpose_xz(nb_components);
    Transpose * transpose_yz_fwd =
        this->get_transpose_yz_forward(nb_components);
    Transpose * transpose_yz_bwd =
        this->get_transpose_yz_backward(nb_components);

    bool need_mpi_path =
        (transpose_xz != nullptr || transpose_yz_fwd != nullptr);

    constexpr bool is_device = is_device_space_v<MemorySpace>;
    const Complex * input_ptr =
        static_cast<const Complex *>(input.get_void_data_ptr(!is_device));
    Real * output_ptr =
        static_cast<Real *>(output.get_void_data_ptr(!is_device));

    // Storage order: SoA on GPU, AoS on CPU
    StorageOrder storage_order = output.get_storage_order();
    bool is_soa = (storage_order == StorageOrder::StructureOfArrays);

    Index_t nb_buffer_pixels =
        local_with_ghosts[0] * local_with_ghosts[1] * local_with_ghosts[2];
    Index_t ghost_pixel_offset =
        ghosts_left[0] + ghosts_left[1] * local_with_ghosts[0] +
        ghosts_left[2] * local_with_ghosts[0] * local_with_ghosts[1];

    // Stride helper functions for 3D
    auto get_soa_strides_3d = [&](Index_t nb_pixels, Index_t row_x, Index_t rows_y) {
      return std::make_tuple(nb_pixels, Index_t{1}, row_x, row_x * rows_y);
    };
    auto get_aos_strides_3d = [&](Index_t /*nb_pixels*/, Index_t row_x, Index_t rows_y) {
      return std::make_tuple(Index_t{1}, nb_components, row_x * nb_components,
                             row_x * rows_y * nb_components);
    };

    // Output field strides
    auto [out_comp_factor, out_x_stride, out_y_dist, out_z_dist] =
        is_soa ? get_soa_strides_3d(nb_buffer_pixels, local_with_ghosts[0],
                                    local_with_ghosts[1])
               : get_aos_strides_3d(nb_buffer_pixels, local_with_ghosts[0],
                                    local_with_ghosts[1]);
    Index_t out_base_offset = is_soa ? ghost_pixel_offset : ghost_pixel_offset * nb_components;

    // Input (Fourier) field storage order should match
    StorageOrder in_storage_order = input.get_storage_order();
    bool in_is_soa = (in_storage_order == StorageOrder::StructureOfArrays);

    if (need_mpi_path) {
      // MPI path with transposes
      const DynGridIndex & fourier_local = this->nb_fourier_subdomain_grid_pts;
      Index_t nb_fourier_pixels = fourier_local[0] * fourier_local[1] * fourier_local[2];
      Index_t fourier_size = nb_fourier_pixels * nb_components;

      // Input field strides
      auto [in_comp_factor, in_x_stride, in_y_dist, in_z_dist] =
          in_is_soa ? get_soa_strides_3d(nb_fourier_pixels, fourier_local[0],
                                         fourier_local[1])
                    : get_aos_strides_3d(nb_fourier_pixels, fourier_local[0],
                                         fourier_local[1]);

      WorkBuffer temp(fourier_size);
      deep_copy<Complex, MemorySpace>(temp.data(), input_ptr, fourier_size);

      // Step 1: c2c IFFT along Z for each component
      // For SoA: Batch all local_fx * local_fy transforms together
      // For AoS: Batch local_fx transforms per Y row
      if (in_is_soa) {
        Index_t batch_z = fourier_local[0] * fourier_local[1];
        for (Index_t comp = 0; comp < nb_components; ++comp) {
          Index_t comp_offset = comp * in_comp_factor;
          backend->c2c_backward(Nz, batch_z, temp.data() + comp_offset,
                                in_z_dist, Index_t{1},
                                temp.data() + comp_offset,
                                in_z_dist, Index_t{1});
        }
      } else {
        // AoS: X elements are separated by in_x_stride
        for (Index_t comp = 0; comp < nb_components; ++comp) {
          Index_t comp_offset = comp * in_comp_factor;
          for (Index_t iy = 0; iy < fourier_local[1]; ++iy) {
            Index_t idx = comp_offset + iy * in_y_dist;
            backend->c2c_backward(Nz, fourier_local[0], temp.data() + idx,
                                  in_z_dist, in_x_stride,
                                  temp.data() + idx,
                                  in_z_dist, in_x_stride);
          }
        }
      }

      // Z-pencil work buffer
      Index_t nb_zpencil_pixels = Fx * local_real[1] * local_real[2];
      Index_t zpencil_size = nb_zpencil_pixels * nb_components;

      // Z-pencil strides
      auto [work_z_comp_factor, work_z_x_stride, work_z_y_dist, work_z_z_dist] =
          is_soa ? get_soa_strides_3d(nb_zpencil_pixels, Fx, local_real[1])
                 : get_aos_strides_3d(nb_zpencil_pixels, Fx, local_real[1]);

      WorkBuffer work_z(zpencil_size);
      Complex * work_z_ptr = work_z.data();

      // Step 2: Transpose Z<->X backward (or copy)
      if (transpose_xz != nullptr) {
        // TODO: Transpose needs to handle storage order
        transpose_xz->backward(temp.data(), work_z_ptr);
      } else {
        deep_copy<Complex, MemorySpace>(work_z_ptr, temp.data(), zpencil_size);
      }

      // Y-pencil work buffer
      const DynGridIndex & ypencil_shape =
          this->work_ypencil->get_nb_subdomain_grid_pts_with_ghosts();
      Index_t nb_ypencil_pixels = Fx * Ny * ypencil_shape[2];
      Index_t ypencil_size = nb_ypencil_pixels * nb_components;

      // Y-pencil strides
      auto [work_y_comp_factor, work_y_x_stride, work_y_y_dist, work_y_z_dist] =
          is_soa ? get_soa_strides_3d(nb_ypencil_pixels, Fx, Ny)
                 : get_aos_strides_3d(nb_ypencil_pixels, Fx, Ny);

      WorkBuffer work_y(ypencil_size);
      Complex * work_y_ptr = work_y.data();

      // Step 3a: Transpose Y<->Z backward
      if (transpose_yz_bwd != nullptr) {
        // TODO: Transpose needs to handle storage order
        transpose_yz_bwd->backward(work_z_ptr, work_y_ptr);
      }

      // Step 3b: c2c IFFT along Y for each component
      // Batch Fx transforms per Z plane
      for (Index_t comp = 0; comp < nb_components; ++comp) {
        Index_t comp_offset = comp * work_y_comp_factor;
        for (Index_t iz = 0; iz < ypencil_shape[2]; ++iz) {
          Index_t idx = comp_offset + iz * work_y_z_dist;
          backend->c2c_backward(Ny, Fx, work_y_ptr + idx,
                                work_y_y_dist, work_y_x_stride,
                                work_y_ptr + idx,
                                work_y_y_dist, work_y_x_stride);
        }
      }

      // Step 3c: Transpose Z<->Y backward
      if (transpose_yz_fwd != nullptr) {
        // TODO: Transpose needs to handle storage order
        transpose_yz_fwd->backward(work_y_ptr, work_z_ptr);
      }

      // Step 4: c2r IFFT along X for each component
      for (Index_t comp = 0; comp < nb_components; ++comp) {
        Index_t work_comp_offset = comp * work_z_comp_factor;
        Index_t out_comp_offset = comp * out_comp_factor;

        for (Index_t iz = 0; iz < local_real[2]; ++iz) {
          Index_t in_idx = work_comp_offset + iz * work_z_z_dist;
          Index_t out_idx = out_base_offset + out_comp_offset + iz * out_z_dist;
          backend->c2r(Nx, local_real[1], work_z_ptr + in_idx,
                       work_z_x_stride, work_z_y_dist,
                       output_ptr + out_idx, out_x_stride, out_y_dist);
        }
      }
    } else {
      // Serial path: all dimensions are local
      Index_t nb_fourier_pixels = Fx * Ny * Nz;
      Index_t work_size = nb_fourier_pixels * nb_components;

      // Input strides
      auto [in_comp_factor, in_x_stride, in_y_dist, in_z_dist] =
          in_is_soa ? get_soa_strides_3d(nb_fourier_pixels, Fx, Ny)
                    : get_aos_strides_3d(nb_fourier_pixels, Fx, Ny);

      // Work buffer with same storage order
      WorkBuffer work_buffer(work_size);
      Complex * work_ptr = work_buffer.data();
      deep_copy<Complex, MemorySpace>(work_ptr, input_ptr, work_size);

      // Step 1: c2c IFFT along Z for each component
      // For SoA: Batch all Fx*Ny transforms together with dist=1
      // For AoS: Batch Fx transforms per Y row with dist=in_x_stride
      if (in_is_soa) {
        // SoA: consecutive XY elements are separated by 1
        for (Index_t comp = 0; comp < nb_components; ++comp) {
          Index_t comp_offset = comp * in_comp_factor;
          backend->c2c_backward(Nz, Fx * Ny, work_ptr + comp_offset,
                                in_z_dist, Index_t{1},
                                work_ptr + comp_offset,
                                in_z_dist, Index_t{1});
        }
      } else {
        // AoS: X elements are separated by in_x_stride (nb_components)
        // Batch Fx transforms per Y row
        for (Index_t comp = 0; comp < nb_components; ++comp) {
          Index_t comp_offset = comp * in_comp_factor;
          for (Index_t iy = 0; iy < Ny; ++iy) {
            Index_t idx = comp_offset + iy * in_y_dist;
            backend->c2c_backward(Nz, Fx, work_ptr + idx,
                                  in_z_dist, in_x_stride,
                                  work_ptr + idx,
                                  in_z_dist, in_x_stride);
          }
        }
      }

      // Step 2: c2c IFFT along Y for each component
      // Batch Fx transforms for each Z plane
      // n=Ny, batch=Fx, stride=in_y_dist (between Y elements), dist=in_x_stride (between X batches)
      for (Index_t comp = 0; comp < nb_components; ++comp) {
        Index_t comp_offset = comp * in_comp_factor;
        for (Index_t iz = 0; iz < Nz; ++iz) {
          Index_t idx = comp_offset + iz * in_z_dist;
          backend->c2c_backward(Ny, Fx, work_ptr + idx, in_y_dist, in_x_stride,
                                work_ptr + idx, in_y_dist, in_x_stride);
        }
      }

      // Step 3: c2r IFFT along X for each component
      for (Index_t comp = 0; comp < nb_components; ++comp) {
        Index_t in_comp_offset = comp * in_comp_factor;
        Index_t out_comp_offset = comp * out_comp_factor;

        for (Index_t iz = 0; iz < Nz; ++iz) {
          Index_t in_idx = in_comp_offset + iz * in_z_dist;
          Index_t out_idx = out_base_offset + out_comp_offset + iz * out_z_dist;
          backend->c2r(Nx, Ny, work_ptr + in_idx,
                       in_x_stride, in_y_dist,
                       output_ptr + out_idx, out_x_stride, out_y_dist);
        }
      }
    }
  }

  //! FFT backend for this memory space
  std::unique_ptr<FFT1DBackend> backend;
};

// Explicit template instantiation declarations for common memory spaces
extern template class FFTEngine<HostSpace>;
#if defined(MUGRID_ENABLE_CUDA)
extern template class FFTEngine<CUDASpace>;
#endif
#if defined(MUGRID_ENABLE_HIP)
extern template class FFTEngine<ROCmSpace>;
#endif

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_FFT_FFT_ENGINE_HH_

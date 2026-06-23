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

#include <array>

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
            Device device = memory_space_to_device<MemorySpace>(),
            const std::string & decomposition = "auto")
      : Parent_t{nb_domain_grid_pts, comm, nb_ghosts_left, nb_ghosts_right,
                 nb_sub_pts, device, decomposition},
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
    bool input_on_device{input.is_on_device()};
    bool output_on_device{output.is_on_device()};
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

    if (this->spatial_dim == 1) {
      fft_1d(input, output);
    } else if (this->spatial_dim == 2) {
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
    bool input_on_device{input.is_on_device()};
    bool output_on_device{output.is_on_device()};
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

    if (this->spatial_dim == 1) {
      ifft_1d(input, output);
    } else if (this->spatial_dim == 2) {
      ifft_2d(input, output);
    } else {
      ifft_3d(input, output);
    }
  }

  const char * get_backend_name() const override {
    return fft_backend_name<MemorySpace>();
  }

 protected:
  void fft_1d(const Field & input, Field & output) {
    const DynGridIndex & nb_grid_pts = this->get_nb_domain_grid_pts();
    const DynGridIndex & ghosts_left = this->get_nb_ghosts_left();
    const DynGridIndex & local_with_ghosts =
        this->get_nb_subdomain_grid_pts_with_ghosts();

    Index_t nb_components{input.get_nb_components()};
    if (output.get_nb_components() != nb_components) {
      throw RuntimeError(
          "Input and output fields must have the same number of components");
    }

    constexpr bool is_device = is_device_space_v<MemorySpace>;
    const Real * input_ptr =
        static_cast<const Real *>(input.get_void_data_ptr(!is_device));
    Complex * output_ptr =
        static_cast<Complex *>(output.get_void_data_ptr(!is_device));

    Index_t Nx{nb_grid_pts[0]};
    Index_t Fx{Nx / 2 + 1};

    StorageOrder storage_order{input.get_storage_order()};
    bool is_soa{(storage_order == StorageOrder::StructureOfArrays)};

    Index_t nb_buffer_pixels{local_with_ghosts[0]};
    Index_t ghost_offset{ghosts_left[0]};

    if (is_soa) {
      // SoA: components in separate blocks
      for (Index_t comp{0}; comp < nb_components; ++comp) {
        backend->r2c(Nx, 1,
                     input_ptr + ghost_offset + comp * nb_buffer_pixels,
                     1, Nx,
                     output_ptr + comp * Fx,
                     1, Fx);
      }
    } else {
      // AoS: components interleaved
      for (Index_t comp{0}; comp < nb_components; ++comp) {
        backend->r2c(Nx, 1,
                     input_ptr + ghost_offset * nb_components + comp,
                     nb_components, Nx * nb_components,
                     output_ptr + comp,
                     nb_components, Fx * nb_components);
      }
    }
  }

  void ifft_1d(const Field & input, Field & output) {
    const DynGridIndex & nb_grid_pts = this->get_nb_domain_grid_pts();
    const DynGridIndex & ghosts_left = this->get_nb_ghosts_left();
    const DynGridIndex & local_with_ghosts =
        this->get_nb_subdomain_grid_pts_with_ghosts();

    Index_t nb_components{input.get_nb_components()};
    if (output.get_nb_components() != nb_components) {
      throw RuntimeError(
          "Input and output fields must have the same number of components");
    }

    constexpr bool is_device = is_device_space_v<MemorySpace>;
    const Complex * input_ptr =
        static_cast<const Complex *>(input.get_void_data_ptr(!is_device));
    Real * output_ptr =
        static_cast<Real *>(output.get_void_data_ptr(!is_device));

    Index_t Nx{nb_grid_pts[0]};
    Index_t Fx{Nx / 2 + 1};

    StorageOrder storage_order{output.get_storage_order()};
    bool is_soa{(storage_order == StorageOrder::StructureOfArrays)};

    Index_t nb_buffer_pixels{local_with_ghosts[0]};
    Index_t ghost_offset{ghosts_left[0]};

    if (is_soa) {
      // SoA: components in separate blocks
      for (Index_t comp{0}; comp < nb_components; ++comp) {
        backend->c2r(Nx, 1,
                     input_ptr + comp * Fx,
                     1, Fx,
                     output_ptr + ghost_offset + comp * nb_buffer_pixels,
                     1, Nx);
      }
    } else {
      // AoS: components interleaved
      for (Index_t comp{0}; comp < nb_components; ++comp) {
        backend->c2r(Nx, 1,
                     input_ptr + comp,
                     nb_components, Fx * nb_components,
                     output_ptr + ghost_offset * nb_components + comp,
                     nb_components, Nx * nb_components);
      }
    }
  }

  void fft_2d(const Field & input, Field & output) {
    const DynGridIndex & nb_grid_pts = this->get_nb_domain_grid_pts();
    DynGridIndex local_real = this->get_nb_subdomain_grid_pts_without_ghosts();
    const DynGridIndex & ghosts_left = this->get_nb_ghosts_left();
    const DynGridIndex & local_with_ghosts =
        this->get_nb_subdomain_grid_pts_with_ghosts();

    Index_t nb_components{input.get_nb_components()};
    if (output.get_nb_components() != nb_components) {
      throw RuntimeError(
          "Input and output fields must have the same number of components");
    }

    constexpr bool is_device = is_device_space_v<MemorySpace>;
    const Real * input_ptr =
        static_cast<const Real *>(input.get_void_data_ptr(!is_device));
    Complex * output_ptr =
        static_cast<Complex *>(output.get_void_data_ptr(!is_device));

    Index_t Nx{nb_grid_pts[0]};
    Index_t Fx{Nx / 2 + 1};
    Index_t Ny{nb_grid_pts[1]};

    // Storage order: SoA on GPU, AoS on CPU
    // For SoA: components are in separate blocks, stride between X elements = 1
    // For AoS: components are interleaved, stride between X elements = nb_components
    StorageOrder storage_order{input.get_storage_order()};
    bool is_soa{(storage_order == StorageOrder::StructureOfArrays)};

    Index_t nb_buffer_pixels{local_with_ghosts[0] * local_with_ghosts[1]};
    Index_t ghost_pixel_offset{
        ghosts_left[0] + ghosts_left[1] * local_with_ghosts[0]};

    Index_t nb_work_pixels{Fx * local_real[1]};
    Index_t nb_fourier_pixels{Fx * local_real[1]};

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
    Index_t in_base_offset{is_soa ? ghost_pixel_offset : ghost_pixel_offset * nb_components};

    // Work buffer strides (same storage order as input/output)
    auto [work_comp_factor, work_x_stride, work_row_dist] =
        is_soa ? get_soa_strides(nb_work_pixels, Fx)
               : get_aos_strides(nb_work_pixels, Fx);

    Index_t work_size{nb_work_pixels * nb_components};
    Complex * work_ptr{this->get_work_buffer(0, work_size)};

    // Step 1: r2c FFT along X for each component
    for (Index_t comp{0}; comp < nb_components; ++comp) {
      Index_t in_comp_offset{comp * in_comp_factor};
      Index_t work_comp_offset{comp * work_comp_factor};
      backend->r2c(Nx, local_real[1],
                   input_ptr + in_base_offset + in_comp_offset,
                   in_x_stride, in_row_dist,
                   work_ptr + work_comp_offset,
                   work_x_stride, work_row_dist);
    }

    // Check output storage order (should match work buffer storage order)
    StorageOrder out_storage_order{output.get_storage_order()};
    bool out_is_soa{(out_storage_order == StorageOrder::StructureOfArrays)};

    // Output field strides
    auto [out_comp_factor, out_x_stride, out_row_dist] =
        out_is_soa ? get_soa_strides(nb_fourier_pixels, Fx)
                   : get_aos_strides(nb_fourier_pixels, Fx);

    // Step 2 & 3: Transpose/copy and c2c FFT along Y
    Transpose * transpose = this->get_transpose_xy(nb_components, storage_order);
    if (transpose != nullptr) {
      // MPI path: transpose to output, then c2c on output
      transpose->forward(work_ptr, output_ptr);

      Index_t local_fx{this->nb_fourier_subdomain_grid_pts[0]};
      Index_t local_fourier_pixels{local_fx * Ny};

      // Recalculate output strides for MPI path (different shape)
      auto [mpi_out_comp_factor, mpi_out_x_stride, mpi_out_row_dist] =
          out_is_soa ? get_soa_strides(local_fourier_pixels, local_fx)
                     : get_aos_strides(local_fourier_pixels, local_fx);

      // For c2c along Y: stride is between Y elements (row distance)
      // batch dimension is X
      for (Index_t comp{0}; comp < nb_components; ++comp) {
        Index_t comp_offset{comp * mpi_out_comp_factor};
        backend->c2c_forward(Ny, local_fx,
                             output_ptr + comp_offset, mpi_out_row_dist, mpi_out_x_stride,
                             output_ptr + comp_offset, mpi_out_row_dist, mpi_out_x_stride);
      }
    } else {
      // Serial path: c2c on work buffer, then copy to output
      Index_t local_fy{local_real[1]};

      // Step 2: c2c FFT along Y on work buffer
      // For c2c along Y: stride is between Y elements, batch is X
      for (Index_t comp{0}; comp < nb_components; ++comp) {
        Index_t work_comp_offset{comp * work_comp_factor};
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

    Index_t nb_components{input.get_nb_components()};
    if (output.get_nb_components() != nb_components) {
      throw RuntimeError(
          "Input and output fields must have the same number of components");
    }

    Index_t Nx{nb_grid_pts[0]};
    Index_t Ny{nb_grid_pts[1]};
    Index_t Nz{nb_grid_pts[2]};
    Index_t Fx{Nx / 2 + 1};

    constexpr bool is_device = is_device_space_v<MemorySpace>;
    const Real * input_ptr =
        static_cast<const Real *>(input.get_void_data_ptr(!is_device));
    Complex * output_ptr =
        static_cast<Complex *>(output.get_void_data_ptr(!is_device));

    // Storage order: SoA on GPU, AoS on CPU
    StorageOrder storage_order{input.get_storage_order()};
    bool is_soa{(storage_order == StorageOrder::StructureOfArrays)};

    Transpose * transpose_xy =
        this->get_transpose_xy(nb_components, storage_order);
    Transpose * transpose_yz =
        this->get_transpose_yz(nb_components, storage_order);

    bool need_mpi_path{(transpose_xy != nullptr || transpose_yz != nullptr)};

    Index_t nb_buffer_pixels{
        local_with_ghosts[0] * local_with_ghosts[1] * local_with_ghosts[2]};
    Index_t ghost_pixel_offset{
        ghosts_left[0] + ghosts_left[1] * local_with_ghosts[0] +
        ghosts_left[2] * local_with_ghosts[0] * local_with_ghosts[1]};

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
    Index_t in_base_offset{is_soa ? ghost_pixel_offset : ghost_pixel_offset * nb_components};

    if (need_mpi_path) {
      // MPI path with transposes. Set up the Y-pencil layout [Fx/P2, Ny, Nz/P1]
      // and its work buffer (target of the X<->Y stage) first, so the slab fast
      // path can write straight into it.
      const DynGridIndex & ypencil_shape =
          this->work_ypencil->get_nb_subdomain_grid_pts_with_ghosts();
      Index_t local_yfx{ypencil_shape[0]};
      Index_t nb_ypencil_pixels{local_yfx * Ny * ypencil_shape[2]};
      Index_t ypencil_size{nb_ypencil_pixels * nb_components};

      auto [work_y_comp_factor, work_y_x_stride, work_y_y_dist, work_y_z_dist] =
          is_soa ? get_soa_strides_3d(nb_ypencil_pixels, local_yfx, Ny)
                 : get_aos_strides_3d(nb_ypencil_pixels, local_yfx, Ny);

      Complex * work_y_ptr{this->get_work_buffer(1, ypencil_size)};

      if (transpose_xy == nullptr && backend->supports_nd()) {
        // Slab fast path (P2 == 1: X and Y are both local). Transform the two
        // local axes (Y, and the half-complex X last) in a single planned
        // rank-2 r2c per component, batched over the local Z planes, written
        // straight into the Y-pencil layout. This replaces the per-Z-plane
        // r2c-X loop, the X<->Y copy and the per-Z-plane c2c-Y loop with one
        // native cuFFT/rocFFT/pocketFFT N-D transform per component.
        std::vector<Index_t> shape{local_real[2], Ny, Nx};
        std::vector<Index_t> axes{1, 2};
        std::vector<Index_t> in_strides{in_z_dist, in_y_dist, in_x_stride};
        std::vector<Index_t> out_strides{work_y_z_dist, work_y_y_dist,
                                         work_y_x_stride};
        for (Index_t comp{0}; comp < nb_components; ++comp) {
          backend->r2c_nd(
              shape, axes,
              input_ptr + in_base_offset + comp * in_comp_factor, in_strides,
              work_y_ptr + comp * work_y_comp_factor, out_strides);
        }
      } else {
        // General pencil path: r2c along X into a Z-pencil, redistribute X<->Y,
        // then c2c along Y, axis-by-axis.
        Index_t nb_zpencil_pixels{Fx * local_real[1] * local_real[2]};
        Index_t zpencil_size{nb_zpencil_pixels * nb_components};
        auto [work_z_comp_factor, work_z_x_stride, work_z_y_dist,
              work_z_z_dist] =
            is_soa ? get_soa_strides_3d(nb_zpencil_pixels, Fx, local_real[1])
                   : get_aos_strides_3d(nb_zpencil_pixels, Fx, local_real[1]);
        Complex * work_z_ptr{this->get_work_buffer(0, zpencil_size)};

        // Step 1: r2c FFT along X for each component
        for (Index_t comp{0}; comp < nb_components; ++comp) {
          Index_t in_comp_offset{comp * in_comp_factor};
          Index_t work_comp_offset{comp * work_z_comp_factor};
          for (Index_t iz{0}; iz < local_real[2]; ++iz) {
            Index_t in_idx{in_base_offset + in_comp_offset + iz * in_z_dist};
            Index_t out_idx{work_comp_offset + iz * work_z_z_dist};
            backend->r2c(Nx, local_real[1], input_ptr + in_idx,
                         in_x_stride, in_y_dist,
                         work_z_ptr + out_idx,
                         work_z_x_stride, work_z_y_dist);
          }
        }

        // Step 2a: Transpose X<->Y (scatter X across P2, gather Y)
        if (transpose_xy != nullptr) {
          transpose_xy->forward(work_z_ptr, work_y_ptr);
        } else {
          // No X<->Y redistribution (P2 == 1): same shape, copy through.
          deep_copy<Complex, MemorySpace>(work_y_ptr, work_z_ptr, ypencil_size);
        }

        // Step 2b: c2c FFT along Y for each component
        for (Index_t comp{0}; comp < nb_components; ++comp) {
          Index_t comp_offset{comp * work_y_comp_factor};
          for (Index_t iz{0}; iz < ypencil_shape[2]; ++iz) {
            Index_t idx{comp_offset + iz * work_y_z_dist};
            backend->c2c_forward(Ny, local_yfx, work_y_ptr + idx,
                                 work_y_y_dist, work_y_x_stride,
                                 work_y_ptr + idx,
                                 work_y_y_dist, work_y_x_stride);
          }
        }
      }

      // Step 3: Transpose Y<->Z (scatter Y across P1, gather Z) into the
      // final Fourier X-pencil layout [Fx/P2, Ny/P1, Nz]
      const DynGridIndex & fourier_local = this->nb_fourier_subdomain_grid_pts;
      Index_t nb_fourier_pixels{fourier_local[0] * fourier_local[1] * fourier_local[2]};
      Index_t fourier_size{nb_fourier_pixels * nb_components};

      // Output field strides
      StorageOrder out_storage_order{output.get_storage_order()};
      bool out_is_soa{(out_storage_order == StorageOrder::StructureOfArrays)};
      auto [out_comp_factor, out_x_stride, out_y_dist, out_z_dist] =
          out_is_soa ? get_soa_strides_3d(nb_fourier_pixels, fourier_local[0],
                                          fourier_local[1])
                     : get_aos_strides_3d(nb_fourier_pixels, fourier_local[0],
                                          fourier_local[1]);

      if (transpose_yz != nullptr) {
        transpose_yz->forward(work_y_ptr, output_ptr);
      } else {
        // No Y<->Z redistribution needed (P1 == 1): Z is already local and
        // the Fourier layout has the same shape as the Y-pencil.
        deep_copy<Complex, MemorySpace>(output_ptr, work_y_ptr, fourier_size);
      }

      // Step 4: c2c FFT along Z for each component
      // For SoA: Batch all local_fx * local_fy transforms together
      // For AoS: Batch local_fx transforms per Y row
      if (out_is_soa) {
        Index_t batch_z{fourier_local[0] * fourier_local[1]};
        for (Index_t comp{0}; comp < nb_components; ++comp) {
          Index_t comp_offset{comp * out_comp_factor};
          backend->c2c_forward(Nz, batch_z, output_ptr + comp_offset,
                               out_z_dist, Index_t{1},
                               output_ptr + comp_offset,
                               out_z_dist, Index_t{1});
        }
      } else {
        // AoS: X elements are separated by out_x_stride
        for (Index_t comp{0}; comp < nb_components; ++comp) {
          Index_t comp_offset{comp * out_comp_factor};
          for (Index_t iy{0}; iy < fourier_local[1]; ++iy) {
            Index_t idx{comp_offset + iy * out_y_dist};
            backend->c2c_forward(Nz, fourier_local[0], output_ptr + idx,
                                 out_z_dist, out_x_stride,
                                 output_ptr + idx,
                                 out_z_dist, out_x_stride);
          }
        }
      }
    } else {
      // Serial path: all dimensions are local
      Index_t nb_fourier_pixels{Fx * Ny * Nz};
      Index_t work_size{nb_fourier_pixels * nb_components};

      // Output (Fourier) field strides
      StorageOrder out_storage_order{output.get_storage_order()};
      bool out_is_soa{(out_storage_order == StorageOrder::StructureOfArrays)};
      auto [out_comp_factor, out_x_stride, out_y_dist, out_z_dist] =
          out_is_soa ? get_soa_strides_3d(nb_fourier_pixels, Fx, Ny)
                     : get_aos_strides_3d(nb_fourier_pixels, Fx, Ny);

      if (backend->supports_nd()) {
        // Fast path: one planned transform over all three axes plus the
        // component batch, written straight into the output. No work buffer,
        // no intermediate copy, no axis-by-axis re-dispatch. The half-complex
        // (x) axis must be last in `axes`; the component axis (0) is a
        // non-transformed batch axis. Strides fold in the input ghost layout
        // and the AoS/SoA component layout, so no repacking is needed.
        std::vector<Index_t> shape{nb_components, Nz, Ny, Nx};
        std::vector<Index_t> axes{1, 2, 3};
        std::vector<Index_t> in_strides{in_comp_factor, in_z_dist, in_y_dist,
                                        in_x_stride};
        std::vector<Index_t> out_strides{out_comp_factor, out_z_dist, out_y_dist,
                                         out_x_stride};
        backend->r2c_nd(shape, axes, input_ptr + in_base_offset, in_strides,
                        output_ptr, out_strides);
        return;
      }

      // Work/output buffer strides (same storage order as input)
      auto [work_comp_factor, work_x_stride, work_y_dist, work_z_dist] =
          is_soa ? get_soa_strides_3d(nb_fourier_pixels, Fx, Ny)
                 : get_aos_strides_3d(nb_fourier_pixels, Fx, Ny);

      Complex * work_ptr{this->get_work_buffer(0, work_size)};

      // Step 1: r2c FFT along X for each component
      for (Index_t comp{0}; comp < nb_components; ++comp) {
        Index_t in_comp_offset{comp * in_comp_factor};
        Index_t work_comp_offset{comp * work_comp_factor};

        for (Index_t iz{0}; iz < Nz; ++iz) {
          Index_t in_idx{in_base_offset + in_comp_offset + iz * in_z_dist};
          Index_t work_idx{work_comp_offset + iz * work_z_dist};
          backend->r2c(Nx, Ny, input_ptr + in_idx,
                       in_x_stride, in_y_dist,
                       work_ptr + work_idx,
                       work_x_stride, work_y_dist);
        }
      }

      // Step 2: c2c FFT along Y for each component
      // Batch Fx transforms for each Z plane
      // n=Ny, batch=Fx, stride=work_y_dist (between Y elements), dist=work_x_stride (between X batches)
      for (Index_t comp{0}; comp < nb_components; ++comp) {
        Index_t comp_offset{comp * work_comp_factor};
        for (Index_t iz{0}; iz < Nz; ++iz) {
          Index_t idx{comp_offset + iz * work_z_dist};
          backend->c2c_forward(Ny, Fx, work_ptr + idx, work_y_dist, work_x_stride,
                               work_ptr + idx, work_y_dist, work_x_stride);
        }
      }

      // Step 3: c2c FFT along Z for each component
      // For SoA: Batch all Fx*Ny transforms together with dist=1
      // For AoS: Batch Fx transforms per Y row with dist=work_x_stride
      if (is_soa) {
        // SoA: consecutive XY elements are separated by 1
        for (Index_t comp{0}; comp < nb_components; ++comp) {
          Index_t comp_offset{comp * work_comp_factor};
          backend->c2c_forward(Nz, Fx * Ny, work_ptr + comp_offset,
                               work_z_dist, Index_t{1},
                               work_ptr + comp_offset,
                               work_z_dist, Index_t{1});
        }
      } else {
        // AoS: X elements are separated by work_x_stride (nb_components)
        // Batch Fx transforms per Y row
        for (Index_t comp{0}; comp < nb_components; ++comp) {
          Index_t comp_offset{comp * work_comp_factor};
          for (Index_t iy{0}; iy < Ny; ++iy) {
            Index_t idx{comp_offset + iy * work_y_dist};
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

    Index_t nb_components{input.get_nb_components()};
    if (output.get_nb_components() != nb_components) {
      throw RuntimeError(
          "Input and output fields must have the same number of components");
    }

    constexpr bool is_device = is_device_space_v<MemorySpace>;
    const Complex * input_ptr =
        static_cast<const Complex *>(input.get_void_data_ptr(!is_device));
    Real * output_ptr =
        static_cast<Real *>(output.get_void_data_ptr(!is_device));

    Index_t Nx{nb_grid_pts[0]};
    Index_t Fx{Nx / 2 + 1};
    Index_t Ny{nb_grid_pts[1]};

    // Storage order: SoA on GPU, AoS on CPU
    StorageOrder storage_order{output.get_storage_order()};
    bool is_soa{(storage_order == StorageOrder::StructureOfArrays)};

    Index_t nb_buffer_pixels{local_with_ghosts[0] * local_with_ghosts[1]};
    Index_t ghost_pixel_offset{
        ghosts_left[0] + ghosts_left[1] * local_with_ghosts[0]};

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
    Index_t out_base_offset{is_soa ? ghost_pixel_offset : ghost_pixel_offset * nb_components};

    // Input (Fourier) field storage order should match
    StorageOrder in_storage_order{input.get_storage_order()};
    bool in_is_soa{(in_storage_order == StorageOrder::StructureOfArrays)};

    Transpose * transpose = this->get_transpose_xy(nb_components, storage_order);
    if (transpose != nullptr) {
      Index_t local_fx{this->nb_fourier_subdomain_grid_pts[0]};
      Index_t local_fourier_pixels{local_fx * Ny};
      Index_t local_fourier_size{local_fourier_pixels * nb_components};

      // Input strides for MPI path
      auto [in_comp_factor, in_x_stride, in_row_dist] =
          in_is_soa ? get_soa_strides(local_fourier_pixels, local_fx)
                    : get_aos_strides(local_fourier_pixels, local_fx);

      Complex * temp_ptr{this->get_work_buffer(0, local_fourier_size)};
      deep_copy<Complex, MemorySpace>(temp_ptr, input_ptr, local_fourier_size);

      // Step 1: c2c IFFT along Y for each component
      for (Index_t comp{0}; comp < nb_components; ++comp) {
        Index_t comp_offset{comp * in_comp_factor};
        backend->c2c_backward(Ny, local_fx,
                              temp_ptr + comp_offset, in_row_dist, in_x_stride,
                              temp_ptr + comp_offset, in_row_dist, in_x_stride);
      }

      // Step 2: Transpose X<->Y backward
      Index_t nb_work_pixels{Fx * local_real[1]};
      Index_t work_size{nb_work_pixels * nb_components};
      Complex * work_buffer_ptr{this->get_work_buffer(1, work_size)};

      transpose->backward(temp_ptr, work_buffer_ptr);

      // Work buffer strides (same storage order)
      auto [work_comp_factor, work_x_stride, work_row_dist] =
          is_soa ? get_soa_strides(nb_work_pixels, Fx)
                 : get_aos_strides(nb_work_pixels, Fx);

      // Step 3: c2r IFFT along X for each component
      for (Index_t comp{0}; comp < nb_components; ++comp) {
        Index_t work_comp_offset{comp * work_comp_factor};
        Index_t out_comp_offset{comp * out_comp_factor};
        backend->c2r(Nx, local_real[1],
                     work_buffer_ptr + work_comp_offset,
                     work_x_stride, work_row_dist,
                     output_ptr + out_base_offset + out_comp_offset,
                     out_x_stride, out_row_dist);
      }
    } else {
      // Serial path
      Index_t local_fy{local_real[1]};
      Index_t nb_fourier_pixels{Fx * local_fy};
      Index_t fourier_size{nb_fourier_pixels * nb_components};

      // Input strides
      auto [in_comp_factor, in_x_stride, in_row_dist] =
          in_is_soa ? get_soa_strides(nb_fourier_pixels, Fx)
                    : get_aos_strides(nb_fourier_pixels, Fx);

      // Work buffer uses same storage order
      Complex * temp_ptr{this->get_work_buffer(0, fourier_size)};
      deep_copy<Complex, MemorySpace>(temp_ptr, input_ptr, fourier_size);

      // Step 1: c2c IFFT along Y for each component
      for (Index_t comp{0}; comp < nb_components; ++comp) {
        Index_t comp_offset{comp * in_comp_factor};
        backend->c2c_backward(local_fy, Fx,
                              temp_ptr + comp_offset, in_row_dist, in_x_stride,
                              temp_ptr + comp_offset, in_row_dist, in_x_stride);
      }

      // Step 2: c2r IFFT along X for each component
      for (Index_t comp{0}; comp < nb_components; ++comp) {
        Index_t in_comp_offset{comp * in_comp_factor};
        Index_t out_comp_offset{comp * out_comp_factor};
        backend->c2r(Nx, local_fy,
                     temp_ptr + in_comp_offset,
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

    Index_t nb_components{input.get_nb_components()};
    if (output.get_nb_components() != nb_components) {
      throw RuntimeError(
          "Input and output fields must have the same number of components");
    }

    Index_t Nx{nb_grid_pts[0]};
    Index_t Ny{nb_grid_pts[1]};
    Index_t Nz{nb_grid_pts[2]};
    Index_t Fx{Nx / 2 + 1};

    constexpr bool is_device = is_device_space_v<MemorySpace>;
    const Complex * input_ptr =
        static_cast<const Complex *>(input.get_void_data_ptr(!is_device));
    Real * output_ptr =
        static_cast<Real *>(output.get_void_data_ptr(!is_device));

    // Storage order: SoA on GPU, AoS on CPU
    StorageOrder storage_order{output.get_storage_order()};
    bool is_soa{(storage_order == StorageOrder::StructureOfArrays)};

    Index_t nb_buffer_pixels{
        local_with_ghosts[0] * local_with_ghosts[1] * local_with_ghosts[2]};
    Index_t ghost_pixel_offset{
        ghosts_left[0] + ghosts_left[1] * local_with_ghosts[0] +
        ghosts_left[2] * local_with_ghosts[0] * local_with_ghosts[1]};

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
    Index_t out_base_offset{is_soa ? ghost_pixel_offset : ghost_pixel_offset * nb_components};

    // Input (Fourier) field storage order should match
    StorageOrder in_storage_order{input.get_storage_order()};
    bool in_is_soa{(in_storage_order == StorageOrder::StructureOfArrays)};

    Transpose * transpose_xy =
        this->get_transpose_xy(nb_components, storage_order);
    Transpose * transpose_yz =
        this->get_transpose_yz(nb_components, storage_order);

    bool need_mpi_path{(transpose_xy != nullptr || transpose_yz != nullptr)};

    if (need_mpi_path) {
      // MPI path with transposes
      const DynGridIndex & fourier_local = this->nb_fourier_subdomain_grid_pts;
      Index_t nb_fourier_pixels{fourier_local[0] * fourier_local[1] *
                                fourier_local[2]};
      Index_t fourier_size{nb_fourier_pixels * nb_components};

      // Input field strides
      auto [in_comp_factor, in_x_stride, in_y_dist, in_z_dist] =
          in_is_soa ? get_soa_strides_3d(nb_fourier_pixels, fourier_local[0],
                                         fourier_local[1])
                    : get_aos_strides_3d(nb_fourier_pixels, fourier_local[0],
                                         fourier_local[1]);

      Complex * temp_ptr{this->get_work_buffer(0, fourier_size)};
      deep_copy<Complex, MemorySpace>(temp_ptr, input_ptr, fourier_size);

      // Step 1: c2c IFFT along Z for each component
      // For SoA: Batch all local_fx * local_fy transforms together
      // For AoS: Batch local_fx transforms per Y row
      if (in_is_soa) {
        Index_t batch_z{fourier_local[0] * fourier_local[1]};
        for (Index_t comp{0}; comp < nb_components; ++comp) {
          Index_t comp_offset{comp * in_comp_factor};
          backend->c2c_backward(Nz, batch_z, temp_ptr + comp_offset,
                                in_z_dist, Index_t{1},
                                temp_ptr + comp_offset,
                                in_z_dist, Index_t{1});
        }
      } else {
        // AoS: X elements are separated by in_x_stride
        for (Index_t comp{0}; comp < nb_components; ++comp) {
          Index_t comp_offset{comp * in_comp_factor};
          for (Index_t iy{0}; iy < fourier_local[1]; ++iy) {
            Index_t idx{comp_offset + iy * in_y_dist};
            backend->c2c_backward(Nz, fourier_local[0], temp_ptr + idx,
                                  in_z_dist, in_x_stride,
                                  temp_ptr + idx,
                                  in_z_dist, in_x_stride);
          }
        }
      }

      // Y-pencil work buffer [Fx/P2, Ny, Nz/P1]
      const DynGridIndex & ypencil_shape =
          this->work_ypencil->get_nb_subdomain_grid_pts_with_ghosts();
      Index_t local_yfx{ypencil_shape[0]};
      Index_t nb_ypencil_pixels{local_yfx * Ny * ypencil_shape[2]};
      Index_t ypencil_size{nb_ypencil_pixels * nb_components};

      // Y-pencil strides
      auto [work_y_comp_factor, work_y_x_stride, work_y_y_dist, work_y_z_dist] =
          is_soa ? get_soa_strides_3d(nb_ypencil_pixels, local_yfx, Ny)
                 : get_aos_strides_3d(nb_ypencil_pixels, local_yfx, Ny);

      Complex * work_y_ptr{this->get_work_buffer(1, ypencil_size)};

      // Step 2: Transpose Y<->Z backward (gather Y, scatter Z across P1)
      if (transpose_yz != nullptr) {
        transpose_yz->backward(temp_ptr, work_y_ptr);
      } else {
        // No Y<->Z redistribution needed (P1 == 1): the Fourier layout has
        // the same shape as the Y-pencil; copy the data through so the
        // Y-IFFT below does not run on an uninitialised work_y.
        deep_copy<Complex, MemorySpace>(work_y_ptr, temp_ptr, ypencil_size);
      }

      if (transpose_xy == nullptr && backend->supports_nd()) {
        // Slab fast path (P2 == 1: X and Y both local). Fuse the inverse Y
        // (c2c) and X (c2r, half-complex) transforms into one planned rank-2
        // c2r per component, batched over the local Z planes, reading the
        // Y-pencil and writing the real output directly. Replaces the c2c-Y
        // loop, the X<->Y copy and the c2r-X loop (mirrors the forward path).
        std::vector<Index_t> shape{local_real[2], Ny, Nx};
        std::vector<Index_t> axes{1, 2};
        std::vector<Index_t> in_strides{work_y_z_dist, work_y_y_dist,
                                        work_y_x_stride};
        std::vector<Index_t> out_strides{out_z_dist, out_y_dist, out_x_stride};
        for (Index_t comp{0}; comp < nb_components; ++comp) {
          backend->c2r_nd(
              shape, axes,
              work_y_ptr + comp * work_y_comp_factor, in_strides,
              output_ptr + out_base_offset + comp * out_comp_factor,
              out_strides);
        }
      } else {
        // Step 3: c2c IFFT along Y for each component
        for (Index_t comp{0}; comp < nb_components; ++comp) {
          Index_t comp_offset{comp * work_y_comp_factor};
          for (Index_t iz{0}; iz < ypencil_shape[2]; ++iz) {
            Index_t idx{comp_offset + iz * work_y_z_dist};
            backend->c2c_backward(Ny, local_yfx, work_y_ptr + idx,
                                  work_y_y_dist, work_y_x_stride,
                                  work_y_ptr + idx,
                                  work_y_y_dist, work_y_x_stride);
          }
        }

        // Z-pencil work buffer [Fx, Ny/P2, Nz/P1]
        Index_t nb_zpencil_pixels{Fx * local_real[1] * local_real[2]};
        Index_t zpencil_size{nb_zpencil_pixels * nb_components};
        auto [work_z_comp_factor, work_z_x_stride, work_z_y_dist,
              work_z_z_dist] =
            is_soa ? get_soa_strides_3d(nb_zpencil_pixels, Fx, local_real[1])
                   : get_aos_strides_3d(nb_zpencil_pixels, Fx, local_real[1]);
        Complex * work_z_ptr{this->get_work_buffer(2, zpencil_size)};

        // Step 4a: Transpose X<->Y backward (gather X, scatter Y across P2)
        if (transpose_xy != nullptr) {
          transpose_xy->backward(work_y_ptr, work_z_ptr);
        } else {
          deep_copy<Complex, MemorySpace>(work_z_ptr, work_y_ptr, zpencil_size);
        }

        // Step 4b: c2r IFFT along X for each component
        for (Index_t comp{0}; comp < nb_components; ++comp) {
          Index_t work_comp_offset{comp * work_z_comp_factor};
          Index_t out_comp_offset{comp * out_comp_factor};
          for (Index_t iz{0}; iz < local_real[2]; ++iz) {
            Index_t in_idx{work_comp_offset + iz * work_z_z_dist};
            Index_t out_idx{out_base_offset + out_comp_offset +
                            iz * out_z_dist};
            backend->c2r(Nx, local_real[1], work_z_ptr + in_idx,
                         work_z_x_stride, work_z_y_dist,
                         output_ptr + out_idx, out_x_stride, out_y_dist);
          }
        }
      }
    } else {
      // Serial path: all dimensions are local
      Index_t nb_fourier_pixels{Fx * Ny * Nz};
      Index_t work_size{nb_fourier_pixels * nb_components};

      // Input strides
      auto [in_comp_factor, in_x_stride, in_y_dist, in_z_dist] =
          in_is_soa ? get_soa_strides_3d(nb_fourier_pixels, Fx, Ny)
                    : get_aos_strides_3d(nb_fourier_pixels, Fx, Ny);

      if (backend->supports_nd()) {
        // Fast path: single planned inverse transform over all three axes plus
        // the component batch (mirrors the forward fast path). `shape` is the
        // real-space (output) extent; the c2r (x) axis is last. pocketfft's
        // multi-axis c2r stages the non-real axes into its own temporary, so
        // the const Fourier input is not modified and no work-buffer copy is
        // needed. Output strides fold in the destination ghost layout.
        std::vector<Index_t> shape{nb_components, Nz, Ny, Nx};
        std::vector<Index_t> axes{1, 2, 3};
        std::vector<Index_t> in_strides{in_comp_factor, in_z_dist, in_y_dist,
                                        in_x_stride};
        std::vector<Index_t> out_strides{out_comp_factor, out_z_dist, out_y_dist,
                                         out_x_stride};
        backend->c2r_nd(shape, axes, input_ptr, in_strides,
                        output_ptr + out_base_offset, out_strides);
        return;
      }

      // Work buffer with same storage order
      Complex * work_ptr{this->get_work_buffer(0, work_size)};
      deep_copy<Complex, MemorySpace>(work_ptr, input_ptr, work_size);

      // Step 1: c2c IFFT along Z for each component
      // For SoA: Batch all Fx*Ny transforms together with dist=1
      // For AoS: Batch Fx transforms per Y row with dist=in_x_stride
      if (in_is_soa) {
        // SoA: consecutive XY elements are separated by 1
        for (Index_t comp{0}; comp < nb_components; ++comp) {
          Index_t comp_offset{comp * in_comp_factor};
          backend->c2c_backward(Nz, Fx * Ny, work_ptr + comp_offset,
                                in_z_dist, Index_t{1},
                                work_ptr + comp_offset,
                                in_z_dist, Index_t{1});
        }
      } else {
        // AoS: X elements are separated by in_x_stride (nb_components)
        // Batch Fx transforms per Y row
        for (Index_t comp{0}; comp < nb_components; ++comp) {
          Index_t comp_offset{comp * in_comp_factor};
          for (Index_t iy{0}; iy < Ny; ++iy) {
            Index_t idx{comp_offset + iy * in_y_dist};
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
      for (Index_t comp{0}; comp < nb_components; ++comp) {
        Index_t comp_offset{comp * in_comp_factor};
        for (Index_t iz{0}; iz < Nz; ++iz) {
          Index_t idx{comp_offset + iz * in_z_dist};
          backend->c2c_backward(Ny, Fx, work_ptr + idx, in_y_dist, in_x_stride,
                                work_ptr + idx, in_y_dist, in_x_stride);
        }
      }

      // Step 3: c2r IFFT along X for each component
      for (Index_t comp{0}; comp < nb_components; ++comp) {
        Index_t in_comp_offset{comp * in_comp_factor};
        Index_t out_comp_offset{comp * out_comp_factor};

        for (Index_t iz{0}; iz < Nz; ++iz) {
          Index_t in_idx{in_comp_offset + iz * in_z_dist};
          Index_t out_idx{out_base_offset + out_comp_offset + iz * out_z_dist};
          backend->c2r(Nx, Ny, work_ptr + in_idx,
                       in_x_stride, in_y_dist,
                       output_ptr + out_idx, out_x_stride, out_y_dist);
        }
      }
    }
  }

  //! Return scratch space of at least `size` complex values for `slot`.
  //! The buffers grow on demand and are reused across transforms: a fresh
  //! device allocation per fft/ifft call is expensive and can fail when
  //! the device is near capacity (e.g. when a cupy memory pool holds the
  //! remaining free memory).
  Complex * get_work_buffer(std::size_t slot, Index_t size) {
    WorkBuffer & buf{this->work_buffers.at(slot)};
    if (buf.size() < static_cast<std::size_t>(size)) {
      buf.resize(static_cast<std::size_t>(size));
    }
    return buf.data();
  }

  //! FFT backend for this memory space
  std::unique_ptr<FFT1DBackend> backend;

  //! Scratch buffers reused across transforms; at most three are alive at
  //! once (3D MPI inverse transform)
  std::array<WorkBuffer, 3> work_buffers{};
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

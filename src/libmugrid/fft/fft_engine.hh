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
 * @tparam MemorySpace The memory space for work buffers (HostSpace, CudaSpace,
 * HIPSpace)
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
   */
  FFTEngine(const DynGridIndex & nb_domain_grid_pts,
            const Communicator & comm = Communicator(),
            const DynGridIndex & nb_ghosts_left = DynGridIndex{},
            const DynGridIndex & nb_ghosts_right = DynGridIndex{},
            const SubPtMap_t & nb_sub_pts = {})
      : Parent_t{nb_domain_grid_pts, comm, nb_ghosts_left, nb_ghosts_right,
                 nb_sub_pts, memory_location<MemorySpace>()},
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

    Index_t in_comp_stride = nb_components;
    Index_t in_dist = local_with_ghosts[0] * nb_components;
    Index_t in_base_offset =
        (ghosts_left[0] + ghosts_left[1] * local_with_ghosts[0]) *
        nb_components;

    Index_t work_size = Fx * local_real[1] * nb_components;
    WorkBuffer work_buffer(work_size);
    Complex * work_ptr = work_buffer.data();

    // Step 1: r2c FFT along X for each component
    for (Index_t comp = 0; comp < nb_components; ++comp) {
      backend->r2c(Nx, local_real[1], input_ptr + in_base_offset + comp,
                   in_comp_stride, in_dist, work_ptr + comp, nb_components,
                   Fx * nb_components);
    }

    // Step 2: Transpose (or copy for serial)
    Transpose * transpose = this->get_transpose_xz(nb_components);
    if (transpose != nullptr) {
      transpose->forward(work_ptr, output_ptr);
    } else {
      deep_copy<Complex, MemorySpace>(output_ptr, work_ptr, work_size);
    }

    // Step 3: c2c FFT along Y for all components
    if (transpose != nullptr) {
      Index_t local_fx = this->nb_fourier_subdomain_grid_pts[0];
      Index_t y_stride = local_fx * nb_components;
      Index_t y_dist = nb_components;

      for (Index_t comp = 0; comp < nb_components; ++comp) {
        backend->c2c_forward(Ny, local_fx, output_ptr + comp, y_stride, y_dist,
                             output_ptr + comp, y_stride, y_dist);
      }
    } else {
      Index_t local_fy = local_real[1];
      Index_t y_stride = Fx * nb_components;
      Index_t y_dist = nb_components;

      for (Index_t comp = 0; comp < nb_components; ++comp) {
        backend->c2c_forward(local_fy, Fx, output_ptr + comp, y_stride, y_dist,
                             output_ptr + comp, y_stride, y_dist);
      }
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

    Index_t in_comp_stride = nb_components;
    Index_t in_stride_y = local_with_ghosts[0] * nb_components;
    Index_t in_stride_z = in_stride_y * local_with_ghosts[1];
    Index_t in_base_offset =
        (ghosts_left[0] + ghosts_left[1] * local_with_ghosts[0] +
         ghosts_left[2] * local_with_ghosts[0] * local_with_ghosts[1]) *
        nb_components;

    if (need_mpi_path) {
      // MPI path with transposes
      Index_t zpencil_size = Fx * local_real[1] * local_real[2] * nb_components;
      WorkBuffer work_z(zpencil_size);
      Complex * work_z_ptr = work_z.data();

      // Step 1: r2c FFT along X for each component
      for (Index_t comp = 0; comp < nb_components; ++comp) {
        for (Index_t iz = 0; iz < local_real[2]; ++iz) {
          for (Index_t iy = 0; iy < local_real[1]; ++iy) {
            Index_t in_idx =
                in_base_offset + comp + iy * in_stride_y + iz * in_stride_z;
            Index_t out_idx = comp + iy * Fx * nb_components +
                              iz * Fx * local_real[1] * nb_components;
            backend->r2c(Nx, 1, input_ptr + in_idx, in_comp_stride, 0,
                         work_z_ptr + out_idx, nb_components, 0);
          }
        }
      }

      // Step 2a: Transpose Y<->Z
      const DynGridIndex & ypencil_shape =
          this->work_ypencil->get_nb_subdomain_grid_pts_with_ghosts();
      Index_t ypencil_size = Fx * Ny * ypencil_shape[2] * nb_components;
      WorkBuffer work_y(ypencil_size);
      Complex * work_y_ptr = work_y.data();

      if (transpose_yz_fwd != nullptr) {
        transpose_yz_fwd->forward(work_z_ptr, work_y_ptr);
      }

      // Step 2b: c2c FFT along Y for each component
      for (Index_t comp = 0; comp < nb_components; ++comp) {
        for (Index_t iz = 0; iz < ypencil_shape[2]; ++iz) {
          for (Index_t ix = 0; ix < Fx; ++ix) {
            Index_t idx =
                comp + ix * nb_components + iz * Fx * Ny * nb_components;
            Index_t y_stride = Fx * nb_components;
            backend->c2c_forward(Ny, 1, work_y_ptr + idx, y_stride, 0,
                                 work_y_ptr + idx, y_stride, 0);
          }
        }
      }

      // Step 2c: Transpose Z<->Y
      if (transpose_yz_bwd != nullptr) {
        transpose_yz_bwd->forward(work_y_ptr, work_z_ptr);
      }

      // Step 3: Transpose X<->Z (or copy if no transpose needed)
      const DynGridIndex & fourier_local = this->nb_fourier_subdomain_grid_pts;
      if (transpose_xz != nullptr) {
        transpose_xz->forward(work_z_ptr, output_ptr);
      } else {
        Index_t fourier_size =
            fourier_local[0] * fourier_local[1] * fourier_local[2] *
            nb_components;
        deep_copy<Complex, MemorySpace>(output_ptr, work_z_ptr, fourier_size);
      }

      // Step 4: c2c FFT along Z for each component
      for (Index_t comp = 0; comp < nb_components; ++comp) {
        for (Index_t iy = 0; iy < fourier_local[1]; ++iy) {
          for (Index_t ix = 0; ix < fourier_local[0]; ++ix) {
            Index_t idx = comp + ix * nb_components +
                          iy * fourier_local[0] * nb_components;
            Index_t z_stride =
                fourier_local[0] * fourier_local[1] * nb_components;
            backend->c2c_forward(Nz, 1, output_ptr + idx, z_stride, 0,
                                 output_ptr + idx, z_stride, 0);
          }
        }
      }
    } else {
      // Serial path: all dimensions are local
      Index_t out_comp_stride = nb_components;
      Index_t out_stride_y = Fx * nb_components;
      Index_t out_stride_z = out_stride_y * Ny;

      for (Index_t comp = 0; comp < nb_components; ++comp) {
        // Step 1: r2c FFT along X
        for (Index_t iz = 0; iz < Nz; ++iz) {
          for (Index_t iy = 0; iy < Ny; ++iy) {
            Index_t in_idx =
                in_base_offset + comp + iy * in_stride_y + iz * in_stride_z;
            Index_t out_idx = comp + iy * out_stride_y + iz * out_stride_z;
            backend->r2c(Nx, 1, input_ptr + in_idx, in_comp_stride, 0,
                         output_ptr + out_idx, out_comp_stride, 0);
          }
        }

        // Step 2: c2c FFT along Y
        for (Index_t iz = 0; iz < Nz; ++iz) {
          for (Index_t ix = 0; ix < Fx; ++ix) {
            Index_t idx = comp + ix * nb_components + iz * out_stride_z;
            backend->c2c_forward(Ny, 1, output_ptr + idx, out_stride_y, 0,
                                 output_ptr + idx, out_stride_y, 0);
          }
        }

        // Step 3: c2c FFT along Z
        for (Index_t iy = 0; iy < Ny; ++iy) {
          for (Index_t ix = 0; ix < Fx; ++ix) {
            Index_t idx = comp + ix * nb_components + iy * out_stride_y;
            backend->c2c_forward(Nz, 1, output_ptr + idx, out_stride_z, 0,
                                 output_ptr + idx, out_stride_z, 0);
          }
        }
      }
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

    Index_t out_comp_stride = nb_components;
    Index_t out_dist = local_with_ghosts[0] * nb_components;
    Index_t out_base_offset =
        (ghosts_left[0] + ghosts_left[1] * local_with_ghosts[0]) *
        nb_components;

    Transpose * transpose = this->get_transpose_xz(nb_components);
    if (transpose != nullptr) {
      Index_t local_fx = this->nb_fourier_subdomain_grid_pts[0];
      Index_t local_fourier_size = local_fx * Ny * nb_components;

      WorkBuffer temp(local_fourier_size);
      deep_copy<Complex, MemorySpace>(temp.data(), input_ptr,
                                      local_fourier_size);

      // Step 1: c2c IFFT along Y for each component
      Index_t y_stride = local_fx * nb_components;
      Index_t y_dist = nb_components;

      for (Index_t comp = 0; comp < nb_components; ++comp) {
        backend->c2c_backward(Ny, local_fx, temp.data() + comp, y_stride,
                              y_dist, temp.data() + comp, y_stride, y_dist);
      }

      // Step 2: Transpose X<->Y backward
      Index_t work_size = Fx * local_real[1] * nb_components;
      WorkBuffer work_buffer(work_size);
      transpose->backward(temp.data(), work_buffer.data());

      // Step 3: c2r IFFT along X for each component
      for (Index_t comp = 0; comp < nb_components; ++comp) {
        backend->c2r(Nx, local_real[1], work_buffer.data() + comp,
                     nb_components, Fx * nb_components,
                     output_ptr + out_base_offset + comp, out_comp_stride,
                     out_dist);
      }
    } else {
      Index_t local_fy = local_real[1];
      Index_t fourier_size = Fx * local_fy * nb_components;

      WorkBuffer temp(fourier_size);
      deep_copy<Complex, MemorySpace>(temp.data(), input_ptr, fourier_size);

      // Step 1: c2c IFFT along Y for each component
      Index_t y_stride = Fx * nb_components;
      Index_t y_dist = nb_components;

      for (Index_t comp = 0; comp < nb_components; ++comp) {
        backend->c2c_backward(local_fy, Fx, temp.data() + comp, y_stride,
                              y_dist, temp.data() + comp, y_stride, y_dist);
      }

      // Step 2: c2r IFFT along X for each component
      for (Index_t comp = 0; comp < nb_components; ++comp) {
        backend->c2r(Nx, local_fy, temp.data() + comp, nb_components,
                     Fx * nb_components, output_ptr + out_base_offset + comp,
                     out_comp_stride, out_dist);
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

    Index_t out_comp_stride = nb_components;
    Index_t out_stride_y = local_with_ghosts[0] * nb_components;
    Index_t out_stride_z = out_stride_y * local_with_ghosts[1];
    Index_t out_base_offset =
        (ghosts_left[0] + ghosts_left[1] * local_with_ghosts[0] +
         ghosts_left[2] * local_with_ghosts[0] * local_with_ghosts[1]) *
        nb_components;

    if (need_mpi_path) {
      // MPI path with transposes
      const DynGridIndex & fourier_local = this->nb_fourier_subdomain_grid_pts;
      Index_t fourier_size =
          fourier_local[0] * fourier_local[1] * fourier_local[2] *
          nb_components;
      WorkBuffer temp(fourier_size);
      deep_copy<Complex, MemorySpace>(temp.data(), input_ptr, fourier_size);

      // Step 1: c2c IFFT along Z for each component
      for (Index_t comp = 0; comp < nb_components; ++comp) {
        for (Index_t iy = 0; iy < fourier_local[1]; ++iy) {
          for (Index_t ix = 0; ix < fourier_local[0]; ++ix) {
            Index_t idx = comp + ix * nb_components +
                          iy * fourier_local[0] * nb_components;
            Index_t z_stride =
                fourier_local[0] * fourier_local[1] * nb_components;
            backend->c2c_backward(Nz, 1, temp.data() + idx, z_stride, 0,
                                  temp.data() + idx, z_stride, 0);
          }
        }
      }

      // Z-pencil work buffer
      Index_t zpencil_size = Fx * local_real[1] * local_real[2] * nb_components;
      WorkBuffer work_z(zpencil_size);
      Complex * work_z_ptr = work_z.data();

      // Step 2: Transpose Z<->X backward (or copy)
      if (transpose_xz != nullptr) {
        transpose_xz->backward(temp.data(), work_z_ptr);
      } else {
        deep_copy<Complex, MemorySpace>(work_z_ptr, temp.data(), zpencil_size);
      }

      // Y-pencil work buffer
      const DynGridIndex & ypencil_shape =
          this->work_ypencil->get_nb_subdomain_grid_pts_with_ghosts();
      Index_t ypencil_size = Fx * Ny * ypencil_shape[2] * nb_components;
      WorkBuffer work_y(ypencil_size);
      Complex * work_y_ptr = work_y.data();

      // Step 3a: Transpose Y<->Z backward
      if (transpose_yz_bwd != nullptr) {
        transpose_yz_bwd->backward(work_z_ptr, work_y_ptr);
      }

      // Step 3b: c2c IFFT along Y for each component
      for (Index_t comp = 0; comp < nb_components; ++comp) {
        for (Index_t iz = 0; iz < ypencil_shape[2]; ++iz) {
          for (Index_t ix = 0; ix < Fx; ++ix) {
            Index_t idx =
                comp + ix * nb_components + iz * Fx * Ny * nb_components;
            Index_t y_stride = Fx * nb_components;
            backend->c2c_backward(Ny, 1, work_y_ptr + idx, y_stride, 0,
                                  work_y_ptr + idx, y_stride, 0);
          }
        }
      }

      // Step 3c: Transpose Z<->Y backward
      if (transpose_yz_fwd != nullptr) {
        transpose_yz_fwd->backward(work_y_ptr, work_z_ptr);
      }

      // Step 4: c2r IFFT along X for each component
      for (Index_t comp = 0; comp < nb_components; ++comp) {
        for (Index_t iz = 0; iz < local_real[2]; ++iz) {
          for (Index_t iy = 0; iy < local_real[1]; ++iy) {
            Index_t in_idx = comp + iy * Fx * nb_components +
                             iz * Fx * local_real[1] * nb_components;
            Index_t out_idx =
                out_base_offset + comp + iy * out_stride_y + iz * out_stride_z;
            backend->c2r(Nx, 1, work_z_ptr + in_idx, nb_components, 0,
                         output_ptr + out_idx, out_comp_stride, 0);
          }
        }
      }
    } else {
      // Serial path: all dimensions are local
      Index_t fourier_size = Fx * Ny * Nz;

      for (Index_t comp = 0; comp < nb_components; ++comp) {
        // Copy this component's data to temp buffer
        WorkBuffer temp(fourier_size);
        if constexpr (is_host_space_v<MemorySpace>) {
          for (Index_t i = 0; i < fourier_size; ++i) {
            temp.data()[i] = input_ptr[i * nb_components + comp];
          }
        } else {
          // For device memory, we need a different approach
          // This is a limitation - strided copy on device requires a kernel
          throw RuntimeError(
              "Multi-component 3D serial IFFT on device memory not "
              "yet supported");
        }

        // Step 1: c2c IFFT along Z
        for (Index_t iy = 0; iy < Ny; ++iy) {
          for (Index_t ix = 0; ix < Fx; ++ix) {
            Index_t idx = ix + iy * Fx;
            Index_t stride = Fx * Ny;
            backend->c2c_backward(Nz, 1, temp.data() + idx, stride, 0,
                                  temp.data() + idx, stride, 0);
          }
        }

        // Step 2: c2c IFFT along Y
        for (Index_t iz = 0; iz < Nz; ++iz) {
          for (Index_t ix = 0; ix < Fx; ++ix) {
            Index_t idx = ix + iz * Fx * Ny;
            backend->c2c_backward(Ny, 1, temp.data() + idx, Fx, 0,
                                  temp.data() + idx, Fx, 0);
          }
        }

        // Step 3: c2r IFFT along X
        for (Index_t iz = 0; iz < Nz; ++iz) {
          for (Index_t iy = 0; iy < Ny; ++iy) {
            Index_t in_idx = iy * Fx + iz * Fx * Ny;
            Index_t out_idx =
                out_base_offset + comp + iy * out_stride_y + iz * out_stride_z;
            backend->c2r(Nx, 1, temp.data() + in_idx, 1, 0,
                         output_ptr + out_idx, out_comp_stride, 0);
          }
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
extern template class FFTEngine<CudaSpace>;
#endif
#if defined(MUGRID_ENABLE_HIP)
extern template class FFTEngine<HIPSpace>;
#endif

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_FFT_FFT_ENGINE_HH_

/**
 * @file   fft/fft_engine.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   18 Dec 2024
 *
 * @brief  Distributed FFT engine with pencil decomposition
 *
 * Copyright © 2024 Lars Pastewka
 *
 * µGrid is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µGrid is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µGrid; see the file COPYING. If not, write to the
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

#include "fft_engine.hh"
#include "core/exception.hh"
#include "field/field_typed.hh"

#ifdef WITH_MPI
#include <mpi.h>
#endif

#include <cmath>
#include <sstream>

namespace muGrid {

FFTEngine::FFTEngine(const IntCoord_t & nb_domain_grid_pts,
                     const Communicator & comm,
                     const IntCoord_t & nb_ghosts_left,
                     const IntCoord_t & nb_ghosts_right,
                     const SubPtMap_t & nb_sub_pts,
                     MemoryLocation memory_location)
    : Parent_t{comm, nb_domain_grid_pts.get_dim(), nb_sub_pts, memory_location},
      spatial_dim{nb_domain_grid_pts.get_dim()} {
  // Validate dimensions
  if (spatial_dim != 2 && spatial_dim != 3) {
    throw RuntimeError("FFTEngine only supports 2D and 3D grids");
  }

  // Initialize FFT backends
  this->host_backend = get_host_fft_backend();
  this->device_backend = get_device_fft_backend();

  // Set up the process grid for pencil decomposition
  int num_ranks = comm.size();
  select_process_grid(num_ranks, nb_domain_grid_pts, this->proc_grid_p1,
                      this->proc_grid_p2);

  // Compute this rank's position in the process grid
  int rank = comm.rank();
  this->proc_coord_p2 = rank % this->proc_grid_p2;
  this->proc_coord_p1 = rank / this->proc_grid_p2;

#ifdef WITH_MPI
  // Create row and column subcommunicators
  MPI_Comm mpi_comm = comm.get_mpi_comm();
  if (mpi_comm != MPI_COMM_NULL && num_ranks > 1) {
    MPI_Comm row_mpi_comm, col_mpi_comm;

    // Row communicator: ranks with same p1 (for Y redistribution in 3D)
    MPI_Comm_split(mpi_comm, this->proc_coord_p1, this->proc_coord_p2,
                   &row_mpi_comm);
    this->row_comm = Communicator(row_mpi_comm);

    // Column communicator: ranks with same p2 (for X↔Z transpose)
    MPI_Comm_split(mpi_comm, this->proc_coord_p2, this->proc_coord_p1,
                   &col_mpi_comm);
    this->col_comm = Communicator(col_mpi_comm);
  }
#endif

  // Compute the real-space subdomain distribution
  // For 2D: Y is distributed across P2 ranks
  // For 3D: Y is distributed across P2, Z across P1
  IntCoord_t nb_subdivisions(spatial_dim);
  nb_subdivisions[0] = 1;  // X is not distributed in real space
  nb_subdivisions[1] = this->proc_grid_p2;
  if (spatial_dim == 3) {
    nb_subdivisions[2] = this->proc_grid_p1;
  }

  // Initialize the parent CartesianDecomposition
  // This sets up the real-space field collection with ghosts
  IntCoord_t effective_ghosts_left = nb_ghosts_left;
  IntCoord_t effective_ghosts_right = nb_ghosts_right;
  if (effective_ghosts_left.get_dim() == 0) {
    effective_ghosts_left = IntCoord_t(spatial_dim, 0);
  }
  if (effective_ghosts_right.get_dim() == 0) {
    effective_ghosts_right = IntCoord_t(spatial_dim, 0);
  }

  Parent_t::initialise(nb_domain_grid_pts, nb_subdivisions,
                       effective_ghosts_left, effective_ghosts_right);

  // Compute Fourier-space dimensions and distribution
  // Fourier space has X as the half-complex dimension (Nx/2+1)
  // and is distributed as X-pencils: X across P1, Y across P2
  this->nb_fourier_grid_pts = get_hermitian_grid_pts(nb_domain_grid_pts, 0);

  // Compute local Fourier subdomain
  Index_t local_fx, offset_fx;
  Index_t local_fy, offset_fy;
  Index_t local_fz = 1, offset_fz = 0;

  if (spatial_dim == 2) {
    // 2D: After transpose, X is distributed across P2, Y is full
    distribute_dimension(this->nb_fourier_grid_pts[0], this->proc_grid_p2,
                         this->proc_coord_p2, local_fx, offset_fx);
    local_fy = nb_domain_grid_pts[1];  // Full Y
    offset_fy = 0;
  } else {
    // 3D: X distributed across P1, Y distributed across P2
    distribute_dimension(this->nb_fourier_grid_pts[0], this->proc_grid_p1,
                         this->proc_coord_p1, local_fx, offset_fx);
    distribute_dimension(nb_domain_grid_pts[1], this->proc_grid_p2,
                         this->proc_coord_p2, local_fy, offset_fy);
    // Z is fully local in Fourier space (X-pencils)
    local_fz = nb_domain_grid_pts[2];
    offset_fz = 0;
  }

  this->nb_fourier_subdomain_grid_pts = IntCoord_t(spatial_dim);
  this->nb_fourier_subdomain_grid_pts[0] = local_fx;
  this->nb_fourier_subdomain_grid_pts[1] = local_fy;
  if (spatial_dim == 3) {
    this->nb_fourier_subdomain_grid_pts[2] = local_fz;
  }

  this->fourier_subdomain_locations = IntCoord_t(spatial_dim);
  this->fourier_subdomain_locations[0] = offset_fx;
  this->fourier_subdomain_locations[1] = offset_fy;
  if (spatial_dim == 3) {
    this->fourier_subdomain_locations[2] = offset_fz;
  }

  // Initialize the Fourier collection (no ghosts in Fourier space)
  IntCoord_t fourier_no_ghosts(spatial_dim, 0);
  this->fourier_collection = std::make_unique<GlobalFieldCollection>(
      this->nb_fourier_grid_pts, this->nb_fourier_subdomain_grid_pts,
      this->fourier_subdomain_locations, nb_sub_pts,
      StorageOrder::ArrayOfStructures, fourier_no_ghosts,
      fourier_no_ghosts, memory_location);

  // Compute normalization factor
  this->norm_factor = fft_normalization(nb_domain_grid_pts);

  // Initialize FFT infrastructure (work buffers and transposes)
  this->initialise_fft();
}

void FFTEngine::initialise_fft() {
  const IntCoord_t & nb_grid_pts = this->get_nb_domain_grid_pts();
  auto memory_location = this->get_memory_location();

  // Get local real-space dimensions (without ghosts)
  IntCoord_t local_real = this->get_nb_subdomain_grid_pts_without_ghosts();

  // Work buffer 1: After first FFT (r2c along X), Z-pencil layout
  // Shape: [Nx/2+1, Ny/P2, Nz/P1] - same Y,Z distribution as real space
  IntCoord_t zpencil_shape(this->spatial_dim);
  zpencil_shape[0] = nb_grid_pts[0] / 2 + 1;
  zpencil_shape[1] = local_real[1];
  if (this->spatial_dim == 3) {
    zpencil_shape[2] = local_real[2];
  }

  IntCoord_t zpencil_loc(this->spatial_dim);
  zpencil_loc[0] = 0;
  zpencil_loc[1] = this->get_subdomain_locations_without_ghosts()[1];
  if (this->spatial_dim == 3) {
    zpencil_loc[2] = this->get_subdomain_locations_without_ghosts()[2];
  }

  IntCoord_t no_ghosts(this->spatial_dim, 0);
  this->work_zpencil = std::make_unique<GlobalFieldCollection>(
      get_hermitian_grid_pts(nb_grid_pts, 0), zpencil_shape, zpencil_loc,
      SubPtMap_t{}, StorageOrder::ArrayOfStructures,
      no_ghosts, no_ghosts, memory_location);

  if (this->spatial_dim == 3) {
    // For 3D, we need additional work buffers for the Y-FFT step

    // After Y↔Z transpose: [Nx/2+1, Ny, Nz/(P1*P2)]
    // This gives us full Y for the Y-FFT
    IntCoord_t ypencil_global(3);
    ypencil_global[0] = nb_grid_pts[0] / 2 + 1;
    ypencil_global[1] = nb_grid_pts[1];
    ypencil_global[2] = nb_grid_pts[2];

    Index_t local_z_ypencil, offset_z_ypencil;
    int total_ranks = this->proc_grid_p1 * this->proc_grid_p2;
    int linear_rank = this->proc_coord_p1 * this->proc_grid_p2 +
                      this->proc_coord_p2;
    distribute_dimension(nb_grid_pts[2], total_ranks, linear_rank,
                         local_z_ypencil, offset_z_ypencil);

    IntCoord_t ypencil_shape(3);
    ypencil_shape[0] = nb_grid_pts[0] / 2 + 1;
    ypencil_shape[1] = nb_grid_pts[1];  // Full Y
    ypencil_shape[2] = local_z_ypencil;

    IntCoord_t ypencil_loc(3);
    ypencil_loc[0] = 0;
    ypencil_loc[1] = 0;  // Full Y starts at 0
    ypencil_loc[2] = offset_z_ypencil;

    this->work_ypencil = std::make_unique<GlobalFieldCollection>(
        ypencil_global, ypencil_shape, ypencil_loc, SubPtMap_t{},
        StorageOrder::ArrayOfStructures, no_ghosts,
        no_ghosts, memory_location);

    // Set up Y↔Z transpose configuration (within row communicator)
#ifdef WITH_MPI
    // Only create transpose if we have multiple ranks that need data exchange
    if (this->row_comm.size() > 1) {
      this->need_transpose_yz = true;
      this->transpose_yz_fwd_config = {
          zpencil_shape, ypencil_shape, nb_grid_pts[1], nb_grid_pts[2], 1, 2,
          true  // use row_comm
      };
      this->transpose_yz_bwd_config = {
          ypencil_shape, zpencil_shape, nb_grid_pts[2], nb_grid_pts[1], 2, 1,
          true  // use row_comm
      };
    }
#endif

    // Set up X↔Z transpose configuration (within column communicator)
#ifdef WITH_MPI
    // Only create transpose if we have multiple ranks that need data exchange
    if (this->col_comm.size() > 1) {
      this->need_transpose_xz = true;
      this->transpose_xz_config = {
          zpencil_shape, this->nb_fourier_subdomain_grid_pts,
          this->nb_fourier_grid_pts[0], nb_grid_pts[2], 2, 0,
          false  // use col_comm
      };
    }
#endif
  } else {
    // 2D case: simpler, just one transpose
#ifdef WITH_MPI
    // For 2D, use row_comm (ranks with same P1 coordinate) to transpose Y↔X
    // With process grid (1, P2), all ranks have P1=0, so row_comm contains all ranks
    if (this->row_comm.size() > 1) {
      this->need_transpose_xz = true;
      // global_in = Ny (distributed in input, becomes local in output)
      // global_out = Fx (local in input, becomes distributed in output)
      this->transpose_xz_config = {
          zpencil_shape, this->nb_fourier_subdomain_grid_pts,
          nb_grid_pts[1], this->nb_fourier_grid_pts[0], 1, 0,
          true  // use row_comm
      };
    }
#endif
  }

  // Register persistent work fields
  this->work_zpencil->register_complex_field("work", 1);
  if (this->work_ypencil) {
    this->work_ypencil->register_complex_field("work", 1);
  }
}

FFT1DBackend * FFTEngine::select_backend(bool is_device_memory) const {
  if (is_device_memory) {
    if (this->device_backend == nullptr) {
      throw RuntimeError("No GPU FFT backend available for device memory. "
                         "Compile with CUDA or HIP support.");
    }
    return this->device_backend.get();
  } else {
    if (this->host_backend == nullptr) {
      throw RuntimeError("No host FFT backend available");
    }
    return this->host_backend.get();
  }
}

#ifdef WITH_MPI
DatatypeTranspose * FFTEngine::get_transpose_xz(Index_t nb_components) {
  if (!this->need_transpose_xz) {
    return nullptr;
  }

  auto it = this->transpose_xz_cache.find(nb_components);
  if (it != this->transpose_xz_cache.end()) {
    return it->second.get();
  }

  // Create new transpose for this nb_components
  const auto & cfg = this->transpose_xz_config;
  const Communicator & comm =
      cfg.use_row_comm ? this->row_comm : this->col_comm;

  auto transpose = std::make_unique<DatatypeTranspose>(
      comm, cfg.local_in, cfg.local_out, cfg.global_in, cfg.global_out,
      cfg.axis_in, cfg.axis_out, nb_components);

  auto * ptr = transpose.get();
  this->transpose_xz_cache[nb_components] = std::move(transpose);
  return ptr;
}

DatatypeTranspose * FFTEngine::get_transpose_yz_forward(Index_t nb_components) {
  if (!this->need_transpose_yz) {
    return nullptr;
  }

  auto it = this->transpose_yz_fwd_cache.find(nb_components);
  if (it != this->transpose_yz_fwd_cache.end()) {
    return it->second.get();
  }

  // Create new transpose for this nb_components
  const auto & cfg = this->transpose_yz_fwd_config;

  auto transpose = std::make_unique<DatatypeTranspose>(
      this->row_comm, cfg.local_in, cfg.local_out, cfg.global_in, cfg.global_out,
      cfg.axis_in, cfg.axis_out, nb_components);

  auto * ptr = transpose.get();
  this->transpose_yz_fwd_cache[nb_components] = std::move(transpose);
  return ptr;
}

DatatypeTranspose * FFTEngine::get_transpose_yz_backward(Index_t nb_components) {
  if (!this->need_transpose_yz) {
    return nullptr;
  }

  auto it = this->transpose_yz_bwd_cache.find(nb_components);
  if (it != this->transpose_yz_bwd_cache.end()) {
    return it->second.get();
  }

  // Create new transpose for this nb_components
  const auto & cfg = this->transpose_yz_bwd_config;

  auto transpose = std::make_unique<DatatypeTranspose>(
      this->row_comm, cfg.local_in, cfg.local_out, cfg.global_in, cfg.global_out,
      cfg.axis_in, cfg.axis_out, nb_components);

  auto * ptr = transpose.get();
  this->transpose_yz_bwd_cache[nb_components] = std::move(transpose);
  return ptr;
}
#else
// Non-MPI stubs
DatatypeTranspose * FFTEngine::get_transpose_xz(Index_t) { return nullptr; }
DatatypeTranspose * FFTEngine::get_transpose_yz_forward(Index_t) {
  return nullptr;
}
DatatypeTranspose * FFTEngine::get_transpose_yz_backward(Index_t) {
  return nullptr;
}
#endif

void FFTEngine::fft(const Field & input, Field & output) {
  // Verify fields belong to correct collections
  if (&input.get_collection() != &this->get_collection()) {
    throw RuntimeError("Input field must belong to the real-space collection");
  }
  if (&output.get_collection() != this->fourier_collection.get()) {
    throw RuntimeError(
        "Output field must belong to the Fourier-space collection");
  }

  // Verify memory spaces match
  bool input_on_device = input.is_on_device();
  bool output_on_device = output.is_on_device();
  if (input_on_device != output_on_device) {
    throw RuntimeError("Input and output must be in the same memory space");
  }

  if (this->spatial_dim == 2) {
    fft_2d(input, output);
  } else {
    fft_3d(input, output);
  }
}

void FFTEngine::ifft(const Field & input, Field & output) {
  // Verify fields belong to correct collections
  if (&input.get_collection() != this->fourier_collection.get()) {
    throw RuntimeError(
        "Input field must belong to the Fourier-space collection");
  }
  if (&output.get_collection() != &this->get_collection()) {
    throw RuntimeError("Output field must belong to the real-space collection");
  }

  // Verify memory spaces match
  bool input_on_device = input.is_on_device();
  bool output_on_device = output.is_on_device();
  if (input_on_device != output_on_device) {
    throw RuntimeError("Input and output must be in the same memory space");
  }

  if (this->spatial_dim == 2) {
    ifft_2d(input, output);
  } else {
    ifft_3d(input, output);
  }
}

void FFTEngine::fft_2d(const Field & input, Field & output) {
  // 2D Forward FFT:
  // 1. r2c FFT along X (local, strided to skip ghosts)
  // 2. Transpose X↔Y (all-to-all) - serial: just reorganize data
  // 3. c2c FFT along Y (local)

  bool is_device = input.is_on_device();
  FFT1DBackend * backend = select_backend(is_device);

  const IntCoord_t & nb_grid_pts = this->get_nb_domain_grid_pts();
  IntCoord_t local_real = this->get_nb_subdomain_grid_pts_without_ghosts();
  const IntCoord_t & ghosts_left = this->get_nb_ghosts_left();
  const IntCoord_t & local_with_ghosts =
      this->get_nb_subdomain_grid_pts_with_ghosts();

  // Get number of components (handle multi-component fields)
  Index_t nb_components = input.get_nb_components();
  if (output.get_nb_components() != nb_components) {
    throw RuntimeError("Input and output fields must have the same number of "
                       "components");
  }

  // Get pointers
  const Real * input_ptr =
      static_cast<const Real *>(input.get_void_data_ptr(!is_device));
  Complex * output_ptr =
      static_cast<Complex *>(output.get_void_data_ptr(!is_device));

  Index_t Nx = nb_grid_pts[0];
  Index_t Fx = Nx / 2 + 1;
  Index_t Ny = nb_grid_pts[1];

  // For multi-component fields with Array-of-Structures layout,
  // components are interleaved: [c0, c1, c0, c1, ...] for each pixel
  // Input stride: nb_components (to skip to next X value for same component)
  Index_t in_comp_stride = nb_components;
  // Input distance between Y rows (includes ghost padding and component stride)
  Index_t in_dist = local_with_ghosts[0] * nb_components;
  // Offset to start of core data for component 0
  Index_t in_base_offset =
      (ghosts_left[0] + ghosts_left[1] * local_with_ghosts[0]) * nb_components;

  // Work buffer size: [Fx, Ny_local] with AoS layout for multi-component
  Index_t work_size = Fx * local_real[1] * nb_components;

  // Allocate work buffer for all components (AoS layout)
  std::vector<Complex> work_buffer(work_size);
  Complex * work_ptr = work_buffer.data();

  // Step 1: r2c FFT along X for each component
  // Output to work buffer in AoS layout: [comp0, comp1, ...] for each (x, y)
  for (Index_t comp = 0; comp < nb_components; ++comp) {
    // Work buffer AoS: stride between X values is nb_components
    // Distance between Y rows is Fx * nb_components
    backend->r2c(Nx, local_real[1], input_ptr + in_base_offset + comp,
                 in_comp_stride, in_dist, work_ptr + comp, nb_components,
                 Fx * nb_components);
  }

  // Step 2: Transpose (or copy for serial)
  DatatypeTranspose * transpose = this->get_transpose_xz(nb_components);
  if (transpose != nullptr) {
    // MPI case: transpose all components at once
    transpose->forward(work_ptr, output_ptr);
  } else {
    // Serial case: copy to output (same layout, just copy)
    std::copy(work_ptr, work_ptr + work_size, output_ptr);
  }

  // Step 3: c2c FFT along Y for all components
  // Data layout depends on whether we did a transpose:
  // - MPI with transpose: X-pencil layout [Fx_local, Ny_full] with Y fully local
  // - Serial: zpencil layout [Fx_full, Ny_local] with X fully local
  if (transpose != nullptr) {
    // MPI case: after transpose, X is distributed, Y is full
    Index_t local_fx = this->nb_fourier_subdomain_grid_pts[0];
    Index_t y_stride = local_fx * nb_components;
    Index_t y_dist = nb_components;

    for (Index_t comp = 0; comp < nb_components; ++comp) {
      backend->c2c_forward(Ny, local_fx, output_ptr + comp, y_stride, y_dist,
                           output_ptr + comp, y_stride, y_dist);
    }
  } else {
    // Serial case: X is full, Y is local (same as real space distribution)
    Index_t local_fy = local_real[1];
    Index_t y_stride = Fx * nb_components;
    Index_t y_dist = nb_components;

    for (Index_t comp = 0; comp < nb_components; ++comp) {
      backend->c2c_forward(local_fy, Fx, output_ptr + comp, y_stride, y_dist,
                           output_ptr + comp, y_stride, y_dist);
    }
  }
}

void FFTEngine::fft_3d(const Field & input, Field & output) {
  // 3D Forward FFT:
  // 1. r2c FFT along X (local, strided to skip ghosts)
  // 2. c2c FFT along Y (serial: direct, MPI: with transpose)
  // 3. c2c FFT along Z (serial: direct, MPI: with transpose)

  bool is_device = input.is_on_device();
  FFT1DBackend * backend = select_backend(is_device);

  const IntCoord_t & nb_grid_pts = this->get_nb_domain_grid_pts();
  IntCoord_t local_real = this->get_nb_subdomain_grid_pts_without_ghosts();
  const IntCoord_t & ghosts_left = this->get_nb_ghosts_left();
  const IntCoord_t & local_with_ghosts =
      this->get_nb_subdomain_grid_pts_with_ghosts();

  // Get number of components (handle multi-component fields)
  Index_t nb_components = input.get_nb_components();
  if (output.get_nb_components() != nb_components) {
    throw RuntimeError("Input and output fields must have the same number of "
                       "components");
  }

  Index_t Nx = nb_grid_pts[0];
  Index_t Ny = nb_grid_pts[1];
  Index_t Nz = nb_grid_pts[2];
  Index_t Fx = Nx / 2 + 1;

  DatatypeTranspose * transpose_xz = this->get_transpose_xz(nb_components);
  DatatypeTranspose * transpose_yz_fwd =
      this->get_transpose_yz_forward(nb_components);
  DatatypeTranspose * transpose_yz_bwd =
      this->get_transpose_yz_backward(nb_components);

  // MPI path: needed if any transpose is required (comm.size > 1)
  // With 2D process grid (P1, P2), we may need Y↔Z transposes (row_comm)
  // and/or X↔Z transpose (col_comm) depending on rank count.
  bool need_mpi_path = (transpose_xz != nullptr || transpose_yz_fwd != nullptr);

  if (need_mpi_path) {
    // MPI path with transposes - TODO: implement multi-component support
    if (nb_components > 1) {
      throw RuntimeError(
          "Multi-component 3D FFT not yet supported in MPI mode");
    }

    const Real * input_ptr =
        static_cast<const Real *>(input.get_void_data_ptr(!is_device));
    Complex * output_ptr =
        static_cast<Complex *>(output.get_void_data_ptr(!is_device));

    Index_t stride_y = local_with_ghosts[0];
    Index_t stride_z = stride_y * local_with_ghosts[1];
    Index_t in_offset =
        ghosts_left[0] + ghosts_left[1] * stride_y + ghosts_left[2] * stride_z;

    // Z-pencil work buffer
    Index_t zpencil_size = Fx * local_real[1] * local_real[2];
    std::vector<Complex> work_z(zpencil_size);
    Complex * work_z_ptr = work_z.data();

    // Step 1: r2c FFT along X
    for (Index_t iz = 0; iz < local_real[2]; ++iz) {
      for (Index_t iy = 0; iy < local_real[1]; ++iy) {
        Index_t in_idx = in_offset + iy * stride_y + iz * stride_z;
        Index_t out_idx = iy * Fx + iz * Fx * local_real[1];
        backend->r2c(Nx, 1, input_ptr + in_idx, 1, 0, work_z_ptr + out_idx, 1,
                     0);
      }
    }

    // Step 2a: Transpose Y↔Z
    const IntCoord_t & ypencil_shape =
        this->work_ypencil->get_nb_subdomain_grid_pts_with_ghosts();
    Index_t ypencil_size = Fx * Ny * ypencil_shape[2];
    std::vector<Complex> work_y(ypencil_size);
    Complex * work_y_ptr = work_y.data();

    if (transpose_yz_fwd != nullptr) {
      transpose_yz_fwd->forward(work_z_ptr, work_y_ptr);
    }

    // Step 2b: c2c FFT along Y
    for (Index_t iz = 0; iz < ypencil_shape[2]; ++iz) {
      for (Index_t ix = 0; ix < Fx; ++ix) {
        Index_t idx = ix + iz * Fx * Ny;
        backend->c2c_forward(Ny, 1, work_y_ptr + idx, Fx, 0, work_y_ptr + idx,
                             Fx, 0);
      }
    }

    // Step 2c: Transpose Z↔Y
    if (transpose_yz_bwd != nullptr) {
      transpose_yz_bwd->forward(work_y_ptr, work_z_ptr);
    }

    // Step 3: Transpose X↔Z (or copy if no transpose needed)
    const IntCoord_t & fourier_local = this->nb_fourier_subdomain_grid_pts;
    if (transpose_xz != nullptr) {
      transpose_xz->forward(work_z_ptr, output_ptr);
    } else {
      // No X↔Z transpose needed (P1=1), just copy from work_z to output
      // work_z layout: [Fx, Ny_local, Nz] = fourier_local
      Index_t fourier_size =
          fourier_local[0] * fourier_local[1] * fourier_local[2];
      std::copy(work_z_ptr, work_z_ptr + fourier_size, output_ptr);
    }

    // Step 4: c2c FFT along Z
    for (Index_t iy = 0; iy < fourier_local[1]; ++iy) {
      for (Index_t ix = 0; ix < fourier_local[0]; ++ix) {
        Index_t idx = ix + iy * fourier_local[0];
        Index_t stride = fourier_local[0] * fourier_local[1];
        backend->c2c_forward(Nz, 1, output_ptr + idx, stride, 0,
                             output_ptr + idx, stride, 0);
      }
    }
  } else
  {
    // Serial path: all dimensions are local, no transposes needed
    const Real * input_ptr =
        static_cast<const Real *>(input.get_void_data_ptr(!is_device));
    Complex * output_ptr =
        static_cast<Complex *>(output.get_void_data_ptr(!is_device));

    // Input strides for multi-component interleaved data
    Index_t in_comp_stride = nb_components;
    Index_t in_stride_y = local_with_ghosts[0] * nb_components;
    Index_t in_stride_z = in_stride_y * local_with_ghosts[1];
    Index_t in_base_offset =
        (ghosts_left[0] + ghosts_left[1] * local_with_ghosts[0] +
         ghosts_left[2] * local_with_ghosts[0] * local_with_ghosts[1]) *
        nb_components;

    // Output strides for multi-component interleaved data
    Index_t out_comp_stride = nb_components;
    Index_t out_stride_y = Fx * nb_components;
    Index_t out_stride_z = out_stride_y * Ny;

    // Process each component separately
    for (Index_t comp = 0; comp < nb_components; ++comp) {
      // Step 1: r2c FFT along X directly to output
      // Output layout: [Fx, Ny, Nz] with interleaved components
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
      // For each (ix, iz), FFT the Y dimension
      // Stride between Y elements = Fx * nb_components
      for (Index_t iz = 0; iz < Nz; ++iz) {
        for (Index_t ix = 0; ix < Fx; ++ix) {
          Index_t idx = comp + ix * nb_components + iz * out_stride_z;
          backend->c2c_forward(Ny, 1, output_ptr + idx, out_stride_y, 0,
                               output_ptr + idx, out_stride_y, 0);
        }
      }

      // Step 3: c2c FFT along Z
      // For each (ix, iy), FFT the Z dimension
      // Stride between Z elements = Fx * Ny * nb_components
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

void FFTEngine::ifft_2d(const Field & input, Field & output) {
  // 2D Inverse FFT (reverse of forward):
  // 1. c2c IFFT along Y (local)
  // 2. Transpose Y↔X (all-to-all, reverse) - serial: just copy
  // 3. c2r IFFT along X (local, strided for ghosts)

  bool is_device = input.is_on_device();
  FFT1DBackend * backend = select_backend(is_device);

  const IntCoord_t & nb_grid_pts = this->get_nb_domain_grid_pts();
  IntCoord_t local_real = this->get_nb_subdomain_grid_pts_without_ghosts();
  const IntCoord_t & ghosts_left = this->get_nb_ghosts_left();
  const IntCoord_t & local_with_ghosts =
      this->get_nb_subdomain_grid_pts_with_ghosts();

  // Get number of components (handle multi-component fields)
  Index_t nb_components = input.get_nb_components();
  if (output.get_nb_components() != nb_components) {
    throw RuntimeError("Input and output fields must have the same number of "
                       "components");
  }

  // Get pointers
  const Complex * input_ptr =
      static_cast<const Complex *>(input.get_void_data_ptr(!is_device));
  Real * output_ptr =
      static_cast<Real *>(output.get_void_data_ptr(!is_device));

  Index_t Nx = nb_grid_pts[0];
  Index_t Fx = Nx / 2 + 1;
  Index_t Ny = nb_grid_pts[1];

  // Output strides for multi-component interleaved data
  Index_t out_comp_stride = nb_components;
  Index_t out_dist = local_with_ghosts[0] * nb_components;
  Index_t out_base_offset =
      (ghosts_left[0] + ghosts_left[1] * local_with_ghosts[0]) * nb_components;

  DatatypeTranspose * transpose = this->get_transpose_xz(nb_components);
  if (transpose != nullptr) {
    // MPI case: input is in X-pencil layout [Fx_local, Ny] with AoS components
    Index_t local_fx = this->nb_fourier_subdomain_grid_pts[0];
    Index_t local_fourier_size = local_fx * Ny * nb_components;

    // Allocate temp buffer for IFFT (we need to preserve input)
    std::vector<Complex> temp(local_fourier_size);
    std::copy(input_ptr, input_ptr + local_fourier_size, temp.data());

    // Step 1: c2c IFFT along Y for each component
    // Input layout: [Fx_local, Ny] × nb_components in AoS
    Index_t y_stride = local_fx * nb_components;
    Index_t y_dist = nb_components;

    for (Index_t comp = 0; comp < nb_components; ++comp) {
      backend->c2c_backward(Ny, local_fx, temp.data() + comp, y_stride, y_dist,
                            temp.data() + comp, y_stride, y_dist);
    }

    // Step 2: Transpose X↔Y (backward) to get zpencil layout [Fx, Ny_local]
    // Work buffer for transposed data
    Index_t work_size = Fx * local_real[1] * nb_components;
    std::vector<Complex> work_buffer(work_size);
    transpose->backward(temp.data(), work_buffer.data());

    // Step 3: c2r IFFT along X for each component
    // work_buffer layout: [Fx, Ny_local] × nb_components in AoS
    for (Index_t comp = 0; comp < nb_components; ++comp) {
      backend->c2r(Nx, local_real[1], work_buffer.data() + comp, nb_components,
                   Fx * nb_components, output_ptr + out_base_offset + comp,
                   out_comp_stride, out_dist);
    }
  } else {
    // Serial case: input is in zpencil layout [Fx, Ny_local] with AoS components
    Index_t local_fy = local_real[1];
    Index_t fourier_size = Fx * local_fy * nb_components;

    // Allocate temp buffer for IFFT (we need to preserve input)
    std::vector<Complex> temp(fourier_size);
    std::copy(input_ptr, input_ptr + fourier_size, temp.data());

    // Step 1: c2c IFFT along Y for each component
    Index_t y_stride = Fx * nb_components;
    Index_t y_dist = nb_components;

    for (Index_t comp = 0; comp < nb_components; ++comp) {
      backend->c2c_backward(local_fy, Fx, temp.data() + comp, y_stride, y_dist,
                            temp.data() + comp, y_stride, y_dist);
    }

    // Step 2: c2r IFFT along X for each component
    for (Index_t comp = 0; comp < nb_components; ++comp) {
      backend->c2r(Nx, local_fy, temp.data() + comp, nb_components,
                   Fx * nb_components, output_ptr + out_base_offset + comp,
                   out_comp_stride, out_dist);
    }
  }
}

void FFTEngine::ifft_3d(const Field & input, Field & output) {
  // 3D Inverse FFT (reverse of forward):
  // 1. c2c IFFT along Z
  // 2. c2c IFFT along Y
  // 3. c2r IFFT along X

  bool is_device = input.is_on_device();
  FFT1DBackend * backend = select_backend(is_device);

  const IntCoord_t & nb_grid_pts = this->get_nb_domain_grid_pts();
  IntCoord_t local_real = this->get_nb_subdomain_grid_pts_without_ghosts();
  const IntCoord_t & ghosts_left = this->get_nb_ghosts_left();
  const IntCoord_t & local_with_ghosts =
      this->get_nb_subdomain_grid_pts_with_ghosts();

  // Get number of components (handle multi-component fields)
  Index_t nb_components = input.get_nb_components();
  if (output.get_nb_components() != nb_components) {
    throw RuntimeError("Input and output fields must have the same number of "
                       "components");
  }

  Index_t Nx = nb_grid_pts[0];
  Index_t Ny = nb_grid_pts[1];
  Index_t Nz = nb_grid_pts[2];
  Index_t Fx = Nx / 2 + 1;

  DatatypeTranspose * transpose_xz = this->get_transpose_xz(nb_components);
  DatatypeTranspose * transpose_yz_fwd =
      this->get_transpose_yz_forward(nb_components);
  DatatypeTranspose * transpose_yz_bwd =
      this->get_transpose_yz_backward(nb_components);

  // MPI path: needed if any transpose is required (comm.size > 1)
  bool need_mpi_path = (transpose_xz != nullptr || transpose_yz_fwd != nullptr);

  if (need_mpi_path) {
    // MPI path with transposes - TODO: implement multi-component support
    if (nb_components > 1) {
      throw RuntimeError(
          "Multi-component 3D FFT not yet supported in MPI mode");
    }

    const Complex * input_ptr =
        static_cast<const Complex *>(input.get_void_data_ptr(!is_device));
    Real * output_ptr =
        static_cast<Real *>(output.get_void_data_ptr(!is_device));

    // Copy input to temp
    const IntCoord_t & fourier_local = this->nb_fourier_subdomain_grid_pts;
    Index_t fourier_size =
        fourier_local[0] * fourier_local[1] * fourier_local[2];
    std::vector<Complex> temp(fourier_size);
    std::copy(input_ptr, input_ptr + fourier_size, temp.data());

    // Step 1: c2c IFFT along Z
    for (Index_t iy = 0; iy < fourier_local[1]; ++iy) {
      for (Index_t ix = 0; ix < fourier_local[0]; ++ix) {
        Index_t idx = ix + iy * fourier_local[0];
        Index_t stride = fourier_local[0] * fourier_local[1];
        backend->c2c_backward(Nz, 1, temp.data() + idx, stride, 0,
                              temp.data() + idx, stride, 0);
      }
    }

    // Z-pencil work buffer
    Index_t zpencil_size = Fx * local_real[1] * local_real[2];
    std::vector<Complex> work_z(zpencil_size);
    Complex * work_z_ptr = work_z.data();

    // Step 2: Transpose Z↔X (backward) or copy if no transpose needed
    if (transpose_xz != nullptr) {
      transpose_xz->backward(temp.data(), work_z_ptr);
    } else {
      // No X↔Z transpose needed (P1=1), just copy from temp to work_z
      std::copy(temp.data(), temp.data() + zpencil_size, work_z_ptr);
    }

    // Y-pencil work buffer
    const IntCoord_t & ypencil_shape =
        this->work_ypencil->get_nb_subdomain_grid_pts_with_ghosts();
    Index_t ypencil_size = Fx * Ny * ypencil_shape[2];
    std::vector<Complex> work_y(ypencil_size);
    Complex * work_y_ptr = work_y.data();

    // Step 3a: Transpose Y↔Z (backward)
    if (transpose_yz_bwd != nullptr) {
      transpose_yz_bwd->backward(work_z_ptr, work_y_ptr);
    }

    // Step 3b: c2c IFFT along Y
    for (Index_t iz = 0; iz < ypencil_shape[2]; ++iz) {
      for (Index_t ix = 0; ix < Fx; ++ix) {
        Index_t idx = ix + iz * Fx * Ny;
        backend->c2c_backward(Ny, 1, work_y_ptr + idx, Fx, 0, work_y_ptr + idx,
                              Fx, 0);
      }
    }

    // Step 3c: Transpose Z↔Y (backward)
    if (transpose_yz_fwd != nullptr) {
      transpose_yz_fwd->backward(work_y_ptr, work_z_ptr);
    }

    // Step 4: c2r IFFT along X
    Index_t stride_y = local_with_ghosts[0];
    Index_t stride_z = stride_y * local_with_ghosts[1];
    Index_t out_offset =
        ghosts_left[0] + ghosts_left[1] * stride_y + ghosts_left[2] * stride_z;

    for (Index_t iz = 0; iz < local_real[2]; ++iz) {
      for (Index_t iy = 0; iy < local_real[1]; ++iy) {
        Index_t in_idx = iy * Fx + iz * Fx * local_real[1];
        Index_t out_idx = out_offset + iy * stride_y + iz * stride_z;
        backend->c2r(Nx, 1, work_z_ptr + in_idx, 1, 0, output_ptr + out_idx, 1,
                     0);
      }
    }
  } else
  {
    // Serial path: all dimensions are local
    const Complex * input_ptr =
        static_cast<const Complex *>(input.get_void_data_ptr(!is_device));
    Real * output_ptr =
        static_cast<Real *>(output.get_void_data_ptr(!is_device));

    // Input strides for multi-component interleaved data
    Index_t in_comp_stride = nb_components;
    Index_t in_stride_y = Fx * nb_components;
    Index_t in_stride_z = in_stride_y * Ny;

    // Output strides for multi-component interleaved data
    Index_t out_comp_stride = nb_components;
    Index_t out_stride_y = local_with_ghosts[0] * nb_components;
    Index_t out_stride_z = out_stride_y * local_with_ghosts[1];
    Index_t out_base_offset =
        (ghosts_left[0] + ghosts_left[1] * local_with_ghosts[0] +
         ghosts_left[2] * local_with_ghosts[0] * local_with_ghosts[1]) *
        nb_components;

    // Size of Fourier data for one component
    Index_t fourier_size = Fx * Ny * Nz;

    // Process each component separately
    for (Index_t comp = 0; comp < nb_components; ++comp) {
      // Copy this component's data to temp buffer
      std::vector<Complex> temp(fourier_size);
      for (Index_t i = 0; i < fourier_size; ++i) {
        temp[i] = input_ptr[i * nb_components + comp];
      }

      // Step 1: c2c IFFT along Z
      // Stride = Fx * Ny (in temp, which has no component interleaving)
      for (Index_t iy = 0; iy < Ny; ++iy) {
        for (Index_t ix = 0; ix < Fx; ++ix) {
          Index_t idx = ix + iy * Fx;
          Index_t stride = Fx * Ny;
          backend->c2c_backward(Nz, 1, temp.data() + idx, stride, 0,
                                temp.data() + idx, stride, 0);
        }
      }

      // Step 2: c2c IFFT along Y
      // Stride = Fx (in temp)
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
          backend->c2r(Nx, 1, temp.data() + in_idx, 1, 0, output_ptr + out_idx,
                       out_comp_stride, 0);
        }
      }
    }
  }
}

Field & FFTEngine::register_real_space_field(const std::string & name,
                                             Index_t nb_components) {
  return this->get_collection().register_real_field(name, nb_components);
}

Field & FFTEngine::register_real_space_field(const std::string & name,
                                             const Shape_t & components) {
  return this->get_collection().register_real_field(name, components);
}

Field & FFTEngine::register_fourier_space_field(const std::string & name,
                                                Index_t nb_components) {
  return this->fourier_collection->register_complex_field(name, nb_components);
}

Field & FFTEngine::register_fourier_space_field(const std::string & name,
                                                const Shape_t & components) {
  return this->fourier_collection->register_complex_field(name, components);
}

GlobalFieldCollection & FFTEngine::get_real_space_collection() {
  return this->get_collection();
}

const GlobalFieldCollection & FFTEngine::get_real_space_collection() const {
  return this->get_collection();
}

GlobalFieldCollection & FFTEngine::get_fourier_space_collection() {
  return *this->fourier_collection;
}

const GlobalFieldCollection & FFTEngine::get_fourier_space_collection() const {
  return *this->fourier_collection;
}

const char * FFTEngine::get_backend_name() const {
  if (this->host_backend) {
    return this->host_backend->name();
  }
  if (this->device_backend) {
    return this->device_backend->name();
  }
  return "none";
}

}  // namespace muGrid

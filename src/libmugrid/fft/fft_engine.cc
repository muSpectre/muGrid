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
#include "libmugrid/exception.hh"
#include "libmugrid/field_typed.hh"

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

  // X is distributed across P1 in Fourier space
  distribute_dimension(this->nb_fourier_grid_pts[0], this->proc_grid_p1,
                       this->proc_coord_p1, local_fx, offset_fx);

  // Y is distributed across P2 in Fourier space (same as real space)
  distribute_dimension(nb_domain_grid_pts[1], this->proc_grid_p2,
                       this->proc_coord_p2, local_fy, offset_fy);

  if (spatial_dim == 3) {
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

    // Set up Y↔Z transpose operations (within row communicator)
#ifdef WITH_MPI
    this->transpose_yz_forward = std::make_unique<PencilTranspose>(
        this->row_comm, zpencil_shape, ypencil_shape,
        nb_grid_pts[1], nb_grid_pts[2], 1, 2);

    this->transpose_yz_backward = std::make_unique<PencilTranspose>(
        this->row_comm, ypencil_shape, zpencil_shape,
        nb_grid_pts[2], nb_grid_pts[1], 2, 1);
#endif

    // Set up X↔Z transpose (within column communicator)
#ifdef WITH_MPI
    this->transpose_xz = std::make_unique<PencilTranspose>(
        this->col_comm, zpencil_shape, this->nb_fourier_subdomain_grid_pts,
        this->nb_fourier_grid_pts[0], nb_grid_pts[2], 2, 0);
#endif
  } else {
    // 2D case: simpler, just one transpose
#ifdef WITH_MPI
    this->transpose_xz = std::make_unique<PencilTranspose>(
        this->col_comm, zpencil_shape, this->nb_fourier_subdomain_grid_pts,
        this->nb_fourier_grid_pts[0], nb_grid_pts[1], 1, 0);
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

  // Get work buffer field
  Field & work = this->work_zpencil->get_field("work");

  // Get pointers
  const Real * input_ptr =
      static_cast<const Real *>(input.get_void_data_ptr(!is_device));
  Complex * work_ptr =
      static_cast<Complex *>(work.get_void_data_ptr(!is_device));
  Complex * output_ptr =
      static_cast<Complex *>(output.get_void_data_ptr(!is_device));

  // Step 1: r2c FFT along X for each Y row
  // Input is strided due to ghosts
  Index_t Nx = nb_grid_pts[0];
  Index_t Fx = Nx / 2 + 1;
  Index_t Ny = nb_grid_pts[1];

  // Input stride: elements are contiguous along X
  Index_t in_stride = 1;
  // Input distance between batches: stride to next Y (includes ghost padding)
  Index_t in_dist = local_with_ghosts[0];
  // Offset to start of core data
  Index_t in_offset = ghosts_left[0] + ghosts_left[1] * local_with_ghosts[0];

  // Output: zpencil layout [Fx, Ny_local], row-major (Y changes slowest)
  // work_ptr[ix + iy * Fx]
  Index_t out_stride = 1;
  Index_t out_dist = Fx;

  backend->r2c(Nx, local_real[1], input_ptr + in_offset, in_stride, in_dist,
               work_ptr, out_stride, out_dist);

  // Step 2: In serial mode, no actual data redistribution needed
  // In MPI mode, transpose X↔Y
#ifdef WITH_MPI
  if (this->transpose_xz) {
    this->transpose_xz->forward(work_ptr, output_ptr);
  } else {
    // Serial case: copy to output
    Index_t total = Fx * local_real[1];
    for (Index_t i = 0; i < total; ++i) {
      output_ptr[i] = work_ptr[i];
    }
  }
#else
  // Serial case: copy to output
  Index_t total = Fx * local_real[1];
  for (Index_t i = 0; i < total; ++i) {
    output_ptr[i] = work_ptr[i];
  }
#endif

  // Step 3: c2c FFT along Y
  // Data layout in output: [Fx, Ny] row-major
  // For each X frequency bin, FFT along Y
  // Elements for Y: output[ix], output[ix + Fx], output[ix + 2*Fx], ...
  // So stride between Y elements = Fx, and we have Fx independent transforms
  Index_t y_stride = Fx;  // Stride between consecutive Y values
  Index_t y_dist = 1;     // Distance between different X transforms

  backend->c2c_forward(Ny, Fx, output_ptr, y_stride, y_dist, output_ptr,
                       y_stride, y_dist);
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

  Index_t Nx = nb_grid_pts[0];
  Index_t Ny = nb_grid_pts[1];
  Index_t Nz = nb_grid_pts[2];
  Index_t Fx = Nx / 2 + 1;

#ifdef WITH_MPI
  if (this->transpose_xz) {
    // MPI path with transposes
    Field & work_z = this->work_zpencil->get_field("work");
    Field & work_y = this->work_ypencil->get_field("work");

    const Real * input_ptr =
        static_cast<const Real *>(input.get_void_data_ptr(!is_device));
    Complex * work_z_ptr =
        static_cast<Complex *>(work_z.get_void_data_ptr(!is_device));
    Complex * work_y_ptr =
        static_cast<Complex *>(work_y.get_void_data_ptr(!is_device));
    Complex * output_ptr =
        static_cast<Complex *>(output.get_void_data_ptr(!is_device));

    Index_t stride_y = local_with_ghosts[0];
    Index_t stride_z = stride_y * local_with_ghosts[1];
    Index_t in_offset =
        ghosts_left[0] + ghosts_left[1] * stride_y + ghosts_left[2] * stride_z;

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
    if (this->transpose_yz_forward) {
      this->transpose_yz_forward->forward(work_z_ptr, work_y_ptr);
    }

    // Step 2b: c2c FFT along Y
    const IntCoord_t & ypencil_shape =
        this->work_ypencil->get_nb_subdomain_grid_pts_with_ghosts();
    for (Index_t iz = 0; iz < ypencil_shape[2]; ++iz) {
      for (Index_t ix = 0; ix < Fx; ++ix) {
        Index_t idx = ix + iz * Fx * Ny;
        backend->c2c_forward(Ny, 1, work_y_ptr + idx, Fx, 0, work_y_ptr + idx,
                             Fx, 0);
      }
    }

    // Step 2c: Transpose Z↔Y
    if (this->transpose_yz_backward) {
      this->transpose_yz_backward->forward(work_y_ptr, work_z_ptr);
    }

    // Step 3: Transpose X↔Z
    this->transpose_xz->forward(work_z_ptr, output_ptr);

    // Step 4: c2c FFT along Z
    const IntCoord_t & fourier_local = this->nb_fourier_subdomain_grid_pts;
    for (Index_t iy = 0; iy < fourier_local[1]; ++iy) {
      for (Index_t ix = 0; ix < fourier_local[0]; ++ix) {
        Index_t idx = ix + iy * fourier_local[0];
        Index_t stride = fourier_local[0] * fourier_local[1];
        backend->c2c_forward(Nz, 1, output_ptr + idx, stride, 0,
                             output_ptr + idx, stride, 0);
      }
    }
  } else
#endif
  {
    // Serial path: all dimensions are local, no transposes needed
    const Real * input_ptr =
        static_cast<const Real *>(input.get_void_data_ptr(!is_device));
    Complex * output_ptr =
        static_cast<Complex *>(output.get_void_data_ptr(!is_device));

    Index_t stride_y = local_with_ghosts[0];
    Index_t stride_z = stride_y * local_with_ghosts[1];
    Index_t in_offset =
        ghosts_left[0] + ghosts_left[1] * stride_y + ghosts_left[2] * stride_z;

    // Step 1: r2c FFT along X directly to output
    // Output layout: [Fx, Ny, Nz] row-major
    for (Index_t iz = 0; iz < Nz; ++iz) {
      for (Index_t iy = 0; iy < Ny; ++iy) {
        Index_t in_idx = in_offset + iy * stride_y + iz * stride_z;
        Index_t out_idx = iy * Fx + iz * Fx * Ny;
        backend->r2c(Nx, 1, input_ptr + in_idx, 1, 0, output_ptr + out_idx, 1,
                     0);
      }
    }

    // Step 2: c2c FFT along Y
    // For each (ix, iz), FFT the Y dimension
    // Elements: output[ix + iy*Fx + iz*Fx*Ny] for iy = 0..Ny-1
    // Stride between Y elements = Fx
    for (Index_t iz = 0; iz < Nz; ++iz) {
      for (Index_t ix = 0; ix < Fx; ++ix) {
        Index_t idx = ix + iz * Fx * Ny;
        backend->c2c_forward(Ny, 1, output_ptr + idx, Fx, 0, output_ptr + idx,
                             Fx, 0);
      }
    }

    // Step 3: c2c FFT along Z
    // For each (ix, iy), FFT the Z dimension
    // Elements: output[ix + iy*Fx + iz*Fx*Ny] for iz = 0..Nz-1
    // Stride between Z elements = Fx * Ny
    for (Index_t iy = 0; iy < Ny; ++iy) {
      for (Index_t ix = 0; ix < Fx; ++ix) {
        Index_t idx = ix + iy * Fx;
        Index_t stride = Fx * Ny;
        backend->c2c_forward(Nz, 1, output_ptr + idx, stride, 0,
                             output_ptr + idx, stride, 0);
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

  // Get work buffer field
  Field & work = this->work_zpencil->get_field("work");

  // Get pointers
  const Complex * input_ptr =
      static_cast<const Complex *>(input.get_void_data_ptr(!is_device));
  Complex * work_ptr =
      static_cast<Complex *>(work.get_void_data_ptr(!is_device));
  Real * output_ptr =
      static_cast<Real *>(output.get_void_data_ptr(!is_device));

  Index_t Nx = nb_grid_pts[0];
  Index_t Fx = Nx / 2 + 1;
  Index_t Ny = nb_grid_pts[1];

  // We need to copy input to a temp buffer since c2c is in-place here
  Index_t fourier_size = Fx * Ny;
  std::vector<Complex> temp(fourier_size);
  for (Index_t i = 0; i < fourier_size; ++i) {
    temp[i] = input_ptr[i];
  }

  // Step 1: c2c IFFT along Y
  // Data layout: [Fx, Ny] row-major
  // Elements for Y: temp[ix], temp[ix + Fx], temp[ix + 2*Fx], ...
  Index_t y_stride = Fx;  // Stride between consecutive Y values
  Index_t y_dist = 1;     // Distance between different X transforms

  backend->c2c_backward(Ny, Fx, temp.data(), y_stride, y_dist, temp.data(),
                        y_stride, y_dist);

  // Step 2: Transpose Y↔X (backward)
#ifdef WITH_MPI
  if (this->transpose_xz) {
    this->transpose_xz->backward(temp.data(), work_ptr);
  } else {
    for (Index_t i = 0; i < fourier_size; ++i) {
      work_ptr[i] = temp[i];
    }
  }
#else
  for (Index_t i = 0; i < fourier_size; ++i) {
    work_ptr[i] = temp[i];
  }
#endif

  // Step 3: c2r IFFT along X
  // work_ptr layout: [Fx, Ny_local] row-major
  Index_t in_stride = 1;
  Index_t in_dist = Fx;

  Index_t out_stride = 1;
  Index_t out_dist = local_with_ghosts[0];
  Index_t out_offset = ghosts_left[0] + ghosts_left[1] * local_with_ghosts[0];

  backend->c2r(Nx, local_real[1], work_ptr, in_stride, in_dist,
               output_ptr + out_offset, out_stride, out_dist);
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

  Index_t Nx = nb_grid_pts[0];
  Index_t Ny = nb_grid_pts[1];
  Index_t Nz = nb_grid_pts[2];
  Index_t Fx = Nx / 2 + 1;

#ifdef WITH_MPI
  if (this->transpose_xz) {
    // MPI path with transposes
    Field & work_z = this->work_zpencil->get_field("work");
    Field & work_y = this->work_ypencil->get_field("work");

    const Complex * input_ptr =
        static_cast<const Complex *>(input.get_void_data_ptr(!is_device));
    Complex * work_z_ptr =
        static_cast<Complex *>(work_z.get_void_data_ptr(!is_device));
    Complex * work_y_ptr =
        static_cast<Complex *>(work_y.get_void_data_ptr(!is_device));
    Real * output_ptr =
        static_cast<Real *>(output.get_void_data_ptr(!is_device));

    // Copy input to temp
    const IntCoord_t & fourier_local = this->nb_fourier_subdomain_grid_pts;
    Index_t fourier_size =
        fourier_local[0] * fourier_local[1] * fourier_local[2];
    std::vector<Complex> temp(fourier_size);
    for (Index_t i = 0; i < fourier_size; ++i) {
      temp[i] = input_ptr[i];
    }

    // Step 1: c2c IFFT along Z
    for (Index_t iy = 0; iy < fourier_local[1]; ++iy) {
      for (Index_t ix = 0; ix < fourier_local[0]; ++ix) {
        Index_t idx = ix + iy * fourier_local[0];
        Index_t stride = fourier_local[0] * fourier_local[1];
        backend->c2c_backward(Nz, 1, temp.data() + idx, stride, 0,
                              temp.data() + idx, stride, 0);
      }
    }

    // Step 2: Transpose Z↔X (backward)
    this->transpose_xz->backward(temp.data(), work_z_ptr);

    // Step 3a: Transpose Y↔Z (backward)
    if (this->transpose_yz_backward) {
      this->transpose_yz_backward->backward(work_z_ptr, work_y_ptr);
    }

    // Step 3b: c2c IFFT along Y
    const IntCoord_t & ypencil_shape =
        this->work_ypencil->get_nb_subdomain_grid_pts_with_ghosts();
    for (Index_t iz = 0; iz < ypencil_shape[2]; ++iz) {
      for (Index_t ix = 0; ix < Fx; ++ix) {
        Index_t idx = ix + iz * Fx * Ny;
        backend->c2c_backward(Ny, 1, work_y_ptr + idx, Fx, 0, work_y_ptr + idx,
                              Fx, 0);
      }
    }

    // Step 3c: Transpose Z↔Y (backward)
    if (this->transpose_yz_forward) {
      this->transpose_yz_forward->backward(work_y_ptr, work_z_ptr);
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
#endif
  {
    // Serial path: all dimensions are local
    const Complex * input_ptr =
        static_cast<const Complex *>(input.get_void_data_ptr(!is_device));
    Real * output_ptr =
        static_cast<Real *>(output.get_void_data_ptr(!is_device));

    // Copy input to temp (we need to modify it)
    Index_t fourier_size = Fx * Ny * Nz;
    std::vector<Complex> temp(fourier_size);
    for (Index_t i = 0; i < fourier_size; ++i) {
      temp[i] = input_ptr[i];
    }

    // Step 1: c2c IFFT along Z
    // Elements: temp[ix + iy*Fx + iz*Fx*Ny] for iz = 0..Nz-1
    // Stride = Fx * Ny
    for (Index_t iy = 0; iy < Ny; ++iy) {
      for (Index_t ix = 0; ix < Fx; ++ix) {
        Index_t idx = ix + iy * Fx;
        Index_t stride = Fx * Ny;
        backend->c2c_backward(Nz, 1, temp.data() + idx, stride, 0,
                              temp.data() + idx, stride, 0);
      }
    }

    // Step 2: c2c IFFT along Y
    // Elements: temp[ix + iy*Fx + iz*Fx*Ny] for iy = 0..Ny-1
    // Stride = Fx
    for (Index_t iz = 0; iz < Nz; ++iz) {
      for (Index_t ix = 0; ix < Fx; ++ix) {
        Index_t idx = ix + iz * Fx * Ny;
        backend->c2c_backward(Ny, 1, temp.data() + idx, Fx, 0,
                              temp.data() + idx, Fx, 0);
      }
    }

    // Step 3: c2r IFFT along X
    Index_t stride_y = local_with_ghosts[0];
    Index_t stride_z = stride_y * local_with_ghosts[1];
    Index_t out_offset =
        ghosts_left[0] + ghosts_left[1] * stride_y + ghosts_left[2] * stride_z;

    for (Index_t iz = 0; iz < Nz; ++iz) {
      for (Index_t iy = 0; iy < Ny; ++iy) {
        Index_t in_idx = iy * Fx + iz * Fx * Ny;
        Index_t out_idx = out_offset + iy * stride_y + iz * stride_z;
        backend->c2r(Nx, 1, temp.data() + in_idx, 1, 0, output_ptr + out_idx, 1,
                     0);
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

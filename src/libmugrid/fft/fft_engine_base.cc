/**
 * @file   fft/fft_engine_base.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   21 Dec 2024
 *
 * @brief  Non-templated base class for distributed FFT engine
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

#include "fft_engine_base.hh"
#include "core/exception.hh"

#ifdef WITH_MPI
#include <mpi.h>
#endif

namespace muGrid {

FFTEngineBase::FFTEngineBase(const DynGridIndex & nb_domain_grid_pts,
                             const Communicator & comm,
                             const DynGridIndex & nb_ghosts_left,
                             const DynGridIndex & nb_ghosts_right,
                             const SubPtMap_t & nb_sub_pts,
                             MemoryLocation memory_location)
    : Parent_t{comm, nb_domain_grid_pts.get_dim(), nb_sub_pts, memory_location},
      spatial_dim{nb_domain_grid_pts.get_dim()} {
  // Validate dimensions
  if (spatial_dim != 2 && spatial_dim != 3) {
    throw RuntimeError("FFTEngine only supports 2D and 3D grids");
  }

  // Set up the process grid for pencil decomposition
  int num_ranks = comm.size();
  select_process_grid(num_ranks, nb_domain_grid_pts, this->proc_grid_p1,
                      this->proc_grid_p2);

  // Compute this rank's position in the process grid
  // The CartesianDecomposition uses subdivisions [1, P2, P1], which means
  // Z varies fastest in the rank ordering: rank = Y_idx * P1 + Z_idx
  int rank = comm.rank();
  this->proc_coord_p1 = rank % this->proc_grid_p1;  // Z index (varies fastest)
  this->proc_coord_p2 = rank / this->proc_grid_p1;  // Y index

#ifdef WITH_MPI
  // Create row and column subcommunicators
  MPI_Comm mpi_comm = comm.get_mpi_comm();
  if (mpi_comm != MPI_COMM_NULL && num_ranks > 1) {
    MPI_Comm row_mpi_comm, col_mpi_comm;

    // Row communicator: ranks with same p1 (for Y redistribution in 3D)
    MPI_Comm_split(mpi_comm, this->proc_coord_p1, this->proc_coord_p2,
                   &row_mpi_comm);
    this->row_comm = Communicator(row_mpi_comm);

    // Column communicator: ranks with same p2 (for X<->Z transpose)
    MPI_Comm_split(mpi_comm, this->proc_coord_p2, this->proc_coord_p1,
                   &col_mpi_comm);
    this->col_comm = Communicator(col_mpi_comm);
  }
#endif

  // Compute the real-space subdomain distribution
  // For 2D: Y is distributed across P2 ranks
  // For 3D: Y is distributed across P2, Z across P1
  DynGridIndex nb_subdivisions(spatial_dim);
  nb_subdivisions[0] = 1;  // X is not distributed in real space
  nb_subdivisions[1] = this->proc_grid_p2;
  if (spatial_dim == 3) {
    nb_subdivisions[2] = this->proc_grid_p1;
  }

  // Initialize the parent CartesianDecomposition
  // This sets up the real-space field collection with ghosts
  DynGridIndex effective_ghosts_left = nb_ghosts_left;
  DynGridIndex effective_ghosts_right = nb_ghosts_right;
  if (effective_ghosts_left.get_dim() == 0) {
    effective_ghosts_left = DynGridIndex(spatial_dim, 0);
  }
  if (effective_ghosts_right.get_dim() == 0) {
    effective_ghosts_right = DynGridIndex(spatial_dim, 0);
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

  this->nb_fourier_subdomain_grid_pts = DynGridIndex(spatial_dim);
  this->nb_fourier_subdomain_grid_pts[0] = local_fx;
  this->nb_fourier_subdomain_grid_pts[1] = local_fy;
  if (spatial_dim == 3) {
    this->nb_fourier_subdomain_grid_pts[2] = local_fz;
  }

  this->fourier_subdomain_locations = DynGridIndex(spatial_dim);
  this->fourier_subdomain_locations[0] = offset_fx;
  this->fourier_subdomain_locations[1] = offset_fy;
  if (spatial_dim == 3) {
    this->fourier_subdomain_locations[2] = offset_fz;
  }

  // Initialize the Fourier collection (no ghosts in Fourier space)
  DynGridIndex fourier_no_ghosts(spatial_dim, 0);
  this->fourier_collection = std::make_unique<GlobalFieldCollection>(
      this->nb_fourier_grid_pts, this->nb_fourier_subdomain_grid_pts,
      this->fourier_subdomain_locations, nb_sub_pts,
      StorageOrder::ArrayOfStructures, fourier_no_ghosts,
      fourier_no_ghosts, memory_location);

  // Compute normalization factor
  this->norm_factor = fft_normalization(nb_domain_grid_pts);

  // Initialize FFT infrastructure (work buffer collections and transposes)
  this->initialise_fft_base();
}

void FFTEngineBase::initialise_fft_base() {
  const DynGridIndex & nb_grid_pts = this->get_nb_domain_grid_pts();
  auto memory_location = this->get_memory_location();

  // Get local real-space dimensions (without ghosts)
  DynGridIndex local_real = this->get_nb_subdomain_grid_pts_without_ghosts();

  // Work buffer 1: After first FFT (r2c along X), Z-pencil layout
  // Shape: [Nx/2+1, Ny/P2, Nz/P1] - same Y,Z distribution as real space
  DynGridIndex zpencil_shape(this->spatial_dim);
  zpencil_shape[0] = nb_grid_pts[0] / 2 + 1;
  zpencil_shape[1] = local_real[1];
  if (this->spatial_dim == 3) {
    zpencil_shape[2] = local_real[2];
  }

  DynGridIndex zpencil_loc(this->spatial_dim);
  zpencil_loc[0] = 0;
  zpencil_loc[1] = this->get_subdomain_locations_without_ghosts()[1];
  if (this->spatial_dim == 3) {
    zpencil_loc[2] = this->get_subdomain_locations_without_ghosts()[2];
  }

  DynGridIndex no_ghosts(this->spatial_dim, 0);
  this->work_zpencil = std::make_unique<GlobalFieldCollection>(
      get_hermitian_grid_pts(nb_grid_pts, 0), zpencil_shape, zpencil_loc,
      SubPtMap_t{}, StorageOrder::ArrayOfStructures,
      no_ghosts, no_ghosts, memory_location);

  if (this->spatial_dim == 3) {
    // For 3D, we need additional work buffers for the Y-FFT step

    // After Y-gather transpose: [Nx/2+1, Ny, Nz/P1]
    // This gives us full Y for the Y-FFT, Z remains distributed across P1
    // Note: The YZ transpose within row_comm only gathers Y; Z stays the same
    // because all ranks in a row have the same Z portion.
    DynGridIndex ypencil_global(3);
    ypencil_global[0] = nb_grid_pts[0] / 2 + 1;
    ypencil_global[1] = nb_grid_pts[1];
    ypencil_global[2] = nb_grid_pts[2];

    // Z in Y-pencil layout is the same as in Z-pencil (distributed across P1)
    Index_t local_z_ypencil = local_real[2];  // Nz/P1
    Index_t offset_z_ypencil = zpencil_loc[2];  // Same offset as Z-pencil

    DynGridIndex ypencil_shape(3);
    ypencil_shape[0] = nb_grid_pts[0] / 2 + 1;
    ypencil_shape[1] = nb_grid_pts[1];  // Full Y
    ypencil_shape[2] = local_z_ypencil;

    DynGridIndex ypencil_loc(3);
    ypencil_loc[0] = 0;
    ypencil_loc[1] = 0;  // Full Y starts at 0
    ypencil_loc[2] = offset_z_ypencil;

    this->work_ypencil = std::make_unique<GlobalFieldCollection>(
        ypencil_global, ypencil_shape, ypencil_loc, SubPtMap_t{},
        StorageOrder::ArrayOfStructures, no_ghosts,
        no_ghosts, memory_location);

    // Set up Y-gather transpose configuration (within row communicator)
    // Note: This is a Y-gather operation. Within a row, all ranks have the
    // same Z portion (Z is distributed across P1, not P2). We only need to
    // gather Y across the P2 ranks in the row.
    // We use axis_out=0 (X) with global_out=Fx as a "dummy" scatter dimension
    // that doesn't actually redistribute X (since all ranks have full Fx).
#ifdef WITH_MPI
    if (this->row_comm.size() > 1) {
      this->need_transpose_yz = true;
      // Y gather: axis_in=1 (Y gathered), axis_out=0 (X as dummy scatter)
      this->transpose_yz_fwd_config = {
          zpencil_shape, ypencil_shape, nb_grid_pts[1], zpencil_shape[0], 1, 0,
          true  // use row_comm
      };
      // Y scatter: reverse of gather
      this->transpose_yz_bwd_config = {
          ypencil_shape, zpencil_shape, zpencil_shape[0], nb_grid_pts[1], 0, 1,
          true  // use row_comm
      };
    }
#endif

    // Set up X<->Z transpose configuration (within column communicator)
    // Z is gathered (from Nz/P1 to full Nz), X is scattered (from full Fx to Fx/P1)
#ifdef WITH_MPI
    if (this->col_comm.size() > 1) {
      this->need_transpose_xz = true;
      // Note: global_in = size of gathered dim, global_out = size of scattered dim
      this->transpose_xz_config = {
          zpencil_shape, this->nb_fourier_subdomain_grid_pts,
          nb_grid_pts[2], this->nb_fourier_grid_pts[0], 2, 0,
          false  // use col_comm
      };
    }
#endif
  } else {
    // 2D case: simpler, just one transpose
#ifdef WITH_MPI
    if (this->row_comm.size() > 1) {
      this->need_transpose_xz = true;
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

#ifdef WITH_MPI
Transpose * FFTEngineBase::get_transpose_xz(Index_t nb_components) {
  if (!this->need_transpose_xz) {
    return nullptr;
  }

  auto it = this->transpose_xz_cache.find(nb_components);
  if (it != this->transpose_xz_cache.end()) {
    return it->second.get();
  }

  const auto & cfg = this->transpose_xz_config;
  const Communicator & comm =
      cfg.use_row_comm ? this->row_comm : this->col_comm;

  auto transpose = std::make_unique<Transpose>(
      comm, cfg.local_in, cfg.local_out, cfg.global_in, cfg.global_out,
      cfg.axis_in, cfg.axis_out, nb_components);

  auto * ptr = transpose.get();
  this->transpose_xz_cache[nb_components] = std::move(transpose);
  return ptr;
}

Transpose * FFTEngineBase::get_transpose_yz_forward(
    Index_t nb_components) {
  if (!this->need_transpose_yz) {
    return nullptr;
  }

  auto it = this->transpose_yz_fwd_cache.find(nb_components);
  if (it != this->transpose_yz_fwd_cache.end()) {
    return it->second.get();
  }

  const auto & cfg = this->transpose_yz_fwd_config;

  auto transpose = std::make_unique<Transpose>(
      this->row_comm, cfg.local_in, cfg.local_out, cfg.global_in,
      cfg.global_out, cfg.axis_in, cfg.axis_out, nb_components);

  auto * ptr = transpose.get();
  this->transpose_yz_fwd_cache[nb_components] = std::move(transpose);
  return ptr;
}

Transpose * FFTEngineBase::get_transpose_yz_backward(
    Index_t nb_components) {
  if (!this->need_transpose_yz) {
    return nullptr;
  }

  auto it = this->transpose_yz_bwd_cache.find(nb_components);
  if (it != this->transpose_yz_bwd_cache.end()) {
    return it->second.get();
  }

  const auto & cfg = this->transpose_yz_bwd_config;

  auto transpose = std::make_unique<Transpose>(
      this->row_comm, cfg.local_in, cfg.local_out, cfg.global_in,
      cfg.global_out, cfg.axis_in, cfg.axis_out, nb_components);

  auto * ptr = transpose.get();
  this->transpose_yz_bwd_cache[nb_components] = std::move(transpose);
  return ptr;
}
#else
// Non-MPI stubs
Transpose * FFTEngineBase::get_transpose_xz(Index_t) {
  return nullptr;
}
Transpose * FFTEngineBase::get_transpose_yz_forward(Index_t) {
  return nullptr;
}
Transpose * FFTEngineBase::get_transpose_yz_backward(Index_t) {
  return nullptr;
}
#endif

Field & FFTEngineBase::register_real_space_field(const std::string & name,
                                                 Index_t nb_components) {
  return this->get_collection().register_real_field(name, nb_components);
}

Field & FFTEngineBase::register_real_space_field(const std::string & name,
                                                 const Shape_t & components) {
  return this->get_collection().register_real_field(name, components);
}

Field & FFTEngineBase::register_fourier_space_field(const std::string & name,
                                                    Index_t nb_components) {
  return this->fourier_collection->register_complex_field(name, nb_components);
}

Field & FFTEngineBase::register_fourier_space_field(const std::string & name,
                                                    const Shape_t & components) {
  return this->fourier_collection->register_complex_field(name, components);
}

GlobalFieldCollection & FFTEngineBase::get_real_space_collection() {
  return this->get_collection();
}

const GlobalFieldCollection & FFTEngineBase::get_real_space_collection() const {
  return this->get_collection();
}

GlobalFieldCollection & FFTEngineBase::get_fourier_space_collection() {
  return *this->fourier_collection;
}

const GlobalFieldCollection &
FFTEngineBase::get_fourier_space_collection() const {
  return *this->fourier_collection;
}

}  // namespace muGrid

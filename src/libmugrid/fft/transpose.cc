/**
 * @file   fft/transpose.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   21 Dec 2024
 *
 * @brief  MPI transpose using derived datatypes (no explicit pack/unpack)
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

#include "transpose.hh"
#include "core/exception.hh"

#include <algorithm>
#include <numeric>

namespace muGrid {

void Transpose::compute_distribution(Index_t global_size, int comm_size,
                                             std::vector<Index_t> & counts,
                                             std::vector<Index_t> & displs) {
  counts.resize(comm_size);
  displs.resize(comm_size);

  // Distribute global_size as evenly as possible across ranks
  Index_t base_count = global_size / comm_size;
  Index_t remainder = global_size % comm_size;

  Index_t offset = 0;
  for (int r = 0; r < comm_size; ++r) {
    // First 'remainder' ranks get one extra element
    counts[r] = base_count + (r < remainder ? 1 : 0);
    displs[r] = offset;
    offset += counts[r];
  }
}

Transpose::Transpose(const Communicator & comm,
                                     const DynGridIndex & local_in,
                                     const DynGridIndex & local_out,
                                     Index_t global_in, Index_t global_out,
                                     Index_t axis_in, Index_t axis_out,
                                     Index_t nb_components, StorageOrder layout)
    : comm{comm}, local_in{local_in}, local_out{local_out},
      global_in{global_in}, global_out{global_out}, axis_in{axis_in},
      axis_out{axis_out}, nb_components{nb_components}, layout{layout} {
  if (local_in.get_dim() != local_out.get_dim()) {
    throw RuntimeError("Input and output must have same dimensionality");
  }

  int comm_size = comm.size();

  // Check if this is an allgather operation (no scatter needed)
  // This happens when the "scatter" dimension is already full locally
  // in BOTH input and output (i.e., axis_out is not being redistributed)
  this->is_allgather = (local_in[axis_out] == global_out) &&
                       (local_out[axis_out] == global_out);

  // Check if this is a scatter-only operation (no gather needed)
  // This happens when the "gather" dimension is already full locally
  // in BOTH input and output (i.e., axis_in is not being redistributed)
  this->is_scatter_only = (local_in[axis_in] == global_in) &&
                          (local_out[axis_in] == global_in);

  // Compute how the distributed dimensions are split across ranks
  compute_distribution(global_in, comm_size, this->in_counts, this->in_displs);
  compute_distribution(global_out, comm_size, this->out_counts,
                       this->out_displs);

#ifdef WITH_MPI
  // Initialize MPI infrastructure
  this->send_counts.resize(comm_size, 1);  // Always 1 with derived types
  this->recv_counts.resize(comm_size, 1);
  this->send_displs.resize(comm_size, 0);  // Offset encoded in type
  this->recv_displs.resize(comm_size, 0);

  this->send_types_fwd.resize(comm_size, MPI_DATATYPE_NULL);
  this->recv_types_fwd.resize(comm_size, MPI_DATATYPE_NULL);
  this->send_types_bwd.resize(comm_size, MPI_DATATYPE_NULL);
  this->recv_types_bwd.resize(comm_size, MPI_DATATYPE_NULL);

  // Build the datatypes
  if (comm.size() > 1) {
    this->init_forward_types();
    this->init_backward_types();
    this->types_initialized = true;
  }
#endif
}

Transpose::Transpose(Transpose && other) noexcept
    : comm{std::move(other.comm)}, local_in{std::move(other.local_in)},
      local_out{std::move(other.local_out)}, global_in{other.global_in},
      global_out{other.global_out}, axis_in{other.axis_in},
      axis_out{other.axis_out}, nb_components{other.nb_components},
      layout{other.layout}, in_counts{std::move(other.in_counts)},
      in_displs{std::move(other.in_displs)},
      out_counts{std::move(other.out_counts)},
      out_displs{std::move(other.out_displs)}
#ifdef WITH_MPI
      ,
      send_types_fwd{std::move(other.send_types_fwd)},
      recv_types_fwd{std::move(other.recv_types_fwd)},
      send_types_bwd{std::move(other.send_types_bwd)},
      recv_types_bwd{std::move(other.recv_types_bwd)},
      send_counts{std::move(other.send_counts)},
      recv_counts{std::move(other.recv_counts)},
      send_displs{std::move(other.send_displs)},
      recv_displs{std::move(other.recv_displs)}
#endif
      ,
      types_initialized{other.types_initialized},
      is_allgather{other.is_allgather},
      is_scatter_only{other.is_scatter_only} {
  other.types_initialized = false;  // Prevent double-free
}

Transpose &
Transpose::operator=(Transpose && other) noexcept {
  if (this != &other) {
#ifdef WITH_MPI
    free_datatypes();
#endif
    comm = std::move(other.comm);
    local_in = std::move(other.local_in);
    local_out = std::move(other.local_out);
    global_in = other.global_in;
    global_out = other.global_out;
    axis_in = other.axis_in;
    axis_out = other.axis_out;
    nb_components = other.nb_components;
    layout = other.layout;
    in_counts = std::move(other.in_counts);
    in_displs = std::move(other.in_displs);
    out_counts = std::move(other.out_counts);
    out_displs = std::move(other.out_displs);
#ifdef WITH_MPI
    send_types_fwd = std::move(other.send_types_fwd);
    recv_types_fwd = std::move(other.recv_types_fwd);
    send_types_bwd = std::move(other.send_types_bwd);
    recv_types_bwd = std::move(other.recv_types_bwd);
    send_counts = std::move(other.send_counts);
    recv_counts = std::move(other.recv_counts);
    send_displs = std::move(other.send_displs);
    recv_displs = std::move(other.recv_displs);
#endif
    types_initialized = other.types_initialized;
    is_allgather = other.is_allgather;
    is_scatter_only = other.is_scatter_only;
    other.types_initialized = false;
  }
  return *this;
}

Transpose::~Transpose() {
#ifdef WITH_MPI
  free_datatypes();
#endif
}

#ifdef WITH_MPI
void Transpose::free_datatypes() {
  if (!types_initialized) {
    return;
  }

  for (auto & type : send_types_fwd) {
    if (type != MPI_DATATYPE_NULL) {
      MPI_Type_free(&type);
      type = MPI_DATATYPE_NULL;
    }
  }
  for (auto & type : recv_types_fwd) {
    if (type != MPI_DATATYPE_NULL) {
      MPI_Type_free(&type);
      type = MPI_DATATYPE_NULL;
    }
  }
  for (auto & type : send_types_bwd) {
    if (type != MPI_DATATYPE_NULL) {
      MPI_Type_free(&type);
      type = MPI_DATATYPE_NULL;
    }
  }
  for (auto & type : recv_types_bwd) {
    if (type != MPI_DATATYPE_NULL) {
      MPI_Type_free(&type);
      type = MPI_DATATYPE_NULL;
    }
  }

  types_initialized = false;
}

MPI_Datatype
Transpose::build_block_type(const DynGridIndex & local_shape,
                                    const DynGridIndex & block_shape,
                                    const DynGridIndex & block_start) const {
  Dim_t dim = local_shape.get_dim();
  MPI_Datatype result;

  // For AoS layout, we first create an element type for n_comp complex values
  // For SoA layout, we create a spatial type and replicate across components
  if (this->layout == StorageOrder::ArrayOfStructures) {
    // AoS: [c0, c1, c2, c0, c1, c2, ...] - components interleaved
    // Create element type: nb_components complex values
    MPI_Datatype element_type;
    MPI_Type_contiguous(static_cast<int>(this->nb_components),
                        mpi_type<Complex>(), &element_type);
    MPI_Type_commit(&element_type);

    if (dim == 2) {
      // 2D subarray: [local_shape[0], local_shape[1]] elements
      // Extract block at [block_start[0], block_start[1]] with size
      // [block_shape[0], block_shape[1]]

      // In column-major (Fortran) order:
      // - Stride along dim 0 is 1 element
      // - Stride along dim 1 is local_shape[0] elements

      // Build using MPI_Type_create_subarray
      int sizes[2] = {static_cast<int>(local_shape[0]),
                      static_cast<int>(local_shape[1])};
      int subsizes[2] = {static_cast<int>(block_shape[0]),
                         static_cast<int>(block_shape[1])};
      int starts[2] = {static_cast<int>(block_start[0]),
                       static_cast<int>(block_start[1])};

      MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_FORTRAN,
                               element_type, &result);
    } else if (dim == 3) {
      // 3D subarray
      int sizes[3] = {static_cast<int>(local_shape[0]),
                      static_cast<int>(local_shape[1]),
                      static_cast<int>(local_shape[2])};
      int subsizes[3] = {static_cast<int>(block_shape[0]),
                         static_cast<int>(block_shape[1]),
                         static_cast<int>(block_shape[2])};
      int starts[3] = {static_cast<int>(block_start[0]),
                       static_cast<int>(block_start[1]),
                       static_cast<int>(block_start[2])};

      MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_FORTRAN,
                               element_type, &result);
    } else {
      MPI_Type_free(&element_type);
      throw RuntimeError("Transpose only supports 2D and 3D");
    }

    MPI_Type_commit(&result);
    MPI_Type_free(&element_type);
  } else {
    // SoA: [x0, x1, ..., y0, y1, ..., z0, z1, ...]
    // Create spatial type for single component, then replicate

    MPI_Datatype spatial_type;

    if (dim == 2) {
      int sizes[2] = {static_cast<int>(local_shape[0]),
                      static_cast<int>(local_shape[1])};
      int subsizes[2] = {static_cast<int>(block_shape[0]),
                         static_cast<int>(block_shape[1])};
      int starts[2] = {static_cast<int>(block_start[0]),
                       static_cast<int>(block_start[1])};

      MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_FORTRAN,
                               mpi_type<Complex>(), &spatial_type);
    } else if (dim == 3) {
      int sizes[3] = {static_cast<int>(local_shape[0]),
                      static_cast<int>(local_shape[1]),
                      static_cast<int>(local_shape[2])};
      int subsizes[3] = {static_cast<int>(block_shape[0]),
                         static_cast<int>(block_shape[1]),
                         static_cast<int>(block_shape[2])};
      int starts[3] = {static_cast<int>(block_start[0]),
                       static_cast<int>(block_start[1]),
                       static_cast<int>(block_start[2])};

      MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_FORTRAN,
                               mpi_type<Complex>(), &spatial_type);
    } else {
      throw RuntimeError("Transpose only supports 2D and 3D");
    }
    MPI_Type_commit(&spatial_type);

    // Replicate across components with grid-sized stride
    Index_t grid_size = 1;
    for (Dim_t d = 0; d < dim; ++d) {
      grid_size *= local_shape[d];
    }
    MPI_Aint comp_stride =
        static_cast<MPI_Aint>(grid_size) * sizeof(Complex);

    MPI_Type_create_hvector(static_cast<int>(this->nb_components), 1,
                            comp_stride, spatial_type, &result);
    MPI_Type_commit(&result);
    MPI_Type_free(&spatial_type);
  }

  return result;
}

void Transpose::init_forward_types() {
  int comm_size = this->comm.size();
  Dim_t dim = this->local_in.get_dim();

  if (this->is_allgather) {
    // Allgather mode: send same data to all peers, receive at different offsets
    // Build one send type (full local input) used for all peers
    DynGridIndex send_block_shape = this->local_in;
    DynGridIndex send_block_start(dim, 0);

    MPI_Datatype send_type =
        build_block_type(this->local_in, send_block_shape, send_block_start);

    for (int r = 0; r < comm_size; ++r) {
      // All ranks get the same send type (our full local data)
      if (r == 0) {
        this->send_types_fwd[r] = send_type;
      } else {
        MPI_Type_dup(send_type, &this->send_types_fwd[r]);
      }

      // Build recv type: place incoming data at the correct position
      DynGridIndex recv_block_shape(dim);
      DynGridIndex recv_block_start(dim);

      for (Dim_t d = 0; d < dim; ++d) {
        if (d == this->axis_in) {
          // This dimension is gathered - place each rank's portion at its offset
          recv_block_shape[d] = this->in_counts[r];
          recv_block_start[d] = this->in_displs[r];
        } else {
          // Other dimensions: same as output (axis_out is unchanged in allgather)
          recv_block_shape[d] = this->local_out[d];
          recv_block_start[d] = 0;
        }
      }

      this->recv_types_fwd[r] =
          build_block_type(this->local_out, recv_block_shape, recv_block_start);
    }
  } else if (this->is_scatter_only) {
    // Scatter-only mode: send different portions to different peers,
    // receive our own portion from all peers (they all send the same)

    // Build one recv type (full local output) used for all peers
    DynGridIndex recv_block_shape = this->local_out;
    DynGridIndex recv_block_start(dim, 0);

    MPI_Datatype recv_type =
        build_block_type(this->local_out, recv_block_shape, recv_block_start);

    for (int r = 0; r < comm_size; ++r) {
      // Build send type: extract rank r's portion along axis_out
      DynGridIndex send_block_shape(dim);
      DynGridIndex send_block_start(dim);

      for (Dim_t d = 0; d < dim; ++d) {
        if (d == this->axis_out) {
          // This dimension is scattered - send the portion for rank r
          send_block_shape[d] = this->out_counts[r];
          send_block_start[d] = this->out_displs[r];
        } else {
          // Other dimensions: same as input (axis_in is unchanged in scatter-only)
          send_block_shape[d] = this->local_in[d];
          send_block_start[d] = 0;
        }
      }

      this->send_types_fwd[r] =
          build_block_type(this->local_in, send_block_shape, send_block_start);

      // All ranks use the same recv type (our full local output)
      if (r == 0) {
        this->recv_types_fwd[r] = recv_type;
      } else {
        MPI_Type_dup(recv_type, &this->recv_types_fwd[r]);
      }
    }
  } else {
    // Standard transpose mode: scatter-gather
    for (int r = 0; r < comm_size; ++r) {
      // For forward transpose:
      // - We send a block from our input to rank r
      // - The block covers out_counts[r] elements along axis_out
      //   (the dimension that becomes distributed after transpose)
      // - And all of our local elements along axis_in
      //   (the dimension that is distributed in input)

      // Build send type: extract block from input
      DynGridIndex send_block_shape(dim);
      DynGridIndex send_block_start(dim);

      for (Dim_t d = 0; d < dim; ++d) {
        if (d == this->axis_out) {
          // This dimension gets distributed after transpose
          // Send only the portion that will belong to rank r
          send_block_shape[d] = this->out_counts[r];
          send_block_start[d] = this->out_displs[r];
        } else {
          // Other dimensions: send all our local data
          send_block_shape[d] = this->local_in[d];
          send_block_start[d] = 0;
        }
      }

      this->send_types_fwd[r] =
          build_block_type(this->local_in, send_block_shape, send_block_start);

      // Build recv type: place block into output
      DynGridIndex recv_block_shape(dim);
      DynGridIndex recv_block_start(dim);

      for (Dim_t d = 0; d < dim; ++d) {
        if (d == this->axis_in) {
          // This dimension becomes local after transpose
          // Receive the portion that rank r owned before
          recv_block_shape[d] = this->in_counts[r];
          recv_block_start[d] = this->in_displs[r];
        } else {
          // Other dimensions: receive all data
          recv_block_shape[d] = this->local_out[d];
          recv_block_start[d] = 0;
        }
      }

      this->recv_types_fwd[r] =
          build_block_type(this->local_out, recv_block_shape, recv_block_start);
    }
  }
}

void Transpose::init_backward_types() {
  int comm_size = this->comm.size();
  Dim_t dim = this->local_out.get_dim();

  if (this->is_allgather) {
    // Scatter mode (reverse of allgather):
    // Each rank sends different portions to different peers,
    // receives the same data from all peers (but from different positions)

    // Build one recv type (full local output) used for all peers
    DynGridIndex recv_block_shape = this->local_in;
    DynGridIndex recv_block_start(dim, 0);

    MPI_Datatype recv_type =
        build_block_type(this->local_in, recv_block_shape, recv_block_start);

    for (int r = 0; r < comm_size; ++r) {
      // Build send type: extract rank r's portion from our gathered data
      DynGridIndex send_block_shape(dim);
      DynGridIndex send_block_start(dim);

      for (Dim_t d = 0; d < dim; ++d) {
        if (d == this->axis_in) {
          // Send the portion that belongs to rank r
          send_block_shape[d] = this->in_counts[r];
          send_block_start[d] = this->in_displs[r];
        } else {
          // Other dimensions: same as output (axis_out is unchanged in allgather)
          send_block_shape[d] = this->local_out[d];
          send_block_start[d] = 0;
        }
      }

      this->send_types_bwd[r] =
          build_block_type(this->local_out, send_block_shape, send_block_start);

      // All ranks use the same recv type (our full local data)
      if (r == 0) {
        this->recv_types_bwd[r] = recv_type;
      } else {
        MPI_Type_dup(recv_type, &this->recv_types_bwd[r]);
      }
    }
  } else if (this->is_scatter_only) {
    // Allgather mode (reverse of scatter-only):
    // Each rank sends same data to all peers,
    // receives different portions at different positions

    // Build one send type (full local output) used for all peers
    DynGridIndex send_block_shape = this->local_out;
    DynGridIndex send_block_start(dim, 0);

    MPI_Datatype send_type =
        build_block_type(this->local_out, send_block_shape, send_block_start);

    for (int r = 0; r < comm_size; ++r) {
      // All ranks use the same send type (our full local data)
      if (r == 0) {
        this->send_types_bwd[r] = send_type;
      } else {
        MPI_Type_dup(send_type, &this->send_types_bwd[r]);
      }

      // Build recv type: place rank r's data at the correct position
      DynGridIndex recv_block_shape(dim);
      DynGridIndex recv_block_start(dim);

      for (Dim_t d = 0; d < dim; ++d) {
        if (d == this->axis_out) {
          // This dimension is gathered back - place at the original position
          recv_block_shape[d] = this->out_counts[r];
          recv_block_start[d] = this->out_displs[r];
        } else {
          // Other dimensions: same as input (axis_in is unchanged in scatter-only)
          recv_block_shape[d] = this->local_in[d];
          recv_block_start[d] = 0;
        }
      }

      this->recv_types_bwd[r] =
          build_block_type(this->local_in, recv_block_shape, recv_block_start);
    }
  } else {
    // Standard transpose mode (reverse of scatter-gather)
    // Backward is the reverse of forward:
    // - Send from output layout (X-distributed) to input layout (Y-distributed)

    for (int r = 0; r < comm_size; ++r) {
      // Build send type: extract block from output (backward input)
      DynGridIndex send_block_shape(dim);
      DynGridIndex send_block_start(dim);

      for (Dim_t d = 0; d < dim; ++d) {
        if (d == this->axis_in) {
          // In backward, axis_in is now local (was distributed)
          // Send the portion that will belong to rank r
          send_block_shape[d] = this->in_counts[r];
          send_block_start[d] = this->in_displs[r];
        } else {
          send_block_shape[d] = this->local_out[d];
          send_block_start[d] = 0;
        }
      }

      this->send_types_bwd[r] =
          build_block_type(this->local_out, send_block_shape, send_block_start);

      // Build recv type: place block into input (backward output)
      DynGridIndex recv_block_shape(dim);
      DynGridIndex recv_block_start(dim);

      for (Dim_t d = 0; d < dim; ++d) {
        if (d == this->axis_out) {
          // In backward, axis_out becomes local again
          recv_block_shape[d] = this->out_counts[r];
          recv_block_start[d] = this->out_displs[r];
        } else {
          recv_block_shape[d] = this->local_in[d];
          recv_block_start[d] = 0;
        }
      }

      this->recv_types_bwd[r] =
          build_block_type(this->local_in, recv_block_shape, recv_block_start);
    }
  }
}
#endif  // WITH_MPI

void Transpose::forward(const Complex * input, Complex * output) const {
#ifdef WITH_MPI
  MPI_Comm mpi_comm = this->comm.get_mpi_comm();

  if (mpi_comm == MPI_COMM_NULL || this->comm.size() == 1) {
    // Serial case: direct copy (with reordering if needed)
    // For serial, input and output have the same total size
    Index_t total_size = 1;
    for (Dim_t d = 0; d < this->local_in.get_dim(); ++d) {
      total_size *= this->local_in[d];
    }
    total_size *= this->nb_components;
    std::copy(input, input + total_size, output);
    return;
  }

  // Use MPI_Alltoallw with derived datatypes
  MPI_Alltoallw(input, this->send_counts.data(), this->send_displs.data(),
                this->send_types_fwd.data(), output, this->recv_counts.data(),
                this->recv_displs.data(), this->recv_types_fwd.data(),
                mpi_comm);

#else   // WITH_MPI
  // Serial case: direct copy
  Index_t total_size = 1;
  for (Dim_t d = 0; d < this->local_in.get_dim(); ++d) {
    total_size *= this->local_in[d];
  }
  total_size *= this->nb_components;
  std::copy(input, input + total_size, output);
#endif  // WITH_MPI
}

void Transpose::backward(const Complex * input,
                                 Complex * output) const {
#ifdef WITH_MPI
  MPI_Comm mpi_comm = this->comm.get_mpi_comm();

  if (mpi_comm == MPI_COMM_NULL || this->comm.size() == 1) {
    // Serial case: direct copy
    Index_t total_size = 1;
    for (Dim_t d = 0; d < this->local_out.get_dim(); ++d) {
      total_size *= this->local_out[d];
    }
    total_size *= this->nb_components;
    std::copy(input, input + total_size, output);
    return;
  }

  // Use MPI_Alltoallw with derived datatypes
  MPI_Alltoallw(input, this->send_counts.data(), this->send_displs.data(),
                this->send_types_bwd.data(), output, this->recv_counts.data(),
                this->recv_displs.data(), this->recv_types_bwd.data(),
                mpi_comm);

#else   // WITH_MPI
  // Serial case: direct copy
  Index_t total_size = 1;
  for (Dim_t d = 0; d < this->local_out.get_dim(); ++d) {
    total_size *= this->local_out[d];
  }
  total_size *= this->nb_components;
  std::copy(input, input + total_size, output);
#endif  // WITH_MPI
}

}  // namespace muGrid

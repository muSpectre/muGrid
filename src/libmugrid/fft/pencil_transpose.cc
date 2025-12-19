/**
 * @file   fft/pencil_transpose.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   18 Dec 2024
 *
 * @brief  MPI transpose operations for pencil decomposition
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

#include "pencil_transpose.hh"
#include "core/exception.hh"

#include <algorithm>
#include <numeric>

#ifdef WITH_MPI
#include <mpi.h>
#endif

namespace muGrid {

void PencilTranspose::compute_distribution(Index_t global_size, int comm_size,
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

PencilTranspose::PencilTranspose(const Communicator & comm,
                                 const IntCoord_t & local_in,
                                 const IntCoord_t & local_out,
                                 Index_t global_in, Index_t global_out,
                                 Index_t axis_in, Index_t axis_out)
    : comm{comm}, local_in{local_in}, local_out{local_out},
      global_in{global_in}, global_out{global_out}, axis_in{axis_in},
      axis_out{axis_out} {
  if (local_in.get_dim() != local_out.get_dim()) {
    throw RuntimeError("Input and output must have same dimensionality");
  }

  int comm_size = comm.size();

  // Compute how the distributed dimensions are split across ranks
  compute_distribution(global_in, comm_size, this->in_counts, this->in_displs);
  compute_distribution(global_out, comm_size, this->out_counts,
                       this->out_displs);

  // Compute the "other" dimensions that remain unchanged during transpose
  // These are all dimensions except axis_in and axis_out
  Index_t other_size = 1;
  for (Dim_t d = 0; d < local_in.get_dim(); ++d) {
    if (d != axis_in && d != axis_out) {
      other_size *= local_in[d];
    }
  }

  // Compute send counts: we send slabs of axis_out to each rank
  // Each rank r receives out_counts[r] slabs along axis_out
  // Each slab contains local_in[axis_in] * other_size elements
  this->send_counts.resize(comm_size);
  this->send_displs.resize(comm_size);
  this->total_send_size = 0;

  for (int r = 0; r < comm_size; ++r) {
    // Number of elements to send to rank r:
    // - out_counts[r] slices along axis_out (what rank r will own)
    // - local_in[axis_in] along axis_in (our current local portion)
    // - other_size for the remaining dimensions
    this->send_counts[r] =
        static_cast<int>(out_counts[r] * local_in[axis_in] * other_size);
    this->send_displs[r] = static_cast<int>(this->total_send_size);
    this->total_send_size += this->send_counts[r];
  }

  // Compute recv counts: we receive slabs of axis_in from each rank
  // Each rank r sends in_counts[r] slabs along axis_in
  // Each slab contains local_out[axis_out] * other_size elements
  this->recv_counts.resize(comm_size);
  this->recv_displs.resize(comm_size);
  this->total_recv_size = 0;

  for (int r = 0; r < comm_size; ++r) {
    // Number of elements to receive from rank r:
    // - in_counts[r] slices along axis_in (what rank r owned)
    // - local_out[axis_out] along axis_out (our new local portion)
    // - other_size for the remaining dimensions
    this->recv_counts[r] =
        static_cast<int>(in_counts[r] * local_out[axis_out] * other_size);
    this->recv_displs[r] = static_cast<int>(this->total_recv_size);
    this->total_recv_size += this->recv_counts[r];
  }
}

void PencilTranspose::pack_send_buffer(const Complex * input,
                                       Complex * send_buf) const {
  Dim_t dim = this->local_in.get_dim();
  int comm_size = this->comm.size();

  // Compute strides for input array (column-major order)
  std::vector<Index_t> in_strides(dim);
  in_strides[0] = 1;
  for (Dim_t d = 1; d < dim; ++d) {
    in_strides[d] = in_strides[d - 1] * local_in[d - 1];
  }

  // Pack data for each destination rank
  Index_t buf_offset = 0;

  for (int r = 0; r < comm_size; ++r) {
    // Range of axis_out that goes to rank r
    Index_t out_start = this->out_displs[r];
    Index_t out_end = out_start + this->out_counts[r];

    // Iterate over all elements in our local input
    // and copy those with axis_out in [out_start, out_end)
    if (dim == 2) {
      // 2D case: axes are 0 and 1
      for (Index_t i1 = 0; i1 < local_in[1]; ++i1) {
        for (Index_t i0 = 0; i0 < local_in[0]; ++i0) {
          Index_t idx_out = (axis_out == 0) ? i0 : i1;
          if (idx_out >= out_start && idx_out < out_end) {
            Index_t src_idx = i0 * in_strides[0] + i1 * in_strides[1];
            send_buf[buf_offset++] = input[src_idx];
          }
        }
      }
    } else if (dim == 3) {
      // 3D case
      for (Index_t i2 = 0; i2 < local_in[2]; ++i2) {
        for (Index_t i1 = 0; i1 < local_in[1]; ++i1) {
          for (Index_t i0 = 0; i0 < local_in[0]; ++i0) {
            Index_t idx_out;
            if (axis_out == 0) {
              idx_out = i0;
            } else if (axis_out == 1) {
              idx_out = i1;
            } else {
              idx_out = i2;
            }
            if (idx_out >= out_start && idx_out < out_end) {
              Index_t src_idx = i0 * in_strides[0] + i1 * in_strides[1] +
                                i2 * in_strides[2];
              send_buf[buf_offset++] = input[src_idx];
            }
          }
        }
      }
    }
  }
}

void PencilTranspose::unpack_recv_buffer(const Complex * recv_buf,
                                         Complex * output) const {
  Dim_t dim = this->local_out.get_dim();
  int comm_size = this->comm.size();

  // Compute strides for output array (column-major order)
  std::vector<Index_t> out_strides(dim);
  out_strides[0] = 1;
  for (Dim_t d = 1; d < dim; ++d) {
    out_strides[d] = out_strides[d - 1] * local_out[d - 1];
  }

  // Unpack data from each source rank
  Index_t buf_offset = 0;

  for (int r = 0; r < comm_size; ++r) {
    // Range of axis_in that came from rank r
    Index_t in_start = this->in_displs[r];
    Index_t in_end = in_start + this->in_counts[r];

    // Place elements from rank r into the appropriate positions
    if (dim == 2) {
      // 2D case
      for (Index_t i1 = 0; i1 < local_out[1]; ++i1) {
        for (Index_t i0 = 0; i0 < local_out[0]; ++i0) {
          Index_t idx_in = (axis_in == 0) ? i0 : i1;
          if (idx_in >= in_start && idx_in < in_end) {
            Index_t dst_idx = i0 * out_strides[0] + i1 * out_strides[1];
            output[dst_idx] = recv_buf[buf_offset++];
          }
        }
      }
    } else if (dim == 3) {
      // 3D case
      for (Index_t i2 = 0; i2 < local_out[2]; ++i2) {
        for (Index_t i1 = 0; i1 < local_out[1]; ++i1) {
          for (Index_t i0 = 0; i0 < local_out[0]; ++i0) {
            Index_t idx_in;
            if (axis_in == 0) {
              idx_in = i0;
            } else if (axis_in == 1) {
              idx_in = i1;
            } else {
              idx_in = i2;
            }
            if (idx_in >= in_start && idx_in < in_end) {
              Index_t dst_idx = i0 * out_strides[0] + i1 * out_strides[1] +
                                i2 * out_strides[2];
              output[dst_idx] = recv_buf[buf_offset++];
            }
          }
        }
      }
    }
  }
}

void PencilTranspose::forward(const Complex * input, Complex * output) const {
#ifdef WITH_MPI
  MPI_Comm mpi_comm = this->comm.get_mpi_comm();

  if (mpi_comm == MPI_COMM_NULL || this->comm.size() == 1) {
    // Serial case: just copy (with potential reordering)
    std::vector<Complex> temp(this->total_send_size);
    pack_send_buffer(input, temp.data());
    unpack_recv_buffer(temp.data(), output);
    return;
  }

  // Allocate send and receive buffers
  std::vector<Complex> send_buf(this->total_send_size);
  std::vector<Complex> recv_buf(this->total_recv_size);

  // Pack data into send buffer
  pack_send_buffer(input, send_buf.data());

  // Perform all-to-all exchange
  MPI_Alltoallv(send_buf.data(), this->send_counts.data(),
                this->send_displs.data(), mpi_type<Complex>(), recv_buf.data(),
                this->recv_counts.data(), this->recv_displs.data(),
                mpi_type<Complex>(), mpi_comm);

  // Unpack receive buffer into output
  unpack_recv_buffer(recv_buf.data(), output);

#else   // WITH_MPI
  // Serial case: just copy (with potential reordering)
  std::vector<Complex> temp(this->total_send_size);
  pack_send_buffer(input, temp.data());
  unpack_recv_buffer(temp.data(), output);
#endif  // WITH_MPI
}

void PencilTranspose::backward(const Complex * input, Complex * output) const {
#ifdef WITH_MPI
  MPI_Comm mpi_comm = this->comm.get_mpi_comm();

  if (mpi_comm == MPI_COMM_NULL || this->comm.size() == 1) {
    // Serial case: reverse the pack/unpack order
    std::vector<Complex> temp(this->total_recv_size);
    // In backward, output layout is like forward's input
    // So we need to pack from "forward output" layout
    // and unpack into "forward input" layout
    // This is the reverse: pack using recv pattern, unpack using send pattern

    // For backward, we reuse the same transpose but swap roles:
    // - Pack from local_out layout (which was forward's output)
    // - Unpack into local_in layout (which was forward's input)

    // Create a temporary transpose with swapped parameters
    PencilTranspose reverse(this->comm, this->local_out, this->local_in,
                            this->global_out, this->global_in, this->axis_out,
                            this->axis_in);
    reverse.forward(input, output);
    return;
  }

  // For backward, swap send and receive
  std::vector<Complex> send_buf(this->total_recv_size);
  std::vector<Complex> recv_buf(this->total_send_size);

  // Pack data using the reverse pattern
  // We need to pack from local_out layout, so create reverse transpose
  PencilTranspose reverse(this->comm, this->local_out, this->local_in,
                          this->global_out, this->global_in, this->axis_out,
                          this->axis_in);
  reverse.pack_send_buffer(input, send_buf.data());

  // Perform all-to-all exchange with swapped counts
  MPI_Alltoallv(send_buf.data(), this->recv_counts.data(),
                this->recv_displs.data(), mpi_type<Complex>(), recv_buf.data(),
                this->send_counts.data(), this->send_displs.data(),
                mpi_type<Complex>(), mpi_comm);

  // Unpack using reverse pattern
  reverse.unpack_recv_buffer(recv_buf.data(), output);

#else   // WITH_MPI
  // Serial case
  PencilTranspose reverse(this->comm, this->local_out, this->local_in,
                          this->global_out, this->global_in, this->axis_out,
                          this->axis_in);
  reverse.forward(input, output);
#endif  // WITH_MPI
}

}  // namespace muGrid

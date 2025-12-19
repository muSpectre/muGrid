/**
 * @file   fft/pencil_transpose.hh
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

#ifndef SRC_LIBMUGRID_FFT_PENCIL_TRANSPOSE_HH_
#define SRC_LIBMUGRID_FFT_PENCIL_TRANSPOSE_HH_

#include "core/grid_common.hh"
#include "mpi/communicator.hh"

#include <vector>

namespace muGrid {

/**
 * Handles MPI transpose operations for pencil decomposition in distributed FFT.
 *
 * In pencil decomposition, data needs to be redistributed between ranks to
 * switch which dimension is "local" (not distributed). This class performs
 * the all-to-all communication needed for this redistribution.
 *
 * For example, transforming from Z-pencils (Z fully local) to X-pencils
 * (X fully local) requires gathering X data that is distributed across ranks
 * while scattering Z data that was previously local.
 *
 * The transpose operation is characterized by:
 * - Input: Data distributed along axis_in, full along axis_out
 * - Output: Data distributed along axis_out, full along axis_in
 *
 * Implementation uses MPI_Alltoallv for efficient non-uniform data exchange.
 */
class PencilTranspose {
 public:
  /**
   * Configure a transpose operation between two pencil orientations.
   *
   * @param comm           MPI communicator (typically a row or column
   *                       subcommunicator of the process grid)
   * @param local_in       Local input shape (before transpose)
   * @param local_out      Local output shape (after transpose)
   * @param global_in      Global size of the dimension that is distributed
   *                       in input but becomes local in output
   * @param global_out     Global size of the dimension that is local
   *                       in input but becomes distributed in output
   * @param axis_in        Axis that is distributed in input (becomes local)
   * @param axis_out       Axis that is local in input (becomes distributed)
   */
  PencilTranspose(const Communicator & comm, const IntCoord_t & local_in,
                  const IntCoord_t & local_out, Index_t global_in,
                  Index_t global_out, Index_t axis_in, Index_t axis_out);

  PencilTranspose() = delete;
  PencilTranspose(const PencilTranspose & other) = default;
  PencilTranspose(PencilTranspose && other) = default;
  ~PencilTranspose() = default;

  PencilTranspose & operator=(const PencilTranspose & other) = default;
  PencilTranspose & operator=(PencilTranspose && other) = default;

  /**
   * Perform forward transpose (gather axis_in, scatter axis_out).
   *
   * @param input   Pointer to input complex data
   * @param output  Pointer to output complex data
   */
  void forward(const Complex * input, Complex * output) const;

  /**
   * Perform backward transpose (reverse of forward).
   *
   * @param input   Pointer to input complex data
   * @param output  Pointer to output complex data
   */
  void backward(const Complex * input, Complex * output) const;

  /**
   * Get the local input shape.
   */
  const IntCoord_t & get_local_in() const { return this->local_in; }

  /**
   * Get the local output shape.
   */
  const IntCoord_t & get_local_out() const { return this->local_out; }

  /**
   * Get the input axis that is distributed (will become local after forward).
   */
  Index_t get_axis_in() const { return this->axis_in; }

  /**
   * Get the output axis that is local in input (will become distributed
   * after forward).
   */
  Index_t get_axis_out() const { return this->axis_out; }

 protected:
  /**
   * Compute send/recv counts and displacements for MPI_Alltoallv.
   *
   * @param global_size    Global size of the dimension being redistributed
   * @param comm_size      Number of ranks in communicator
   * @param counts         Output: number of elements per rank
   * @param displs         Output: displacement for each rank
   */
  static void compute_distribution(Index_t global_size, int comm_size,
                                   std::vector<Index_t> & counts,
                                   std::vector<Index_t> & displs);

  /**
   * Pack data from input layout into contiguous send buffer.
   *
   * The input data has shape local_in with axis_out being local.
   * We need to pack slabs destined for each rank contiguously.
   *
   * @param input      Pointer to input data
   * @param send_buf   Pointer to send buffer (contiguous, packed)
   */
  void pack_send_buffer(const Complex * input, Complex * send_buf) const;

  /**
   * Unpack data from receive buffer into output layout.
   *
   * The receive buffer contains slabs from each rank that need to be
   * interleaved along axis_in to form the output.
   *
   * @param recv_buf   Pointer to receive buffer (contiguous, packed)
   * @param output     Pointer to output data
   */
  void unpack_recv_buffer(const Complex * recv_buf, Complex * output) const;

  //! MPI communicator for this transpose
  Communicator comm;

  //! Local shape before transpose
  IntCoord_t local_in;

  //! Local shape after transpose
  IntCoord_t local_out;

  //! Global size of dimension distributed in input
  Index_t global_in;

  //! Global size of dimension local in input (becomes distributed)
  Index_t global_out;

  //! Axis that is distributed in input (will become local)
  Index_t axis_in;

  //! Axis that is local in input (will become distributed)
  Index_t axis_out;

  //! Number of elements to send to each rank
  std::vector<int> send_counts;

  //! Displacement in send buffer for each rank
  std::vector<int> send_displs;

  //! Number of elements to receive from each rank
  std::vector<int> recv_counts;

  //! Displacement in receive buffer for each rank
  std::vector<int> recv_displs;

  //! Distribution of global_in across ranks (how much each rank owns in input)
  std::vector<Index_t> in_counts;
  std::vector<Index_t> in_displs;

  //! Distribution of global_out across ranks (how much each rank will own
  //! after)
  std::vector<Index_t> out_counts;
  std::vector<Index_t> out_displs;

  //! Total size of send/recv buffers
  Index_t total_send_size;
  Index_t total_recv_size;
};

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_FFT_PENCIL_TRANSPOSE_HH_

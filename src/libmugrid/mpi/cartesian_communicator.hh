/**
 * @file   cartesian_communicator.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @brief  Cartesian domain decomposition communicator for structured grids
 *
 * Copyright © 2017 Till Junge
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

#ifndef SRC_LIBMUGRID_CARTESIAN_COMMUNICATOR_HH_
#define SRC_LIBMUGRID_CARTESIAN_COMMUNICATOR_HH_

#ifdef WITH_MPI
#include <mpi.h>
#endif

#include "core/coordinates.hh"
#include "core/type_descriptor.hh"
#include "mpi/communicator.hh"

namespace muGrid {

    /**
     * @class CartesianCommunicator
     *
     * @brief Manages communication for Cartesian domain decomposition on
     * structured grids.
     *
     * This class provides MPI-based communication abstractions for distributed
     * memory parallelization on Cartesian grids. It creates a Cartesian MPI
     * topology, automatically computing neighbor ranks and allowing efficient
     * halo/boundary exchange along each spatial dimension.
     *
     * The class distinguishes between left and right neighbors: in a given
     * direction, the left neighbor is at coordinate[direction] - 1 and the
     * right neighbor is at coordinate[direction] + 1. With periodic boundaries,
     * these wrap around at the domain edges.
     *
     * For efficient structured grid stencil operations, the class provides:
     * - Template methods for scalar value exchange
     * (sendrecv_left/sendrecv_right)
     * - Bulk data transfer methods that handle complex memory layouts via MPI
     *   derived types
     */
    class CartesianCommunicator : public Communicator {
       public:
        using Parent_t = Communicator;

        /**
         * @brief Constructor for a Cartesian communicator with automatic
         * topology creation.
         *
         * @details Initializes a `CartesianCommunicator` with the parent
         * communicator and the number of subdivisions in each spatial
         * direction. This constructor creates a Cartesian MPI topology with
         * periodic boundaries in all directions. The Cartesian communicator
         * automatically computes this rank's coordinates and the ranks of its
         * left and right neighbors in each dimension.
         *
         * @param parent The parent communicator from which the MPI communicator
         *               is derived.
         * @param nb_subdivisions The number of subdivisions (ranks) in each
         *                        spatial direction. The product of all elements
         *                        must equal the size of the parent
         * communicator.
         *
         * @throw RuntimeError if the product of nb_subdivisions does not match
         *                     the communicator size.
         */
        explicit CartesianCommunicator(const Parent_t & parent,
                                       const DynGridIndex & nb_subdivisions);
        /**
         * @brief Construct a Cartesian communicator with explicit topology.
         *
         * @details This constructor initializes a CartesianCommunicator object
         * with pre-computed coordinates and neighbor ranks. This is useful for
         * reconstructing a CartesianCommunicator from previously computed
         * decomposition information or for testing purposes.
         *
         * @param parent The parent communicator from which the MPI communicator
         *               is derived.
         * @param nb_subdivisions The number of subdivisions (ranks) in each
         *                        spatial direction.
         * @param coordinates The Cartesian coordinates (0-indexed) of this rank
         *                    within the decomposition. Must be consistent with
         *                    the parent communicator's rank.
         * @param left_ranks A vector of neighbor ranks to the left (lower
         *                   coordinate) in each spatial dimension. Use
         *                   MPI_PROC_NULL for boundaries (in periodic mode,
         *                   wraps to the opposite side).
         * @param right_ranks A vector of neighbor ranks to the right (higher
         *                    coordinate) in each spatial dimension. Use
         *                    MPI_PROC_NULL for boundaries (in periodic mode,
         *                    wraps to the opposite side).
         */
        explicit CartesianCommunicator(const Parent_t & parent,
                                       const DynGridIndex & nb_subdivisions,
                                       const DynGridIndex & coordinates,
                                       const std::vector<int> & left_ranks,
                                       const std::vector<int> & right_ranks);

        CartesianCommunicator() = delete;

        virtual ~CartesianCommunicator() {}

        /**
         * @brief Assignment operator.
         *
         * @details Assigns the MPI communicator from another
         * CartesianCommunicator. Note that other members (nb_subdivisions,
         * coordinates, ranks) are not copied.
         *
         * @param other The CartesianCommunicator to copy the communicator from.
         * @return Reference to this CartesianCommunicator.
         */
        CartesianCommunicator & operator=(const CartesianCommunicator & other);

        /**
         * @brief Get the number of subdivisions in each spatial dimension.
         *
         * @return Const reference to the vector of subdivision counts.
         */
        const DynGridIndex & get_nb_subdivisions() const;

        /**
         * @brief Get the Cartesian coordinates of this rank.
         *
         * @details Returns the 0-indexed coordinates of this rank in the
         * Cartesian topology grid.
         *
         * @return Const reference to the vector of coordinates.
         */
        const DynGridIndex & get_coordinates() const;

        /**
         * @brief Send data to the right neighbor; receive from the left.
         *
         * @details Performs a synchronous send-receive operation using MPI
         * derived types to handle non-contiguous memory layouts. Data is sent
         * to the right neighbor (higher coordinate) and received from the left
         * neighbor (lower coordinate) in the specified spatial dimension. The
         * MPI derived type efficiently describes strided block data.
         *
         * @param direction The spatial dimension in which to communicate
         *                  (0 <= direction < spatial_dim).
         * @param block_stride Stride to the next block in the next
         *                     dimension (in elements).
         * @param nb_send_blocks Number of blocks to send.
         * @param send_block_len Length of each contiguous block to send
         *                       (in elements).
         * @param send_offset Offset of the first block in the send buffer
         *                    (in blocks, computed as offset *
         * stride_in_direction).
         * @param nb_recv_blocks Number of blocks to receive.
         * @param recv_block_len Length of each contiguous block to receive
         *                       (in elements).
         * @param recv_offset Offset of the first block in the receive buffer
         *                    (in blocks, computed as offset *
         * stride_in_direction).
         * @param data Base address of the data buffer.
         * @param stride_in_direction Stride in the communication direction
         *                            (in elements).
         * @param elem_size_in_bytes Size of each element in bytes.
         * @param type_desc TypeDescriptor identifying the element type. Used
         *                  for MPI type conversion; ignored in serial mode.
         * @param is_device_memory If true, data resides on GPU device memory.
         *                         Used in serial mode to select appropriate
         *                         memory copy method (CUDA/HIP for GPU).
         */
        void sendrecv_right(int direction, int block_stride, int nb_send_blocks,
                            int send_block_len, Index_t send_offset,
                            int nb_recv_blocks, int recv_block_len,
                            Index_t recv_offset, char * data,
                            int stride_in_direction, int elem_size_in_bytes,
                            TypeDescriptor type_desc,
                            bool is_device_memory = false) const;

        /**
         * @brief Send data to the left neighbor; receive from the right.
         *
         * @details Performs a synchronous send-receive operation using MPI
         * derived types to handle non-contiguous memory layouts. Data is sent
         * to the left neighbor (lower coordinate) and received from the right
         * neighbor (higher coordinate) in the specified spatial dimension. The
         * MPI derived type efficiently describes strided block data.
         *
         * @param direction The spatial dimension in which to communicate
         *                  (0 <= direction < spatial_dim).
         * @param block_stride Stride to the next block in the next
         *                     dimension (in elements).
         * @param nb_send_blocks Number of blocks to send.
         * @param send_block_len Length of each contiguous block to send
         *                       (in elements).
         * @param send_offset Offset of the first block in the send buffer
         *                    (in blocks, computed as offset *
         * stride_in_direction).
         * @param nb_recv_blocks Number of blocks to receive.
         * @param recv_block_len Length of each contiguous block to receive
         *                       (in elements).
         * @param recv_offset Offset of the first block in the receive buffer
         *                    (in blocks, computed as offset *
         * stride_in_direction).
         * @param data Base address of the data buffer.
         * @param stride_in_direction Stride in the communication direction
         *                            (in elements).
         * @param elem_size_in_bytes Size of each element in bytes.
         * @param type_desc TypeDescriptor identifying the element type. Used
         *                  for MPI type conversion; ignored in serial mode.
         * @param is_device_memory If true, data resides on GPU device memory.
         *                         Used in serial mode to select appropriate
         *                         memory copy method (CUDA/HIP for GPU).
         */
        void sendrecv_left(int direction, int block_stride, int nb_send_blocks,
                           int send_block_len, Index_t send_offset,
                           int nb_recv_blocks, int recv_block_len,
                           Index_t recv_offset, char * data,
                           int stride_in_direction, int elem_size_in_bytes,
                           TypeDescriptor type_desc,
                           bool is_device_memory = false) const;

        /**
         * @brief Template method for sending a scalar to the right neighbor and
         * receiving from the left.
         *
         * @details Efficiently exchanges a single scalar value with neighbors.
         * The data is sent to the right neighbor (higher coordinate) and the
         * return value is received from the left neighbor (lower coordinate) in
         * the specified spatial dimension. This is suitable for point-wise
         * values rather than large data transfers.
         *
         * @tparam T Scalar type (must have an mpi_type<T>() specialization).
         * @param direction The spatial dimension in which to communicate
         *                  (0 <= direction < spatial_dim).
         * @param data The scalar value to send to the right neighbor.
         * @return The scalar value received from the left neighbor.
         */
        template <typename T>
        T sendrecv_right(int direction, T data) const {
#ifdef WITH_MPI
            // Check if MPI is available (comm may be NULL if MPI was not
            // initialized)
            if (this->comm == MPI_COMM_NULL) {
                return data;
            }
            MPI_Status status;
            T value;
            MPI_Sendrecv(&data, 1, mpi_type<T>(), this->right_ranks[direction],
                         0, &value, 1, mpi_type<T>(),
                         this->left_ranks[direction], 0, this->comm, &status);
            return value;
#else
            return data;
#endif
        }

        /**
         * @brief Template method for sending a scalar to the left neighbor and
         * receiving from the right.
         *
         * @details Efficiently exchanges a single scalar value with neighbors.
         * The data is sent to the left neighbor (lower coordinate) and the
         * return value is received from the right neighbor (higher coordinate)
         * in the specified spatial dimension. This is suitable for point-wise
         * values rather than large data transfers.
         *
         * @tparam T Scalar type (must have an mpi_type<T>() specialization).
         * @param direction The spatial dimension in which to communicate
         *                  (0 <= direction < spatial_dim).
         * @param data The scalar value to send to the left neighbor.
         * @return The scalar value received from the right neighbor.
         */
        template <typename T>
        T sendrecv_left(int direction, T data) const {
#ifdef WITH_MPI
            // Check if MPI is available (comm may be NULL if MPI was not
            // initialized)
            if (this->comm == MPI_COMM_NULL) {
                return data;
            }
            MPI_Status status;
            T value;
            MPI_Sendrecv(&data, 1, mpi_type<T>(), this->left_ranks[direction],
                         0, &value, 1, mpi_type<T>(),
                         this->right_ranks[direction], 0, this->comm, &status);
            return value;
#else
            return data;
#endif
        }

        /**
         * @brief Send to right neighbor, receive from left and accumulate (add).
         *
         * @details Like sendrecv_right, but received values are added to the
         * destination rather than overwriting it. This is the adjoint operation
         * for ghost reduction in transpose operations with periodic BCs.
         *
         * @param direction The spatial dimension in which to communicate.
         * @param block_stride Stride to the next block.
         * @param nb_send_blocks Number of blocks to send.
         * @param send_block_len Length of each block to send.
         * @param send_offset Offset of send buffer in blocks.
         * @param nb_recv_blocks Number of blocks to receive.
         * @param recv_block_len Length of each block to receive.
         * @param recv_offset Offset of receive buffer in blocks.
         * @param data Base address of the data buffer.
         * @param stride_in_direction Stride in communication direction.
         * @param elem_size_in_bytes Size of each element in bytes.
         * @param type_desc TypeDescriptor identifying the element type.
         * @param is_device_memory If true, data is on GPU device memory.
         */
        void sendrecv_right_accumulate(int direction, int block_stride,
                                       int nb_send_blocks, int send_block_len,
                                       Index_t send_offset, int nb_recv_blocks,
                                       int recv_block_len, Index_t recv_offset,
                                       char * data, int stride_in_direction,
                                       int elem_size_in_bytes,
                                       TypeDescriptor type_desc,
                                       bool is_device_memory = false) const;

        /**
         * @brief Send to left neighbor, receive from right and accumulate (add).
         *
         * @details Like sendrecv_left, but received values are added to the
         * destination rather than overwriting it. This is the adjoint operation
         * for ghost reduction in transpose operations with periodic BCs.
         *
         * @param direction The spatial dimension in which to communicate.
         * @param block_stride Stride to the next block.
         * @param nb_send_blocks Number of blocks to send.
         * @param send_block_len Length of each block to send.
         * @param send_offset Offset of send buffer in blocks.
         * @param nb_recv_blocks Number of blocks to receive.
         * @param recv_block_len Length of each block to receive.
         * @param recv_offset Offset of receive buffer in blocks.
         * @param data Base address of the data buffer.
         * @param stride_in_direction Stride in communication direction.
         * @param elem_size_in_bytes Size of each element in bytes.
         * @param type_desc TypeDescriptor identifying the element type.
         * @param is_device_memory If true, data is on GPU device memory.
         */
        void sendrecv_left_accumulate(int direction, int block_stride,
                                      int nb_send_blocks, int send_block_len,
                                      Index_t send_offset, int nb_recv_blocks,
                                      int recv_block_len, Index_t recv_offset,
                                      char * data, int stride_in_direction,
                                      int elem_size_in_bytes,
                                      TypeDescriptor type_desc,
                                      bool is_device_memory = false) const;

       protected:
        //! The parent communicator from which this Cartesian communicator
        //! was derived.
        Parent_t parent;

        //! The number of subdivisions (ranks) in each spatial dimension.
        DynGridIndex nb_subdivisions;

        //! The Cartesian coordinates of this rank (0-indexed in each
        //! dimension).
        DynGridIndex coordinates;

#if WITH_MPI
        //! Ranks of the left neighbors (lower coordinate) in each dimension.
        //! Uses MPI_PROC_NULL for periodic boundaries.
        std::vector<int> left_ranks;

        //! Ranks of the right neighbors (higher coordinate) in each dimension.
        //! Uses MPI_PROC_NULL for periodic boundaries.
        std::vector<int> right_ranks;
#endif  // WITH_MPI
    };
}  // namespace muGrid

#endif  // SRC_LIBMUGRID_CARTESIAN_COMMUNICATOR_HH_

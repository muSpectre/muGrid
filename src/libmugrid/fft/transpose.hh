/**
 * @file   fft/transpose.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   21 Dec 2024
 *
 * @brief  MPI transpose using derived datatypes (no explicit pack/unpack)
 *
 * This implementation uses MPI derived datatypes (MPI_Type_create_subarray,
 * MPI_Type_vector, MPI_Type_create_hvector) with MPI_Alltoallw to perform
 * transpose operations without explicit pack/unpack buffers. This approach:
 *
 * - Eliminates memory overhead from temporary buffers
 * - Allows MPI to optimize non-contiguous memory access
 * - Works seamlessly with GPU-aware MPI implementations
 * - Supports multi-component fields (AoS and SoA layouts)
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

#ifndef SRC_LIBMUGRID_FFT_DATATYPE_TRANSPOSE_HH_
#define SRC_LIBMUGRID_FFT_DATATYPE_TRANSPOSE_HH_

#include "core/types.hh"
#include "core/enums.hh"
#include "core/coordinates.hh"
#include "mpi/communicator.hh"

#include <vector>

#ifdef WITH_MPI
#include <mpi.h>
#endif

namespace muGrid {

    /**
     * Handles MPI transpose operations using derived datatypes.
     *
     * This class uses MPI derived datatypes instead of explicit pack/unpack
     * buffers. The MPI library handles the non-contiguous memory access
     * patterns, which can be more efficient and works with GPU-aware MPI.
     *
     * For pencil decomposition, data needs to be redistributed between ranks to
     * switch which dimension is "local" (not distributed). The transpose
     * operation is characterized by:
     * - Input: Data distributed along axis_in, full along axis_out
     * - Output: Data distributed along axis_out, full along axis_in
     */
    class Transpose {
       public:
        /**
         * Configure a transpose operation between two pencil orientations.
         *
         * @param comm           MPI communicator (typically a row or column
         *                       subcommunicator of the process grid)
         * @param local_in       Local input shape (before transpose)
         * @param local_out      Local output shape (after transpose)
         * @param global_in      Global size of the dimension that is
         * distributed in input but becomes local in output
         * @param global_out     Global size of the dimension that is local
         *                       in input but becomes distributed in output
         * @param axis_in        Axis that is distributed in input (becomes
         * local)
         * @param axis_out       Axis that is local in input (becomes
         * distributed)
         * @param nb_components  Number of field components (default: 1)
         * @param layout         Memory layout for multi-component fields
         */
        Transpose(const Communicator & comm, const DynGridIndex & local_in,
                  const DynGridIndex & local_out, Index_t global_in,
                  Index_t global_out, Index_t axis_in, Index_t axis_out,
                  Index_t nb_components = 1,
                  StorageOrder layout = StorageOrder::ArrayOfStructures);

        Transpose() = delete;
        Transpose(const Transpose & other) = delete;
        Transpose(Transpose && other) noexcept;
        ~Transpose();

        Transpose & operator=(const Transpose & other) = delete;
        Transpose & operator=(Transpose && other) noexcept;

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
        const DynGridIndex & get_local_in() const { return this->local_in; }

        /**
         * Get the local output shape.
         */
        const DynGridIndex & get_local_out() const { return this->local_out; }

        /**
         * Get the input axis that is distributed (will become local after
         * forward).
         */
        Index_t get_axis_in() const { return this->axis_in; }

        /**
         * Get the output axis that is local in input (will become distributed
         * after forward).
         */
        Index_t get_axis_out() const { return this->axis_out; }

        /**
         * Get the number of components.
         */
        Index_t get_nb_components() const { return this->nb_components; }

       protected:
#ifdef WITH_MPI
        /**
         * Build MPI datatype for a block in a multi-dimensional array.
         *
         * For 2D array of shape [nx, ny] (column-major), extracts block at
         * position [start_x, start_y] with size [block_x, block_y].
         *
         * @param local_shape   Full local array shape
         * @param block_shape   Shape of block to extract
         * @param block_start   Starting position of block
         * @return Committed MPI datatype (caller must free)
         */
        MPI_Datatype build_block_type(const DynGridIndex & local_shape,
                                      const DynGridIndex & block_shape,
                                      const DynGridIndex & block_start) const;

        /**
         * Free all MPI datatypes.
         */
        void free_datatypes();

        /**
         * Initialize datatypes for forward transpose.
         */
        void init_forward_types();

        /**
         * Initialize datatypes for backward transpose.
         */
        void init_backward_types();
#endif

        /**
         * Compute how a dimension is distributed across ranks.
         *
         * @param global_size    Global size of the dimension
         * @param comm_size      Number of ranks
         * @param counts         Output: number of elements per rank
         * @param displs         Output: starting offset for each rank
         */
        static void compute_distribution(Index_t global_size, int comm_size,
                                         std::vector<Index_t> & counts,
                                         std::vector<Index_t> & displs);

        //! MPI communicator for this transpose
        Communicator comm;

        //! Local shape before transpose
        DynGridIndex local_in;

        //! Local shape after transpose
        DynGridIndex local_out;

        //! Global size of dimension distributed in input
        Index_t global_in;

        //! Global size of dimension local in input (becomes distributed)
        Index_t global_out;

        //! Axis that is distributed in input (will become local)
        Index_t axis_in;

        //! Axis that is local in input (will become distributed)
        Index_t axis_out;

        //! Number of field components
        Index_t nb_components;

        //! Memory layout for multi-component fields
        StorageOrder layout;

        //! Distribution of global_in across ranks
        std::vector<Index_t> in_counts;
        std::vector<Index_t> in_displs;

        //! Distribution of global_out across ranks
        std::vector<Index_t> out_counts;
        std::vector<Index_t> out_displs;

#ifdef WITH_MPI
        //! Send datatypes for forward transpose (one per peer rank)
        std::vector<MPI_Datatype> send_types_fwd;

        //! Receive datatypes for forward transpose (one per peer rank)
        std::vector<MPI_Datatype> recv_types_fwd;

        //! Send datatypes for backward transpose (one per peer rank)
        std::vector<MPI_Datatype> send_types_bwd;

        //! Receive datatypes for backward transpose (one per peer rank)
        std::vector<MPI_Datatype> recv_types_bwd;

        //! Send counts (always 1 when using derived types)
        std::vector<int> send_counts;

        //! Receive counts (always 1 when using derived types)
        std::vector<int> recv_counts;

        //! Send displacements (always 0 when offset is in datatype)
        std::vector<int> send_displs;

        //! Receive displacements (always 0 when offset is in datatype)
        std::vector<int> recv_displs;
#endif

        //! Flag indicating if datatypes have been initialized
        bool types_initialized{false};

        //! Flag indicating if this is an allgather (no scatter) operation
        bool is_allgather{false};

        //! Flag indicating if this is a scatter-only (no gather) operation
        bool is_scatter_only{false};
    };

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_FFT_DATATYPE_TRANSPOSE_HH_

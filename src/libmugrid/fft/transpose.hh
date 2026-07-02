/**
 * @file   fft/transpose.hh
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   21 Dec 2024
 *
 * @brief  MPI transpose: derived datatypes on host, contiguous staging on
 *         device
 *
 * Host transposes use MPI derived datatypes (MPI_Type_create_subarray,
 * MPI_Type_create_hvector) with MPI_Alltoallw, avoiding explicit
 * pack/unpack buffers and letting MPI optimize the non-contiguous access.
 * Device transposes instead gather each peer's block into a contiguous
 * staging buffer, exchange flat messages pairwise, and scatter on the
 * receiver: MPI implementations pack strided datatypes on device memory
 * block by block, which is orders of magnitude slower than a device-side
 * gather. Both paths support multi-component fields (AoS and SoA layouts)
 * and are wire-compatible with each other's block serialization.
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

#include <array>
#include <cstddef>
#include <vector>

#ifdef WITH_MPI
#include <mpi.h>
#endif

namespace muGrid {

    /**
     * Handles MPI transpose operations.
     *
     * Host data is exchanged with MPI derived datatypes (no explicit
     * pack/unpack; the MPI library handles the non-contiguous access).
     * Device data is staged through cached contiguous device buffers (see
     * the constructor's on_device parameter).
     *
     * For pencil decomposition, data needs to be redistributed between ranks to
     * switch which dimension is "local" (not distributed). The transpose
     * operation is characterized by:
     * - Input: Data distributed along axis_in, full along axis_out
     * - Output: Data distributed along axis_out, full along axis_in
     *
     * The transpose is a genuine scatter-gather: every rank sends a disjoint
     * block to every peer and receives into a disjoint block of the output.
     * All send and receive buffers are non-overlapping, as required by the
     * MPI standard for MPI_Alltoallw.
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
         * @param on_device      True if the data pointers passed to
         *                       forward()/backward() are device memory. Device
         *                       transposes are staged through contiguous
         *                       buffers instead of derived datatypes: MPI
         *                       implementations pack strided datatypes on
         *                       device memory block by block, which is orders
         *                       of magnitude slower than a device-side gather
         *                       followed by a contiguous all-to-all.
         */
        Transpose(const Communicator & comm, const DynGridIndex & local_in,
                  const DynGridIndex & local_out, Index_t global_in,
                  Index_t global_out, Index_t axis_in, Index_t axis_out,
                  Index_t nb_components = 1,
                  StorageOrder layout = StorageOrder::ArrayOfStructures,
                  bool on_device = false, bool single_precision = false);

        Transpose() = delete;
        Transpose(const Transpose & other) = delete;
        // Transposes live behind unique_ptr in the engine's cache and are
        // never moved or copied
        Transpose(Transpose && other) = delete;
        ~Transpose();

        Transpose & operator=(const Transpose & other) = delete;
        Transpose & operator=(Transpose && other) = delete;

        /**
         * Perform forward transpose (gather axis_in, scatter axis_out).
         *
         * @param input   Pointer to input complex data
         * @param output  Pointer to output complex data
         */
        void forward(const Complex * input, Complex * output) const {
            this->forward_impl(input, output);
        }
        //! Single-precision (Complex32) forward transpose. Valid only when the
        //! transpose was constructed with single_precision = true.
        void forward(const Complex32 * input, Complex32 * output) const {
            this->forward_impl(input, output);
        }

        /**
         * Perform backward transpose (reverse of forward).
         *
         * @param input   Pointer to input complex data
         * @param output  Pointer to output complex data
         */
        void backward(const Complex * input, Complex * output) const {
            this->backward_impl(input, output);
        }
        //! Single-precision (Complex32) backward transpose.
        void backward(const Complex32 * input, Complex32 * output) const {
            this->backward_impl(input, output);
        }

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
        //! Element-type-agnostic forward/backward transpose. Operates on raw
        //! bytes (element size + MPI datatype are stored members), so it serves
        //! both the double (Complex) and single (Complex32) overloads above.
        void forward_impl(const void * input, void * output) const;
        void backward_impl(const void * input, void * output) const;

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

        /**
         * All-to-all through contiguous device staging buffers: gather each
         * peer's subarray block of `input` into a flat send buffer
         * (one strided 2D copy per block), MPI_Alltoallv, scatter the flat
         * receive buffer into `output`. Block geometry: full extent in all
         * dimensions except `src_axis`/`dst_axis`, where peer r owns
         * `src_counts[r]` slices starting at `src_displs[r]`.
         */
        void staged_alltoall(const void * input, void * output,
                             const DynGridIndex & src_shape, Index_t src_axis,
                             const std::vector<Index_t> & src_counts,
                             const std::vector<Index_t> & src_displs,
                             const DynGridIndex & dst_shape, Index_t dst_axis,
                             const std::vector<Index_t> & dst_counts,
                             const std::vector<Index_t> & dst_displs) const;

        //! Cached contiguous device staging buffer (slot 0: send, 1: recv)
        char * get_device_staging(std::size_t slot, std::size_t size) const;
#endif

        //! Release the device staging buffers
        void free_staging();

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

        //! Size in bytes of one transported element (one complex value times
        //! nb_components is folded into the MPI datatype, so this is the size of
        //! a single complex scalar: 16 for Complex, 8 for Complex32).
        std::size_t elem_size{sizeof(Complex)};
#ifdef WITH_MPI
        //! MPI datatype of one complex scalar (MPI_C_DOUBLE_COMPLEX /
        //! MPI_C_FLOAT_COMPLEX); the base from which the per-peer block
        //! datatypes are built.
        MPI_Datatype elem_mpi_type{MPI_C_DOUBLE_COMPLEX};
#endif

        //! Flag indicating if datatypes have been initialized
        bool types_initialized{false};

        //! True if forward()/backward() receive device pointers
        bool on_device{false};

        //! On the host, route the exchange through the contiguous-staging
        //! Alltoallv path (explicit cache-friendly pack/unpack) instead of
        //! MPI_Alltoallw with derived datatypes. Opt-in via the environment
        //! variable MUGRID_STAGED_TRANSPOSE=1.
        bool staged_host{false};

        //! Contiguous device staging buffers (send, recv), grown on demand
        mutable std::array<void *, 2> device_staging{{nullptr, nullptr}};
        //! Sizes of the staging buffers in bytes
        mutable std::array<std::size_t, 2> device_staging_size{{0, 0}};
        //! Host bounce buffers (send, recv) used when the MPI library is
        //! not GPU-aware (see mpi/gpu_aware_mpi.hh)
        mutable std::array<std::vector<char>, 2> host_staging{};
    };

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_FFT_DATATYPE_TRANSPOSE_HH_

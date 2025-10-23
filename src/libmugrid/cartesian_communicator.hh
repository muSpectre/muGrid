#ifndef SRC_LIBMUGRID_CARTESIAN_COMMUNICATOR_HH_
#define SRC_LIBMUGRID_CARTESIAN_COMMUNICATOR_HH_

#ifdef WITH_MPI
#include <mpi.h>
#endif

#include "grid_common.hh"
#include "communicator.hh"

namespace muGrid {

    class CartesianCommunicator : public Communicator {
       public:
        using Parent_t = Communicator;

        /**
         * @brief Constructor for a Cartesian communicator.
         * @details Initializes a `CartesianCommunicator` with the parent
         * communicator and the number of subdivisions in each direction.
         * @param parent The parent communicator.
         * @param nb_subdivisions The number of subdivisions in each direction.
         */
        explicit CartesianCommunicator(const Parent_t & parent,
                                       const IntCoord_t & nb_subdivisions);
        /**
         * @brief Construct a Cartesian communicator.
         * @details This constructor initializes a CartesianCommunicator object
         * with the given parent communicator, number of subdivisions,
         * coordinates, and ranks for communication in the left and right
         * directions.
         * @param parent The communicator of parent type.
         * @param nb_subdivisions The number of subdivisions in each direction.
         * @param coordinates The preassigned coordinates of the communicator.
         * @param left_ranks A vector of ranks for communication to the left.
         * @param right_ranks A vector of ranks for communication to the right.
         */
        explicit CartesianCommunicator(const Parent_t & parent,
                                       const IntCoord_t & nb_subdivisions,
                                       const IntCoord_t & coordinates,
                                       const std::vector<int> & left_ranks,
                                       const std::vector<int> & right_ranks);

        CartesianCommunicator() = delete;

        virtual ~CartesianCommunicator() {}

        CartesianCommunicator & operator=(const CartesianCommunicator & other);

        const IntCoord_t & get_nb_subdivisions() const;

        const IntCoord_t & get_coordinates() const;

#ifdef WITH_MPI
        //! send to right; receive from left
        void sendrecv_right(int direction, int block_len,
                            int stride_in_next_dim, int nb_block,
                            Index_t send_offset, Index_t recv_offset,
                            char * begin_addr, int stride_in_direction,
                            int elem_size_in_bytes,
                            MPI_Datatype elem_mpi_t) const;

        //! send to left; receive from right
        void sendrecv_left(int direction, int block_len, int stride_in_next_dim,
                           int nb_block, Index_t send_offset,
                           Index_t recv_offset, char * begin_addr,
                           int stride_in_direction, int elem_size_in_bytes,
                           MPI_Datatype elem_mpi_t) const;
#else   // not WITH_MPI
        void sendrecv_right(int direction, int block_len,
                            int stride_in_next_dim, int nb_block,
                            Index_t send_offset, Index_t recv_offset,
                            char * begin_addr, int stride_in_direction,
                            int elem_size_in_bytes) const;

        void sendrecv_left(int direction, int block_len, int stride_in_next_dim,
                           int nb_block, Index_t send_offset,
                           Index_t recv_offset, char * begin_addr,
                           int stride_in_direction,
                           int elem_size_in_bytes) const;
#endif  // WITH_MPI

       protected:
        Parent_t parent;
        IntCoord_t nb_subdivisions;
        IntCoord_t coordinates;
#if WITH_MPI
        std::vector<int> left_ranks;
        std::vector<int> right_ranks;
#endif  // WITH_MPI
    };
}  // namespace muGrid

#endif  // SRC_LIBMUGRID_CARTESIAN_COMMUNICATOR_HH_

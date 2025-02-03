#ifndef SRC_LIBMUGRID_CARTESIAN_COMMUNICATOR_HH_
#define SRC_LIBMUGRID_CARTESIAN_COMMUNICATOR_HH_

#ifdef WITH_MPI
#include <mpi.h>
#endif

#include "grid_common.hh"
#include "communicator.hh"

namespace muGrid {

#ifdef WITH_MPI
  class CartesianCommunicator : public Communicator {
   public:
    using Parent_t = Communicator;

    /**
     * @brief Construct a Cartesian communicator from the parent communicator
     * with specified shape.
     * @param parent the communicator of parent type
     * @param nb_subdivisions number of subdivisions in each direction
     */
    explicit CartesianCommunicator(const Parent_t & parent,
                                   const DynCcoord_t & nb_subdivisions);

    CartesianCommunicator(const CartesianCommunicator & other);

    virtual ~CartesianCommunicator() {}

    CartesianCommunicator & operator=(const CartesianCommunicator & other);

    const DynCcoord_t & get_nb_subdivisions() const;

    const DynCcoord_t & get_coordinates() const;

    template <typename T>
    void sendrecv_right(const int direction, const int count,
                        const int blocklength, const int strides,
                        void * send_offset, void * recv_offset) {
      MPI_Datatype new_type;
      MPI_Type_vector(count, blocklength, strides, mpi_type<T>(), &new_type);
      MPI_Status status;
      MPI_Sendrecv(send_offset, count, new_type, right_dest_ranks[direction], 0,
                   recv_offset, count, new_type, right_src_ranks[direction], 0,
                   this->comm, &status);
    }

    template <typename T>
    void sendrecv_left(const int direction, const int count,
                        const int blocklength, const int strides,
                        void * send_offset, void * recv_offset) {
      MPI_Datatype new_type;
      MPI_Type_vector(count, blocklength, strides, mpi_type<T>(), &new_type);
      MPI_Status status;
      MPI_Sendrecv(send_offset, count, new_type, left_dest_ranks[direction], 0,
                   recv_offset, count, new_type, left_src_ranks[direction], 0,
                   this->comm, &status);
    }

   protected:
    Parent_t parent;
    DynCcoord_t nb_subdivisions;
    DynCcoord_t coordinates;
    std::vector<int> right_dest_ranks;
    std::vector<int> right_src_ranks;
    std::vector<int> left_dest_ranks;
    std::vector<int> left_src_ranks;
  };

#endif  // WITH_MPI
}  // namespace muGrid

#endif  // SRC_LIBMUGRID_CARTESIAN_COMMUNICATOR_HH_

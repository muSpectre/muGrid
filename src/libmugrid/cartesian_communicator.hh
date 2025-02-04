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

    CartesianCommunicator() = delete;

    virtual ~CartesianCommunicator() {}

    CartesianCommunicator & operator=(const CartesianCommunicator & other);

    const DynCcoord_t & get_nb_subdivisions() const;

    const DynCcoord_t & get_coordinates() const;

    void sendrecv_right(const int direction, const int count,
                        const int blocklength, const int strides,
                        MPI_Datatype old_type, void * send_offset,
                        void * recv_offset) const;

    void sendrecv_left(const int direction, const int count,
                       const int blocklength, const int strides,
                       MPI_Datatype old_type, void * send_offset,
                       void * recv_offset) const;

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

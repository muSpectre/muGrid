#ifdef WITH_MPI
#include <mpi.h>
#endif

#include "grid_common.hh"
#include "communicator.hh"
#include "cartesian_communicator.hh"

namespace muGrid {

  CartesianCommunicator::CartesianCommunicator(
      const Parent_t & parent, const DynCcoord_t & nb_subdivisions)
      : Parent_t{parent.get_mpi_comm()}, parent{parent},
        nb_subdivisions{nb_subdivisions},
        coordinates(nb_subdivisions.size(), -1),
        right_dest_ranks(nb_subdivisions.size(), MPI_PROC_NULL),
        right_src_ranks(nb_subdivisions.size(), MPI_PROC_NULL),
        left_dest_ranks(nb_subdivisions.size(), MPI_PROC_NULL),
        left_src_ranks(nb_subdivisions.size(), MPI_PROC_NULL) {
    // the spatial dimension of the topology
    const int spatial_dim{static_cast<int>(nb_subdivisions.size())};
    // the domain is periodic in all directions
    std::vector<int> is_periodic(spatial_dim, true);
    // reordering is allowed
    const bool reoder_is_allowed{true};

    // create the new communicator with cartesian topology
    std::vector<int> narr(spatial_dim);
    std::copy(nb_subdivisions.begin(), nb_subdivisions.end(), narr.begin());
    MPI_Cart_create(this->comm, spatial_dim, narr.data(), is_periodic.data(),
                    reoder_is_allowed, &this->comm);

    // get coordinates of current rank
    MPI_Cart_coords(this->comm, this->rank(), spatial_dim, narr.data());
    std::copy(narr.begin(), narr.end(), this->coordinates.begin());

    // get the ranks of the neighbors
    for (auto direction{0}; direction < spatial_dim; ++direction) {
      MPI_Cart_shift(this->comm, direction, 1,
                     &this->right_src_ranks[direction],
                     &this->right_dest_ranks[direction]);
      MPI_Cart_shift(this->comm, direction, -1,
                     &this->left_src_ranks[direction],
                     &this->left_dest_ranks[direction]);
    }
  }

  CartesianCommunicator &
  CartesianCommunicator::operator=(const CartesianCommunicator & other) {
    this->comm = other.comm;
    return *this;
  }

  void CartesianCommunicator::sendrecv_right(
      const int direction, const int count, const int blocklength,
      const int strides, MPI_Datatype old_type, void * send_offset,
      void * recv_offset) const {
    MPI_Datatype new_type;
    MPI_Type_vector(count, blocklength, strides, old_type, &new_type);
    MPI_Status status;
    MPI_Sendrecv(send_offset, count, new_type,
                 this->right_dest_ranks[direction], 0, recv_offset, count,
                 new_type, this->right_src_ranks[direction], 0, this->comm,
                 &status);
  }

  void CartesianCommunicator::sendrecv_left(
      const int direction, const int count, const int blocklength,
      const int strides, MPI_Datatype old_type, void * send_offset,
      void * recv_offset) const {
    MPI_Datatype new_type;
    MPI_Type_vector(count, blocklength, strides, old_type, &new_type);
    MPI_Status status;
    MPI_Sendrecv(send_offset, count, new_type, this->left_dest_ranks[direction],
                 0, recv_offset, count, new_type,
                 this->left_src_ranks[direction], 0, this->comm, &status);
  }

  const DynCcoord_t & CartesianCommunicator::get_nb_subdivisions() const {
    return this->nb_subdivisions;
  }

  const DynCcoord_t & CartesianCommunicator::get_coordinates() const {
    return this->coordinates;
  }

}  // namespace muGrid

#ifdef WITH_MPI
#include <mpi.h>
#endif

#include "grid_common.hh"
#include "communicator.hh"
#include "cartesian_communicator.hh"

namespace muGrid {

  CartesianCommunicator::CartesianCommunicator(
      const Parent_t & parent, const DynCcoord_t & nb_subdivisions)
      : Parent_t{parent.get_mpi_comm()}, parent{parent} {
    // the spatial dimension of the topology
    const int spatial_dim{static_cast<int>(nb_subdivisions.size())};
    // the domain is periodic in all directions
    std::vector<int> is_periodic{};
    is_periodic.resize(spatial_dim, true);
    // reordering is allowed
    const bool reoder_is_allowed{true};
    // create the new communicator with cartesian topology
    MPI_Cart_create(this->comm, spatial_dim,
                    reinterpret_cast<const int *>(nb_subdivisions.data()),
                    is_periodic.data(), reoder_is_allowed, &this->comm);

    // get coordinates of current rank
    MPI_Cart_coords(this->comm, this->rank(), spatial_dim,
                    reinterpret_cast<int *>(this->coordinates.data()));

    // get the ranks of the neighbors
    this->right_dest_ranks.reserve(spatial_dim);
    this->right_src_ranks.reserve(spatial_dim);
    this->left_dest_ranks.reserve(spatial_dim);
    this->left_src_ranks.reserve(spatial_dim);
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

  const DynCcoord_t & CartesianCommunicator::get_nb_subdivisions() const {
    return this->nb_subdivisions;
  }

  const DynCcoord_t & CartesianCommunicator::get_coordinates() const {
    return this->coordinates;
  }

}  // namespace muGrid

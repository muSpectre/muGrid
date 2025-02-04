#include <iterator>

#ifdef WITH_MPI
#include <mpi.h>
#endif

#include "grid_common.hh"
#include "field.hh"
#include "field_collection_global.hh"
#include "cartesian_communicator.hh"
#include "cartesian_decomposition.hh"

namespace muGrid {
#ifdef WITH_MPI

  CartesianDecomposition::CartesianDecomposition(
      const Communicator & comm, const DynCcoord_t & nb_domain_grid_pts,
      const DynCcoord_t & nb_subdivisions, const DynCcoord_t & nb_ghosts_left,
      const DynCcoord_t & nb_ghosts_right, const SubPtMap_t & nb_sub_pts)
      : Parent_t{}, nb_ghosts_left{nb_ghosts_left},
        nb_ghosts_right{nb_ghosts_right},
        comm{CartesianCommunicator(comm, nb_subdivisions)} {
    // get spatial dimensions
    auto spatial_dims{nb_domain_grid_pts.size()};
    // check spatial dimensions are matching
    if (nb_ghosts_left.size() != spatial_dims) {
      throw RuntimeError("The spatial dimension of the left ghost buffer is "
                         "not compatible with the decomposition.");
    }
    if (nb_ghosts_right.size() != spatial_dims) {
      throw RuntimeError("The spatial dimension of the right ghost buffer is "
                         "not compatible with the decomposition.");
    }

    // compute bare domain decomposition without ghosts
    // FIXME(Lars): This is a suboptimal decomposition. We actually want each
    // the number of grid point per MPI process to vary by 1 in each direction
    // at most.
    auto nb_subdomain_grid_pts{nb_domain_grid_pts / nb_subdivisions};
    auto coordinates{this->comm.get_coordinates()};
    auto subdomain_locations{coordinates * nb_subdomain_grid_pts};
    for (int dim{0}; dim < spatial_dims; ++dim) {
      if (coordinates[dim] == nb_subdivisions[dim] - 1) {
        // We are the last MPI process on the right, add missing grid points
        // here
        nb_subdomain_grid_pts[dim] +=
            nb_domain_grid_pts[dim] -
            nb_subdomain_grid_pts[dim] * nb_subdivisions[dim];
      }
    }

    // Adjust domain decomposition for ghosts
    nb_subdomain_grid_pts += nb_ghosts_left + nb_ghosts_right;
    subdomain_locations -= nb_ghosts_left;

    // Initialize field collection (we now know the subdivision)
    this->collection = std::make_unique<GlobalFieldCollection>(
        nb_domain_grid_pts, nb_subdomain_grid_pts, subdomain_locations,
        nb_sub_pts);
  }

  void
  CartesianDecomposition::communicate_ghosts(std::string field_name) const {
    // Get shape of the fields on this processor
    auto nb_subdomain_grid_pts{this->get_nb_subdomain_grid_pts()};
    auto nb_subdomain_grid_pts_without_ghosts{
        nb_subdomain_grid_pts - this->nb_ghosts_right - this->nb_ghosts_left};

    // Get spatial dimensions
    int spatial_dims{nb_subdomain_grid_pts.size()};

    // Get field
    auto & field{this->collection->get_field(field_name)};

    // Get shape
    auto shape{field.get_shape(IterUnit::SubPt)};

    // Get strides (in unit: elements)
    auto strides{field.get_strides(IterUnit::SubPt)};

    // Get element size (only useful for pointer arithmetic in finding the right
    // offset, but in the MPI routine, it uses elements as the unit.)
    auto element_size{field.get_element_size_in_bytes()};

    // Blocklength (number of elements at one location)
    Index_t blocklength{strides[strides.size() - spatial_dims]};

    // For each direction...
    for (int direction{0}; direction < spatial_dims; ++direction) {
      // Number of blocks in ghost buffer
      Index_t count{1};
      for (int i{0}; i < spatial_dims; ++i) {
        if (i != direction)
          count *= nb_subdomain_grid_pts[i];
      }

      // Stride for sending slices at the border
      auto sendrecv_strides{strides};
      auto it{sendrecv_strides.begin()};
      std::advance(it, sendrecv_strides.size() - spatial_dims + direction);
      sendrecv_strides.erase(it);

      // Offset of send and receive buffers
      auto strides_in_direction{
          strides[strides.size() - spatial_dims + direction]};

      // Sending things to the RIGHT
      // When sending right, we need the size of the ghost layer on the left
      for (int ghost_layer{0}; ghost_layer < this->nb_ghosts_left[direction];
           ++ghost_layer) {
        Index_t send_layer{ghost_layer +
                           nb_subdomain_grid_pts_without_ghosts[direction]};
        Index_t send_offset{strides_in_direction * send_layer};
        Index_t recv_offset{strides_in_direction * ghost_layer};

        // send to right, receive from left
        this->comm.sendrecv_right<field.get_stored_typeid()>(
            direction, count, blocklength, strides_in_direction,
            static_cast<void *>(static_cast<char *>(field.get_void_data_ptr()) +
                                send_offset * element_size),
            static_cast<void *>(static_cast<char *>(field.get_void_data_ptr()) +
                                recv_offset * element_size));
      }

      // Sending things to the LEFT
      for (int ghost_layer{0}; ghost_layer < this->nb_ghosts_right[direction];
           ++ghost_layer) {
        Index_t send_offset{strides_in_direction * ghost_layer};
        Index_t recv_layer{ghost_layer +
                           nb_subdomain_grid_pts_without_ghosts[direction]};
        Index_t recv_offset{strides_in_direction * recv_layer};

        // send to left, receive from right
        this->comm.sendrecv_left<field.get_stored_typeid()>(
            direction, count, blocklength, strides_in_direction,
            static_cast<void *>(static_cast<char *>(field.get_void_data_ptr()) +
                                send_offset * element_size),
            static_cast<void *>(static_cast<char *>(field.get_void_data_ptr()) +
                                recv_offset * element_size));
      }
    }
  }

  GlobalFieldCollection & CartesianDecomposition::get_collection() const {
    return *this->collection;
  }

  const DynCcoord_t CartesianDecomposition::get_nb_subdivisions() const {
    return this->comm.get_nb_subdivisions();
  }

  const DynCcoord_t CartesianDecomposition::get_nb_subdomain_grid_pts() const {
    return (this->collection)->get_nb_subdomain_grid_pts();
  }
  const DynCcoord_t CartesianDecomposition::get_subdomain_locations() const {
    return (this->collection)->get_subdomain_locations();
  }

#endif  // WITH_MPI
}  // namespace muGrid

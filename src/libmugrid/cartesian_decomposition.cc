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
    // Get spatial dimensions
    auto spatial_dims{nb_domain_grid_pts.size()};
    // Check spatial dimensions are matching
    if (nb_subdivisions.size() != spatial_dims) {
      throw RuntimeError("The spatial dimension of the subdivisions is "
                         "not compatible with the decomposition.");
    }
    if (nb_ghosts_left.size() != spatial_dims) {
      throw RuntimeError("The spatial dimension of the left ghost buffer is "
                         "not compatible with the decomposition.");
    }
    if (nb_ghosts_right.size() != spatial_dims) {
      throw RuntimeError("The spatial dimension of the right ghost buffer is "
                         "not compatible with the decomposition.");
    }

    // Compute bare domain decomposition without ghosts
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

    // Check if the ghost buffer covers more than one subdomain (process)
    for (int dim{0}; dim < spatial_dims; ++dim) {
      if (nb_ghosts_left[dim] > nb_subdomain_grid_pts[dim] ||
          nb_ghosts_right[dim] > nb_subdomain_grid_pts[dim]) {
        throw RuntimeError("It is not allowed to have ghost buffers covering "
                           "more than one subdomain.");
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

    // Get spatial dimensions
    int spatial_dims{nb_subdomain_grid_pts.size()};

    // Get field
    auto & field{this->collection->get_field(field_name)};

    // Get strides (in unit: elements)
    auto strides{field.get_strides(IterUnit::SubPt)};

    // Get the begin address of the field data (cast into char * for pointer
    // arithemtic)
    auto * begin_addr{static_cast<char *>(field.get_void_data_ptr())};

    // Get element size (only useful for pointer arithmetic in finding the
    // correct offset)
    auto element_size{static_cast<Index_t>(field.get_element_size_in_bytes())};

    // For each direction...
    for (int direction{0}; direction < spatial_dims; ++direction) {
      // Stride in the send/recv direction
      auto stride_in_direction{
          strides[strides.size() - spatial_dims + direction]};
      // Stride inside the ghost buffer
      auto stride_between_contiguous{stride_in_direction *
                                     nb_subdomain_grid_pts[direction]};
      // Number of strides inside the ghost buffer
      auto nb_strided{strides[strides.size() - 1] *
                      nb_subdomain_grid_pts[spatial_dims - 1] /
                      stride_between_contiguous};

      // Sending things to the RIGHT

      // When sending right, we need the ghost buffer on left to receive
      auto nb_contiguous_l{stride_in_direction *
                           this->nb_ghosts_left[direction]};

      // Create an MPI type for the ghost buffer left
      MPI_Datatype buffer_l_mpi_t;
      MPI_Type_vector(nb_strided, nb_contiguous_l, stride_between_contiguous,
                      field.get_mpi_type(), &buffer_l_mpi_t);
      MPI_Type_commit(&buffer_l_mpi_t);

      // Offset of send and receive buffers
      Index_t send_offset_r{nb_subdomain_grid_pts[direction] -
                            nb_ghosts_right[direction] -
                            nb_ghosts_left[direction]};
      Index_t recv_offset_r{0};

      // Send to right, receive from left
      this->comm.sendrecv_right(
          direction, 1,
          static_cast<void *>(begin_addr + element_size * stride_in_direction *
                                               send_offset_r),
          static_cast<void *>(begin_addr + element_size * stride_in_direction *
                                               recv_offset_r),
          buffer_l_mpi_t);

      // Sending things to the LEFT

      // When sending right, we need the ghost buffer on left to receive
      auto nb_contiguous_r{stride_in_direction *
                           this->nb_ghosts_right[direction]};

      // Create an MPI type for the ghost buffer right
      MPI_Datatype buffer_r_mpi_t;
      MPI_Type_vector(nb_strided, nb_contiguous_r, stride_between_contiguous,
                      field.get_mpi_type(), &buffer_r_mpi_t);
      MPI_Type_commit(&buffer_r_mpi_t);

      // Offset of send and receive buffers
      Index_t send_offset_l{nb_ghosts_left[direction]};
      Index_t recv_offset_l{nb_subdomain_grid_pts[direction] -
                            nb_ghosts_right[direction]};
      // Send to left, receive from right
      this->comm.sendrecv_left(
          direction, 1,
          static_cast<void *>(begin_addr + element_size * stride_in_direction *
                                               send_offset_l),
          static_cast<void *>(begin_addr + element_size * stride_in_direction *
                                               recv_offset_l),
          buffer_r_mpi_t);
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

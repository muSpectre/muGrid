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

    // Blocklength (number of elements at one location)
    auto blocklength{strides[strides.size() - spatial_dims - 1]};

    // Create an MPI type for field data at one location
    MPI_Datatype block_mpi_t;
    MPI_Type_contiguous(blocklength, field.get_mpi_type(), &block_mpi_t);
    MPI_Type_commit(&block_mpi_t);

    // Get the begin address of the field data (cast into char * for pointer
    // arithemtic)
    auto * begin_addr{static_cast<char *>(field.get_void_data_ptr())};

    // Get element size (only useful for pointer arithmetic in finding the
    // correct offset)
    auto element_size{static_cast<Index_t>(field.get_element_size_in_bytes())};

    // Sending things to the RIGHT
    // For each direction...
    for (int direction{0}; direction < spatial_dims; ++direction) {
      // When sending right, we need the ghost buffer on left to receive

      // Figuring the memory layout of the ghost buffer...
      // All dimensions faster than ghost are contiguous in memory
      auto nb_contiguous{this->nb_ghosts_left[direction]};
      auto stride_between_contiguous{nb_subdomain_grid_pts[direction]};
      for (int faster_dim{0}; faster_dim < direction; ++faster_dim) {
        nb_contiguous *= nb_subdomain_grid_pts[faster_dim];
        stride_between_contiguous *= nb_subdomain_grid_pts[faster_dim];
      }
      // All dimensions slower than ghost are not, hence strided
      Index_t nb_strided{1};
      for (int slower_dim{direction + 1}; slower_dim < spatial_dims;
           ++slower_dim) {
        nb_strided *= nb_subdomain_grid_pts[slower_dim];
      }

      // Create an MPI type for the ghost buffer
      MPI_Datatype buffer_mpi_t;
      MPI_Type_vector(nb_strided, nb_contiguous, stride_between_contiguous,
                      block_mpi_t, &buffer_mpi_t);
      MPI_Type_commit(&buffer_mpi_t);

      // Stride between layers of the ghost buffer (unit: bytes!), used in
      // pointer arithmetic
      auto stride_in_direction{
          element_size * strides[strides.size() - spatial_dims + direction]};

      // Offset of send and receive buffers
      Index_t send_layer{nb_subdomain_grid_pts[direction] -
                         nb_ghosts_right[direction] -
                         nb_ghosts_left[direction]};
      Index_t recv_layer{0};

      // Send to right, receive from left
      this->comm.sendrecv_right(
          direction, 1,
          static_cast<void *>(begin_addr + stride_in_direction * send_layer),
          static_cast<void *>(begin_addr + stride_in_direction * recv_layer),
          buffer_mpi_t);
    }

    // Sending things to the LEFT
    // For each direction...
    for (int direction{0}; direction < spatial_dims; ++direction) {
      // When sending to left, we need the ghost buffer on right to recevie

      // Figuring the memory layout of the ghost buffer...
      // All dimensions faster than ghost are contiguous in memory
      auto nb_contiguous{this->nb_ghosts_right[direction]};
      auto stride_between_contiguous{nb_subdomain_grid_pts[direction]};
      for (int faster_dim{0}; faster_dim < direction; ++faster_dim) {
        nb_contiguous *= nb_subdomain_grid_pts[faster_dim];
        stride_between_contiguous *= nb_subdomain_grid_pts[faster_dim];
      }
      // All dimensions slower than ghost are not, hence strided
      Index_t nb_strided{1};
      for (int slower_dim{direction + 1}; slower_dim < spatial_dims;
           ++slower_dim) {
        nb_strided *= nb_subdomain_grid_pts[slower_dim];
      }

      // Create an MPI type for the ghost buffer
      MPI_Datatype buffer_mpi_t;
      MPI_Type_vector(nb_strided, nb_contiguous, stride_between_contiguous,
                      block_mpi_t, &buffer_mpi_t);
      MPI_Type_commit(&buffer_mpi_t);

      // Stride between layers of the ghost buffer (unit: bytes!), used in
      // pointer arithmetic
      auto stride_in_direction{
          element_size * strides[strides.size() - spatial_dims + direction]};

      // Offset of send and receive buffers
      Index_t send_layer{nb_ghosts_left[direction]};
      Index_t recv_layer{nb_subdomain_grid_pts[direction] -
                         nb_ghosts_right[direction]};
      // Send to left, receive from right
      this->comm.sendrecv_left(
          direction, 1,
          static_cast<void *>(begin_addr + stride_in_direction * send_layer),
          static_cast<void *>(begin_addr + stride_in_direction * recv_layer),
          buffer_mpi_t);
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

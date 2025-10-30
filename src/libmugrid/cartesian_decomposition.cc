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
    CartesianDecomposition::CartesianDecomposition(
        const Communicator & comm, Index_t spatial_dimension,
        const SubPtMap_t & nb_sub_pts)
        : Parent_t{}, comm{comm}, collection(spatial_dimension, nb_sub_pts) {}

    CartesianDecomposition::CartesianDecomposition(
        const Communicator & comm, const IntCoord_t & nb_domain_grid_pts,
        const IntCoord_t & nb_subdivisions, const IntCoord_t & nb_ghosts_left,
        const IntCoord_t & nb_ghosts_right, const SubPtMap_t & nb_sub_pts)
        : Parent_t{}, comm{comm},
          collection(nb_domain_grid_pts.size(), nb_sub_pts) {
        this->initialise(nb_domain_grid_pts, nb_subdivisions, nb_ghosts_left,
                         nb_ghosts_right);
    }

    void
    CartesianDecomposition::check_dimension(const IntCoord_t & n,
                                            const std::string & name) const {
        if (this->collection.get_spatial_dim() != n.get_dim()) {
            std::stringstream s;
            s << "The number of spatial dimensions of argument `" << name
              << "` during does not match the "
                 "number of spatial dimensions of the field collection.";
            throw RuntimeError(s.str());
        }
    }

    void CartesianDecomposition::initialise(
        const IntCoord_t & nb_domain_grid_pts,
        const IntCoord_t & nb_subdivisions,
        const IntCoord_t & nb_subdomain_grid_pts_without_ghosts,
        const IntCoord_t & subdomain_locations_without_ghosts,
        const IntCoord_t & nb_ghosts_left,
        const IntCoord_t & nb_ghosts_right,
        const IntCoord_t & subdomain_strides) {
        // Idiot checks
        this->check_dimension(nb_domain_grid_pts, "nb_domain_grid_pts");
        this->check_dimension(nb_subdivisions, "nb_subdivisions");
        this->check_dimension(nb_subdomain_grid_pts_without_ghosts,
                              "nb_subdomain_grid_pts_without_ghosts");
        this->check_dimension(subdomain_locations_without_ghosts,
                              "subdomain_locations_without_ghosts");
        this->check_dimension(nb_ghosts_left, "nb_ghosts_left");
        this->check_dimension(nb_ghosts_right, "nb_ghosts_right");

        // Create Cartesian communicator if this has not already happened
        if (this->cart_comm) {
            // Since we don't have a Cartesian communicator, we assume that the
            // subdivision information does not come from the communicator but
            // some auxiliary source (e.g. the FFT library).
            this->cart_comm = std::make_unique<CartesianCommunicator>(
                this->comm, nb_subdivisions);
        }

        // Grid points and locations
        auto nb_subdomain_grid_pts{nb_subdomain_grid_pts_without_ghosts};
        auto subdomain_locations{subdomain_locations_without_ghosts};

        // Adjust domain decomposition for ghosts
        subdomain_locations -= nb_ghosts_left;
        nb_subdomain_grid_pts += nb_ghosts_left + nb_ghosts_right;

        // Initialize field collection (we know the subdivision)
        if (subdomain_strides.get_dim() == 0) {
            this->collection.initialise(
                nb_domain_grid_pts, nb_subdomain_grid_pts, subdomain_locations,
                StorageOrder::ArrayOfStructures,
                nb_ghosts_left, nb_ghosts_right);
        } else {
            this->check_dimension(subdomain_strides, "subdomain_strides");
            this->collection.initialise(
                nb_domain_grid_pts, nb_subdomain_grid_pts, subdomain_locations,
                subdomain_strides, nb_ghosts_left, nb_ghosts_right);
        }
    }

    void
    CartesianDecomposition::initialise(const IntCoord_t & nb_domain_grid_pts,
                                       const IntCoord_t & nb_subdivisions,
                                       const IntCoord_t & nb_ghosts_left,
                                       const IntCoord_t & nb_ghosts_right) {
        // Idiot checks
        this->check_dimension(nb_domain_grid_pts, "nb_domain_grid_pts");
        this->check_dimension(nb_subdivisions, "nb_subdivisions");
        this->check_dimension(nb_ghosts_left, "nb_ghosts_left");
        this->check_dimension(nb_ghosts_right, "nb_ghosts_right");

        // Get spatial dimensions
        auto spatial_dims{nb_domain_grid_pts.size()};

        // Create Cartesian communicator
        this->cart_comm = std::make_unique<CartesianCommunicator>(
            this->comm, nb_subdivisions);

        // Compute bare domain decomposition without ghosts
        auto nb_subdomain_grid_pts{nb_domain_grid_pts / nb_subdivisions};
        auto coordinates{this->cart_comm->get_coordinates()};
        auto subdomain_locations{coordinates * nb_subdomain_grid_pts};
        auto nb_residual_grid_pts{nb_domain_grid_pts % nb_subdivisions};
        for (int dim{0}; dim < spatial_dims; ++dim) {
            // Adjust domain decomposition for the residual grid points
            if (coordinates[dim] < nb_residual_grid_pts[dim]) {
                nb_subdomain_grid_pts[dim] += 1;
                subdomain_locations[dim] += coordinates[dim];
            } else {
                subdomain_locations[dim] += nb_residual_grid_pts[dim];
            }
        }

        this->initialise(nb_domain_grid_pts, nb_subdivisions,
                         nb_subdomain_grid_pts, subdomain_locations,
                         nb_ghosts_left, nb_ghosts_right);
    }

    void CartesianDecomposition::communicate_ghosts(const Field & field) const {
        // Get shape of the fields on this processor
        auto nb_subdomain_grid_pts{
            this->get_nb_subdomain_grid_pts_with_ghosts()};

        // Get spatial dimensions
        int spatial_dims{nb_subdomain_grid_pts.size()};

        // Get strides (in unit: elements)
        auto strides{field.get_strides(IterUnit::SubPt)};

        // Total number of elements in the field
        auto nb_total_elements{strides[strides.size() - 1] *
                               nb_subdomain_grid_pts[spatial_dims - 1]};

        // Get the begin address of the field data (cast into char * for pointer
        // arithemtic)
        auto * begin_addr{static_cast<char *>(field.get_void_data_ptr())};

        // Get element size (only useful for pointer arithmetic in finding the
        // correct offset)
        auto element_size{
            static_cast<Index_t>(field.get_element_size_in_bytes())};

        // Get ghost buffer sizes
        auto nb_ghosts_left{this->collection.get_nb_ghosts_left()};
        auto nb_ghosts_right{this->collection.get_nb_ghosts_right()};

        // Get the number of grid points without ghosts
        auto nb_subdomain_grid_pts_no_ghosts{
            this->get_nb_subdomain_grid_pts_without_ghosts()};

        // For each direction...
        for (int direction{0}; direction < spatial_dims; ++direction) {
            // Stride in the send/recv direction
            auto stride_in_direction{
                strides[strides.size() - spatial_dims + direction]};
            // Stride in the very next dimension
            auto stride_in_next_dim{stride_in_direction *
                                    nb_subdomain_grid_pts[direction]};
            // Number of blocks inside the ghost buffer
            auto nb_blocks_seen_in_next_dim{nb_total_elements /
                                            stride_in_next_dim};

            // Calculate number of communication steps needed
            // Each step can exchange at most subdomain_size worth of data
            auto nb_subdomain_size{nb_subdomain_grid_pts_no_ghosts[direction]};

            // Handle zero grid points case: still need to do communication
            // if we have ghost buffers
            int nb_steps_left;
            if (nb_subdomain_size > 0) {
                nb_steps_left = static_cast<int>(
                    (nb_ghosts_left[direction] + nb_subdomain_size - 1) /
                    nb_subdomain_size);
            } else {
                nb_steps_left = (nb_ghosts_left[direction] > 0 ? 1 : 0);
            }

            int nb_steps_right;
            if (nb_subdomain_size > 0) {
                nb_steps_right = static_cast<int>(
                    (nb_ghosts_right[direction] + nb_subdomain_size - 1) /
                    nb_subdomain_size);
            } else {
                nb_steps_right = (nb_ghosts_right[direction] > 0 ? 1 : 0);
            }

            // Perform multiple communication steps to the RIGHT
            for (int step{0}; step < nb_steps_right; ++step) {
                // For each step, we exchange the real data + previously received
                // ghost data
                auto block_len_ghost_left{stride_in_direction *
                                          nb_ghosts_left[direction]};

                // Send offset: start from the real data (adjusted for step)
                Index_t send_offset_right{nb_subdomain_grid_pts[direction] -
                                          nb_ghosts_right[direction] -
                                          nb_ghosts_left[direction]};
                // Receive offset: into the left ghost buffer
                Index_t recv_offset_right{0};

#ifdef WITH_MPI
                MPI_Datatype mpi_type{field.get_mpi_type()};
                void * mpi_type_ptr{static_cast<void *>(&mpi_type)};
#else
                void * mpi_type_ptr{nullptr};
#endif
                this->cart_comm->sendrecv_right(
                    direction, block_len_ghost_left, stride_in_next_dim,
                    nb_blocks_seen_in_next_dim, send_offset_right,
                    recv_offset_right, begin_addr, stride_in_direction,
                    element_size, mpi_type_ptr);
            }

            // Perform multiple communication steps to the LEFT
            for (int step{0}; step < nb_steps_left; ++step) {
                auto block_len_ghost_right{stride_in_direction *
                                           nb_ghosts_right[direction]};

                // Send offset: start from the real data (adjusted for step)
                Index_t send_offset_left{nb_ghosts_left[direction]};
                // Receive offset: into the right ghost buffer
                Index_t recv_offset_left{nb_subdomain_grid_pts[direction] -
                                         nb_ghosts_right[direction]};

#ifdef WITH_MPI
                MPI_Datatype mpi_type_left{field.get_mpi_type()};
                void * mpi_type_ptr_left{static_cast<void *>(&mpi_type_left)};
#else
                void * mpi_type_ptr_left{nullptr};
#endif
                this->cart_comm->sendrecv_left(
                    direction, block_len_ghost_right, stride_in_next_dim,
                    nb_blocks_seen_in_next_dim, send_offset_left, recv_offset_left,
                    begin_addr, stride_in_direction, element_size,
                    mpi_type_ptr_left);
            }
        }
    }

    void CartesianDecomposition::communicate_ghosts(
        const std::string & field_name) const {
        this->communicate_ghosts(this->collection.get_field(field_name));
    }

    GlobalFieldCollection & CartesianDecomposition::get_collection() {
        return this->collection;
    }

    const GlobalFieldCollection &
    CartesianDecomposition::get_collection() const {
        return this->collection;
    }

    Index_t CartesianDecomposition::get_spatial_dim() const {
        return this->collection.get_spatial_dim();
    }

    const IntCoord_t & CartesianDecomposition::get_nb_subdivisions() const {
        return this->cart_comm->get_nb_subdivisions();
    }

    const IntCoord_t & CartesianDecomposition::get_nb_domain_grid_pts() const {
        return this->collection.get_nb_domain_grid_pts();
    }

    const IntCoord_t &
    CartesianDecomposition::get_nb_subdomain_grid_pts_with_ghosts() const {
        return this->collection.get_nb_subdomain_grid_pts_with_ghosts();
    }

    IntCoord_t
    CartesianDecomposition::get_nb_subdomain_grid_pts_without_ghosts() const {
        return this->collection.get_nb_subdomain_grid_pts_without_ghosts();
    }

    const IntCoord_t &
    CartesianDecomposition::get_subdomain_locations_with_ghosts() const {
        return this->collection.get_subdomain_locations_with_ghosts();
    }

    IntCoord_t
    CartesianDecomposition::get_subdomain_locations_without_ghosts() const {
        return this->collection.get_subdomain_locations_without_ghosts();
    }
}  // namespace muGrid

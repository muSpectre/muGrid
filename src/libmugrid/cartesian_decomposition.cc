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
        const IntCoord_t & nb_ghosts_left, const IntCoord_t & nb_ghosts_right,
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
        if (!this->cart_comm) {
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
                StorageOrder::ArrayOfStructures, nb_ghosts_left,
                nb_ghosts_right);
        } else {
            this->check_dimension(subdomain_strides, "subdomain_strides");
            this->collection.initialise(
                nb_domain_grid_pts, nb_subdomain_grid_pts, subdomain_locations,
                subdomain_strides, nb_ghosts_left, nb_ghosts_right);
        }

        // Determine communication strategy
        this->recv_right_sequence.resize(this->get_spatial_dim());
        this->recv_left_sequence.resize(this->get_spatial_dim());
        this->nb_sendrecv_steps.resize(this->get_spatial_dim());
        for (int direction{0}; direction < this->get_spatial_dim();
             ++direction) {
            // Compute the sequence of sendrecv events required to fill the
            // ghost buffer
            this->recv_right_sequence[direction].resize(0);
            Index_t nb_cum_send_right{0}, nb_cum_send_left{0};

            // Ghost buffers in direction
            auto nb_ghosts_right{this->get_nb_ghosts_right()[direction]};
            auto nb_ghosts_left{this->get_nb_ghosts_left()[direction]};

            // We can send this many slices to the right now; we need to fill
            // the left ghost buffer of the right rank
            auto nb_send_right{
                std::min(nb_subdomain_grid_pts_without_ghosts[direction],
                         nb_ghosts_left)};

            // We can send this many slices to the left now; we need to fill
            // the right ghost buffer of the left rank
            auto nb_send_left{
                std::min(nb_subdomain_grid_pts_without_ghosts[direction],
                         nb_ghosts_right)};

            int step{0};
            while (this->cart_comm->any(nb_cum_send_right < nb_ghosts_left ||
                                        nb_cum_send_left < nb_ghosts_right)) {
                auto nb_recv_left{
                    this->cart_comm->sendrecv_right(direction, nb_send_right)};
                auto nb_recv_right{
                    this->cart_comm->sendrecv_left(direction, nb_send_left)};
                this->recv_left_sequence[direction].push_back(nb_recv_left);
                this->recv_right_sequence[direction].push_back(nb_recv_right);

                // Update how many slices we have already sent to the right/left
                nb_cum_send_right += nb_send_right;
                nb_cum_send_left += nb_send_left;

                // Determine how much additional data we can now send
                nb_send_right =
                    std::min(nb_ghosts_left - nb_cum_send_right, nb_recv_left);
                nb_send_left =
                    std::min(nb_ghosts_right - nb_cum_send_left, nb_recv_right);

                // Count how many send/recv cycles we need
                step++;
            }

            this->nb_sendrecv_steps[direction] = step;
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
        // Get spatial dimensions
        auto spatial_dims{this->get_spatial_dim()};

        // Get strides (in unit: elements)
        auto strides{field.get_strides(IterUnit::SubPt)};

        // Total number of elements in the field
        // FIXME: This appears to be assuming a column-major field
        auto nb_total_elements{
            strides[strides.size() - 1] *
            this->get_nb_subdomain_grid_pts_with_ghosts()[spatial_dims - 1]};

        // Get the begin address of the field data (cast into char * for pointer
        // arithmetics)
        auto * data{static_cast<char *>(field.get_void_data_ptr())};

        // Get element size (only useful for pointer arithmetic in finding the
        // correct offset)
        auto element_size{
            static_cast<Index_t>(field.get_element_size_in_bytes())};

#ifdef WITH_MPI
        MPI_Datatype mpi_type{field.get_mpi_type()};
        void * mpi_type_ptr{static_cast<void *>(&mpi_type)};
#else
        void * mpi_type_ptr{nullptr};
#endif

        // FIXME! The code below assumes a specific form of data layout,
        // essentially column-major but with potentially varying slides
        // --- i.e. the first axis needs to be fastest and then it becomes
        // slower in order of the axes. It also assumes an array of structures
        // layout. We should either generalize or introduce a guard that fails
        // if the data layout is wrong.

        // For each direction...
        for (int direction{0}; direction < spatial_dims; ++direction) {
            // Grid size
            auto nb_subdomain_grid_pts_without_ghosts{
                this->get_nb_subdomain_grid_pts_without_ghosts()[direction]};

            // Calculate memory layout; we assume column-major layout
            // possibly with padding (first index is fastest). The
            // following contains the instructions to send a single
            // D-1 dimensional slice of the buffer. The slice has a normal
            // in `direction`.

            // The block length equals the strides; this is also the stride
            // between slices which means to send multiple consecutive slices
            // we just send more blocks.
            auto block_len{strides[strides.size() - spatial_dims + direction]};
            // Block stride
            auto block_stride{
                direction < spatial_dims - 1
                    ? strides[strides.size() - spatial_dims + direction + 1]
                    : nb_total_elements};
            /*
             auto block_stride{
                 block_len *
                 this->get_nb_subdomain_grid_pts_with_ghosts()[direction]};
            */

            // Number of blocks for single slice
            auto nb_blocks{nb_total_elements / block_stride};

            // Calculate number of communication steps needed
            auto nb_ghosts_right{this->get_nb_ghosts_right()[direction]};
            auto nb_ghosts_left{this->get_nb_ghosts_left()[direction]};

            // Compute the sequence of sendrecv events required to fill the
            // ghost buffer
            Index_t nb_cum_send_right{0}, nb_cum_send_left{0};
            Index_t nb_cum_recv_right{0}, nb_cum_recv_left{0};

            // We can send this many slices to the right now; we need to fill
            // the left ghost buffer of the right rank
            auto nb_send_right{
                std::min(nb_subdomain_grid_pts_without_ghosts, nb_ghosts_left)};

            // We can send this many slices to the left now; we need to fill
            // the right ghost buffer of the left rank
            auto nb_send_left{std::min(nb_subdomain_grid_pts_without_ghosts,
                                       nb_ghosts_right)};

            // Loop until ghost buffers have been filled
            for (Index_t step{0}; step < this->nb_sendrecv_steps[direction];
                 ++step) {
                // Idiot check that there is still stuff left to send
                assert(
                    this->cart_comm->any(nb_cum_send_right < nb_ghosts_left ||
                                         nb_cum_send_left < nb_ghosts_right));

                // Get the number of elements that we will receive
                auto nb_recv_left{this->recv_left_sequence[direction][step]};
                auto nb_recv_right{this->recv_right_sequence[direction][step]};

                // Idiot check the cached receive sequence
                assert(nb_recv_left == this->cart_comm->sendrecv_right(
                                           direction, nb_send_right));
                assert(nb_recv_right ==
                       this->cart_comm->sendrecv_left(direction, nb_send_left));

                // Perform send to the RIGHT, receive from the LEFT
                this->cart_comm->sendrecv_right(
                    // send direction, i.e. 0, 1 or 2 (x, y or z)
                    direction,
                    // block stride
                    block_stride,
                    // number of blocks to send
                    nb_blocks,
                    // block length
                    nb_send_right * block_len,
                    // slice to send from
                    nb_ghosts_left + nb_subdomain_grid_pts_without_ghosts -
                        nb_cum_send_right - nb_send_right,
                    // number of blocks to receive
                    nb_blocks,
                    // block length
                    nb_recv_left * block_len,
                    // slice to receive into
                    nb_ghosts_left - nb_cum_recv_left - nb_recv_left,
                    // data buffer
                    data,
                    // stride in send direction
                    block_len,
                    // type information
                    element_size, mpi_type_ptr);

                // Perform send to the LEFT, receive from the RIGHT
                this->cart_comm->sendrecv_left(
                    // send direction, i.e. 0, 1 or 2 (x, y or z)
                    direction,
                    // block stride
                    block_stride,
                    // number of blocks to send
                    nb_blocks,
                    // block length
                    nb_send_left * block_len,
                    // slice to send from
                    nb_ghosts_left + nb_cum_send_left,
                    // number of blocks to receive
                    nb_blocks,
                    // block length
                    nb_recv_right * block_len,
                    // slice to receive into
                    nb_ghosts_left + nb_subdomain_grid_pts_without_ghosts +
                        nb_cum_recv_right,
                    // data buffer
                    data,
                    // stride in send direction
                    block_len,
                    // type information
                    element_size, mpi_type_ptr);

                // Update how many blocks we have already sent to the right/left
                nb_cum_send_right += nb_send_right;
                nb_cum_send_left += nb_send_left;
                nb_cum_recv_right += nb_recv_right;
                nb_cum_recv_left += nb_recv_left;

                // Determine how much additional data we can now send
                nb_send_right =
                    std::min(nb_ghosts_left - nb_cum_send_right, nb_recv_left);
                nb_send_left =
                    std::min(nb_ghosts_right - nb_cum_send_left, nb_recv_right);
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

#include <cassert>
#include <cstring>
#include <iterator>
#include <vector>

#ifdef WITH_MPI
#include <mpi.h>
#endif

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
#include "memory/gpu_runtime.hh"
#endif

#include "core/coordinates.hh"
#include "core/enums.hh"
#include "core/type_descriptor.hh"
#include "field/field.hh"
#include "collection/field_collection_global.hh"
#include "mpi/cartesian_communicator.hh"
#include "mpi/cartesian_decomposition.hh"

namespace muGrid {
    CartesianDecomposition::CartesianDecomposition(
        const Communicator & comm, Dim_t spatial_dimension,
        const SubPtMap_t & nb_sub_pts, Device device)
        : Parent_t{}, comm{comm},
          collection(spatial_dimension, nb_sub_pts,
                     StorageOrder::ArrayOfStructures, device) {}

    CartesianDecomposition::CartesianDecomposition(
        const Communicator & comm, const DynGridIndex & nb_domain_grid_pts,
        const DynGridIndex & nb_subdivisions, const DynGridIndex & nb_ghosts_left,
        const DynGridIndex & nb_ghosts_right, const SubPtMap_t & nb_sub_pts,
        Device device)
        : Parent_t{}, comm{comm},
          collection(nb_domain_grid_pts.size(), nb_sub_pts,
                     StorageOrder::ArrayOfStructures, device) {
        this->initialise(nb_domain_grid_pts, nb_subdivisions, nb_ghosts_left,
                         nb_ghosts_right);
    }

    void
    CartesianDecomposition::check_dimension(const DynGridIndex & n,
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
        const DynGridIndex & nb_domain_grid_pts,
        const DynGridIndex & nb_subdivisions,
        const DynGridIndex & nb_subdomain_grid_pts_without_ghosts,
        const DynGridIndex & subdomain_locations_without_ghosts,
        const DynGridIndex & nb_ghosts_left, const DynGridIndex & nb_ghosts_right,
        const DynGridIndex & subdomain_strides) {
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
        this->send_right_sequence.resize(this->get_spatial_dim());
        this->send_left_sequence.resize(this->get_spatial_dim());
        this->nb_sendrecv_steps.resize(this->get_spatial_dim());
        for (Dim_t direction{0}; direction < this->get_spatial_dim();
             ++direction) {
            // Compute the sequence of sendrecv events required to fill the
            // ghost buffer
            this->recv_right_sequence[direction].resize(0);
            this->recv_left_sequence[direction].resize(0);
            this->send_right_sequence[direction].resize(0);
            this->send_left_sequence[direction].resize(0);
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
                this->send_right_sequence[direction].push_back(nb_send_right);
                this->send_left_sequence[direction].push_back(nb_send_left);

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
    CartesianDecomposition::initialise(const DynGridIndex & nb_domain_grid_pts,
                                       const DynGridIndex & nb_subdivisions,
                                       const DynGridIndex & nb_ghosts_left,
                                       const DynGridIndex & nb_ghosts_right) {
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
        for (Dim_t dim{0}; dim < spatial_dims; ++dim) {
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

    void CartesianDecomposition::check_field_is_of_this_collection(
        const Field & field, const std::string & operation) const {
        if (&field.get_collection() != &this->collection) {
            std::stringstream s;
            s << "Field `" << field.get_name() << "` does not belong to this "
              << "decomposition's field collection. " << operation
              << " is a real-space operation on the decomposed grid and is "
                 "only defined for fields of this decomposition; applied to "
                 "a field of another collection (e.g. a Fourier-space field "
                 "of an FFT engine) it would scramble interior data.";
            throw RuntimeError(s.str());
        }
    }

    void CartesianDecomposition::communicate_ghosts(const Field & field) const {
        this->check_field_is_of_this_collection(field, "communicate_ghosts");

        // Get spatial dimensions
        auto spatial_dims{this->get_spatial_dim()};

        // Get strides (in unit: elements)
        auto strides{field.get_strides(IterUnit::SubPt)};

        // Total number of elements in the field.
        // For SoA (Structure of Arrays) layout on GPU, components are stored
        // separately, so we need to use get_buffer_size() to get the true
        // total, not just spatial elements × last_stride.
        auto nb_total_elements{static_cast<Index_t>(field.get_buffer_size())};

        // Get the begin address of the field data (cast into char * for pointer
        // arithmetics). Pass false to allow device pointers for CUDA-aware MPI.
        auto * data{static_cast<char *>(field.get_void_data_ptr(false))};

        // Check if field is on device memory (needed for serial ghost comm)
        bool is_device_memory{field.is_on_device()};

        // Get element size (only useful for pointer arithmetic in finding the
        // correct offset)
        auto element_size{
            static_cast<Index_t>(field.get_element_size_in_bytes())};

        // Get type descriptor for communication
        TypeDescriptor type_desc{field.get_type_descriptor()};

        // FIXME! The code below assumes a specific form of data layout,
        // essentially column-major but with potentially varying slides
        // --- i.e. the first axis needs to be fastest and then it becomes
        // slower in order of the axes. It also assumes an array of structures
        // layout. We should either generalize or introduce a guard that fails
        // if the data layout is wrong.

        // For each direction...
        for (Dim_t direction{0}; direction < spatial_dims; ++direction) {
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

            // Detect SoA (Structure of Arrays) layout vs AoS (Array of
            // Structures). In SoA, spatial indices are fastest (stride = 1),
            // components are slowest. In AoS, components are fastest
            // (stride = 1), spatial indices are slower.
            auto first_spatial_stride{strides[strides.size() - spatial_dims]};
            bool is_soa{strides[0] > first_spatial_stride};

            // Block stride: for non-last directions, use the next spatial
            // stride. For the last direction:
            // - AoS: use nb_total_elements (one big block)
            // - SoA: use the stride of the last non-spatial dimension (just
            //   before the spatial dimensions). This ensures we get the right
            //   number of blocks to cover all component/sub_pt combinations.
            Index_t last_non_spatial_stride{
                strides.size() > static_cast<size_t>(spatial_dims)
                    ? strides[strides.size() - spatial_dims - 1]
                    : nb_total_elements};
            auto block_stride{
                direction < spatial_dims - 1
                    ? strides[strides.size() - spatial_dims + direction + 1]
                    : (is_soa ? last_non_spatial_stride : nb_total_elements)};

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
                // NB: do not validate the cached sequence here with collective
                // calls (any()/sendrecv) inside assert(): those vanish under
                // NDEBUG, so a debug/release build mix across ranks would
                // deadlock (and debug builds would do redundant communication).
                // The sequence was already validated when it was built.

                // Get the number of elements that we will receive
                auto nb_recv_left{this->recv_left_sequence[direction][step]};
                auto nb_recv_right{this->recv_right_sequence[direction][step]};

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
                    element_size, type_desc,
                    // device memory flag for serial GPU memory copy
                    is_device_memory);

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
                    element_size, type_desc,
                    // device memory flag for serial GPU memory copy
                    is_device_memory);

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

    void CartesianDecomposition::reduce_ghosts(const Field & field) const {
        this->check_field_is_of_this_collection(field, "reduce_ghosts");

        // Get spatial dimensions
        auto spatial_dims{this->get_spatial_dim()};

        // Get strides (in unit: elements)
        auto strides{field.get_strides(IterUnit::SubPt)};

        // Total number of elements in the field.
        // For SoA (Structure of Arrays) layout on GPU, components are stored
        // separately, so we need to use get_buffer_size() to get the true
        // total, not just spatial elements × last_stride.
        auto nb_total_elements{static_cast<Index_t>(field.get_buffer_size())};

        // Get the begin address of the field data
        auto * data{static_cast<char *>(field.get_void_data_ptr(false))};

        // Check if field is on device memory
        bool is_device_memory{field.is_on_device()};

        // Get element size
        auto element_size{
            static_cast<Index_t>(field.get_element_size_in_bytes())};

        // Get type descriptor for communication
        TypeDescriptor type_desc{field.get_type_descriptor()};

        // For each direction (in reverse order to handle corners correctly)
        for (Dim_t direction{static_cast<int>(spatial_dims) - 1}; direction >= 0;
             --direction) {
            // Grid size
            auto nb_subdomain_grid_pts_without_ghosts{
                this->get_nb_subdomain_grid_pts_without_ghosts()[direction]};

            // Calculate memory layout
            auto block_len{strides[strides.size() - spatial_dims + direction]};

            // Detect SoA (Structure of Arrays) layout vs AoS (Array of
            // Structures). In SoA, spatial indices are fastest (stride = 1),
            // components are slowest. In AoS, components are fastest
            // (stride = 1), spatial indices are slower.
            auto first_spatial_stride{strides[strides.size() - spatial_dims]};
            bool is_soa{strides[0] > first_spatial_stride};

            // Block stride: for non-last directions, use the next spatial
            // stride. For the last direction:
            // - AoS: use nb_total_elements (one big block)
            // - SoA: use the stride of the last non-spatial dimension (just
            //   before the spatial dimensions). This ensures we get the right
            //   number of blocks to cover all component/sub_pt combinations.
            Index_t last_non_spatial_stride{
                strides.size() > static_cast<size_t>(spatial_dims)
                    ? strides[strides.size() - spatial_dims - 1]
                    : nb_total_elements};
            auto block_stride{
                direction < spatial_dims - 1
                    ? strides[strides.size() - spatial_dims + direction + 1]
                    : (is_soa ? last_non_spatial_stride : nb_total_elements)};

            // Number of blocks for single slice
            auto nb_blocks{nb_total_elements / block_stride};

            // Ghost counts
            auto nb_ghosts_right{this->get_nb_ghosts_right()[direction]};
            auto nb_ghosts_left{this->get_nb_ghosts_left()[direction]};

            // reduce_ghosts is the transpose of communicate_ghosts. Every
            // step of the forward cascade is a copy "ghost slices B :=
            // neighbour slices A"; its transpose is "A += B, then B := 0".
            // The transpose of the whole cascade is therefore the sequence
            // of transposed steps in REVERSE order, with the roles of
            // sender and receiver swapped: what a rank received into its
            // ghost in forward step k is sent back to the rank it came from,
            // which adds it into the slices it sent in that step. For relay
            // ranks those slices are themselves ghost slices (received in
            // step k-1); they accumulate the downstream contributions on
            // top of their own and are forwarded in the next (earlier)
            // step, so contributions travel back along the exact path the
            // data took forwards, however many ranks that path spans.
            //
            // Forward step k reads/writes at slice offsets that depend on
            // the cumulative counts of the preceding steps; rebuild those
            // prefix sums from the schedule recorded in initialise().
            const auto nb_steps{this->nb_sendrecv_steps[direction]};
            const auto & send_right{this->send_right_sequence[direction]};
            const auto & send_left{this->send_left_sequence[direction]};
            const auto & recv_left{this->recv_left_sequence[direction]};
            const auto & recv_right{this->recv_right_sequence[direction]};
            std::vector<Index_t> cum_send_right(nb_steps + 1, 0),
                cum_send_left(nb_steps + 1, 0), cum_recv_left(nb_steps + 1, 0),
                cum_recv_right(nb_steps + 1, 0);
            for (Index_t step{0}; step < nb_steps; ++step) {
                cum_send_right[step + 1] =
                    cum_send_right[step] + send_right[step];
                cum_send_left[step + 1] =
                    cum_send_left[step] + send_left[step];
                cum_recv_left[step + 1] =
                    cum_recv_left[step] + recv_left[step];
                cum_recv_right[step + 1] =
                    cum_recv_right[step] + recv_right[step];
            }

            for (Index_t step{nb_steps - 1}; step >= 0; --step) {
                // Forward step `step`, send to the RIGHT / receive from the
                // LEFT: this rank sent `send_right[step]` slices starting at
                // `src_right` and received `recv_left[step]` slices into
                // its left ghost starting at `dst_left`.
                auto src_right{nb_ghosts_left +
                               nb_subdomain_grid_pts_without_ghosts -
                               cum_send_right[step] - send_right[step]};
                auto dst_left{nb_ghosts_left - cum_recv_left[step] -
                              recv_left[step]};

                // Forward step `step`, send to the LEFT / receive from the
                // RIGHT: sent `send_left[step]` slices from `src_left`,
                // received `recv_right[step]` slices into `dst_right`.
                auto src_left{nb_ghosts_left + cum_send_left[step]};
                auto dst_right{nb_ghosts_left +
                               nb_subdomain_grid_pts_without_ghosts +
                               cum_recv_right[step]};

                // Transposed step: return the left-ghost slices to the LEFT
                // neighbour and add what the RIGHT neighbour returns into
                // the slices we had sent it. The guard is on the (global)
                // ghost count only: a rank-local skip would leave the
                // neighbour's matching zero-length sendrecv unanswered.
                if (nb_ghosts_left > 0) {
                    this->cart_comm->sendrecv_left_accumulate(
                        direction, block_stride, nb_blocks,
                        // send: slices received in the forward step
                        recv_left[step] * block_len, dst_left,
                        // receive and accumulate: slices sent in the
                        // forward step
                        nb_blocks, send_right[step] * block_len, src_right,
                        data, block_len, element_size, type_desc,
                        is_device_memory);
                }

                // Mirror image for the right ghost.
                if (nb_ghosts_right > 0) {
                    this->cart_comm->sendrecv_right_accumulate(
                        direction, block_stride, nb_blocks,
                        recv_right[step] * block_len, dst_right, nb_blocks,
                        send_left[step] * block_len, src_left, data,
                        block_len, element_size, type_desc, is_device_memory);
                }
            }

            // Zero out the ghost buffers after reduction. Every ghost slice
            // of this direction has been sent back exactly once above, and
            // no transposed step accumulates into a ghost slice after it
            // has been sent back, so the whole halo can be cleared in one
            // go rather than per step. Within a block the
            // ghost slices are contiguous (slice s at s * block_len), so each
            // side's ghost region is one contiguous run of
            // nb_ghosts * block_len elements per block, repeated nb_blocks
            // times with stride block_stride -- i.e. a single 2D memset
            // (device) instead of one call per (block, slice), which on the
            // device is otherwise thousands of tiny cudaMemset calls.
            auto right_ghost_start{nb_ghosts_left +
                                   nb_subdomain_grid_pts_without_ghosts};
            auto zero_region{[&](Index_t first_slice, Index_t nb_slices) {
                if (nb_slices == 0 || nb_blocks == 0) {
                    return;
                }
                Index_t base_offset{first_slice * block_len};
                std::size_t width{static_cast<std::size_t>(nb_slices) *
                                  block_len * element_size};
                std::size_t pitch{static_cast<std::size_t>(block_stride) *
                                  element_size};
                char * region{data + base_offset * element_size};
                if (is_device_memory) {
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
                    GPU_MEMSET_2D(region, pitch, 0, width,
                                  static_cast<std::size_t>(nb_blocks));
#endif
                } else {
                    for (Index_t block{0}; block < nb_blocks; ++block) {
                        std::memset(region + block * pitch, 0, width);
                    }
                }
            }};
            zero_region(0, nb_ghosts_left);                 // left ghost region
            zero_region(right_ghost_start, nb_ghosts_right);  // right ghost
        }
    }

    void CartesianDecomposition::reduce_ghosts(
        const std::string & field_name) const {
        this->reduce_ghosts(this->collection.get_field(field_name));
    }

    GlobalFieldCollection & CartesianDecomposition::get_collection() {
        return this->collection;
    }

    const GlobalFieldCollection &
    CartesianDecomposition::get_collection() const {
        return this->collection;
    }

    Dim_t CartesianDecomposition::get_spatial_dim() const {
        return this->collection.get_spatial_dim();
    }

    const DynGridIndex & CartesianDecomposition::get_nb_subdivisions() const {
        return this->cart_comm->get_nb_subdivisions();
    }

    const DynGridIndex & CartesianDecomposition::get_nb_domain_grid_pts() const {
        return this->collection.get_nb_domain_grid_pts();
    }

    const DynGridIndex &
    CartesianDecomposition::get_nb_subdomain_grid_pts_with_ghosts() const {
        return this->collection.get_nb_subdomain_grid_pts_with_ghosts();
    }

    DynGridIndex
    CartesianDecomposition::get_nb_subdomain_grid_pts_without_ghosts() const {
        return this->collection.get_nb_subdomain_grid_pts_without_ghosts();
    }

    const DynGridIndex &
    CartesianDecomposition::get_subdomain_locations_with_ghosts() const {
        return this->collection.get_subdomain_locations_with_ghosts();
    }

    DynGridIndex
    CartesianDecomposition::get_subdomain_locations_without_ghosts() const {
        return this->collection.get_subdomain_locations_without_ghosts();
    }
}  // namespace muGrid

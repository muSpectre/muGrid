#ifdef WITH_MPI
#include <mpi.h>
#endif

#include "core/coordinates.hh"
#include "core/exception.hh"
#include "core/types.hh"
#include "mpi/communicator.hh"

#include <cstring>
#include <numeric>
#include <vector>
#include "mpi/cartesian_communicator.hh"

#if defined(MUGRID_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

#if defined(MUGRID_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace muGrid {
#ifdef WITH_MPI
    CartesianCommunicator::CartesianCommunicator(
        const Parent_t & parent, const DynGridIndex & nb_subdivisions)
        : Parent_t{parent.get_mpi_comm()}, parent{parent},
          nb_subdivisions{nb_subdivisions},
          coordinates(nb_subdivisions.size(), 0),
          left_ranks(nb_subdivisions.size(), MPI_PROC_NULL),
          right_ranks(nb_subdivisions.size(), MPI_PROC_NULL) {
        // Check if MPI is initialized - if not, operate in serial mode
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if (!mpi_initialized || this->comm == MPI_COMM_NULL) {
            // Serial mode: single rank with coordinates all zero
            this->comm = MPI_COMM_NULL;
            return;
        }

        // idiot check
        auto nb_total_subdivisions{std::accumulate(nb_subdivisions.begin(),
                                                   nb_subdivisions.end(), 1,
                                                   std::multiplies<Index_t>())};
        if (nb_total_subdivisions != static_cast<Index_t>(this->size())) {
            std::stringstream s;
            s << "The total number of subdivisions (" << nb_total_subdivisions
              << ") does not match the size of the communicator ("
              << this->size() << ").";
            throw RuntimeError(s.str());
        }

        // the spatial dimension of the topology
        const int spatial_dim{static_cast<int>(nb_subdivisions.size())};
        // the domain is periodic in all directions
        std::vector<int> is_periodic(spatial_dim, true);
        // reordering is allowed
        const bool reoder_is_allowed{false};

        // create the new communicator with cartesian topology
        std::vector<int> narr(spatial_dim);
        std::copy(nb_subdivisions.begin(), nb_subdivisions.end(), narr.begin());
        MPI_Cart_create(this->comm, spatial_dim, narr.data(),
                        is_periodic.data(), reoder_is_allowed, &this->comm);

        // get coordinates of current rank
        MPI_Cart_coords(this->comm, this->rank(), spatial_dim, narr.data());
        std::copy(narr.begin(), narr.end(), this->coordinates.begin());

        // get the ranks of the neighbors
        for (auto direction{0}; direction < spatial_dim; ++direction) {
            MPI_Cart_shift(this->comm, direction, 1,
                           &this->left_ranks[direction],
                           &this->right_ranks[direction]);
        }
    }

    CartesianCommunicator::CartesianCommunicator(
        const Parent_t & parent, const DynGridIndex & nb_subdivisions,
        const DynGridIndex & coordinates, const std::vector<int> & left_ranks,
        const std::vector<int> & right_ranks)
        : Parent_t{parent.get_mpi_comm()}, parent{parent},
          nb_subdivisions{nb_subdivisions}, coordinates{coordinates},
          left_ranks{left_ranks}, right_ranks{right_ranks} {}

    CartesianCommunicator &
    CartesianCommunicator::operator=(const CartesianCommunicator & other) {
        this->comm = other.comm;
        return *this;
    }

    void CartesianCommunicator::sendrecv_right(
        int direction, int block_stride, int nb_send_blocks, int send_block_len,
        Index_t send_offset, int nb_recv_blocks, int recv_block_len,
        Index_t recv_offset, char * data, int stride_in_direction,
        int elem_size_in_bytes, void * elem_mpi_t,
        bool is_device_memory [[maybe_unused]]) const {
        // Note: is_device_memory is not used in MPI mode - CUDA-aware MPI
        // handles device pointers directly.
        // Cast void pointer to MPI_Datatype for MPI implementation
        MPI_Datatype mpi_datatype{*static_cast<MPI_Datatype *>(elem_mpi_t)};
        MPI_Datatype send_buffer_mpi_t, recv_buffer_mpi_t;
        MPI_Type_vector(nb_send_blocks, send_block_len, block_stride,
                        mpi_datatype, &send_buffer_mpi_t);
        MPI_Type_commit(&send_buffer_mpi_t);
        MPI_Type_vector(nb_recv_blocks, recv_block_len, block_stride,
                        mpi_datatype, &recv_buffer_mpi_t);
        MPI_Type_commit(&recv_buffer_mpi_t);
        auto recv_addr{static_cast<void *>(
            data + recv_offset * stride_in_direction * elem_size_in_bytes)};
        auto send_addr{static_cast<void *>(
            data + send_offset * stride_in_direction * elem_size_in_bytes)};

        MPI_Status status;
        MPI_Sendrecv(send_addr, 1, send_buffer_mpi_t,
                     this->right_ranks[direction], 0, recv_addr, 1,
                     recv_buffer_mpi_t, this->left_ranks[direction], 0,
                     this->comm, &status);
        MPI_Type_free(&send_buffer_mpi_t);
        MPI_Type_free(&recv_buffer_mpi_t);
    }

    void CartesianCommunicator::sendrecv_left(
        int direction, int block_stride, int nb_send_blocks, int send_block_len,
        Index_t send_offset, int nb_recv_blocks, int recv_block_len,
        Index_t recv_offset, char * data, int stride_in_direction,
        int elem_size_in_bytes, void * elem_mpi_t,
        bool is_device_memory [[maybe_unused]]) const {
        // Note: is_device_memory is not used in MPI mode - CUDA-aware MPI
        // handles device pointers directly.
        // Cast void pointer to MPI_Datatype for MPI implementation
        MPI_Datatype mpi_datatype{*static_cast<MPI_Datatype *>(elem_mpi_t)};
        MPI_Datatype send_buffer_mpi_t, recv_buffer_mpi_t;
        MPI_Type_vector(nb_send_blocks, send_block_len, block_stride,
                        mpi_datatype, &send_buffer_mpi_t);
        MPI_Type_commit(&send_buffer_mpi_t);
        MPI_Type_vector(nb_recv_blocks, recv_block_len, block_stride,
                        mpi_datatype, &recv_buffer_mpi_t);
        MPI_Type_commit(&recv_buffer_mpi_t);
        auto recv_addr{static_cast<void *>(
            data + recv_offset * stride_in_direction * elem_size_in_bytes)};
        auto send_addr{static_cast<void *>(
            data + send_offset * stride_in_direction * elem_size_in_bytes)};

        MPI_Status status;
        MPI_Sendrecv(send_addr, 1, send_buffer_mpi_t,
                     this->left_ranks[direction], 0, recv_addr, 1,
                     recv_buffer_mpi_t, this->right_ranks[direction], 0,
                     this->comm, &status);
        MPI_Type_free(&send_buffer_mpi_t);
        MPI_Type_free(&recv_buffer_mpi_t);
    }

    void CartesianCommunicator::sendrecv_right_accumulate(
        int direction, int block_stride, int nb_send_blocks, int send_block_len,
        Index_t send_offset, int nb_recv_blocks, int recv_block_len,
        Index_t recv_offset, char * data, int stride_in_direction,
        int elem_size_in_bytes, void * elem_mpi_t,
        bool is_device_memory [[maybe_unused]]) const {
        // Cast void pointer to MPI_Datatype for MPI implementation
        MPI_Datatype mpi_datatype{*static_cast<MPI_Datatype *>(elem_mpi_t)};

        // Create send type
        MPI_Datatype send_buffer_mpi_t;
        MPI_Type_vector(nb_send_blocks, send_block_len, block_stride,
                        mpi_datatype, &send_buffer_mpi_t);
        MPI_Type_commit(&send_buffer_mpi_t);

        // Calculate total receive size for temporary buffer
        auto total_recv_elems{static_cast<size_t>(nb_recv_blocks) *
                              static_cast<size_t>(recv_block_len)};

        // Allocate contiguous temporary buffer for receiving
        std::vector<char> recv_buffer(total_recv_elems * elem_size_in_bytes);

        auto send_addr{static_cast<void *>(
            data + send_offset * stride_in_direction * elem_size_in_bytes)};

        MPI_Status status;
        MPI_Sendrecv(send_addr, 1, send_buffer_mpi_t,
                     this->right_ranks[direction], 0,
                     recv_buffer.data(), total_recv_elems, mpi_datatype,
                     this->left_ranks[direction], 0,
                     this->comm, &status);

        MPI_Type_free(&send_buffer_mpi_t);

        // Accumulate received data into destination
        // The receive buffer is contiguous; we need to scatter to strided destination
        auto * recv_ptr{recv_buffer.data()};
        for (int block{0}; block < nb_recv_blocks; ++block) {
            auto dest_addr{data + (recv_offset * stride_in_direction +
                                   block * block_stride) * elem_size_in_bytes};
            // Add element by element (assuming Real = double)
            auto * dest{reinterpret_cast<Real *>(dest_addr)};
            auto * src{reinterpret_cast<Real *>(recv_ptr)};
            for (int i{0}; i < recv_block_len; ++i) {
                dest[i] += src[i];
            }
            recv_ptr += recv_block_len * elem_size_in_bytes;
        }
    }

    void CartesianCommunicator::sendrecv_left_accumulate(
        int direction, int block_stride, int nb_send_blocks, int send_block_len,
        Index_t send_offset, int nb_recv_blocks, int recv_block_len,
        Index_t recv_offset, char * data, int stride_in_direction,
        int elem_size_in_bytes, void * elem_mpi_t,
        bool is_device_memory [[maybe_unused]]) const {
        // Cast void pointer to MPI_Datatype for MPI implementation
        MPI_Datatype mpi_datatype{*static_cast<MPI_Datatype *>(elem_mpi_t)};

        // Create send type
        MPI_Datatype send_buffer_mpi_t;
        MPI_Type_vector(nb_send_blocks, send_block_len, block_stride,
                        mpi_datatype, &send_buffer_mpi_t);
        MPI_Type_commit(&send_buffer_mpi_t);

        // Calculate total receive size for temporary buffer
        auto total_recv_elems{static_cast<size_t>(nb_recv_blocks) *
                              static_cast<size_t>(recv_block_len)};

        // Allocate contiguous temporary buffer for receiving
        std::vector<char> recv_buffer(total_recv_elems * elem_size_in_bytes);

        auto send_addr{static_cast<void *>(
            data + send_offset * stride_in_direction * elem_size_in_bytes)};

        MPI_Status status;
        MPI_Sendrecv(send_addr, 1, send_buffer_mpi_t,
                     this->left_ranks[direction], 0,
                     recv_buffer.data(), total_recv_elems, mpi_datatype,
                     this->right_ranks[direction], 0,
                     this->comm, &status);

        MPI_Type_free(&send_buffer_mpi_t);

        // Accumulate received data into destination
        auto * recv_ptr{recv_buffer.data()};
        for (int block{0}; block < nb_recv_blocks; ++block) {
            auto dest_addr{data + (recv_offset * stride_in_direction +
                                   block * block_stride) * elem_size_in_bytes};
            // Add element by element (assuming Real = double)
            auto * dest{reinterpret_cast<Real *>(dest_addr)};
            auto * src{reinterpret_cast<Real *>(recv_ptr)};
            for (int i{0}; i < recv_block_len; ++i) {
                dest[i] += src[i];
            }
            recv_ptr += recv_block_len * elem_size_in_bytes;
        }
    }
#else   // not WITH_MPI
    CartesianCommunicator::CartesianCommunicator(
        const Parent_t & parent, const DynGridIndex & nb_subdivisions)
        : Parent_t{}, parent{parent}, nb_subdivisions{nb_subdivisions},
          coordinates(nb_subdivisions.size(), 0) {}

    namespace {
        /**
         * @brief Copy memory block using appropriate method for memory space.
         *
         * Uses CUDA/HIP runtime for device memory copies. This is necessary
         * because std::memcpy cannot access GPU device memory.
         *
         * @param dst Destination address
         * @param src Source address
         * @param size_in_bytes Number of bytes to copy
         * @param is_device_memory If true, use GPU device copy
         */
        void device_memcpy(void * dst, const void * src, size_t size_in_bytes,
                           bool is_device_memory) {
            if (is_device_memory) {
#if defined(MUGRID_ENABLE_CUDA)
                (void)cudaMemcpy(dst, src, size_in_bytes, cudaMemcpyDeviceToDevice);
#elif defined(MUGRID_ENABLE_HIP)
                (void)hipMemcpy(dst, src, size_in_bytes, hipMemcpyDeviceToDevice);
#else
                // Fallback to memcpy if no GPU backend (shouldn't happen if
                // is_device_memory is true, but handle gracefully)
                std::memcpy(dst, src, size_in_bytes);
#endif
            } else {
                std::memcpy(dst, src, size_in_bytes);
            }
        }
    }  // anonymous namespace

    void CartesianCommunicator::sendrecv_right(
        int direction, int block_stride, int nb_send_blocks, int send_block_len,
        Index_t send_offset, int nb_recv_blocks, int recv_block_len,
        Index_t recv_offset, char * data, int stride_in_direction,
        int elem_size_in_bytes, void * elem_mpi_t,
        bool is_device_memory) const {
        // Note: elem_mpi_t and direction are not used in serial mode
        (void)elem_mpi_t;
        (void)direction;
        if (nb_send_blocks != nb_recv_blocks) {
            throw std::runtime_error("nb_send_blocks != nb_recv_blocks");
        }
        if (send_block_len != recv_block_len) {
            throw std::runtime_error("send_block_len != recv_block_len");
        }
        for (int count{0}; count < nb_send_blocks; ++count) {
            auto recv_addr{static_cast<void *>(
                data + recv_offset * stride_in_direction * elem_size_in_bytes)};
            auto send_addr{static_cast<void *>(
                data + send_offset * stride_in_direction * elem_size_in_bytes)};
            device_memcpy(recv_addr, send_addr,
                          send_block_len * elem_size_in_bytes, is_device_memory);
            data += block_stride * elem_size_in_bytes;
        }
    }

    void CartesianCommunicator::sendrecv_left(
        int direction, int block_stride, int nb_send_blocks, int send_block_len,
        Index_t send_offset, int nb_recv_blocks, int recv_block_len,
        Index_t recv_offset, char * data, int stride_in_direction,
        int elem_size_in_bytes, void * elem_mpi_t,
        bool is_device_memory) const {
        // Note: elem_mpi_t and direction are not used in serial mode
        (void)elem_mpi_t;
        (void)direction;
        if (nb_send_blocks != nb_recv_blocks) {
            throw std::runtime_error("nb_send_blocks != nb_recv_blocks");
        }
        if (send_block_len != recv_block_len) {
            throw std::runtime_error("send_block_len != recv_block_len");
        }
        for (int count{0}; count < nb_send_blocks; ++count) {
            auto recv_addr{static_cast<void *>(
                data + recv_offset * stride_in_direction * elem_size_in_bytes)};
            auto send_addr{static_cast<void *>(
                data + send_offset * stride_in_direction * elem_size_in_bytes)};
            device_memcpy(recv_addr, send_addr,
                          send_block_len * elem_size_in_bytes, is_device_memory);
            data += block_stride * elem_size_in_bytes;
        }
    }

    namespace {
        /**
         * @brief Accumulate (add) memory block using appropriate method for
         * memory space.
         *
         * For host memory, adds source values to destination.
         * For device memory, uses GPU kernels.
         *
         * @param dst Destination address (values are added here)
         * @param src Source address
         * @param nb_elements Number of Real elements to accumulate
         * @param is_device_memory If true, use GPU device accumulation
         */
        void device_accumulate(Real * dst, const Real * src, size_t nb_elements,
                               bool is_device_memory) {
            if (is_device_memory) {
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
                // For device memory, copy to host, accumulate, copy back
                // This is not optimal but works for correctness.
                // A proper implementation would use a GPU kernel.
                std::vector<Real> host_dst(nb_elements);
                std::vector<Real> host_src(nb_elements);
#if defined(MUGRID_ENABLE_CUDA)
                cudaMemcpy(host_dst.data(), dst, nb_elements * sizeof(Real),
                           cudaMemcpyDeviceToHost);
                cudaMemcpy(host_src.data(), src, nb_elements * sizeof(Real),
                           cudaMemcpyDeviceToHost);
#elif defined(MUGRID_ENABLE_HIP)
                hipMemcpy(host_dst.data(), dst, nb_elements * sizeof(Real),
                          hipMemcpyDeviceToHost);
                hipMemcpy(host_src.data(), src, nb_elements * sizeof(Real),
                          hipMemcpyDeviceToHost);
#endif
                for (size_t i{0}; i < nb_elements; ++i) {
                    host_dst[i] += host_src[i];
                }
#if defined(MUGRID_ENABLE_CUDA)
                cudaMemcpy(dst, host_dst.data(), nb_elements * sizeof(Real),
                           cudaMemcpyHostToDevice);
#elif defined(MUGRID_ENABLE_HIP)
                hipMemcpy(dst, host_dst.data(), nb_elements * sizeof(Real),
                          hipMemcpyHostToDevice);
#endif
#else
                // Fallback: should not happen
                for (size_t i{0}; i < nb_elements; ++i) {
                    dst[i] += src[i];
                }
#endif
            } else {
                for (size_t i{0}; i < nb_elements; ++i) {
                    dst[i] += src[i];
                }
            }
        }
    }  // anonymous namespace

    void CartesianCommunicator::sendrecv_right_accumulate(
        int direction, int block_stride, int nb_send_blocks, int send_block_len,
        Index_t send_offset, int nb_recv_blocks, int recv_block_len,
        Index_t recv_offset, char * data, int stride_in_direction,
        int elem_size_in_bytes, void * elem_mpi_t,
        bool is_device_memory) const {
        // Note: elem_mpi_t and direction are not used in serial mode
        (void)elem_mpi_t;
        (void)direction;
        if (nb_send_blocks != nb_recv_blocks) {
            throw std::runtime_error("nb_send_blocks != nb_recv_blocks");
        }
        if (send_block_len != recv_block_len) {
            throw std::runtime_error("send_block_len != recv_block_len");
        }
        char * base_data{data};
        for (int count{0}; count < nb_send_blocks; ++count) {
            auto recv_addr{reinterpret_cast<Real *>(
                base_data + recv_offset * stride_in_direction * elem_size_in_bytes)};
            auto send_addr{reinterpret_cast<Real *>(
                base_data + send_offset * stride_in_direction * elem_size_in_bytes)};
            device_accumulate(recv_addr, send_addr, send_block_len,
                              is_device_memory);
            base_data += block_stride * elem_size_in_bytes;
        }
    }

    void CartesianCommunicator::sendrecv_left_accumulate(
        int direction, int block_stride, int nb_send_blocks, int send_block_len,
        Index_t send_offset, int nb_recv_blocks, int recv_block_len,
        Index_t recv_offset, char * data, int stride_in_direction,
        int elem_size_in_bytes, void * elem_mpi_t,
        bool is_device_memory) const {
        // Note: elem_mpi_t and direction are not used in serial mode
        (void)elem_mpi_t;
        (void)direction;
        if (nb_send_blocks != nb_recv_blocks) {
            throw std::runtime_error("nb_send_blocks != nb_recv_blocks");
        }
        if (send_block_len != recv_block_len) {
            throw std::runtime_error("send_block_len != recv_block_len");
        }
        char * base_data{data};
        for (int count{0}; count < nb_send_blocks; ++count) {
            auto recv_addr{reinterpret_cast<Real *>(
                base_data + recv_offset * stride_in_direction * elem_size_in_bytes)};
            auto send_addr{reinterpret_cast<Real *>(
                base_data + send_offset * stride_in_direction * elem_size_in_bytes)};
            device_accumulate(recv_addr, send_addr, send_block_len,
                              is_device_memory);
            base_data += block_stride * elem_size_in_bytes;
        }
    }
#endif  // WITH_MPI

    const DynGridIndex & CartesianCommunicator::get_nb_subdivisions() const {
        return this->nb_subdivisions;
    }

    const DynGridIndex & CartesianCommunicator::get_coordinates() const {
        return this->coordinates;
    }
}  // namespace muGrid

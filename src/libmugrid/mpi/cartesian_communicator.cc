#ifdef WITH_MPI
#include <mpi.h>
#include "mpi/type_descriptor_mpi.hh"
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

    namespace {
        /**
         * @brief Copy strided memory blocks using appropriate method.
         *
         * Copies nb_blocks blocks of data, each of size block_len bytes,
         * with blocks separated by pitch bytes in both source and destination.
         *
         * For GPU memory, uses hipMemcpy2D/cudaMemcpy2D for efficiency.
         * For host memory, falls back to individual memcpy calls.
         *
         * @param dst Destination address
         * @param src Source address
         * @param block_len Size of each block in bytes (width)
         * @param nb_blocks Number of blocks to copy (height)
         * @param pitch Stride between blocks in bytes
         * @param is_device_memory If true, use GPU device copy
         */
        void device_memcpy_strided(void * dst, const void * src,
                                   size_t block_len, size_t nb_blocks,
                                   size_t pitch, bool is_device_memory) {
            if (is_device_memory) {
#if defined(MUGRID_ENABLE_CUDA)
                // Use cudaMemcpy2D for efficient strided copy
                // cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind)
                (void)cudaMemcpy2D(dst, pitch, src, pitch, block_len, nb_blocks,
                                   cudaMemcpyDeviceToDevice);
#elif defined(MUGRID_ENABLE_HIP)
                // Use hipMemcpy2D for efficient strided copy
                // hipMemcpy2D(dst, dpitch, src, spitch, width, height, kind)
                (void)hipMemcpy2D(dst, pitch, src, pitch, block_len, nb_blocks,
                                  hipMemcpyDeviceToDevice);
#else
                // Fallback to loop if no GPU backend
                auto * dst_ptr = static_cast<char *>(dst);
                auto * src_ptr = static_cast<const char *>(src);
                for (size_t i{0}; i < nb_blocks; ++i) {
                    std::memcpy(dst_ptr, src_ptr, block_len);
                    dst_ptr += pitch;
                    src_ptr += pitch;
                }
#endif
            } else {
                // Host memory: use loop of memcpy
                auto * dst_ptr = static_cast<char *>(dst);
                auto * src_ptr = static_cast<const char *>(src);
                for (size_t i{0}; i < nb_blocks; ++i) {
                    std::memcpy(dst_ptr, src_ptr, block_len);
                    dst_ptr += pitch;
                    src_ptr += pitch;
                }
            }
        }

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
        template <typename T>
        void device_accumulate_typed(T * dst, const T * src, size_t nb_elements,
                                     bool is_device_memory) {
            if (is_device_memory) {
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
                // For device memory, copy to host, accumulate, copy back
                // This is not optimal but works for correctness.
                // A proper implementation would use a GPU kernel.
                std::vector<T> host_dst(nb_elements);
                std::vector<T> host_src(nb_elements);
#if defined(MUGRID_ENABLE_CUDA)
                cudaMemcpy(host_dst.data(), dst, nb_elements * sizeof(T),
                           cudaMemcpyDeviceToHost);
                cudaMemcpy(host_src.data(), src, nb_elements * sizeof(T),
                           cudaMemcpyDeviceToHost);
#elif defined(MUGRID_ENABLE_HIP)
                (void)hipMemcpy(host_dst.data(), dst, nb_elements * sizeof(T),
                                hipMemcpyDeviceToHost);
                (void)hipMemcpy(host_src.data(), src, nb_elements * sizeof(T),
                                hipMemcpyDeviceToHost);
#endif
                for (size_t i{0}; i < nb_elements; ++i) {
                    host_dst[i] += host_src[i];
                }
#if defined(MUGRID_ENABLE_CUDA)
                cudaMemcpy(dst, host_dst.data(), nb_elements * sizeof(T),
                           cudaMemcpyHostToDevice);
#elif defined(MUGRID_ENABLE_HIP)
                (void)hipMemcpy(dst, host_dst.data(), nb_elements * sizeof(T),
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

        /**
         * @brief Accumulate nb_elements of the given type from src into dst,
         * honouring device memory and the actual element type.
         *
         * dst/src are byte pointers to (possibly device) memory; nb_elements is
         * the number of field elements (not bytes). Dispatching on the type
         * descriptor keeps Complex/Int reductions correct (the previous code
         * hard-coded Real, so e.g. a Complex field only had its real part
         * accumulated, and device pointers were dereferenced on the host).
         */
        void accumulate_elements(char * dst, const char * src,
                                 size_t nb_elements, TypeDescriptor type_desc,
                                 bool is_device_memory) {
            switch (type_desc) {
            case TypeDescriptor::Real:
                device_accumulate_typed(reinterpret_cast<Real *>(dst),
                                        reinterpret_cast<const Real *>(src),
                                        nb_elements, is_device_memory);
                break;
            case TypeDescriptor::Complex:
                device_accumulate_typed(reinterpret_cast<Complex *>(dst),
                                        reinterpret_cast<const Complex *>(src),
                                        nb_elements, is_device_memory);
                break;
            case TypeDescriptor::Int:
                device_accumulate_typed(reinterpret_cast<Int *>(dst),
                                        reinterpret_cast<const Int *>(src),
                                        nb_elements, is_device_memory);
                break;
            case TypeDescriptor::Uint:
                device_accumulate_typed(reinterpret_cast<Uint *>(dst),
                                        reinterpret_cast<const Uint *>(src),
                                        nb_elements, is_device_memory);
                break;
            case TypeDescriptor::Index:
                device_accumulate_typed(reinterpret_cast<Index_t *>(dst),
                                        reinterpret_cast<const Index_t *>(src),
                                        nb_elements, is_device_memory);
                break;
            default:
                throw std::runtime_error(
                    "accumulate_elements: unsupported type descriptor");
            }
        }

        /**
         * @brief Serial implementation of sendrecv (local copy).
         *
         * Used when MPI is not available or not initialized.
         */
        void serial_sendrecv(
            int block_stride, int nb_send_blocks, int send_block_len,
            Index_t send_offset, int nb_recv_blocks, int recv_block_len,
            Index_t recv_offset, char * data, int stride_in_direction,
            int elem_size_in_bytes, bool is_device_memory) {
            if (nb_send_blocks != nb_recv_blocks) {
                throw std::runtime_error(
                    "serial_sendrecv: nb_send_blocks != nb_recv_blocks");
            }
            if (send_block_len != recv_block_len) {
                throw std::runtime_error(
                    "serial_sendrecv: send_block_len != recv_block_len");
            }
            auto * recv_addr{data +
                             recv_offset * stride_in_direction * elem_size_in_bytes};
            auto * send_addr{data +
                             send_offset * stride_in_direction * elem_size_in_bytes};
            auto block_len_bytes{static_cast<size_t>(send_block_len) *
                                 elem_size_in_bytes};
            auto pitch_bytes{static_cast<size_t>(block_stride) * elem_size_in_bytes};
            device_memcpy_strided(recv_addr, send_addr, block_len_bytes,
                                  nb_send_blocks, pitch_bytes, is_device_memory);
        }

        /**
         * @brief Serial implementation of sendrecv with accumulation.
         *
         * Used when MPI is not available or not initialized.
         */
        void serial_sendrecv_accumulate(
            int block_stride, int nb_send_blocks, int send_block_len,
            Index_t send_offset, int nb_recv_blocks, int recv_block_len,
            Index_t recv_offset, char * data, int stride_in_direction,
            int elem_size_in_bytes, TypeDescriptor type_desc,
            bool is_device_memory) {
            if (nb_send_blocks != nb_recv_blocks) {
                throw std::runtime_error(
                    "serial_sendrecv_accumulate: nb_send_blocks != nb_recv_blocks");
            }
            if (send_block_len != recv_block_len) {
                throw std::runtime_error(
                    "serial_sendrecv_accumulate: send_block_len != recv_block_len");
            }
            char * base_data{data};
            for (int count{0}; count < nb_send_blocks; ++count) {
                auto * recv_addr{
                    base_data +
                    recv_offset * stride_in_direction * elem_size_in_bytes};
                auto * send_addr{
                    base_data +
                    send_offset * stride_in_direction * elem_size_in_bytes};
                accumulate_elements(recv_addr, send_addr, send_block_len,
                                    type_desc, is_device_memory);
                base_data += block_stride * elem_size_in_bytes;
            }
        }
    }  // anonymous namespace

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
        // We created this communicator, so we are responsible for freeing it.
        this->owns_comm = true;

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

    CartesianCommunicator::CartesianCommunicator(
        const CartesianCommunicator & other)
        : Parent_t{other.comm}, parent{other.parent},
          nb_subdivisions{other.nb_subdivisions},
          coordinates{other.coordinates}, left_ranks{other.left_ranks},
          right_ranks{other.right_ranks} {
        // The copy shares the handle but does not own it; only the original
        // (owns_comm == true) frees it. owns_comm defaults to false.
    }

    CartesianCommunicator::~CartesianCommunicator() {
        if (this->owns_comm && this->comm != MPI_COMM_NULL) {
            int initialized{0}, finalized{0};
            MPI_Initialized(&initialized);
            MPI_Finalized(&finalized);
            if (initialized && !finalized) {
                MPI_Comm_free(&this->comm);
            }
        }
    }

    CartesianCommunicator &
    CartesianCommunicator::operator=(const CartesianCommunicator & other) {
        if (this == &other) {
            return *this;
        }
        // Release any communicator we currently own before overwriting it.
        if (this->owns_comm && this->comm != MPI_COMM_NULL &&
            this->comm != other.comm) {
            int initialized{0}, finalized{0};
            MPI_Initialized(&initialized);
            MPI_Finalized(&finalized);
            if (initialized && !finalized) {
                MPI_Comm_free(&this->comm);
            }
        }
        this->comm = other.comm;
        // We only share the handle; ownership stays with the original.
        this->owns_comm = false;
        return *this;
    }

    void CartesianCommunicator::sendrecv_right(
        int direction, int block_stride, int nb_send_blocks, int send_block_len,
        Index_t send_offset, int nb_recv_blocks, int recv_block_len,
        Index_t recv_offset, char * data, int stride_in_direction,
        int elem_size_in_bytes, TypeDescriptor type_desc,
        bool is_device_memory) const {
        // Check if MPI is available (comm may be NULL if MPI was not initialized)
        if (this->comm == MPI_COMM_NULL) {
            serial_sendrecv(block_stride, nb_send_blocks, send_block_len,
                            send_offset, nb_recv_blocks, recv_block_len,
                            recv_offset, data, stride_in_direction,
                            elem_size_in_bytes, is_device_memory);
            return;
        }
        // Note: is_device_memory is not used in MPI mode - CUDA-aware MPI
        // handles device pointers directly.
        // Convert TypeDescriptor to MPI_Datatype
        MPI_Datatype mpi_datatype{descriptor_to_mpi_type(type_desc)};
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
        int elem_size_in_bytes, TypeDescriptor type_desc,
        bool is_device_memory) const {
        // Check if MPI is available (comm may be NULL if MPI was not initialized)
        if (this->comm == MPI_COMM_NULL) {
            serial_sendrecv(block_stride, nb_send_blocks, send_block_len,
                            send_offset, nb_recv_blocks, recv_block_len,
                            recv_offset, data, stride_in_direction,
                            elem_size_in_bytes, is_device_memory);
            return;
        }
        // Note: is_device_memory is not used in MPI mode - CUDA-aware MPI
        // handles device pointers directly.
        // Convert TypeDescriptor to MPI_Datatype
        MPI_Datatype mpi_datatype{descriptor_to_mpi_type(type_desc)};
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
        int elem_size_in_bytes, TypeDescriptor type_desc,
        bool is_device_memory) const {
        // Check if MPI is available (comm may be NULL if MPI was not initialized)
        if (this->comm == MPI_COMM_NULL) {
            serial_sendrecv_accumulate(block_stride, nb_send_blocks, send_block_len,
                                       send_offset, nb_recv_blocks, recv_block_len,
                                       recv_offset, data, stride_in_direction,
                                       elem_size_in_bytes, type_desc,
                                       is_device_memory);
            return;
        }
        // Convert TypeDescriptor to MPI_Datatype
        MPI_Datatype mpi_datatype{descriptor_to_mpi_type(type_desc)};

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

        // Accumulate received data into destination. The receive buffer is
        // contiguous (host memory); scatter it to the (possibly device) strided
        // destination using a type- and device-aware accumulation. The previous
        // host-side loop hard-coded Real and dereferenced device pointers.
        auto * recv_ptr{recv_buffer.data()};
        for (int block{0}; block < nb_recv_blocks; ++block) {
            auto * dest_addr{data + (recv_offset * stride_in_direction +
                                     block * block_stride) * elem_size_in_bytes};
            accumulate_elements(dest_addr, recv_ptr,
                                static_cast<size_t>(recv_block_len), type_desc,
                                is_device_memory);
            recv_ptr += recv_block_len * elem_size_in_bytes;
        }
    }

    void CartesianCommunicator::sendrecv_left_accumulate(
        int direction, int block_stride, int nb_send_blocks, int send_block_len,
        Index_t send_offset, int nb_recv_blocks, int recv_block_len,
        Index_t recv_offset, char * data, int stride_in_direction,
        int elem_size_in_bytes, TypeDescriptor type_desc,
        bool is_device_memory) const {
        // Check if MPI is available (comm may be NULL if MPI was not initialized)
        if (this->comm == MPI_COMM_NULL) {
            serial_sendrecv_accumulate(block_stride, nb_send_blocks, send_block_len,
                                       send_offset, nb_recv_blocks, recv_block_len,
                                       recv_offset, data, stride_in_direction,
                                       elem_size_in_bytes, type_desc,
                                       is_device_memory);
            return;
        }
        // Convert TypeDescriptor to MPI_Datatype
        MPI_Datatype mpi_datatype{descriptor_to_mpi_type(type_desc)};

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

        // Accumulate received data into destination using a type- and
        // device-aware accumulation (the previous loop hard-coded Real and
        // dereferenced device pointers on the host).
        auto * recv_ptr{recv_buffer.data()};
        for (int block{0}; block < nb_recv_blocks; ++block) {
            auto * dest_addr{data + (recv_offset * stride_in_direction +
                                     block * block_stride) * elem_size_in_bytes};
            accumulate_elements(dest_addr, recv_ptr,
                                static_cast<size_t>(recv_block_len), type_desc,
                                is_device_memory);
            recv_ptr += recv_block_len * elem_size_in_bytes;
        }
    }
#else   // not WITH_MPI
    CartesianCommunicator::CartesianCommunicator(
        const Parent_t & parent, const DynGridIndex & nb_subdivisions)
        : Parent_t{}, parent{parent}, nb_subdivisions{nb_subdivisions},
          coordinates(nb_subdivisions.size(), 0) {}

    CartesianCommunicator::CartesianCommunicator(
        const CartesianCommunicator & other)
        : Parent_t{}, parent{other.parent},
          nb_subdivisions{other.nb_subdivisions},
          coordinates{other.coordinates} {}

    CartesianCommunicator::~CartesianCommunicator() {}

    CartesianCommunicator &
    CartesianCommunicator::operator=(const CartesianCommunicator & other) {
        this->parent = other.parent;
        this->nb_subdivisions = other.nb_subdivisions;
        this->coordinates = other.coordinates;
        return *this;
    }

    void CartesianCommunicator::sendrecv_right(
        int direction, int block_stride, int nb_send_blocks, int send_block_len,
        Index_t send_offset, int nb_recv_blocks, int recv_block_len,
        Index_t recv_offset, char * data, int stride_in_direction,
        int elem_size_in_bytes, TypeDescriptor type_desc,
        bool is_device_memory) const {
        (void)type_desc;
        (void)direction;
        serial_sendrecv(block_stride, nb_send_blocks, send_block_len,
                        send_offset, nb_recv_blocks, recv_block_len,
                        recv_offset, data, stride_in_direction,
                        elem_size_in_bytes, is_device_memory);
    }

    void CartesianCommunicator::sendrecv_left(
        int direction, int block_stride, int nb_send_blocks, int send_block_len,
        Index_t send_offset, int nb_recv_blocks, int recv_block_len,
        Index_t recv_offset, char * data, int stride_in_direction,
        int elem_size_in_bytes, TypeDescriptor type_desc,
        bool is_device_memory) const {
        (void)type_desc;
        (void)direction;
        serial_sendrecv(block_stride, nb_send_blocks, send_block_len,
                        send_offset, nb_recv_blocks, recv_block_len,
                        recv_offset, data, stride_in_direction,
                        elem_size_in_bytes, is_device_memory);
    }

    void CartesianCommunicator::sendrecv_right_accumulate(
        int direction, int block_stride, int nb_send_blocks, int send_block_len,
        Index_t send_offset, int nb_recv_blocks, int recv_block_len,
        Index_t recv_offset, char * data, int stride_in_direction,
        int elem_size_in_bytes, TypeDescriptor type_desc,
        bool is_device_memory) const {
        (void)direction;
        serial_sendrecv_accumulate(block_stride, nb_send_blocks, send_block_len,
                                   send_offset, nb_recv_blocks, recv_block_len,
                                   recv_offset, data, stride_in_direction,
                                   elem_size_in_bytes, type_desc,
                                   is_device_memory);
    }

    void CartesianCommunicator::sendrecv_left_accumulate(
        int direction, int block_stride, int nb_send_blocks, int send_block_len,
        Index_t send_offset, int nb_recv_blocks, int recv_block_len,
        Index_t recv_offset, char * data, int stride_in_direction,
        int elem_size_in_bytes, TypeDescriptor type_desc,
        bool is_device_memory) const {
        (void)direction;
        serial_sendrecv_accumulate(block_stride, nb_send_blocks, send_block_len,
                                   send_offset, nb_recv_blocks, recv_block_len,
                                   recv_offset, data, stride_in_direction,
                                   elem_size_in_bytes, type_desc,
                                   is_device_memory);
    }
#endif  // WITH_MPI

    const DynGridIndex & CartesianCommunicator::get_nb_subdivisions() const {
        return this->nb_subdivisions;
    }

    const DynGridIndex & CartesianCommunicator::get_coordinates() const {
        return this->coordinates;
    }
}  // namespace muGrid

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

#include "memory/device_alloc.hh"
#include "memory/gpu_runtime.hh"
#include "memory/unified_memory.hh"
#include "mpi/gpu_aware_mpi.hh"
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
#include "mpi/ghost_accumulate_gpu.hh"
#endif

namespace muGrid {

    namespace {
        /**
         * @brief Copy strided memory blocks using appropriate method.
         *
         * Copies nb_blocks blocks of data, each of size block_len bytes,
         * with blocks separated by dst_pitch bytes in the destination and
         * src_pitch bytes in the source. Different pitches allow packing a
         * strided layout into a contiguous buffer (dst_pitch == block_len)
         * and unpacking it again.
         *
         * For GPU memory, uses hipMemcpy2D/cudaMemcpy2D for efficiency.
         * For host memory, falls back to individual memcpy calls.
         *
         * @param dst Destination address
         * @param src Source address
         * @param block_len Size of each block in bytes (width)
         * @param nb_blocks Number of blocks to copy (height)
         * @param dst_pitch Stride between blocks in the destination, bytes
         * @param src_pitch Stride between blocks in the source, bytes
         * @param is_device_memory If true, use GPU device copy
         */
        void device_memcpy_strided(void * dst, const void * src,
                                   size_t block_len, size_t nb_blocks,
                                   size_t dst_pitch, size_t src_pitch,
                                   bool is_device_memory) {
            if (is_device_memory) {
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
                GPU_MEMCPY_2D_D2D(dst, dst_pitch, src, src_pitch, block_len,
                                  nb_blocks);
#else
                // GCOVR_EXCL_START -- unreachable: device fields cannot be
                // created in a build without a GPU backend
                throw RuntimeError(
                    "device_memcpy_strided: muGrid was compiled without GPU "
                    "support");
                // GCOVR_EXCL_STOP
#endif
            } else {
                // Host memory: use loop of memcpy
                auto * dst_ptr = static_cast<char *>(dst);
                auto * src_ptr = static_cast<const char *>(src);
                for (size_t i{0}; i < nb_blocks; ++i) {
                    std::memcpy(dst_ptr, src_ptr, block_len);
                    dst_ptr += dst_pitch;
                    src_ptr += src_pitch;
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
         */
        //! Host block-strided accumulation:
        //! dst[b*dst_stride + j] += src[b*src_stride + j]
        template <typename T>
        void host_accumulate_strided(T * dst, const T * src, size_t nb_blocks,
                                     size_t block_len, size_t dst_stride,
                                     size_t src_stride) {
            for (size_t b{0}; b < nb_blocks; ++b) {
                for (size_t j{0}; j < block_len; ++j) {
                    dst[b * dst_stride + j] += src[b * src_stride + j];
                }
            }
        }

        /**
         * @brief Accumulate a block-strided halo region into dst:
         * dst[b*dst_block_stride + j] += src[b*src_block_stride + j] for
         * b in [0, nb_blocks) and j in [0, block_len) (element units of
         * type_desc; dst/src already point at the first block).
         *
         * When the destination is on the device this dispatches to a single
         * device kernel covering the whole region (one launch/sync -- a
         * per-block launch is dominated by launch overhead); otherwise it
         * accumulates on the host. Dispatching on the type descriptor keeps
         * Complex/Int reductions correct.
         */
        void accumulate_blocks(char * dst, const char * src, size_t nb_blocks,
                               size_t block_len, size_t dst_block_stride,
                               size_t src_block_stride,
                               TypeDescriptor type_desc, bool dst_on_device,
                               bool src_on_device) {
            if (dst_on_device) {
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
                // Native device accumulation: the destination never leaves the
                // device. src is read directly when already on the device, or
                // staged to the device once (inside device_accumulate) when it
                // is a host receive buffer.
                device_accumulate(dst, src, nb_blocks, block_len,
                                  dst_block_stride, src_block_stride, type_desc,
                                  src_on_device);
                return;
#else
                // GCOVR_EXCL_START -- unreachable: no device fields without a
                // GPU backend
                throw std::runtime_error(
                    "accumulate_blocks: device memory requires a GPU build");
                // GCOVR_EXCL_STOP
#endif
            }
            // Host destination (the source is host memory as well).
            (void)src_on_device;
            switch (type_desc) {
            case TypeDescriptor::Real:
                host_accumulate_strided(reinterpret_cast<Real *>(dst),
                                        reinterpret_cast<const Real *>(src),
                                        nb_blocks, block_len, dst_block_stride,
                                        src_block_stride);
                break;
            case TypeDescriptor::Complex:
                host_accumulate_strided(reinterpret_cast<Complex *>(dst),
                                        reinterpret_cast<const Complex *>(src),
                                        nb_blocks, block_len, dst_block_stride,
                                        src_block_stride);
                break;
            case TypeDescriptor::Int:
                host_accumulate_strided(reinterpret_cast<Int *>(dst),
                                        reinterpret_cast<const Int *>(src),
                                        nb_blocks, block_len, dst_block_stride,
                                        src_block_stride);
                break;
            case TypeDescriptor::Uint:
                host_accumulate_strided(reinterpret_cast<Uint *>(dst),
                                        reinterpret_cast<const Uint *>(src),
                                        nb_blocks, block_len, dst_block_stride,
                                        src_block_stride);
                break;
            case TypeDescriptor::Index:
                host_accumulate_strided(reinterpret_cast<Index_t *>(dst),
                                        reinterpret_cast<const Index_t *>(src),
                                        nb_blocks, block_len, dst_block_stride,
                                        src_block_stride);
                break;
            default:
                throw std::runtime_error(
                    "accumulate_blocks: unsupported type descriptor");
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
                                  nb_send_blocks, pitch_bytes, pitch_bytes,
                                  is_device_memory);
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
            // Single strided accumulate over all blocks. dst and src are both
            // strided in the field with the same block_stride (self-neighbor
            // wrap); src lives in the same memory space as dst.
            char * recv_base{data + recv_offset * stride_in_direction *
                                        elem_size_in_bytes};
            char * send_base{data + send_offset * stride_in_direction *
                                        elem_size_in_bytes};
            accumulate_blocks(recv_base, send_base,
                              static_cast<size_t>(nb_send_blocks),
                              static_cast<size_t>(send_block_len),
                              static_cast<size_t>(block_stride),
                              static_cast<size_t>(block_stride), type_desc,
                              is_device_memory, is_device_memory);
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
        for (auto & buffer : this->device_staging) {
            if (buffer != nullptr) {
                device_deallocate(buffer);
            }
        }
    }

    char * CartesianCommunicator::get_device_staging(std::size_t slot,
                                                     std::size_t size) const {
        auto & buffer{this->device_staging.at(slot)};
        auto & current_size{this->device_staging_size.at(slot)};
        if (current_size < size) {
            if (buffer != nullptr) {
                device_deallocate(buffer);
                buffer = nullptr;
                current_size = 0;
            }
            buffer = device_allocate(size);
            current_size = size;
        }
        return static_cast<char *>(buffer);
    }

    void CartesianCommunicator::sendrecv_staged(
        int block_stride, int nb_send_blocks, int send_block_len,
        int nb_recv_blocks, int recv_block_len, void * send_addr,
        void * recv_addr, int elem_size_in_bytes, MPI_Datatype mpi_datatype,
        int dest_rank, int src_rank) const {
        auto send_row_bytes{static_cast<std::size_t>(send_block_len) *
                            elem_size_in_bytes};
        auto recv_row_bytes{static_cast<std::size_t>(recv_block_len) *
                            elem_size_in_bytes};
        auto send_bytes{send_row_bytes * nb_send_blocks};
        auto recv_bytes{recv_row_bytes * nb_recv_blocks};
        auto pitch_bytes{static_cast<std::size_t>(block_stride) *
                         elem_size_in_bytes};

        char * send_staging{
            send_bytes > 0 ? this->get_device_staging(0, send_bytes) : nullptr};
        char * recv_staging{
            recv_bytes > 0 ? this->get_device_staging(1, recv_bytes) : nullptr};

        if (send_bytes > 0) {
            // Gather the strided halo into the contiguous send buffer
            device_memcpy_strided(send_staging, send_addr, send_row_bytes,
                                  nb_send_blocks, send_row_bytes, pitch_bytes,
                                  true);
        }
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        // Device-to-device 2D copies are asynchronous with respect to the
        // host; MPI must not read the staging buffer before the gather has
        // completed.
        GPU_DEVICE_SYNCHRONIZE();
#endif
        // Without GPU-aware MPI, bounce the contiguous staging buffers
        // through host memory (correct with any MPI library). On a physically
        // unified-memory device the staging buffer is already host-addressable,
        // so any MPI can read it directly and no bounce is needed.
        const bool bounce{!mpi_is_gpu_aware() && !device_has_unified_memory()};
        char * send_buffer{send_staging};
        char * recv_buffer{recv_staging};
        if (bounce) {
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
            auto & host_send{this->host_staging[0]};
            auto & host_recv{this->host_staging[1]};
            if (host_send.size() < send_bytes) {
                host_send.resize(send_bytes);
            }
            if (host_recv.size() < recv_bytes) {
                host_recv.resize(recv_bytes);
            }
            if (send_bytes > 0) {
                GPU_MEMCPY_D2H(host_send.data(), send_staging, send_bytes);
            }
            send_buffer = host_send.data();
            recv_buffer = host_recv.data();
#endif
        }
        MPI_Status status;
        MPI_Sendrecv(send_buffer, nb_send_blocks * send_block_len,
                     mpi_datatype, dest_rank, 0, recv_buffer,
                     nb_recv_blocks * recv_block_len, mpi_datatype, src_rank, 0,
                     this->comm, &status);
        if (bounce && recv_bytes > 0) {
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
            GPU_MEMCPY_H2D(recv_staging, recv_buffer, recv_bytes);
#endif
        }
        if (recv_bytes > 0) {
            // Scatter the contiguous receive buffer into the strided halo.
            // This runs on the default stream, so subsequent kernels and the
            // next pack are ordered after it.
            device_memcpy_strided(recv_addr, recv_staging, recv_row_bytes,
                                  nb_recv_blocks, pitch_bytes, recv_row_bytes,
                                  true);
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
        // Take the local-copy path when MPI is not initialized (comm is
        // NULL) or when this rank is its own neighbor (direction not
        // subdivided, periodic wrap). Self-communication through
        // MPI_Sendrecv with strided datatypes is functionally correct but
        // catastrophically slow on device memory (UCX packs the vector
        // type block by block).
        if (this->comm == MPI_COMM_NULL ||
            (this->right_ranks[direction] == this->rank() &&
             this->left_ranks[direction] == this->rank())) {
            serial_sendrecv(block_stride, nb_send_blocks, send_block_len,
                            send_offset, nb_recv_blocks, recv_block_len,
                            recv_offset, data, stride_in_direction,
                            elem_size_in_bytes, is_device_memory);
            return;
        }
        auto recv_addr{static_cast<void *>(
            data + recv_offset * stride_in_direction * elem_size_in_bytes)};
        auto send_addr{static_cast<void *>(
            data + send_offset * stride_in_direction * elem_size_in_bytes)};
        MPI_Datatype mpi_datatype{descriptor_to_mpi_type(type_desc)};
        MPI_Status status;

        if (is_device_memory) {
            // Strided derived datatypes on device pointers are packed block
            // by block by the MPI implementation (orders of magnitude
            // slower than a bulk copy); GPU-aware fast paths only apply to
            // contiguous messages. Pack into contiguous device staging
            // buffers, communicate flat, unpack.
            this->sendrecv_staged(block_stride, nb_send_blocks, send_block_len,
                                  nb_recv_blocks, recv_block_len, send_addr,
                                  recv_addr, elem_size_in_bytes, mpi_datatype,
                                  this->right_ranks[direction],
                                  this->left_ranks[direction]);
            return;
        }
        MPI_Datatype send_buffer_mpi_t, recv_buffer_mpi_t;
        MPI_Type_vector(nb_send_blocks, send_block_len, block_stride,
                        mpi_datatype, &send_buffer_mpi_t);
        MPI_Type_commit(&send_buffer_mpi_t);
        MPI_Type_vector(nb_recv_blocks, recv_block_len, block_stride,
                        mpi_datatype, &recv_buffer_mpi_t);
        MPI_Type_commit(&recv_buffer_mpi_t);

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
        // Local-copy path when MPI is uninitialized or this rank is its own
        // neighbor (see sendrecv_right).
        if (this->comm == MPI_COMM_NULL ||
            (this->right_ranks[direction] == this->rank() &&
             this->left_ranks[direction] == this->rank())) {
            serial_sendrecv(block_stride, nb_send_blocks, send_block_len,
                            send_offset, nb_recv_blocks, recv_block_len,
                            recv_offset, data, stride_in_direction,
                            elem_size_in_bytes, is_device_memory);
            return;
        }
        auto recv_addr{static_cast<void *>(
            data + recv_offset * stride_in_direction * elem_size_in_bytes)};
        auto send_addr{static_cast<void *>(
            data + send_offset * stride_in_direction * elem_size_in_bytes)};
        MPI_Datatype mpi_datatype{descriptor_to_mpi_type(type_desc)};
        MPI_Status status;

        if (is_device_memory) {
            // Contiguous staging for strided device halos (see
            // sendrecv_right).
            this->sendrecv_staged(block_stride, nb_send_blocks, send_block_len,
                                  nb_recv_blocks, recv_block_len, send_addr,
                                  recv_addr, elem_size_in_bytes, mpi_datatype,
                                  this->left_ranks[direction],
                                  this->right_ranks[direction]);
            return;
        }
        MPI_Datatype send_buffer_mpi_t, recv_buffer_mpi_t;
        MPI_Type_vector(nb_send_blocks, send_block_len, block_stride,
                        mpi_datatype, &send_buffer_mpi_t);
        MPI_Type_commit(&send_buffer_mpi_t);
        MPI_Type_vector(nb_recv_blocks, recv_block_len, block_stride,
                        mpi_datatype, &recv_buffer_mpi_t);
        MPI_Type_commit(&recv_buffer_mpi_t);

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
        // Local path when MPI is uninitialized or this rank is its own
        // neighbor (see sendrecv_right).
        if (this->comm == MPI_COMM_NULL ||
            (this->right_ranks[direction] == this->rank() &&
             this->left_ranks[direction] == this->rank())) {
            serial_sendrecv_accumulate(block_stride, nb_send_blocks, send_block_len,
                                       send_offset, nb_recv_blocks, recv_block_len,
                                       recv_offset, data, stride_in_direction,
                                       elem_size_in_bytes, type_desc,
                                       is_device_memory);
            return;
        }
        // Convert TypeDescriptor to MPI_Datatype
        MPI_Datatype mpi_datatype{descriptor_to_mpi_type(type_desc)};

        // Number of (contiguous) elements / bytes received
        auto total_recv_elems{static_cast<size_t>(nb_recv_blocks) *
                              static_cast<size_t>(recv_block_len)};
        auto recv_bytes{total_recv_elems * elem_size_in_bytes};

        auto send_addr{static_cast<void *>(
            data + send_offset * stride_in_direction * elem_size_in_bytes)};
        char * dst_base{data +
                        recv_offset * stride_in_direction * elem_size_in_bytes};

        MPI_Status status;
        if (is_device_memory) {
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
            // Pack the strided device send region into contiguous device
            // staging (strided datatypes on device pointers are packed block
            // by block by MPI; see sendrecv_staged).
            auto send_row_bytes{static_cast<std::size_t>(send_block_len) *
                                elem_size_in_bytes};
            auto send_bytes{send_row_bytes * nb_send_blocks};
            auto pitch_bytes{static_cast<std::size_t>(block_stride) *
                             elem_size_in_bytes};
            char * send_staging{
                send_bytes > 0 ? this->get_device_staging(0, send_bytes)
                               : nullptr};
            if (send_bytes > 0) {
                device_memcpy_strided(send_staging, send_addr, send_row_bytes,
                                      nb_send_blocks, send_row_bytes,
                                      pitch_bytes, true);
            }
            GPU_DEVICE_SYNCHRONIZE();

            // GPU-aware MPI receives straight into device staging and the
            // accumulate kernel reads it there: no host transfer at all.
            // Otherwise both sides bounce through host memory.
            const bool gpu_aware{mpi_is_gpu_aware()};
            char * send_buffer{send_staging};
            char * recv_buffer_ptr{nullptr};
            std::vector<char> host_recv{};
            if (gpu_aware) {
                recv_buffer_ptr =
                    recv_bytes > 0 ? this->get_device_staging(1, recv_bytes)
                                   : nullptr;
            } else {
                auto & host_send{this->host_staging[0]};
                if (host_send.size() < send_bytes) {
                    host_send.resize(send_bytes);
                }
                if (send_bytes > 0) {
                    GPU_MEMCPY_D2H(host_send.data(), send_staging, send_bytes);
                }
                send_buffer = host_send.data();
                host_recv.resize(recv_bytes);
                recv_buffer_ptr = host_recv.data();
            }
            MPI_Sendrecv(send_buffer, nb_send_blocks * send_block_len,
                         mpi_datatype, this->right_ranks[direction], 0, recv_buffer_ptr,
                         total_recv_elems, mpi_datatype, this->left_ranks[direction], 0, this->comm,
                         &status);
            // Scatter-accumulate the contiguous receive buffer into the
            // strided destination; src is device memory (gpu-aware) or the
            // host bounce buffer.
            accumulate_blocks(dst_base, recv_buffer_ptr,
                              static_cast<size_t>(nb_recv_blocks),
                              static_cast<size_t>(recv_block_len),
                              static_cast<size_t>(block_stride),
                              static_cast<size_t>(recv_block_len), type_desc,
                              true, gpu_aware);
#endif
        } else {
            // Host: derived-datatype send into a contiguous host receive
            // buffer.
            std::vector<char> recv_buffer(recv_bytes);
            MPI_Datatype send_buffer_mpi_t;
            MPI_Type_vector(nb_send_blocks, send_block_len, block_stride,
                            mpi_datatype, &send_buffer_mpi_t);
            MPI_Type_commit(&send_buffer_mpi_t);
            MPI_Sendrecv(send_addr, 1, send_buffer_mpi_t, this->right_ranks[direction], 0,
                         recv_buffer.data(), total_recv_elems, mpi_datatype,
                         this->left_ranks[direction], 0, this->comm, &status);
            MPI_Type_free(&send_buffer_mpi_t);
            accumulate_blocks(dst_base, recv_buffer.data(),
                              static_cast<size_t>(nb_recv_blocks),
                              static_cast<size_t>(recv_block_len),
                              static_cast<size_t>(block_stride),
                              static_cast<size_t>(recv_block_len), type_desc,
                              false, false);
        }
    }

    void CartesianCommunicator::sendrecv_left_accumulate(
        int direction, int block_stride, int nb_send_blocks, int send_block_len,
        Index_t send_offset, int nb_recv_blocks, int recv_block_len,
        Index_t recv_offset, char * data, int stride_in_direction,
        int elem_size_in_bytes, TypeDescriptor type_desc,
        bool is_device_memory) const {
        // Local path when MPI is uninitialized or this rank is its own
        // neighbor (see sendrecv_right).
        if (this->comm == MPI_COMM_NULL ||
            (this->right_ranks[direction] == this->rank() &&
             this->left_ranks[direction] == this->rank())) {
            serial_sendrecv_accumulate(block_stride, nb_send_blocks, send_block_len,
                                       send_offset, nb_recv_blocks, recv_block_len,
                                       recv_offset, data, stride_in_direction,
                                       elem_size_in_bytes, type_desc,
                                       is_device_memory);
            return;
        }
        // Convert TypeDescriptor to MPI_Datatype
        MPI_Datatype mpi_datatype{descriptor_to_mpi_type(type_desc)};

        // Number of (contiguous) elements / bytes received
        auto total_recv_elems{static_cast<size_t>(nb_recv_blocks) *
                              static_cast<size_t>(recv_block_len)};
        auto recv_bytes{total_recv_elems * elem_size_in_bytes};

        auto send_addr{static_cast<void *>(
            data + send_offset * stride_in_direction * elem_size_in_bytes)};
        char * dst_base{data +
                        recv_offset * stride_in_direction * elem_size_in_bytes};

        MPI_Status status;
        if (is_device_memory) {
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
            // Pack the strided device send region into contiguous device
            // staging (strided datatypes on device pointers are packed block
            // by block by MPI; see sendrecv_staged).
            auto send_row_bytes{static_cast<std::size_t>(send_block_len) *
                                elem_size_in_bytes};
            auto send_bytes{send_row_bytes * nb_send_blocks};
            auto pitch_bytes{static_cast<std::size_t>(block_stride) *
                             elem_size_in_bytes};
            char * send_staging{
                send_bytes > 0 ? this->get_device_staging(0, send_bytes)
                               : nullptr};
            if (send_bytes > 0) {
                device_memcpy_strided(send_staging, send_addr, send_row_bytes,
                                      nb_send_blocks, send_row_bytes,
                                      pitch_bytes, true);
            }
            GPU_DEVICE_SYNCHRONIZE();

            // GPU-aware MPI receives straight into device staging and the
            // accumulate kernel reads it there: no host transfer at all.
            // Otherwise both sides bounce through host memory.
            const bool gpu_aware{mpi_is_gpu_aware()};
            char * send_buffer{send_staging};
            char * recv_buffer_ptr{nullptr};
            std::vector<char> host_recv{};
            if (gpu_aware) {
                recv_buffer_ptr =
                    recv_bytes > 0 ? this->get_device_staging(1, recv_bytes)
                                   : nullptr;
            } else {
                auto & host_send{this->host_staging[0]};
                if (host_send.size() < send_bytes) {
                    host_send.resize(send_bytes);
                }
                if (send_bytes > 0) {
                    GPU_MEMCPY_D2H(host_send.data(), send_staging, send_bytes);
                }
                send_buffer = host_send.data();
                host_recv.resize(recv_bytes);
                recv_buffer_ptr = host_recv.data();
            }
            MPI_Sendrecv(send_buffer, nb_send_blocks * send_block_len,
                         mpi_datatype, this->left_ranks[direction], 0, recv_buffer_ptr,
                         total_recv_elems, mpi_datatype, this->right_ranks[direction], 0, this->comm,
                         &status);
            // Scatter-accumulate the contiguous receive buffer into the
            // strided destination; src is device memory (gpu-aware) or the
            // host bounce buffer.
            accumulate_blocks(dst_base, recv_buffer_ptr,
                              static_cast<size_t>(nb_recv_blocks),
                              static_cast<size_t>(recv_block_len),
                              static_cast<size_t>(block_stride),
                              static_cast<size_t>(recv_block_len), type_desc,
                              true, gpu_aware);
#endif
        } else {
            // Host: derived-datatype send into a contiguous host receive
            // buffer.
            std::vector<char> recv_buffer(recv_bytes);
            MPI_Datatype send_buffer_mpi_t;
            MPI_Type_vector(nb_send_blocks, send_block_len, block_stride,
                            mpi_datatype, &send_buffer_mpi_t);
            MPI_Type_commit(&send_buffer_mpi_t);
            MPI_Sendrecv(send_addr, 1, send_buffer_mpi_t, this->left_ranks[direction], 0,
                         recv_buffer.data(), total_recv_elems, mpi_datatype,
                         this->right_ranks[direction], 0, this->comm, &status);
            MPI_Type_free(&send_buffer_mpi_t);
            accumulate_blocks(dst_base, recv_buffer.data(),
                              static_cast<size_t>(nb_recv_blocks),
                              static_cast<size_t>(recv_block_len),
                              static_cast<size_t>(block_stride),
                              static_cast<size_t>(recv_block_len), type_desc,
                              false, false);
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

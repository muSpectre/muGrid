/**
 * @file   communicator.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   02 Oct 2019
 *
 * @brief  implementation for mpi abstraction layer
 *
 * Copyright © 2019 Till Junge
 *
 * µGrid is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µGrid is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µGrid; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#include "mpi/communicator.hh"
#include "memory/device.hh"
#include "memory/gpu_runtime.hh"

#ifdef WITH_MPI
#include "mpi/gpu_aware_mpi.hh"
#include "mpi/mpi_counts.hh"
#include "mpi/type_descriptor_mpi.hh"
#endif

#include <algorithm>
#include <cstring>
#include <sstream>
#include <utility>

namespace muGrid {

    namespace {
        std::size_t nb_bytes(Index_t count, TypeDescriptor type) {
            if (count < 0) {
                throw RuntimeError("buffer exchange: negative element count");
            }
            return static_cast<std::size_t>(count) *
                   type_descriptor_size(type);
        }

        //! Copy within host or within device memory
        void copy_bytes(void * dst, const void * src, std::size_t bytes,
                        BufferLocation location) {
            if (bytes == 0 || dst == src) {
                return;
            }
            if (location == BufferLocation::Device) {
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
                GPU_MEMCPY_D2D(dst, src, bytes);
                return;
#else
                throw RuntimeError("buffer exchange: device buffers need a "
                                   "build with GPU support");
#endif
            }
            std::memcpy(dst, src, bytes);
        }

        //! This rank's own block of an all-to-all: element counts must agree
        void copy_own_block(const void * send,
                            const std::vector<int> & send_counts,
                            const std::vector<int> & send_displs, void * recv,
                            const std::vector<int> & recv_counts,
                            const std::vector<int> & recv_displs, int rank,
                            std::size_t elem, BufferLocation location) {
            if (send_counts.at(rank) != recv_counts.at(rank)) {
                std::stringstream error{};
                error << "alltoallv: rank " << rank << " sends "
                      << send_counts[rank] << " elements to itself but "
                      << "expects " << recv_counts[rank];
                throw RuntimeError(error.str());
            }
            copy_bytes(static_cast<char *>(recv) +
                           static_cast<std::size_t>(recv_displs.at(rank)) *
                               elem,
                       static_cast<const char *>(send) +
                           static_cast<std::size_t>(send_displs.at(rank)) *
                               elem,
                       static_cast<std::size_t>(send_counts[rank]) * elem,
                       location);
        }
    }  // namespace

    AllreduceRequest::AllreduceRequest(AllreduceRequest && other) noexcept
        : host_staging{std::move(other.host_staging)},
          device_recv{other.device_recv}, recv_bytes{other.recv_bytes} {
        // Moving the vectors keeps their buffers, which MPI may be using
#ifdef WITH_MPI
        this->request = other.request;
        other.request = MPI_REQUEST_NULL;
#endif
        other.device_recv = nullptr;
    }

    AllreduceRequest &
    AllreduceRequest::operator=(AllreduceRequest && other) noexcept {
        if (this != &other) {
            try {
                this->wait();
            } catch (...) {  // NOLINT(bugprone-empty-catch)
            }
#ifdef WITH_MPI
            this->request = other.request;
            other.request = MPI_REQUEST_NULL;
#endif
            this->host_staging = std::move(other.host_staging);
            this->device_recv = other.device_recv;
            this->recv_bytes = other.recv_bytes;
            other.device_recv = nullptr;
        }
        return *this;
    }

    AllreduceRequest::~AllreduceRequest() {
        try {
            this->wait();
        } catch (...) {  // NOLINT(bugprone-empty-catch)
            // A destructor must not throw; the reduction's result is lost.
        }
    }

    void AllreduceRequest::wait() {
#ifdef WITH_MPI
        if (this->request != MPI_REQUEST_NULL) {
            const int status{MPI_Wait(&this->request, MPI_STATUS_IGNORE)};
            if (status != MPI_SUCCESS) {
                std::stringstream error{};
                error << "MPI_Wait on a non-blocking allreduce failed with "
                      << status;
                throw RuntimeError(error.str());
            }
        }
#endif
        if (this->device_recv != nullptr) {
            void * dst{this->device_recv};
            this->device_recv = nullptr;
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
            if (this->recv_bytes > 0) {
                GPU_MEMCPY_H2D(dst, this->host_staging[1].data(),
                               this->recv_bytes);
            }
#else
            (void)dst;
#endif
        }
    }

#ifdef WITH_MPI

    Communicator::Communicator(MPI_Comm comm) : comm{comm} {
        // Check if MPI has been initialized. If not, use MPI_COMM_NULL
        // to allow serial operation without MPI_Init
        if (comm != MPI_COMM_NULL) {
            int initialized = 0;
            MPI_Initialized(&initialized);
            if (!initialized) {
                this->comm = MPI_COMM_NULL;
            }
        }
    }

    Communicator::Communicator(const Communicator & other)
        : comm(other.comm) {}

    Communicator::~Communicator() {}

    Communicator & Communicator::operator=(const Communicator & other) {
        this->comm = other.comm;
        return *this;
    }

    namespace {
        void check_mpi(int status, const char * call, int rank) {
            if (status != MPI_SUCCESS) {
                std::stringstream error{};
                error << call << " failed with " << status << " on rank "
                      << rank;
                throw RuntimeError(error.str());
            }
        }

        MPI_Op mpi_op(ReduceOp op) {
            switch (op) {
            case ReduceOp::Sum:
                return MPI_SUM;
            case ReduceOp::Max:
                return MPI_MAX;
            case ReduceOp::Min:
                return MPI_MIN;
            }
            throw RuntimeError("allreduce: unknown reduction");
        }

        /**
         * The buffers MPI actually reads and writes for one exchange.
         *
         * Device buffers are handed over as they are when MPI is GPU-aware,
         * or when the host can dereference both (managed memory on a
         * unified-memory device), which any MPI can then read. Otherwise
         * they are bounced through the communicator's host buffers: `send`
         * is copied there on construction and the caller copies `recv` back
         * with to_device. Construction drains the default stream first,
         * since MPI does not order itself after the kernels that wrote
         * `send`.
         */
        struct MPIBuffers {
            const void * send;
            void * recv;
            bool bounced{false};

            MPIBuffers(const void * send_buffer, std::size_t send_bytes,
                       void * recv_buffer, std::size_t recv_bytes,
                       BufferLocation location,
                       std::array<std::vector<char>, 2> & host_staging)
                : send{send_buffer}, recv{recv_buffer} {
                if (location != BufferLocation::Device) {
                    return;
                }
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
                GPU_STREAM_SYNCHRONIZE_DEFAULT();
                if (mpi_is_gpu_aware() ||
                    (host_can_read_device_pointer(send_buffer) &&
                     host_can_read_device_pointer(recv_buffer))) {
                    return;
                }
                auto & host_send{host_staging[0]};
                auto & host_recv{host_staging[1]};
                if (host_send.size() < send_bytes) {
                    host_send.resize(send_bytes);
                }
                if (host_recv.size() < recv_bytes) {
                    host_recv.resize(recv_bytes);
                }
                if (send_bytes > 0) {
                    GPU_MEMCPY_D2H(host_send.data(), send_buffer, send_bytes);
                }
                this->send = host_send.data();
                this->recv = host_recv.data();
                this->bounced = true;
#else
                (void)send_bytes;
                (void)recv_bytes;
                (void)host_staging;
                throw RuntimeError("buffer exchange: device buffers need a "
                                   "build with GPU support");
#endif
            }

            //! Copy `bytes` at `offset` of the bounced receive buffer to the
            //! caller's (a no-op unless bounced)
            void to_device(void * dst, std::size_t offset,
                           std::size_t bytes) const {
                if (!this->bounced || bytes == 0) {
                    return;
                }
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
                GPU_MEMCPY_H2D(static_cast<char *>(dst) + offset,
                               static_cast<const char *>(this->recv) + offset,
                               bytes);
#else
                (void)dst;
                (void)offset;
#endif
            }
        };
    }  // namespace

    void Communicator::allreduce(const void * send, void * recv,
                                 Index_t count, TypeDescriptor type,
                                 ReduceOp op, BufferLocation location) const {
        const auto bytes{nb_bytes(count, type)};
        if (this->comm == MPI_COMM_NULL) {
            copy_bytes(recv, send, bytes, location);
            return;
        }
        if (is_complex(type) && op != ReduceOp::Sum) {
            throw RuntimeError("allreduce: complex values have no order, so "
                               "no max or min");
        }
        MPIBuffers buffers{send, bytes, recv, bytes, location,
                           this->host_staging};
        check_mpi(MPI_Allreduce(buffers.send, buffers.recv,
                                checked_mpi_int(count, "allreduce count"),
                                descriptor_to_mpi_type(type), mpi_op(op),
                                this->comm),
                  "MPI_Allreduce", this->rank());
        buffers.to_device(recv, 0, bytes);
    }

    AllreduceRequest Communicator::iallreduce(const void * send, void * recv,
                                              Index_t count,
                                              TypeDescriptor type,
                                              ReduceOp op,
                                              BufferLocation location) const {
        AllreduceRequest request{};
        const auto bytes{nb_bytes(count, type)};
        if (this->comm == MPI_COMM_NULL) {
            copy_bytes(recv, send, bytes, location);
            return request;
        }
        if (is_complex(type) && op != ReduceOp::Sum) {
            throw RuntimeError("iallreduce: complex values have no order, so "
                               "no max or min");
        }
        const void * send_buffer{send};
        void * recv_buffer{recv};
        if (location == BufferLocation::Device) {
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
            GPU_STREAM_SYNCHRONIZE_DEFAULT();
            auto & host_send{request.host_staging[0]};
            auto & host_recv{request.host_staging[1]};
            host_send.resize(bytes);
            host_recv.resize(bytes);
            if (bytes > 0) {
                GPU_MEMCPY_D2H(host_send.data(), send, bytes);
            }
            send_buffer = host_send.data();
            recv_buffer = host_recv.data();
            request.device_recv = recv;
            request.recv_bytes = bytes;
#else
            throw RuntimeError("buffer exchange: device buffers need a "
                               "build with GPU support");
#endif
        }
        check_mpi(MPI_Iallreduce(send_buffer, recv_buffer,
                                 checked_mpi_int(count, "iallreduce count"),
                                 descriptor_to_mpi_type(type), mpi_op(op),
                                 this->comm, &request.request),
                  "MPI_Iallreduce", this->rank());
        return request;
    }

    void Communicator::allgather(const void * send, void * recv,
                                 Index_t count, TypeDescriptor type,
                                 BufferLocation location) const {
        const auto bytes{nb_bytes(count, type)};
        if (this->comm == MPI_COMM_NULL) {
            copy_bytes(recv, send, bytes, location);
            return;
        }
        const auto recv_bytes{bytes * static_cast<std::size_t>(this->size())};
        MPIBuffers buffers{send, bytes, recv, recv_bytes, location,
                           this->host_staging};
        const int mpi_count{checked_mpi_int(count, "allgather count")};
        const auto mpi_type{descriptor_to_mpi_type(type)};
        check_mpi(MPI_Allgather(buffers.send, mpi_count, mpi_type,
                                buffers.recv, mpi_count, mpi_type, this->comm),
                  "MPI_Allgather", this->rank());
        buffers.to_device(recv, 0, recv_bytes);
    }

    void Communicator::sendrecv(const void * send, Index_t send_count,
                                int dest, void * recv, Index_t recv_count,
                                int source, TypeDescriptor type,
                                BufferLocation location, int tag) const {
        const auto send_bytes{nb_bytes(send_count, type)};
        const auto recv_bytes{nb_bytes(recv_count, type)};
        if (this->comm == MPI_COMM_NULL) {
            if (send_count != recv_count) {
                throw RuntimeError("sendrecv: with one rank, the message "
                                   "received is the one sent and must have "
                                   "its size");
            }
            copy_bytes(recv, send, send_bytes, location);
            return;
        }
        MPIBuffers buffers{send, send_bytes, recv, recv_bytes, location,
                           this->host_staging};
        const auto mpi_type{descriptor_to_mpi_type(type)};
        check_mpi(MPI_Sendrecv(buffers.send,
                               checked_mpi_int(send_count, "sendrecv count"),
                               mpi_type, dest, tag, buffers.recv,
                               checked_mpi_int(recv_count, "sendrecv count"),
                               mpi_type, source, tag, this->comm,
                               MPI_STATUS_IGNORE),
                  "MPI_Sendrecv", this->rank());
        buffers.to_device(recv, 0, recv_bytes);
    }

    void Communicator::alltoallv(const void * send,
                                 const std::vector<int> & send_counts,
                                 const std::vector<int> & send_displs,
                                 void * recv,
                                 const std::vector<int> & recv_counts,
                                 const std::vector<int> & recv_displs,
                                 TypeDescriptor type,
                                 BufferLocation location) const {
        const int nb_ranks{this->size()};
        const int my_rank{this->rank()};
        const auto n{static_cast<std::size_t>(nb_ranks)};
        if (send_counts.size() != n || send_displs.size() != n ||
            recv_counts.size() != n || recv_displs.size() != n) {
            throw RuntimeError("alltoallv: counts and displacements need one "
                               "entry per rank");
        }
        const auto elem{type_descriptor_size(type)};
        // The own block never enters MPI (and never leaves the device).
        copy_own_block(send, send_counts, send_displs, recv, recv_counts,
                       recv_displs, my_rank, elem, location);
        if (this->comm == MPI_COMM_NULL) {
            return;
        }

        std::size_t send_bytes{0}, recv_bytes{0};
        for (std::size_t r{0}; r < n; ++r) {
            send_bytes = std::max(
                send_bytes,
                static_cast<std::size_t>(send_displs[r] + send_counts[r]) *
                    elem);
            recv_bytes = std::max(
                recv_bytes,
                static_cast<std::size_t>(recv_displs[r] + recv_counts[r]) *
                    elem);
        }
        MPIBuffers buffers{send, send_bytes, recv, recv_bytes, location,
                           this->host_staging};
        const auto mpi_type{descriptor_to_mpi_type(type)};
        const auto * send_base{static_cast<const char *>(buffers.send)};
        auto * recv_base{static_cast<char *>(buffers.recv)};
        // Post every receive and send up front and wait once, so that the
        // per-peer transfers overlap; each peer has a disjoint slice of the
        // buffers, so the outstanding requests do not alias.
        std::vector<MPI_Request> requests;
        requests.reserve(2 * n);
        for (int peer{0}; peer < nb_ranks; ++peer) {
            if (peer == my_rank) {
                continue;
            }
            if (recv_counts[peer] > 0) {
                requests.emplace_back();
                check_mpi(MPI_Irecv(recv_base +
                                        static_cast<std::size_t>(
                                            recv_displs[peer]) *
                                            elem,
                                    recv_counts[peer], mpi_type, peer, 0,
                                    this->comm, &requests.back()),
                          "MPI_Irecv", my_rank);
            }
            if (send_counts[peer] > 0) {
                requests.emplace_back();
                check_mpi(MPI_Isend(send_base +
                                        static_cast<std::size_t>(
                                            send_displs[peer]) *
                                            elem,
                                    send_counts[peer], mpi_type, peer, 0,
                                    this->comm, &requests.back()),
                          "MPI_Isend", my_rank);
            }
        }
        if (!requests.empty()) {
            check_mpi(MPI_Waitall(static_cast<int>(requests.size()),
                                  requests.data(), MPI_STATUSES_IGNORE),
                      "MPI_Waitall", my_rank);
        }
        // Copy back around the own block, which is already in place
        const auto own_begin{static_cast<std::size_t>(recv_displs[my_rank]) *
                             elem};
        const auto own_end{own_begin +
                           static_cast<std::size_t>(recv_counts[my_rank]) *
                               elem};
        buffers.to_device(recv, 0, std::min(own_begin, recv_bytes));
        if (own_end < recv_bytes) {
            buffers.to_device(recv, own_end, recv_bytes - own_end);
        }
    }

#else  // WITH_MPI

    void Communicator::allreduce(const void * send, void * recv,
                                 Index_t count, TypeDescriptor type,
                                 ReduceOp, BufferLocation location) const {
        copy_bytes(recv, send, nb_bytes(count, type), location);
    }

    AllreduceRequest Communicator::iallreduce(const void * send, void * recv,
                                              Index_t count,
                                              TypeDescriptor type, ReduceOp,
                                              BufferLocation location) const {
        copy_bytes(recv, send, nb_bytes(count, type), location);
        return AllreduceRequest{};
    }

    void Communicator::allgather(const void * send, void * recv,
                                 Index_t count, TypeDescriptor type,
                                 BufferLocation location) const {
        copy_bytes(recv, send, nb_bytes(count, type), location);
    }

    void Communicator::sendrecv(const void * send, Index_t send_count, int,
                                void * recv, Index_t recv_count, int,
                                TypeDescriptor type, BufferLocation location,
                                int) const {
        if (send_count != recv_count) {
            throw RuntimeError("sendrecv: with one rank, the message "
                               "received is the one sent and must have its "
                               "size");
        }
        copy_bytes(recv, send, nb_bytes(send_count, type), location);
    }

    void Communicator::alltoallv(const void * send,
                                 const std::vector<int> & send_counts,
                                 const std::vector<int> & send_displs,
                                 void * recv,
                                 const std::vector<int> & recv_counts,
                                 const std::vector<int> & recv_displs,
                                 TypeDescriptor type,
                                 BufferLocation location) const {
        copy_own_block(send, send_counts, send_displs, recv, recv_counts,
                       recv_displs, 0, type_descriptor_size(type), location);
    }

#endif  // WITH_MPI

}  // namespace muGrid

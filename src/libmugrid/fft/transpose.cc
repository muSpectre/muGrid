/**
 * @file   fft/transpose.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   21 Dec 2025
 *
 * @brief  MPI transpose: derived datatypes on host, contiguous staging
 *         on device
 *
 * Copyright © 2024 Lars Pastewka
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

#include "transpose.hh"
#include "core/exception.hh"
#include "memory/device_alloc.hh"
#include "memory/gpu_runtime.hh"
#include "mpi/gpu_aware_mpi.hh"

#include <algorithm>
#include <numeric>

namespace muGrid {

    void Transpose::compute_distribution(Index_t global_size, int comm_size,
                                         std::vector<Index_t> & counts,
                                         std::vector<Index_t> & displs) {
        counts.resize(comm_size);
        displs.resize(comm_size);

        // Distribute global_size as evenly as possible across ranks
        Index_t base_count{global_size / comm_size};
        Index_t remainder{global_size % comm_size};

        Index_t offset{0};
        for (int r{0}; r < comm_size; ++r) {
            // First 'remainder' ranks get one extra element
            counts[r] = base_count + (r < remainder ? 1 : 0);
            displs[r] = offset;
            offset += counts[r];
        }
    }

    Transpose::Transpose(const Communicator & comm,
                         const DynGridIndex & local_in,
                         const DynGridIndex & local_out, Index_t global_in,
                         Index_t global_out, Index_t axis_in, Index_t axis_out,
                         Index_t nb_components, StorageOrder layout,
                         bool on_device)
        : comm{comm}, local_in{local_in}, local_out{local_out},
          global_in{global_in}, global_out{global_out}, axis_in{axis_in},
          axis_out{axis_out}, nb_components{nb_components}, layout{layout},
          on_device{on_device} {
        if (local_in.get_dim() != local_out.get_dim()) {
            throw RuntimeError(
                "Input and output must have same dimensionality");
        }

        int comm_size{comm.size()};

        // Compute how the distributed dimensions are split across ranks
        compute_distribution(global_in, comm_size, this->in_counts,
                             this->in_displs);
        compute_distribution(global_out, comm_size, this->out_counts,
                             this->out_displs);

#ifdef WITH_MPI
        // Initialize MPI infrastructure
        this->send_counts.resize(comm_size, 1);  // Always 1 with derived types
        this->recv_counts.resize(comm_size, 1);
        this->send_displs.resize(comm_size, 0);  // Offset encoded in type
        this->recv_displs.resize(comm_size, 0);

        this->send_types_fwd.resize(comm_size, MPI_DATATYPE_NULL);
        this->recv_types_fwd.resize(comm_size, MPI_DATATYPE_NULL);
        this->send_types_bwd.resize(comm_size, MPI_DATATYPE_NULL);
        this->recv_types_bwd.resize(comm_size, MPI_DATATYPE_NULL);

        // Build the datatypes
        if (comm.size() > 1) {
            this->init_forward_types();
            this->init_backward_types();
            this->types_initialized = true;
        }
#endif
    }

    void Transpose::free_staging() {
        for (auto & buffer : this->device_staging) {
            if (buffer != nullptr) {
                device_deallocate(buffer);
                buffer = nullptr;
            }
        }
        this->device_staging_size = {{0, 0}};
    }

    Transpose::~Transpose() {
#ifdef WITH_MPI
        free_datatypes();
#endif
        this->free_staging();
    }

#ifdef WITH_MPI
    void Transpose::free_datatypes() {
        if (!types_initialized) {
            return;
        }

        for (auto & type : send_types_fwd) {
            if (type != MPI_DATATYPE_NULL) {
                MPI_Type_free(&type);
                type = MPI_DATATYPE_NULL;
            }
        }
        for (auto & type : recv_types_fwd) {
            if (type != MPI_DATATYPE_NULL) {
                MPI_Type_free(&type);
                type = MPI_DATATYPE_NULL;
            }
        }
        for (auto & type : send_types_bwd) {
            if (type != MPI_DATATYPE_NULL) {
                MPI_Type_free(&type);
                type = MPI_DATATYPE_NULL;
            }
        }
        for (auto & type : recv_types_bwd) {
            if (type != MPI_DATATYPE_NULL) {
                MPI_Type_free(&type);
                type = MPI_DATATYPE_NULL;
            }
        }

        types_initialized = false;
    }

    MPI_Datatype
    Transpose::build_block_type(const DynGridIndex & local_shape,
                                const DynGridIndex & block_shape,
                                const DynGridIndex & block_start) const {
        Dim_t dim{local_shape.get_dim()};
        MPI_Datatype result;

        // MPI_Type_create_subarray requires all subsizes >= 1. If the block
        // is empty (uneven distribution where some rank holds no elements of
        // a dimension), return an empty type instead.
        for (Dim_t d{0}; d < dim; ++d) {
            if (block_shape[d] == 0) {
                MPI_Type_contiguous(0, mpi_type<Complex>(), &result);
                MPI_Type_commit(&result);
                return result;
            }
        }

        // For AoS layout, we first create an element type for n_comp complex
        // values For SoA layout, we create a spatial type and replicate across
        // components
        if (this->layout == StorageOrder::ArrayOfStructures) {
            // AoS: [c0, c1, c2, c0, c1, c2, ...] - components interleaved
            // Create element type: nb_components complex values
            MPI_Datatype element_type;
            MPI_Type_contiguous(static_cast<int>(this->nb_components),
                                mpi_type<Complex>(), &element_type);
            MPI_Type_commit(&element_type);

            if (dim == 2) {
                // 2D subarray: [local_shape[0], local_shape[1]] elements
                // Extract block at [block_start[0], block_start[1]] with size
                // [block_shape[0], block_shape[1]]

                // In column-major (Fortran) order:
                // - Stride along dim 0 is 1 element
                // - Stride along dim 1 is local_shape[0] elements

                // Build using MPI_Type_create_subarray
                int sizes[2] = {static_cast<int>(local_shape[0]),
                                static_cast<int>(local_shape[1])};
                int subsizes[2] = {static_cast<int>(block_shape[0]),
                                   static_cast<int>(block_shape[1])};
                int starts[2] = {static_cast<int>(block_start[0]),
                                 static_cast<int>(block_start[1])};

                MPI_Type_create_subarray(2, sizes, subsizes, starts,
                                         MPI_ORDER_FORTRAN, element_type,
                                         &result);
            } else if (dim == 3) {
                // 3D subarray
                int sizes[3] = {static_cast<int>(local_shape[0]),
                                static_cast<int>(local_shape[1]),
                                static_cast<int>(local_shape[2])};
                int subsizes[3] = {static_cast<int>(block_shape[0]),
                                   static_cast<int>(block_shape[1]),
                                   static_cast<int>(block_shape[2])};
                int starts[3] = {static_cast<int>(block_start[0]),
                                 static_cast<int>(block_start[1]),
                                 static_cast<int>(block_start[2])};

                MPI_Type_create_subarray(3, sizes, subsizes, starts,
                                         MPI_ORDER_FORTRAN, element_type,
                                         &result);
            } else {
                MPI_Type_free(&element_type);
                throw RuntimeError("Transpose only supports 2D and 3D");
            }

            MPI_Type_commit(&result);
            MPI_Type_free(&element_type);
        } else {
            // SoA: [x0, x1, ..., y0, y1, ..., z0, z1, ...]
            // Create spatial type for single component, then replicate

            MPI_Datatype spatial_type;

            if (dim == 2) {
                int sizes[2] = {static_cast<int>(local_shape[0]),
                                static_cast<int>(local_shape[1])};
                int subsizes[2] = {static_cast<int>(block_shape[0]),
                                   static_cast<int>(block_shape[1])};
                int starts[2] = {static_cast<int>(block_start[0]),
                                 static_cast<int>(block_start[1])};

                MPI_Type_create_subarray(2, sizes, subsizes, starts,
                                         MPI_ORDER_FORTRAN, mpi_type<Complex>(),
                                         &spatial_type);
            } else if (dim == 3) {
                int sizes[3] = {static_cast<int>(local_shape[0]),
                                static_cast<int>(local_shape[1]),
                                static_cast<int>(local_shape[2])};
                int subsizes[3] = {static_cast<int>(block_shape[0]),
                                   static_cast<int>(block_shape[1]),
                                   static_cast<int>(block_shape[2])};
                int starts[3] = {static_cast<int>(block_start[0]),
                                 static_cast<int>(block_start[1]),
                                 static_cast<int>(block_start[2])};

                MPI_Type_create_subarray(3, sizes, subsizes, starts,
                                         MPI_ORDER_FORTRAN, mpi_type<Complex>(),
                                         &spatial_type);
            } else {
                throw RuntimeError("Transpose only supports 2D and 3D");
            }
            MPI_Type_commit(&spatial_type);

            // Replicate across components with grid-sized stride
            Index_t grid_size{1};
            for (Dim_t d{0}; d < dim; ++d) {
                grid_size *= local_shape[d];
            }
            MPI_Aint comp_stride{static_cast<MPI_Aint>(grid_size) *
                                 static_cast<MPI_Aint>(sizeof(Complex))};

            MPI_Type_create_hvector(static_cast<int>(this->nb_components), 1,
                                    comp_stride, spatial_type, &result);
            MPI_Type_commit(&result);
            MPI_Type_free(&spatial_type);
        }

        return result;
    }

    void Transpose::init_forward_types() {
        int comm_size{this->comm.size()};
        Dim_t dim{this->local_in.get_dim()};

        // Scatter-gather transpose: every rank sends a disjoint block of its
        // input to each peer and receives a disjoint block of its output from
        // each peer.
        for (int r{0}; r < comm_size; ++r) {
            // For forward transpose:
            // - We send a block from our input to rank r
            // - The block covers out_counts[r] elements along axis_out
            //   (the dimension that becomes distributed after transpose)
            // - And all of our local elements along axis_in
            //   (the dimension that is distributed in input)

            // Build send type: extract block from input
            DynGridIndex send_block_shape(dim);
            DynGridIndex send_block_start(dim);

            for (Dim_t d{0}; d < dim; ++d) {
                if (d == this->axis_out) {
                    // This dimension gets distributed after transpose
                    // Send only the portion that will belong to rank r
                    send_block_shape[d] = this->out_counts[r];
                    send_block_start[d] = this->out_displs[r];
                } else {
                    // Other dimensions: send all our local data
                    send_block_shape[d] = this->local_in[d];
                    send_block_start[d] = 0;
                }
            }

            this->send_types_fwd[r] = build_block_type(
                this->local_in, send_block_shape, send_block_start);

            // Build recv type: place block into output
            DynGridIndex recv_block_shape(dim);
            DynGridIndex recv_block_start(dim);

            for (Dim_t d{0}; d < dim; ++d) {
                if (d == this->axis_in) {
                    // This dimension becomes local after transpose
                    // Receive the portion that rank r owned before
                    recv_block_shape[d] = this->in_counts[r];
                    recv_block_start[d] = this->in_displs[r];
                } else {
                    // Other dimensions: receive all data
                    recv_block_shape[d] = this->local_out[d];
                    recv_block_start[d] = 0;
                }
            }

            this->recv_types_fwd[r] = build_block_type(
                this->local_out, recv_block_shape, recv_block_start);
        }
    }

    void Transpose::init_backward_types() {
        int comm_size{this->comm.size()};
        Dim_t dim{this->local_out.get_dim()};

        // Backward is the exact reverse of forward: send from the output
        // layout back to the input layout.
        for (int r{0}; r < comm_size; ++r) {
            // Build send type: extract block from output (backward input)
            DynGridIndex send_block_shape(dim);
            DynGridIndex send_block_start(dim);

            for (Dim_t d{0}; d < dim; ++d) {
                if (d == this->axis_in) {
                    // In backward, axis_in is now local (was distributed)
                    // Send the portion that will belong to rank r
                    send_block_shape[d] = this->in_counts[r];
                    send_block_start[d] = this->in_displs[r];
                } else {
                    send_block_shape[d] = this->local_out[d];
                    send_block_start[d] = 0;
                }
            }

            this->send_types_bwd[r] = build_block_type(
                this->local_out, send_block_shape, send_block_start);

            // Build recv type: place block into input (backward output)
            DynGridIndex recv_block_shape(dim);
            DynGridIndex recv_block_start(dim);

            for (Dim_t d{0}; d < dim; ++d) {
                if (d == this->axis_out) {
                    // In backward, axis_out becomes local again
                    recv_block_shape[d] = this->out_counts[r];
                    recv_block_start[d] = this->out_displs[r];
                } else {
                    recv_block_shape[d] = this->local_in[d];
                    recv_block_start[d] = 0;
                }
            }

            this->recv_types_bwd[r] = build_block_type(
                this->local_in, recv_block_shape, recv_block_start);
        }
    }

    namespace {
        //! Device-to-device strided 2D copy (pack/unpack building block)
        void strided_device_copy(void * dst, const void * src,
                                 std::size_t width_bytes, std::size_t nb_rows,
                                 std::size_t dst_pitch_bytes,
                                 std::size_t src_pitch_bytes) {
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
            GPU_MEMCPY_2D_D2D(dst, dst_pitch_bytes, src, src_pitch_bytes,
                              width_bytes, nb_rows);
#else
            (void)dst;
            (void)src;
            (void)width_bytes;
            (void)nb_rows;
            (void)dst_pitch_bytes;
            (void)src_pitch_bytes;
            throw RuntimeError("Transpose: device data requires a GPU build");
#endif
        }
    }  // namespace

    char * Transpose::get_device_staging(std::size_t slot,
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

    void Transpose::staged_alltoall(
        const Complex * input, Complex * output, const DynGridIndex & src_shape,
        Index_t src_axis, const std::vector<Index_t> & src_counts,
        const std::vector<Index_t> & src_displs, const DynGridIndex & dst_shape,
        Index_t dst_axis, const std::vector<Index_t> & dst_counts,
        const std::vector<Index_t> & dst_displs) const {
        int comm_size{this->comm.size()};

        // Geometry of a column-major subarray block that is full in all
        // dimensions except `axis`: the block occupies contiguous runs of
        // pre*count elements (pre = product of extents below `axis`),
        // repeated post times (post = product of extents above `axis`) with
        // a uniform stride of pre*extent(axis). Serializing it row by row
        // reproduces exactly the canonical (column-major) subarray order
        // that MPI_Type_create_subarray uses, so the staged exchange is
        // wire-compatible with the datatype-based host path.
        auto pre_post{[](const DynGridIndex & shape, Index_t axis) {
            std::size_t pre{1}, post{1};
            for (Dim_t d{0}; d < shape.get_dim(); ++d) {
                if (d < axis) {
                    pre *= shape[d];
                } else if (d > axis) {
                    post *= shape[d];
                }
            }
            return std::make_pair(pre, post);
        }};
        auto [src_pre, src_post] = pre_post(src_shape, src_axis);
        auto [dst_pre, dst_post] = pre_post(dst_shape, dst_axis);
        std::size_t src_spatial{src_pre * src_shape[src_axis] * src_post};
        std::size_t dst_spatial{dst_pre * dst_shape[dst_axis] * dst_post};
        auto ncomp{static_cast<std::size_t>(this->nb_components)};
        bool aos{this->layout == StorageOrder::ArrayOfStructures};

        // Per-peer element counts (in units of Complex) and displacements
        std::vector<int> send_counts_el(comm_size), send_displs_el(comm_size);
        std::vector<int> recv_counts_el(comm_size), recv_displs_el(comm_size);
        std::size_t send_total{0}, recv_total{0};
        for (int r{0}; r < comm_size; ++r) {
            std::size_t s{src_pre * src_counts[r] * src_post * ncomp};
            std::size_t d{dst_pre * dst_counts[r] * dst_post * ncomp};
            send_counts_el[r] = static_cast<int>(s);
            send_displs_el[r] = static_cast<int>(send_total);
            recv_counts_el[r] = static_cast<int>(d);
            recv_displs_el[r] = static_cast<int>(recv_total);
            send_total += s;
            recv_total += d;
        }

        char * send_staging{
            this->get_device_staging(0, send_total * sizeof(Complex))};
        char * recv_staging{
            this->get_device_staging(1, recv_total * sizeof(Complex))};

        // Without GPU-aware MPI, the flat exchange below operates on host
        // bounce buffers instead of the device staging (correct with any
        // MPI library); the pack/unpack stays on the device either way.
        const bool bounce{!mpi_is_gpu_aware()};
        char * send_buffer{send_staging};
        char * recv_buffer{recv_staging};
        if (bounce) {
            auto & host_send{this->host_staging[0]};
            auto & host_recv{this->host_staging[1]};
            if (host_send.size() < send_total * sizeof(Complex)) {
                host_send.resize(send_total * sizeof(Complex));
            }
            if (host_recv.size() < recv_total * sizeof(Complex)) {
                host_recv.resize(recv_total * sizeof(Complex));
            }
            send_buffer = host_send.data();
            recv_buffer = host_recv.data();
        }

        // Pack: gather each peer's block into the contiguous send buffer
        for (int r{0}; r < comm_size; ++r) {
            if (src_counts[r] == 0) {
                continue;
            }
            auto * staging{send_staging +
                           static_cast<std::size_t>(send_displs_el[r]) *
                               sizeof(Complex)};
            if (aos) {
                std::size_t width{src_pre * src_counts[r] * ncomp *
                                  sizeof(Complex)};
                std::size_t pitch{src_pre * src_shape[src_axis] * ncomp *
                                  sizeof(Complex)};
                const Complex * field{input +
                                      src_displs[r] * src_pre * ncomp};
                strided_device_copy(staging, field, width, src_post, width,
                                    pitch);
            } else {
                std::size_t width{src_pre * src_counts[r] * sizeof(Complex)};
                std::size_t pitch{src_pre * src_shape[src_axis] *
                                  sizeof(Complex)};
                std::size_t comp_block{src_pre * src_counts[r] * src_post};
                for (std::size_t c{0}; c < ncomp; ++c) {
                    const Complex * field{input + c * src_spatial +
                                          src_displs[r] * src_pre};
                    strided_device_copy(staging +
                                            c * comp_block * sizeof(Complex),
                                        field, width, src_post, width, pitch);
                }
            }
        }
#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        // Device-to-device 2D copies are asynchronous with respect to the
        // host; MPI must not read the staging buffer before the gather is
        // complete.
        GPU_DEVICE_SYNCHRONIZE();
        if (bounce && send_total > 0) {
            GPU_MEMCPY_D2H(send_buffer, send_staging,
                           send_total * sizeof(Complex));
        }
#endif
        // Pairwise ring exchange instead of MPI_Alltoallv: collective
        // components are not reliably GPU-aware, while point-to-point on
        // contiguous device buffers is the well-trodden CUDA/ROCm-aware
        // path. At step s, rank r sends to r+s and receives from r-s; the
        // partners match up at the same step on both sides, so the
        // schedule is deadlock-free for any communicator size. (Sending
        // to and receiving from the SAME peer each step deadlocks for
        // size > 2: everyone would wait on a partner that sends to them
        // only at a later step.)
        {
            MPI_Comm mpi_comm{this->comm.get_mpi_comm()};
            int my_rank{this->comm.rank()};
            for (int step{0}; step < comm_size; ++step) {
                if (step == 0) {
                    auto bytes{static_cast<std::size_t>(
                                   send_counts_el[my_rank]) *
                               sizeof(Complex)};
                    if (bytes > 0) {
                        // The self block never leaves the device
                        strided_device_copy(
                            recv_staging + static_cast<std::size_t>(
                                               recv_displs_el[my_rank]) *
                                               sizeof(Complex),
                            send_staging + static_cast<std::size_t>(
                                               send_displs_el[my_rank]) *
                                               sizeof(Complex),
                            bytes, 1, bytes, bytes);
                    }
                    continue;
                }
                int send_peer{(my_rank + step) % comm_size};
                int recv_peer{(my_rank - step + comm_size) % comm_size};
                MPI_Status status;
                MPI_Sendrecv(send_buffer +
                                 static_cast<std::size_t>(
                                     send_displs_el[send_peer]) *
                                     sizeof(Complex),
                             send_counts_el[send_peer], mpi_type<Complex>(),
                             send_peer, 0,
                             recv_buffer +
                                 static_cast<std::size_t>(
                                     recv_displs_el[recv_peer]) *
                                     sizeof(Complex),
                             recv_counts_el[recv_peer], mpi_type<Complex>(),
                             recv_peer, 0, mpi_comm, &status);
            }
        }

#if defined(MUGRID_ENABLE_CUDA) || defined(MUGRID_ENABLE_HIP)
        if (bounce && recv_total > 0) {
            // The self block (still on the device) occupies its slice of
            // recv_staging; copying the bounced host data must not clobber
            // it, so copy the two surrounding extents.
            auto self_begin{static_cast<std::size_t>(
                                recv_displs_el[this->comm.rank()]) *
                            sizeof(Complex)};
            auto self_end{self_begin +
                          static_cast<std::size_t>(
                              recv_counts_el[this->comm.rank()]) *
                              sizeof(Complex)};
            if (self_begin > 0) {
                GPU_MEMCPY_H2D(recv_staging, recv_buffer, self_begin);
            }
            auto total_bytes{recv_total * sizeof(Complex)};
            if (self_end < total_bytes) {
                GPU_MEMCPY_H2D(recv_staging + self_end,
                               recv_buffer + self_end,
                               total_bytes - self_end);
            }
        }
#endif
        // Unpack: scatter each peer's block from the contiguous receive
        // buffer into the output field. Runs on the default stream, so
        // subsequent kernels are ordered after it.
        for (int r{0}; r < comm_size; ++r) {
            if (dst_counts[r] == 0) {
                continue;
            }
            auto * staging{recv_staging +
                           static_cast<std::size_t>(recv_displs_el[r]) *
                               sizeof(Complex)};
            if (aos) {
                std::size_t width{dst_pre * dst_counts[r] * ncomp *
                                  sizeof(Complex)};
                std::size_t pitch{dst_pre * dst_shape[dst_axis] * ncomp *
                                  sizeof(Complex)};
                Complex * field{output + dst_displs[r] * dst_pre * ncomp};
                strided_device_copy(field, staging, width, dst_post, pitch,
                                    width);
            } else {
                std::size_t width{dst_pre * dst_counts[r] * sizeof(Complex)};
                std::size_t pitch{dst_pre * dst_shape[dst_axis] *
                                  sizeof(Complex)};
                std::size_t comp_block{dst_pre * dst_counts[r] * dst_post};
                for (std::size_t c{0}; c < ncomp; ++c) {
                    Complex * field{output + c * dst_spatial +
                                    dst_displs[r] * dst_pre};
                    strided_device_copy(field,
                                        staging +
                                            c * comp_block * sizeof(Complex),
                                        width, dst_post, pitch, width);
                }
            }
        }
    }
#endif  // WITH_MPI

    void Transpose::forward(const Complex * input, Complex * output) const {
#ifdef WITH_MPI
        MPI_Comm mpi_comm = this->comm.get_mpi_comm();

        if (mpi_comm == MPI_COMM_NULL || this->comm.size() == 1) {
            // Serial case: direct copy (with reordering if needed)
            // For serial, input and output have the same total size
            Index_t total_size{1};
            for (Dim_t d{0}; d < this->local_in.get_dim(); ++d) {
                total_size *= this->local_in[d];
            }
            total_size *= this->nb_components;
            std::copy(input, input + total_size, output);
            return;
        }

        if (this->on_device) {
            // Contiguous staging instead of derived datatypes on device
            // pointers (see constructor documentation)
            this->staged_alltoall(input, output, this->local_in,
                                  this->axis_out, this->out_counts,
                                  this->out_displs, this->local_out,
                                  this->axis_in, this->in_counts,
                                  this->in_displs);
            return;
        }

        // Use MPI_Alltoallw with derived datatypes
        MPI_Alltoallw(input, this->send_counts.data(), this->send_displs.data(),
                      this->send_types_fwd.data(), output,
                      this->recv_counts.data(), this->recv_displs.data(),
                      this->recv_types_fwd.data(), mpi_comm);

#else   // WITH_MPI
        // Serial case: direct copy
        Index_t total_size{1};
        for (Dim_t d{0}; d < this->local_in.get_dim(); ++d) {
            total_size *= this->local_in[d];
        }
        total_size *= this->nb_components;
        std::copy(input, input + total_size, output);
#endif  // WITH_MPI
    }

    void Transpose::backward(const Complex * input, Complex * output) const {
#ifdef WITH_MPI
        MPI_Comm mpi_comm = this->comm.get_mpi_comm();

        if (mpi_comm == MPI_COMM_NULL || this->comm.size() == 1) {
            // Serial case: direct copy
            Index_t total_size{1};
            for (Dim_t d{0}; d < this->local_out.get_dim(); ++d) {
                total_size *= this->local_out[d];
            }
            total_size *= this->nb_components;
            std::copy(input, input + total_size, output);
            return;
        }

        if (this->on_device) {
            // Contiguous staging instead of derived datatypes on device
            // pointers (see constructor documentation)
            this->staged_alltoall(input, output, this->local_out,
                                  this->axis_in, this->in_counts,
                                  this->in_displs, this->local_in,
                                  this->axis_out, this->out_counts,
                                  this->out_displs);
            return;
        }

        // Use MPI_Alltoallw with derived datatypes
        MPI_Alltoallw(input, this->send_counts.data(), this->send_displs.data(),
                      this->send_types_bwd.data(), output,
                      this->recv_counts.data(), this->recv_displs.data(),
                      this->recv_types_bwd.data(), mpi_comm);

#else   // WITH_MPI
        // Serial case: direct copy
        Index_t total_size{1};
        for (Dim_t d{0}; d < this->local_out.get_dim(); ++d) {
            total_size *= this->local_out[d];
        }
        total_size *= this->nb_components;
        std::copy(input, input + total_size, output);
#endif  // WITH_MPI
    }

}  // namespace muGrid

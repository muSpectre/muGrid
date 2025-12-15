#ifdef WITH_MPI
#include <mpi.h>
#endif

#include "grid_common.hh"
#include "communicator.hh"
#include "cartesian_communicator.hh"

namespace muGrid {
#ifdef WITH_MPI
    CartesianCommunicator::CartesianCommunicator(
        const Parent_t & parent, const IntCoord_t & nb_subdivisions)
        : Parent_t{parent.get_mpi_comm()}, parent{parent},
          nb_subdivisions{nb_subdivisions},
          coordinates(nb_subdivisions.size(), -1),
          left_ranks(nb_subdivisions.size(), MPI_PROC_NULL),
          right_ranks(nb_subdivisions.size(), MPI_PROC_NULL) {
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
        const Parent_t & parent, const IntCoord_t & nb_subdivisions,
        const IntCoord_t & coordinates, const std::vector<int> & left_ranks,
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
        int elem_size_in_bytes, void * elem_mpi_t) const {
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
        int elem_size_in_bytes, void * elem_mpi_t) const {
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
#else   // not WITH_MPI
    CartesianCommunicator::CartesianCommunicator(
        const Parent_t & parent, const IntCoord_t & nb_subdivisions)
        : Parent_t{}, parent{parent}, nb_subdivisions{nb_subdivisions},
          coordinates(nb_subdivisions.size(), 0) {}

    void CartesianCommunicator::sendrecv_right(
        int direction, int block_stride, int nb_send_blocks, int send_block_len,
        Index_t send_offset, int nb_recv_blocks, int recv_block_len,
        Index_t recv_offset, char * data, int stride_in_direction,
        int elem_size_in_bytes, void * elem_mpi_t) const {
        // Note: elem_mpi_t is not used in serial mode (ignored parameter)
        (void)elem_mpi_t;  // Suppress unused parameter warning
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
            std::memcpy(recv_addr, send_addr, send_block_len * elem_size_in_bytes);
            data += block_stride * elem_size_in_bytes;
        }
    }

    void CartesianCommunicator::sendrecv_left(
        int direction, int block_stride, int nb_send_blocks, int send_block_len,
        Index_t send_offset, int nb_recv_blocks, int recv_block_len,
        Index_t recv_offset, char * data, int stride_in_direction,
        int elem_size_in_bytes, void * elem_mpi_t) const {
        // Note: elem_mpi_t is not used in serial mode (ignored parameter)
        (void)elem_mpi_t;  // Suppress unused parameter warning
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
            std::memcpy(recv_addr, send_addr, send_block_len * elem_size_in_bytes);
            data += block_stride * elem_size_in_bytes;
        }
    }
#endif  // WITH_MPI

    const IntCoord_t & CartesianCommunicator::get_nb_subdivisions() const {
        return this->nb_subdivisions;
    }

    const IntCoord_t & CartesianCommunicator::get_coordinates() const {
        return this->coordinates;
    }
}  // namespace muGrid

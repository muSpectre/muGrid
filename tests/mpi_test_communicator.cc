/**
 * @file   mpi_test_communicator.cc
 *
 * @author Richard Leute <richard.leute@imtek.uni-freiburg.de>
 *
 * @date   02 Dez 2021
 *
 * @brief  testing the communicator functions
 *
 * Copyright © 2021 Till Junge
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

#include "mpi_context.hh"
#include "tests.hh"

#include "libmugrid/field_typed.hh"
#include "libmugrid/field_map.hh"
#include "libmugrid/ccoord_operations.hh"
#include "libmugrid/communicator.hh"
#include "libmugrid/cartesian_decomposition.hh"

namespace muGrid {
  BOOST_AUTO_TEST_SUITE(mpi_communicator_test);

  // ----------------------------------------------------------------------
  BOOST_AUTO_TEST_CASE(sum_test) {
    auto & comm{MPIContext::get_context().comm};
    auto nb_cores{comm.size()};
    Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> send_mat(2, 3);
    send_mat(0, 0) = 1.;
    send_mat(0, 1) = 2.;
    send_mat(0, 2) = 3.;
    send_mat(1, 0) = 4.;
    send_mat(1, 1) = 5.;
    send_mat(1, 2) = 6.;
    auto res{comm.template sum<Real>(send_mat)};
    const auto nb_cols{send_mat.cols()};
    const auto nb_rows{send_mat.rows()};
    for (int row{0}; row < nb_rows; row++) {
      for (int col{0}; col < nb_cols; col++) {
        BOOST_CHECK_EQUAL(res(row, col), (row * nb_cols + col + 1) * nb_cores);
      }
    }
  }

  // ----------------------------------------------------------------------
  BOOST_AUTO_TEST_CASE(cumulative_sum_test) {
    auto & comm{MPIContext::get_context().comm};
    auto rank{comm.rank()};

    // Int values
    Int send_val_int{rank + 1};
    Int res_int{comm.template cumulative_sum<Int>(send_val_int)};
    // 1 + 2 + 3 + ... + n = n*(n+1)/2
    BOOST_CHECK_EQUAL(res_int, rank * (rank + 1) / 2 + rank + 1);

    // Index_t values
    Index_t send_val_ind{rank + 1};
    Index_t res_ind{comm.template cumulative_sum<Index_t>(send_val_ind)};
    // 1 + 2 + 3 + ... + n = n*(n+1)/2
    BOOST_CHECK_EQUAL(res_ind, rank * (rank + 1) / 2 + rank + 1);

    // Real values
    Real send_val_real{static_cast<Real>(rank + 1)};
    Real res_real{comm.template cumulative_sum<Real>(send_val_real)};
    // 1 + 2 + 3 + ... + n = n*(n+1)/2
    BOOST_CHECK_EQUAL(res_real, rank * (rank + 1) / 2 + rank + 1);
  }

  // ----------------------------------------------------------------------
  BOOST_AUTO_TEST_CASE(bcast_test) {
    auto & comm{MPIContext::get_context().comm};
    auto nb_cores{comm.size()};
    Int arg = comm.rank();
    // Check bcast from rank 0
    if (comm.rank() == 0) {
      arg = 2;
    }
    Int res = comm.template bcast<Int>(arg, 0);
    BOOST_CHECK_EQUAL(res, 2);

    // Check bcast from rank = size - 1
    if (nb_cores > 1) {
      if (comm.rank() == nb_cores - 1) {
        arg = 8;
      }
      Int res = comm.template bcast<Int>(arg, nb_cores - 1);
      BOOST_CHECK_EQUAL(res, 8);
    }
  }

  // ----------------------------------------------------------------------
  BOOST_AUTO_TEST_CASE(gather_test) {
    auto & comm{MPIContext::get_context().comm};
    auto rank{comm.rank()};
    const int columns_offset_from_rank{2};
    const Dim_t nb_cols{rank + columns_offset_from_rank};
    const Dim_t nb_rows{2};

    int counter{0};  // for loop internal counting
    Int nb_rows_init{0};
    Int nb_cols_init{0};

    // --- full processors ---
    Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> send_mat(nb_rows,
                                                                 nb_cols);

    for (int col{0}; col < nb_cols; ++col) {
      send_mat.col(col) = send_mat.col(col).Ones(nb_rows, 1) * (rank + col);
    }

    auto res{comm.template gather<Real>(send_mat)};
    counter = 0;
    for (int lrank{0}; lrank < comm.size(); ++lrank) {
      for (int col{0}; col < lrank + columns_offset_from_rank; ++col) {
        BOOST_CHECK_EQUAL(res(0, counter), lrank + col);
        BOOST_CHECK_EQUAL(res(1, counter), lrank + col);
        counter++;
      }
    }

    // --- empty processor at end (last rank) ---
    nb_rows_init = (rank == comm.size() - 1 and comm.size() > 1) ? 0 : nb_rows;
    nb_cols_init = (rank == comm.size() - 1 and comm.size() > 1) ? 0 : nb_cols;
    Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> send_mat_lr(
        nb_rows_init, nb_cols_init);

    for (int col{0}; col < nb_cols; ++col) {
      if (rank == comm.size() - 1 and comm.size() > 1) {
        // you do not have to fill the matrix of size 0x0 on the last rank
        break;
      }
      send_mat_lr.col(col) =
          send_mat_lr.col(col).Ones(nb_rows, 1) * (rank + col);
    }

    auto res_lr{comm.template gather<Real>(send_mat_lr)};
    // check size of gathered matrix
    auto nb_cols_of_res{res_lr.cols()};
    BOOST_CHECK_EQUAL(nb_cols_of_res, comm.size() > 1
                                          ? comm.sum(nb_cols) - comm.size() -
                                                columns_offset_from_rank + 1
                                          : comm.sum(nb_cols));
    auto nb_rows_of_res{res_lr.rows()};
    BOOST_CHECK_EQUAL(nb_rows_of_res, nb_rows);

    // check values of gathered matrix
    counter = 0;
    for (int lrank{0}; lrank < comm.size(); ++lrank) {
      for (int col{0}; col < lrank + columns_offset_from_rank; ++col) {
        if (lrank != comm.size() - 1 or comm.size() == 1) {
          BOOST_CHECK_EQUAL(res_lr(0, counter), lrank + col);
          BOOST_CHECK_EQUAL(res_lr(1, counter), lrank + col);
          counter++;
        } else {
          // there should be no more entry in res_lr
        }
      }
    }

    // --- empty processor in between (rank = comm.size()//2) ---
    nb_rows_init =
        (rank == std::floor(comm.size() / 2) and comm.size() > 1) ? 0 : nb_rows;
    nb_cols_init =
        (rank == std::floor(comm.size() / 2) and comm.size() > 1) ? 0 : nb_cols;
    Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> send_mat_em(
        nb_rows_init, nb_cols_init);

    for (int col{0}; col < nb_cols; ++col) {
      if (rank == std::floor(comm.size() / 2) and comm.size() > 1) {
        // you do not have to fill the matrix of size 0x0 on the middle rank
        break;
      }
      send_mat_em.col(col) =
          send_mat_em.col(col).Ones(nb_rows, 1) * (rank + col);
    }

    auto res_em{comm.template gather<Real>(send_mat_em)};
    // check size of gathered matrix
    auto nb_cols_of_res_em{res_em.cols()};
    auto nb_cols_of_res_em_expected{
        comm.size() > 1 ? comm.sum(nb_cols) - std::floor(comm.size() / 2) -
                              columns_offset_from_rank
                        : comm.sum(nb_cols)};
    BOOST_CHECK_EQUAL(nb_cols_of_res_em, nb_cols_of_res_em_expected);
    auto nb_rows_of_res_em{res_em.rows()};
    BOOST_CHECK_EQUAL(nb_rows_of_res_em, nb_rows);

    // check values of gathered matrix
    counter = 0;
    for (int lrank{0}; lrank < comm.size(); ++lrank) {
      for (int col{0}; col < lrank + columns_offset_from_rank; ++col) {
        if (lrank != std::floor(comm.size() / 2) or comm.size() == 1) {
          BOOST_CHECK_EQUAL(res_em(0, counter), lrank + col);
          BOOST_CHECK_EQUAL(res_em(1, counter), lrank + col);
          counter++;
        } else {
          // there should be no more entry in res_em
        }
      }
    }

    // --- gather empty fields on all processor such that the return should also
    // be an empty matrix ---
    Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> send_mat_empty(0, 0);

    auto res_empty{comm.template gather<Real>(send_mat_empty)};
    BOOST_CHECK_EQUAL(res_empty.rows(), 0);
    BOOST_CHECK_EQUAL(res_empty.cols(), 0);

    // --- calling on row or column vectors ---
    // gather on row vector (not allowed/possible in current implementation)

    // gather on column vector
    int nb_cols_vec{rank + 2};
    Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> send_col_vec(
        1, nb_cols_vec);

    for (int col{0}; col < nb_cols_vec; col++) {
      send_col_vec(col) = rank + col;
    }

    auto res_col_vec{comm.template gather<Real>(send_col_vec)};

    counter = 0;
    int offset{0};
    for (int lrank{0}; lrank < comm.size(); ++lrank) {
      for (int col{0}; col < lrank + 2; ++col) {
        BOOST_CHECK_EQUAL(res_col_vec(col + offset), lrank + col);
        counter++;
      }
      offset += counter;
      counter = 0;  // reset counter
    }
  }

  // ----------------------------------------------------------------------
  BOOST_AUTO_TEST_CASE(cartesian_decomposition) {
    auto & comm{MPIContext::get_context().comm};

    // Decide the number of subdivisions according to number of processes.
    int nb_process{comm.size()};
    DynCcoord_t nb_subdivisions{};
    if (nb_process == 1)
      nb_subdivisions = DynCcoord_t{1};
    else if (nb_process == 2)
      nb_subdivisions = DynCcoord_t{2};
    else if (nb_process == 4)
      nb_subdivisions = DynCcoord_t{2, 2};
    else if (nb_process == 8)
      nb_subdivisions = DynCcoord_t{2, 2, 2};
    else
      throw RuntimeError("Not planned for this number of processes.");

    // Decide the size of the whole domain and some reference values
    int spatial_dims{nb_subdivisions.size()};
    int nb_grid_pts_per_dim{10};
    const DynCcoord_t & nb_domain_grid_pts{
        DynCcoord_t(spatial_dims, nb_grid_pts_per_dim)};

    // A function get referrence values.
    auto && get_ref_value{
        [nb_grid_pts_per_dim](const DynCcoord_t & global_coords) {
          Index_t val{0};
          Index_t coeff{1};
          for (int dim{0}; dim < global_coords.size(); ++dim) {
            val += coeff * global_coords[dim];
            coeff *= nb_grid_pts_per_dim;
          }
          return val;
        }};

    // Create a Cartesian decomposition with ghost buffers
    const DynCcoord_t & nb_ghosts_left{DynCcoord_t(spatial_dims, 1)};
    const DynCcoord_t & nb_ghosts_right{DynCcoord_t(spatial_dims, 2)};
    CartesianDecomposition cart_decomp{comm, nb_domain_grid_pts,
                                       nb_subdivisions, nb_ghosts_left,
                                       nb_ghosts_right};

    // Create a field inside the collection for test
    auto & collection{cart_decomp.get_collection()};
    const Index_t nb_components{1};
    const std::string field_name{"test_field"};
    auto & field{collection.real_field(field_name, nb_components)};

    // Fill the non-ghost cells of the field with some values
    auto & subdomain_locations{cart_decomp.get_subdomain_locations()};
    auto & nb_subdomain_grid_pts{cart_decomp.get_nb_subdomain_grid_pts()};
    auto && field_map{field.get_sub_pt_map(Unknown)};
    CcoordOps::DynamicPixels pixels{nb_subdomain_grid_pts};
    for (auto && pixel_id_coords : pixels.enumerate()) {
      auto && id{std::get<0>(pixel_id_coords)};
      auto && local_coords{std::get<1>(pixel_id_coords)};

      auto && left_check{local_coords - nb_ghosts_left};
      bool is_not_ghost_left{
          std::all_of(left_check.begin(), left_check.end(),
                      [](const auto & elem) { return elem >= 0; })};

      auto && right_check{local_coords + nb_ghosts_right -
                          nb_subdomain_grid_pts};
      bool is_not_ghost_right{
          std::all_of(right_check.begin(), right_check.end(),
                      [](const auto & elem) { return elem < 0; })};

      if (is_not_ghost_left && is_not_ghost_right) {
        auto && global_coords{(subdomain_locations + local_coords) %
                              nb_domain_grid_pts};
        field_map[id] << get_ref_value(global_coords);
        if (comm.rank() == 0) {
          std::cout << "id: " << id << std::endl;
          std::cout << "subdomain_locations: " << subdomain_locations
                    << std::endl;
          std::cout << "local_coords: " << local_coords << std::endl;
          std::cout << "global_coords: " << global_coords << std::endl;
          std::cout << " field_value: " << get_ref_value(global_coords)
                    << std::endl;
        }
      }
    }

    // Communicate the ghost cells
    cart_decomp.communicate_ghosts(field_name);

    // Check the values at the ghost cells are still the same
    for (auto && pixel_id_coords : pixels.enumerate()) {
      auto && id(std::get<0>(pixel_id_coords));
      auto && local_coords{std::get<1>(pixel_id_coords)};

      auto && left_check{local_coords - nb_ghosts_left};
      bool is_ghost_left{
          std::any_of(left_check.begin(), left_check.end(),
                      [](const auto & elem) { return elem < 0; })};

      auto && right_check{local_coords + nb_ghosts_right -
                          nb_subdomain_grid_pts};
      bool is_ghost_right{
          std::any_of(right_check.begin(), right_check.end(),
                      [](const auto & elem) { return elem >= 0; })};

      if (is_ghost_left || is_ghost_right) {
        auto && global_coords{(subdomain_locations + local_coords) %
                              nb_domain_grid_pts};
        if (comm.rank() == 0)
          BOOST_CHECK_EQUAL(field_map[id].coeffRef(0, 0),
                            get_ref_value(global_coords));
      }
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muGrid

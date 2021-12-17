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

#include "libmugrid/communicator.hh"

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
    Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>
        send_mat(nb_rows, nb_cols);

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

    auto res_col_vec{
        comm.template gather<Real>(send_col_vec)};

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

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muGrid

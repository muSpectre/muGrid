/**
 * @file   test_raw_memory_operations.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   29 May 2020
 *
 * @brief  test raw mem functions
 *
 * Copyright © 2020 Till Junge
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

#include "tests.hh"
#include "test_goodies.hh"

#include "libmugrid/raw_memory_operations.hh"

#include <vector>


namespace muGrid {

  BOOST_AUTO_TEST_SUITE(raw_memory_tests);

  BOOST_AUTO_TEST_CASE(replicate_numpy_integer_multiple) {
    using CMatrix =
        Eigen::Matrix<Int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using FMatrix =
        Eigen::Matrix<Int, Eigen::Dynamic, Eigen::Dynamic>;
    const Index_t nb_row{3}, nb_col{2};
    const std::vector<Index_t> logical_shape{nb_row, nb_col};
    const Index_t skip_i{2}, skip_j{2};

    const std::vector<Index_t> large_shape{skip_i * nb_row, skip_j * nb_col};
    CMatrix a(large_shape[0], large_shape[1]);
    a <<      53,  92,  59, 107,
              25,  69, 126, 112,
             121,  19,  15,  43,
              25,  82,  26,  19,
              14,  48,  71,  54,
              88,  68,  73,  12;
    std::vector<Index_t> a_strides{skip_i*large_shape[1], skip_j};
    CMatrix b(nb_row, nb_col);
      b <<    53, 59,
             121, 15,
              14, 71;
    std::vector<Index_t> b_strides{skip_j, 1};
    FMatrix b_out(3, 2);
    b_out <<  53, 59,
             121, 15,
              14, 71;
    std::vector<Index_t> b_out_strides{1, skip_j};

    CMatrix test_rowmaj_b{CMatrix::Zero(nb_row, nb_col)};
    FMatrix test_colmaj_b{FMatrix::Zero(nb_row, nb_col)};

    // copy a into an array like b (row-maj to row-maj)
    raw_mem_ops::strided_copy({nb_row, nb_col}, a_strides, b_strides, a.data(),
                              test_rowmaj_b.data());
    auto rel_error{testGoodies::rel_error(test_rowmaj_b, b)};
    if (rel_error != 0) {
      std::cout << "reference:" << std::endl
                << b << std::endl
                << "result:   " << std::endl
                << test_rowmaj_b << std::endl;
    }
    BOOST_CHECK_EQUAL(rel_error, 0);
  }

  BOOST_AUTO_TEST_CASE(replicate_numpy) {
    using CMatrix =
        Eigen::Matrix<Int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using FMatrix =
        Eigen::Matrix<Int, Eigen::Dynamic, Eigen::Dynamic>;
    const Index_t nb_row{3}, nb_col{2};
    const std::vector<Index_t> logical_shape{nb_row, nb_col};
    const Index_t skip_i{2}, skip_j{3};
    // warning, the following is a tricky case, as the skip_j is not a divisor
    // of the large array's nb of cols
    const std::vector<Index_t> large_shape{skip_i * nb_row, 2 * nb_col};
    CMatrix a(large_shape[0], large_shape[1]);
    a <<      53,  92,  59, 107,
              25,  69, 126, 112,
             121,  19,  15,  43,
              25,  82,  26,  19,
              14,  48,  71,  54,
              88,  68,  73,  12;
    std::vector<Index_t> a_strides{skip_i*large_shape[1], skip_j};
    CMatrix b(nb_row, nb_col);
      b <<    53, 107,
             121,  43,
              14,  54;
    std::vector<Index_t> b_strides{skip_i, 1};
    FMatrix b_out(3, 2);
    b_out <<  53, 107,
             121,  43,
              14,  54;
    std::vector<Index_t> b_out_strides{1, skip_j};

    CMatrix test_rowmaj_b{CMatrix::Zero(nb_row, nb_col)};
    FMatrix test_colmaj_b{FMatrix::Zero(nb_row, nb_col)};

    // copy a into an array like b (row-maj to row-maj)
    raw_mem_ops::strided_copy({nb_row, nb_col}, a_strides, b_strides, a.data(),
                              test_rowmaj_b.data());
    auto rel_error{testGoodies::rel_error(test_rowmaj_b, b)};
    BOOST_CHECK_EQUAL(rel_error, 0);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muGrid

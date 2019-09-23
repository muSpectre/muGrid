/**
 * @file   header_test_eigen_tools.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   07 Mar 2018
 *
 * @brief  test the eigen_tools
 *
 * Copyright © 2018 Till Junge
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
 * * Boston, MA 02111-1307, USA.
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
#include <libmugrid/eigen_tools.hh>
#include <iostream>

namespace muGrid {

  BOOST_AUTO_TEST_SUITE(eigen_tools);

  BOOST_AUTO_TEST_CASE(exponential_test) {
    using Mat_t = Eigen::Matrix<Real, threeD, threeD>;
    Mat_t input{};
    input << 0, .25 * pi, 0, .25 * pi, 0, 0, 0, 0, 1;
    Mat_t output{};

    output << 1.32460909, 0.86867096, 0, 0.86867096, 1.32460909, 0, 0, 0,
        2.71828183;
    auto my_output{expm(input)};
    Real error{(my_output - output).norm()};

    BOOST_CHECK_LT(error, 1e-8);
    if (error >= 1e-8) {
      std::cout << "input:" << std::endl << input << std::endl;
      std::cout << "output:" << std::endl << output << std::endl;
      std::cout << "my_output:" << std::endl << my_output << std::endl;
    }
  }

  BOOST_AUTO_TEST_CASE(log_m_test) {
    using Mat_t = Eigen::Matrix<Real, threeD, threeD>;
    Mat_t input{};
    constexpr Real log_tol{1e-8};
    input << 1.32460909, 0.86867096, 0, 0.86867096, 1.32460909, 0, 0, 0,
        2.71828183;
    Mat_t output{};

    output << 0, .25 * pi, 0, .25 * pi, 0, 0, 0, 0, 1;
    auto my_output{logm(input)};
    Real error{(my_output - output).norm() / output.norm()};

    BOOST_CHECK_LT(error, log_tol);
    if (error >= log_tol) {
      std::cout << "input:" << std::endl << input << std::endl;
      std::cout << "output:" << std::endl << output << std::endl;
      std::cout << "my_output:" << std::endl << my_output << std::endl;
    }

    input << 1.0001000000000002, 0.010000000000000116, 0, 0.010000000000000061,
        1.0000000000000002, 0, 0, 0, 1;

    // from scipy.linalg.logm
    output << 4.99991667e-05, 9.99983334e-03, 0.00000000e+00, 9.99983334e-03,
        -4.99991667e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00;

    my_output = logm(input);
    error = (my_output - output).norm() / output.norm();

    BOOST_CHECK_LT(error, log_tol);
    if (error >= log_tol) {
      std::cout << "input:" << std::endl << input << std::endl;
      std::cout << "output:" << std::endl << output << std::endl;
      std::cout << "my_output:" << std::endl << my_output << std::endl;
    }

    input << 1.0001000000000002, 0.010000000000000116, 0, 0.010000000000000061,
        1.0000000000000002, 0, 0, 0, 1;
    input = input.transpose().eval();

    // from scipy.linalg.logm
    output << 4.99991667e-05, 9.99983334e-03, 0.00000000e+00, 9.99983334e-03,
        -4.99991667e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00;

    my_output = logm(input);

    error = (my_output - output).norm() / output.norm();

    BOOST_CHECK_LT(error, log_tol);
    if (error >= log_tol) {
      std::cout << "input:" << std::endl << input << std::endl;
      std::cout << "output:" << std::endl << output << std::endl;
      std::cout << "my_output:" << std::endl << my_output << std::endl;
    }

    Mat_t my_output_alt{logm_alt(input)};

    error = (my_output_alt - output).norm() / output.norm();

    BOOST_CHECK_LT(error, log_tol);
    if (error >= log_tol) {
      std::cout << "input:" << std::endl << input << std::endl;
      std::cout << "output:" << std::endl << output << std::endl;
      std::cout << "my_output:" << std::endl << my_output_alt << std::endl;
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muGrid

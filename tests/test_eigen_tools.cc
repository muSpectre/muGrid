/**
 * @file   test_eigen_tools.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   07 Mar 2018
 *
 * @brief  test the eigen_tools
 *
 * Copyright © 2018 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Emacs; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#include "common/eigen_tools.hh"
#include "tests.hh"

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(eigen_tools);

  BOOST_AUTO_TEST_CASE(exponential_test) {
    using Mat_t = Eigen::Matrix<Real, threeD, threeD>;
    Mat_t input{};
    input <<      0, .25*pi, 0,
             .25*pi,      0, 0,
                  0,      0, 1;
    Mat_t output{};

    output << 1.32460909, 0.86867096, 0,
              0.86867096, 1.32460909, 0,
                       0,          0, 2.71828183;
    auto my_output{expm(input)};
    Real error{(my_output-output).norm()};

    BOOST_CHECK_LT(error, 1e-8);
    if (error >= 1e-8) {
      std::cout << "input:" << std::endl << input << std::endl;
      std::cout << "output:" << std::endl << output << std::endl;
      std::cout << "my_output:" << std::endl << my_output << std::endl;
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre

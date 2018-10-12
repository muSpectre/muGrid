/**
 * @file   test_fft_utils.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   11 Dec 2017
 *
 * @brief  test the small utility functions used by the fft engines and projections
 *
 * Copyright © 2017 Till Junge
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
 * along with µSpectre; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */
#include <boost/mpl/list.hpp>

#include "tests.hh"
#include "fft/fft_utils.hh"

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(fft_utils);

  BOOST_AUTO_TEST_CASE(fft_freqs_test) {
    //simply comparing to np.fft.fftfreq(12, 1/12.)
    const std::valarray<Real> ref{0.,  1.,  2.,  3.,  4.,  5., -6.,
        -5., -4., -3., -2., -1.};
    auto res{fft_freqs(12)};
    Real error = std::abs(res-ref).sum();
    BOOST_CHECK_EQUAL(error, 0.);
  }

  BOOST_AUTO_TEST_CASE(fft_freqs_test_length) {
    //simply comparing to np.fft.fftfreq(10)
    const std::valarray<Real> ref{ 0. ,  0.1,  0.2,  0.3,  0.4, -0.5,
        -0.4, -0.3, -0.2, -0.1};

    auto res{fft_freqs(10, 10.)};
    Real error = std::abs(res-ref).sum();
    BOOST_CHECK_EQUAL(error, 0.);
  }

  BOOST_AUTO_TEST_CASE(wave_vector_computation) {
    // here, build a FFT_freqs and check it returns the correct xi's
    constexpr Dim_t dim{twoD};
    FFT_freqs<dim> freq_struc{{12, 10}, {1., 10.}};
    Ccoord_t<dim> ccoord1{2, 3};
    auto xi{freq_struc.get_xi(ccoord1)};
    auto unit_xi{freq_struc.get_unit_xi(ccoord1)};
    typename FFT_freqs<dim>::Vector ref;
    ref << 2., .3; // from above tests
    BOOST_CHECK_LT((xi-ref).norm(), tol);
    BOOST_CHECK_LT(std::abs(xi.dot(unit_xi)-xi.norm()), xi.norm()*tol);
    BOOST_CHECK_LT(std::abs(unit_xi.norm()-1.), tol);

    ccoord1={7, 8};
    xi = freq_struc.get_xi(ccoord1);
    unit_xi = freq_struc.get_unit_xi(ccoord1);

    ref << -5., -.2;
    BOOST_CHECK_LT((xi-ref).norm(), tol);
    BOOST_CHECK_LT(std::abs(xi.dot(unit_xi)-xi.norm()), xi.norm()*tol);
    BOOST_CHECK_LT(std::abs(unit_xi.norm()-1.), tol);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre

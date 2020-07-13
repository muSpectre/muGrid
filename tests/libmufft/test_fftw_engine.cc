/**
 * @file   test_fftw_engine.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   05 Dec 2017
 *
 * @brief  tests for the fftw fft engine implementation
 *
 * Copyright © 2017 Till Junge
 *
 * µFFT is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µFFT is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µFFT; see the file COPYING. If not, write to the
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

#include <libmufft/fftw_engine.hh>
#include <libmugrid/ccoord_operations.hh>
#include <libmugrid/field_collection.hh>
#include <libmugrid/field_map_static.hh>
#include <libmugrid/iterators.hh>

#include <boost/mpl/list.hpp>
#include <iostream>

namespace muFFT {
  BOOST_AUTO_TEST_SUITE(fftw_engine);

  /* ---------------------------------------------------------------------- */
  template <Index_t DimS, Index_t DimM, Index_t NbGridPts>
  struct FFTW_fixture {
    constexpr static Index_t BoxNbGridPts{NbGridPts};
    constexpr static Real Boxlength{4.5};
    constexpr static Index_t sdim{DimS};
    constexpr static Index_t mdim{DimM};
    constexpr static Ccoord_t<sdim> res() {
      return muGrid::CcoordOps::get_cube<DimS>(BoxNbGridPts);
    }
    constexpr static Ccoord_t<sdim> loc() {
      return muGrid::CcoordOps::get_cube<DimS>(Index_t{0});
    }
    FFTW_fixture() : engine{DynCcoord_t(res())} {}
    FFTWEngine engine;
  };
  template <Index_t DimS, Index_t DimM, Index_t NbGridPts>
  constexpr Index_t FFTW_fixture<DimS, DimM, NbGridPts>::sdim;
  template <Index_t DimS, Index_t DimM, Index_t NbGridPts>
  constexpr Index_t FFTW_fixture<DimS, DimM, NbGridPts>::mdim;

  struct FFTW_fixture_python_segfault {
    constexpr static Index_t dim{twoD};
    constexpr static Index_t sdim{twoD};
    constexpr static Index_t mdim{twoD};
    constexpr static Ccoord_t<sdim> res() { return {6, 4}; }
    constexpr static Ccoord_t<sdim> loc() { return {0, 0}; }
    FFTW_fixture_python_segfault() : engine{DynCcoord_t(res())} {}
    FFTWEngine engine;
  };
  constexpr Index_t FFTW_fixture_python_segfault::sdim;
  constexpr Index_t FFTW_fixture_python_segfault::mdim;

  using fixlist = boost::mpl::list<
      FFTW_fixture<twoD, twoD, 3>, FFTW_fixture<twoD, threeD, 3>,
      FFTW_fixture<threeD, threeD, 3>, FFTW_fixture<twoD, twoD, 4>,
      FFTW_fixture<twoD, threeD, 4>, FFTW_fixture<threeD, threeD, 4>,
      FFTW_fixture_python_segfault>;

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(Constructor_test, Fix, fixlist, Fix) {
    BOOST_CHECK_NO_THROW(Fix::engine.create_plan(this->mdim * this->mdim));
    BOOST_CHECK_EQUAL(Fix::engine.size(),
                      muGrid::CcoordOps::get_size(Fix::res()));
  }

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(fft_test, Fix, fixlist, Fix) {
    Fix::engine.create_plan(this->mdim * this->mdim);
    using FC_t = muGrid::GlobalFieldCollection;
    FC_t fc(Fix::sdim);
    auto & input{fc.register_real_field("input", Fix::mdim * Fix::mdim)};
    auto & ref{fc.register_real_field("reference", Fix::mdim * Fix::mdim)};
    auto & result{fc.register_real_field("result", Fix::mdim * Fix::mdim)};
    fc.initialise(Fix::res(), Fix::loc());

    using map_t = muGrid::MatrixFieldMap<Real, Mapping::Mut, Fix::mdim,
                                         Fix::mdim, IterUnit::Pixel>;
    map_t inmap{input};
    auto refmap{map_t{ref}};
    auto resultmap{map_t{result}};
    size_t cntr{0};
    for (auto tup : akantu::zip(inmap, refmap)) {
      cntr++;
      auto & in_{std::get<0>(tup)};
      auto & ref_{std::get<1>(tup)};
      in_.setRandom();
      ref_ = in_;
    }
    auto & complex_field{Fix::engine.register_fourier_space_field(
        "fourier work space", Fix::mdim * Fix::mdim)};
    Fix::engine.fft(input, complex_field);
    using cmap_t = muGrid::MatrixFieldMap<Complex, Mapping::Mut, Fix::mdim,
                                          Fix::mdim, IterUnit::Pixel>;
    cmap_t complex_map(complex_field);
    Real error = complex_map[0].imag().norm();
    BOOST_CHECK_LT(error, tol);

    /* make sure, the engine has not modified input (which is
       unfortunately const-casted internally, hence this test) */
    for (auto && tup : akantu::zip(inmap, refmap)) {
      Real error{(std::get<0>(tup) - std::get<1>(tup)).norm()};
      BOOST_CHECK_LT(error, tol);
    }

    /* make sure that the ifft of fft returns the original*/
    Fix::engine.ifft(complex_field, result);
    for (auto && tup : akantu::zip(resultmap, refmap)) {
      Real error{
          (std::get<0>(tup) * Fix::engine.normalisation() - std::get<1>(tup))
              .norm()};
      BOOST_CHECK_LT(error, tol);
      if (error > tol) {
        std::cout << std::get<0>(tup).array() / std::get<1>(tup).array()
                  << std::endl
                  << std::endl;
      }
    }
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muFFT

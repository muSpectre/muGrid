/**
 * @file   test_serial_fft_engines.cc
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

#include <iostream>

#include <boost/mpl/list.hpp>

#include <libmugrid/ccoord_operations.hh>
#include <libmugrid/field_collection.hh>
#include <libmugrid/field_map_static.hh>
#include <libmugrid/iterators.hh>

#include <libmufft/pocketfft_engine.hh>
#ifdef WITH_FFTW
#include <libmufft/fftw_engine.hh>
#endif

#include "tests.hh"

namespace muFFT {
  BOOST_AUTO_TEST_SUITE(serial_fft_engines);

  /* ---------------------------------------------------------------------- */
  template <class Engine, Index_t DimS, Index_t DimM, Index_t NbSubPts,
            Index_t NbGridPts>
  struct FFT_fixture {
    constexpr static Index_t nb_grid_pts{NbGridPts};
    constexpr static Real Boxlength{4.5};
    constexpr static Index_t sdim{DimS};
    constexpr static Index_t mdim{DimM};
    constexpr static Index_t nb_sub_pts{NbSubPts};
    constexpr static Ccoord_t<sdim> res() {
      return muGrid::CcoordOps::get_cube<DimS>(nb_grid_pts);
    }
    constexpr static Ccoord_t<sdim> loc() {
      return muGrid::CcoordOps::get_cube<DimS>(Index_t{0});
    }
    FFT_fixture() : engine{DynCcoord_t(res())} {}
    Engine engine;
  };
  template <class Engine, Index_t DimS, Index_t DimM, Index_t NbSubPts,
            Index_t NbGridPts>
  constexpr Index_t FFT_fixture<Engine, DimS, DimM, NbSubPts, NbGridPts>::sdim;
  template <class Engine, Index_t DimS, Index_t DimM, Index_t NbSubPts,
            Index_t NbGridPts>
  constexpr Index_t FFT_fixture<Engine, DimS, DimM, NbSubPts, NbGridPts>::mdim;
  template <class Engine, Index_t DimS, Index_t DimM, Index_t NbSubPts,
            Index_t NbGridPts>
  constexpr Index_t FFT_fixture<Engine, DimS, DimM, NbSubPts,
                                NbGridPts>::nb_sub_pts;

  template<class Engine>
  struct FFT_fixture_python_segfault {
    constexpr static Index_t dim{twoD};
    constexpr static Index_t sdim{twoD};
    constexpr static Index_t mdim{twoD};
    constexpr static Index_t nb_sub_pts{1};
    constexpr static Ccoord_t<sdim> res() { return {6, 4}; }
    constexpr static Ccoord_t<sdim> loc() { return {0, 0}; }
    FFT_fixture_python_segfault() : engine{DynCcoord_t(res())} {}
    Engine engine;
  };
  template<class Engine>
  constexpr Index_t FFT_fixture_python_segfault<Engine>::sdim;
  template<class Engine>
  constexpr Index_t FFT_fixture_python_segfault<Engine>::mdim;
  template<class Engine>
  constexpr Index_t FFT_fixture_python_segfault<Engine>::nb_sub_pts;

  using fixlist = boost::mpl::list<
#ifdef WITH_FFTW
      FFT_fixture<FFTWEngine, oneD, oneD, OneQuadPt, 3>,
      FFT_fixture<FFTWEngine, oneD, twoD, OneQuadPt, 3>,
      FFT_fixture<FFTWEngine, oneD, threeD, OneQuadPt, 3>,
      FFT_fixture<FFTWEngine, twoD, twoD, OneQuadPt, 3>,
      FFT_fixture<FFTWEngine, twoD, twoD, TwoQuadPts, 3>,
      FFT_fixture<FFTWEngine, twoD, threeD, OneQuadPt, 3>,
      FFT_fixture<FFTWEngine, threeD, threeD, OneQuadPt, 3>,
      FFT_fixture<FFTWEngine, twoD, threeD, OneQuadPt, 4>,
      FFT_fixture<FFTWEngine, threeD, threeD, OneQuadPt, 4>,
      FFT_fixture_python_segfault<FFTWEngine>,
#endif
      FFT_fixture<PocketFFTEngine, oneD, oneD, OneQuadPt, 3>,
      FFT_fixture<PocketFFTEngine, oneD, twoD, OneQuadPt, 3>,
      FFT_fixture<PocketFFTEngine, oneD, threeD, OneQuadPt, 3>,
      FFT_fixture<PocketFFTEngine, twoD, twoD, OneQuadPt, 3>,
      FFT_fixture<PocketFFTEngine, twoD, twoD, TwoQuadPts, 3>,
      FFT_fixture<PocketFFTEngine, twoD, threeD, OneQuadPt, 3>,
      FFT_fixture<PocketFFTEngine, threeD, threeD, OneQuadPt, 3>,
      FFT_fixture<PocketFFTEngine, twoD, threeD, OneQuadPt, 4>,
      FFT_fixture<PocketFFTEngine, threeD, threeD, OneQuadPt, 4>,
      FFT_fixture_python_segfault<PocketFFTEngine>
      >;

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
    FC_t fc(Fix::sdim, FC_t::SubPtMap_t{{"test", Fix::nb_sub_pts}});
    auto & input{fc.register_real_field("input", Fix::mdim * Fix::mdim)};
    auto & ref{fc.register_real_field("reference", Fix::mdim * Fix::mdim)};
    auto & result{fc.register_real_field("result", Fix::mdim * Fix::mdim)};
    fc.initialise(Fix::res(), Fix::res(), Fix::loc());

    using map_t = muGrid::MatrixFieldMap<Real, Mapping::Mut, Fix::mdim,
                                         Fix::mdim, IterUnit::SubPt>;
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
                                          Fix::mdim, IterUnit::SubPt>;
    cmap_t complex_map(complex_field);
    Real error{complex_map[0].imag().norm()};
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

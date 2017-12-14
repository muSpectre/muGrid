/**
 * file   test_system_base.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   14 Dec 2017
 *
 * @brief  Tests for the basic system class
 *
 * @section LICENCE
 *
 * Copyright (C) 2017 Till Junge
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


#include <boost/mpl/list.hpp>

#include "tests.hh"
#include "common/common.hh"
#include "system/system_base.hh"
#include "fft/projection_finite_strain_fast.hh"
#include "fft/fftw_engine.hh"


namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(system_base);
  template <Dim_t DimS>
  struct Sizes {
  };
  template<>
  struct Sizes<twoD> {
    constexpr static Ccoord_t<twoD> get_resolution() {
      return Ccoord_t<twoD>{3, 5};}
    constexpr static Rcoord_t<twoD> get_lengths() {
      return Rcoord_t<twoD>{3.4, 5.8};}
  };
  template<>
  struct Sizes<threeD> {
    constexpr static Ccoord_t<threeD> get_resolution() {
      return Ccoord_t<threeD>{3, 5, 7};}
    constexpr static Rcoord_t<threeD> get_lengths() {
      return Rcoord_t<threeD>{3.4, 5.8, 6.7};}
  };

  template <Dim_t DimS, Dim_t DimM>
  struct SystemBaseFixture: SystemBase<DimS, DimM> {
    SystemBaseFixture()
      :fft_engine{Sizes<DimS>::get_resolution(), Sizes<DimS>::get_lengths()},
       proj{fft_engine},
       SystemBase<DimS, DimM>{std::move(proj)}{}

    FFTW_Engine<DimS, DimM> fft_engine;
    ProjectionFiniteStrainFast<DimS, DimM> proj;
  };

  using fixlist = boost::mpl::list<SystemBaseFixture<twoD, twoD>,
                                   SystemBaseFixture<threeD, threeD>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(constructor_test, fix, fixlist, fix) {
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre

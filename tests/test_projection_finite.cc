/**
 * file   test_projection_finite.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   07 Dec 2017
 *
 * @brief  tests for standard finite strain projection operator
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
#include "fft/fftw_engine.hh"
#include "fft/projection_finite_strain.hh"
#include "common/common.hh"
#include "common/field_collection.hh"

namespace muSpectre {

  BOOST_AUTO_TEST_SUITE(projection_finite_strain);


  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS>
  struct Sizes {
  };
  template<>
  struct Sizes<twoD> {
    constexpr static Ccoord_t<twoD> get_value() {
      return Ccoord_t<twoD>{3, 5};}
  };
  template<>
  struct Sizes<threeD> {
    constexpr static Ccoord_t<threeD> get_value() {
      return Ccoord_t<threeD>{3, 5, 7};}
  };
  /* ---------------------------------------------------------------------- */
  template <Dim_t DimS, Dim_t DimM>
  struct ProjectionFixture {
    using Engine = FFTW_Engine<DimS, DimM>;
    using Parent = ProjectionFiniteStrain<DimS, DimM>;
    ProjectionFixture(): engine{Sizes<DimS>::get_value()},
                         parent(engine){}
    Engine engine;
    Parent parent;
  };

  /* ---------------------------------------------------------------------- */
  using fixlist = boost::mpl::list<ProjectionFixture<twoD, twoD>,
                                   ProjectionFixture<threeD, threeD>>;

  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(constructor_test, fix, fixlist, fix) {
    BOOST_CHECK_NO_THROW(fix::parent.initialise(FFT_PlanFlags::estimate));
  }


  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(Ctest_name, type_name, TL, F)
  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre

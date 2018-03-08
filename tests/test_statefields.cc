/**
 * file   test_statefields.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   01 Mar 2018
 *
 * @brief  Test the StateField abstraction and the associated maps
 *
 * @section LICENSE
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

#include "common/field.hh"
#include "common/field_collection.hh"
#include "common/statefield.hh"
#include "common/ccoord_operations.hh"
#include "tests.hh"

#include <boost/mpl/list.hpp>
#include <type_traits>

namespace muSpectre {

  template <Dim_t DimS, Dim_t DimM, bool Global>
  struct SF_Fixture {
    using FC_t = std::conditional_t<Global,
                                    GlobalFieldCollection<DimS>,
                                    LocalFieldCollection<DimS>>;
    using Field_t = TensorField<FC_t, Real, secondOrder, DimM>;
    using ScalField_t = ScalarField<FC_t, Real>;
    constexpr static size_t nb_mem{2};
    constexpr static Dim_t sdim{DimS};
    constexpr static Dim_t mdim{DimM};
    constexpr static bool global{Global};


    SF_Fixture()
      :fc{}, sf("prefix", fc), scalar_f("scalar", fc), self{*this} {}
    FC_t fc;
    StateField<Field_t, nb_mem> sf;
    StateField<ScalField_t, nb_mem> scalar_f;
    SF_Fixture & self;
  };

  using typelist = boost::mpl::list<SF_Fixture<  twoD,   twoD, false>,
                                    SF_Fixture<  twoD, threeD, false>,
                                    SF_Fixture<threeD, threeD, false>,
                                    SF_Fixture<  twoD,   twoD,  true>,
                                    SF_Fixture<  twoD, threeD,  true>,
                                    SF_Fixture<threeD, threeD,  true>>;

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(constructor_test, Fix, typelist, Fix) {
    BOOST_CHECK_EQUAL("prefix", Fix::sf.get_prefix());
  }

  namespace internal {

    template <bool global, class Fixture_t>
    struct init{
      static void run(Fixture_t & fix) {
        constexpr Dim_t dim{std::remove_reference_t<Fixture_t>::sdim};
        fix.fc.initialise(CcoordOps::get_cube<dim>(3));
      }
    };

    template <class Fixture_t>
    struct init<false, Fixture_t>{
      static void run(Fixture_t & fix) {
        constexpr Dim_t dim{std::remove_reference_t<Fixture_t>::sdim};
        CcoordOps::Pixels<dim> pixels(CcoordOps::get_cube<dim>(3));
        for (auto && pix: pixels) {
          fix.fc.add_pixel(pix);
        }
        fix.fc.initialise();
      }
    };

  }  // internal

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_iteration, Fix, typelist, Fix) {
    internal::init<Fix::global, decltype(Fix::self)>::run(Fix::self);

    constexpr Dim_t mdim{Fix::mdim};
    constexpr bool verbose{false};
    using StateFMap = StateFieldMap<
      MatrixFieldMap<typename Fix::FC_t, Real, mdim, mdim>, Fix::nb_mem>;
    StateFMap matrix_map(Fix::sf);

    for (size_t i = 0; i < Fix::nb_mem+1; ++i) {
      for (auto && wrapper: matrix_map) {
        wrapper.current() += (i+1)*wrapper.current().Identity();
        if (verbose) {
          std::cout << "pixel " << wrapper.get_ccoord() << ", memory cycle " << i << std::endl;
          std::cout << wrapper.current() << std::endl;
          std::cout << wrapper.old() << std::endl;
          std::cout << wrapper.template old<2>() << std::endl << std::endl;
        }
      }
      Fix::sf.cycle();
    }


    for (auto && wrapper: matrix_map) {
      auto I{wrapper.current().Identity()};
      Real error{(wrapper.current() - I).norm()};
      BOOST_CHECK_LT(error, tol);

      error = (wrapper.old() - 3*I).norm();
      BOOST_CHECK_LT(error, tol);

      error = (wrapper.template old<2>() - 2* I).norm();
      BOOST_CHECK_LT(error, tol);

    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_default_map, Fix, typelist, Fix) {
    internal::init<Fix::global, decltype(Fix::self)>::run(Fix::self);

    constexpr bool verbose{false};
    auto matrix_map{Fix::sf.get_map()};

    for (size_t i = 0; i < Fix::nb_mem+1; ++i) {
      for (auto && wrapper: matrix_map) {
        wrapper.current() += (i+1)*wrapper.current().Identity();
        if (verbose) {
          std::cout << "pixel " << wrapper.get_ccoord() << ", memory cycle " << i << std::endl;
          std::cout << wrapper.current() << std::endl;
          std::cout << wrapper.old() << std::endl;
          std::cout << wrapper.template old<2>() << std::endl << std::endl;
        }
      }
      Fix::sf.cycle();
    }

    auto matrix_const_map{Fix::sf.get_const_map()};

    for (auto && wrapper: matrix_const_map) {
      auto I{wrapper.current().Identity()};
      Real error{(wrapper.current() - I).norm()};
      BOOST_CHECK_LT(error, tol);

      error = (wrapper.old() - 3*I).norm();
      BOOST_CHECK_LT(error, tol);

      error = (wrapper.template old<2>() - 2* I).norm();
      BOOST_CHECK_LT(error, tol);

    }
  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(test_scalar_map, Fix, typelist, Fix) {
    internal::init<Fix::global, decltype(Fix::self)>::run(Fix::self);

    constexpr bool verbose{false};
    auto scalar_map{Fix::scalar_f.get_map()};

    for (size_t i = 0; i < Fix::nb_mem+1; ++i) {
      for (auto && wrapper: scalar_map) {
        wrapper.current() += (i+1);
        if (verbose) {
          std::cout << "pixel " << wrapper.get_ccoord() << ", memory cycle " << i << std::endl;
          std::cout << wrapper.current() << std::endl;
          std::cout << wrapper.old() << std::endl;
          std::cout << wrapper.template old<2>() << std::endl << std::endl;
        }
      }
      Fix::scalar_f.cycle();
    }

    auto scalar_const_map{Fix::scalar_f.get_const_map()};

    BOOST_CHECK_EQUAL(scalar_const_map[0].current(), scalar_const_map[1].current());

    for (auto wrapper: scalar_const_map) {
      Real error{wrapper.current() - 1};
      BOOST_CHECK_LT(error, tol);

      error = wrapper.old() - 3;
      BOOST_CHECK_LT(error, tol);

      error = wrapper.template old<2>() - 2;
      BOOST_CHECK_LT(error, tol);

    }

  }

}  // muSpectre

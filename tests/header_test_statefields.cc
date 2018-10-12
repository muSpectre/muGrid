/**
 * file   header_test_statefields.cc
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
 * along with µSpectre; see the file COPYING. If not, write to the
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
    constexpr static size_t get_nb_mem() {return nb_mem;}
    constexpr static Dim_t  get_sdim  () {return sdim;}
    constexpr static Dim_t  get_mdim  () {return mdim;}
    constexpr static bool   get_global() {return global;}

    SF_Fixture()
      :fc{},
       sf{make_statefield<StateField<Field_t, nb_mem>>("prefix", fc)},
       scalar_f{make_statefield<StateField<ScalField_t, nb_mem>>("scalar", fc)},
       self{*this} {}
    FC_t fc;
    StateField<Field_t, nb_mem> & sf;
    StateField<ScalField_t, nb_mem>  & scalar_f;
    SF_Fixture & self;
  };

  using typelist = boost::mpl::list<SF_Fixture<  twoD,   twoD, false>,
                                    SF_Fixture<  twoD, threeD, false>,
                                    SF_Fixture<threeD, threeD, false>,
                                    SF_Fixture<  twoD,   twoD,  true>,
                                    SF_Fixture<  twoD, threeD,  true>,
                                    SF_Fixture<threeD, threeD,  true>>;

  BOOST_AUTO_TEST_SUITE(statefield);

  BOOST_AUTO_TEST_CASE(old_values_test) {
    constexpr Dim_t Dim{twoD};
    constexpr size_t NbMem{2};
    constexpr bool verbose{false};
    using FC_t = LocalFieldCollection<Dim>;
    FC_t fc{};
    using Field_t = ScalarField<FC_t, Int>;
    auto & statefield{make_statefield<StateField<Field_t, NbMem>>("name", fc)};
    fc.add_pixel({});
    fc.initialise();
    for (size_t i{0}; i < NbMem+1; ++i) {
      statefield.current().eigen() = i+1;
      if (verbose) {
        std::cout << "current = " << statefield.current().eigen() << std::endl
                  << "old 1   = " << statefield.old().eigen() << std::endl
                  << "old 2   = " << statefield.template old<2>().eigen()
                  << std::endl
                  << "indices = " << statefield.get_indices() << std::endl
                  << std::endl;
      }
      statefield.cycle();
    }
    BOOST_CHECK_EQUAL(statefield.current().eigen()(0), 1);
    BOOST_CHECK_EQUAL(statefield.old().eigen()(0), 3);
    BOOST_CHECK_EQUAL(statefield.template old<2>().eigen()(0), 2);

  }

  BOOST_FIXTURE_TEST_CASE_TEMPLATE(constructor_test, Fix, typelist, Fix) {
    const std::string ref{"prefix"};
    const std::string & fix{Fix::sf.get_prefix()};
    BOOST_CHECK_EQUAL(ref, fix);
  }

  namespace internal {

    template <bool global, class Fixture_t>
    struct init{
      static void run(Fixture_t & fix) {
        constexpr Dim_t dim{std::remove_reference_t<Fixture_t>::sdim};
        fix.fc.initialise(CcoordOps::get_cube<dim>(3),
                          CcoordOps::get_cube<dim>(0));
      }
    };

    template <class Fixture_t>
    struct init<false, Fixture_t>{
      static void run(Fixture_t & fix) {
        constexpr Dim_t dim{std::remove_reference_t<Fixture_t>::sdim};
        CcoordOps::Pixels<dim> pixels(CcoordOps::get_cube<dim>(3),
                                      CcoordOps::get_cube<dim>(0));
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


  /* ---------------------------------------------------------------------- */
  BOOST_FIXTURE_TEST_CASE_TEMPLATE(Polymorphic_access_by_name, Fix, typelist, Fix) {
    internal::init<Fix::global, decltype(Fix::self)>::run(Fix::self);

    //constexpr bool verbose{true};
    auto & tensor_field = Fix::fc.get_statefield("prefix");
    BOOST_CHECK_EQUAL(tensor_field.get_nb_memory(), Fix::get_nb_mem());

    auto & field = Fix::fc.template get_current<Real>("prefix");
    BOOST_CHECK_EQUAL(field.get_nb_components(),
                      ipow(Fix::get_mdim(), secondOrder));
    BOOST_CHECK_THROW(Fix::fc.template get_current<Int>("prefix"), std::runtime_error);
    auto & old_field = Fix::fc.template get_old<Real>("prefix");
    BOOST_CHECK_EQUAL(old_field.get_nb_components(),
                      field.get_nb_components());
    BOOST_CHECK_THROW(Fix::fc.template get_old<Real>("prefix", Fix::get_nb_mem()+1),
                      std::out_of_range);

    auto & statefield{Fix::fc.get_statefield("prefix")};
    auto & typed_statefield{Fix::fc.template get_typed_statefield<Real>("prefix")};
    auto map{ArrayFieldMap
        <decltype(Fix::fc), Real, ipow(Fix::get_mdim(), secondOrder), 1>
        (typed_statefield.get_current_field())};
    for (auto arr: map) {
      arr.setConstant(1);
    }



    Eigen::ArrayXXd field_copy{field.eigen()};
    statefield.cycle();
    auto & alt_old_field{typed_statefield.get_old_field()};

    Real err{(field_copy - alt_old_field.eigen()).matrix().norm()/
        field_copy.matrix().norm()};
    BOOST_CHECK_LT(err, tol);
    if (not(err<tol)) {
      std::cout << field_copy << std::endl
                << std::endl
                << typed_statefield.get_current_field().eigen() << std::endl
                << std::endl
                << typed_statefield.get_old_field(1).eigen() << std::endl
                << std::endl
                << typed_statefield.get_old_field(2).eigen() << std::endl;
    }


  }

  BOOST_AUTO_TEST_SUITE_END();

}  // muSpectre

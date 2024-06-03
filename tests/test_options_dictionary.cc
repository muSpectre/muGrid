/**
 * @file   test_options_dictionary.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   24 Jun 2021
 *
 * @brief  tests for the options dictionary class
 *
 * Copyright © 2021 Till Junge
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

#include "libmugrid/options_dictionary.hh"

namespace muGrid {

  BOOST_AUTO_TEST_SUITE(option_tests)

  BOOST_AUTO_TEST_CASE(runtime_value_conversion) {
    RuntimeValue int_val(2);
    RuntimeValue real_val(2.5);
    Int recovered_int(int_val.get_int());
    Real recovered_real(real_val.get_real());
    BOOST_CHECK_EQUAL(recovered_int, 2);
    BOOST_CHECK_EQUAL(recovered_real, 2.5);

    BOOST_CHECK_THROW(int_val.get_real(), ValueError);
  }

  BOOST_AUTO_TEST_CASE(construction) {
    const Int int_val{5};
    const Int sub_int_val(3);
    const Real real_val{3.5};
    const Eigen::Vector2d mat_val{(Eigen::Vector2d{} << 1.2, 3.4).finished()};
    Dictionary dict_int{"count", int_val};
    Dictionary dict_real{"intensity", real_val};
    Dictionary dict_mat{"direction", mat_val};

    BOOST_CHECK_THROW(dict_int.get_int(), std::runtime_error);
    auto check_val{dict_int["count"].get_int()};
    BOOST_CHECK_EQUAL(int_val, check_val);
    BOOST_CHECK_THROW(dict_real["intensity"].get_int(), std::runtime_error);
    BOOST_CHECK_THROW(dict_mat["direction"].get_int(), std::runtime_error);

    Int converted_int{dict_int["count"].get_int()};
    BOOST_CHECK_EQUAL(converted_int, int_val);

    BOOST_CHECK_THROW(dict_int["count"].get_real(), std::runtime_error);

    BOOST_CHECK_THROW(dict_int["count"].get_matrix(), ValueError);

    dict_real = int_val;

    // adding new members
    dict_int.add("intensity", real_val);
    BOOST_CHECK_EQUAL(dict_int["intensity"].get_real(), real_val);
    // complain if a member gets added twice
    BOOST_CHECK_THROW(dict_int.add("intensity", real_val), KeyError);
    // change a value's type
    dict_int["intensity"] = int_val;
    BOOST_CHECK_EQUAL(int_val, dict_int["intensity"].get_int());

    // nested dictionary
    dict_int.add("subdictionary", Dictionary("sub_int_val", sub_int_val));
    BOOST_CHECK_EQUAL(dict_int["subdictionary"]["sub_int_val"].get_int(),
                      sub_int_val);
  }

  BOOST_AUTO_TEST_CASE(construction_of_empty_dict) {
    const Int int_val{5};
    Dictionary dict{};
    const std::string key{"intenger value"};
    dict.add(key, int_val);
    BOOST_CHECK_EQUAL(dict[key].get_int(), int_val);
  }

  BOOST_AUTO_TEST_CASE(assignment) {
    const Int int_val{5};
    const Eigen::MatrixXd mat_val{Eigen::MatrixXd::Random(2, 3)};
    const Real real_val{3.2};
    Dictionary dict{};
    const std::string int_key{"int"};

    dict.add(int_key, int_val);
    BOOST_CHECK(dict[int_key].get_value_type() == RuntimeValue::ValueType::Int);
    BOOST_CHECK_EQUAL(dict[int_key].get_int(), int_val);

    dict[int_key] = real_val;
    BOOST_CHECK(dict[int_key].get_value_type() ==
                RuntimeValue::ValueType::Real);
    BOOST_CHECK_EQUAL(dict[int_key].get_real(), real_val);

    dict[int_key] = mat_val;
    BOOST_CHECK(dict[int_key].get_value_type() ==
                RuntimeValue::ValueType::Matrix);
    BOOST_CHECK_EQUAL(
        testGoodies::rel_error(dict[int_key].get_matrix(), mat_val), 0.);
  }

  BOOST_AUTO_TEST_CASE(mut_access) {
    Int initial{2};
    Int clone_modification{3};

    const std::string key{"int"};
    Dictionary dict{key, initial};
    Dictionary clone{dict[key]};

    clone = clone_modification;
    BOOST_CHECK_EQUAL(clone.get_int(), clone_modification);
    BOOST_CHECK_EQUAL(dict[key].get_int(), clone_modification);
  }

  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muGrid

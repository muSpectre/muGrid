/**
 * @file   test_units.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   24 Apr 2020
 *
 * @brief  test for the runtime (physical) units checking
 *
 * Copyright © 2020 Till Junge
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

#include "libmugrid/units.hh"
#include "libmugrid/iterators.hh"

#include <map>

namespace muGrid {

  BOOST_AUTO_TEST_SUITE(unit_checks);

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(prime_generation) {
    const auto primes{testGoodies::generate_primes(7, 28)};
    const std::vector<Uint> primes_ref{7, 11, 13, 17, 19, 23};

    BOOST_CHECK_EQUAL(primes_ref.size(), primes.size());

    for (auto && val_ref : akantu::zip(primes, primes_ref)) {
      const auto & val{std::get<0>(val_ref)};
      const auto & ref{std::get<1>(val_ref)};
      BOOST_CHECK_EQUAL(ref, val);
    }
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(gcd_test) {
    const auto primes{testGoodies::generate_primes(10, 100)};

    // check that gcd of two primes is one
    BOOST_CHECK_EQUAL(1, compute_gcd(primes[0], primes[1]));

    // check that the gcd of two primes times a third prime is that third prime
    BOOST_CHECK_EQUAL(
        primes[2], compute_gcd(primes[0] * primes[2], primes[1] * primes[2]));
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(fraction_reduction) {
    testGoodies::RandRange<Int> rand_int{};

    const auto primes{testGoodies::generate_primes(10, 100)};

    // trivial cases, no reduction possible
    // 1) not a fraction
    auto num{primes[0]};
    UnitExponent sub_unit(num);
    BOOST_CHECK_EQUAL(sub_unit.get_numerator(), num);
    // 2) two primes
    auto den{primes[1]};
    sub_unit = UnitExponent(num, den);
    BOOST_CHECK_EQUAL(sub_unit.get_numerator(), num);
    BOOST_CHECK_EQUAL(sub_unit.get_denominator(), den);
    // 3) no common factor
    auto factor1{primes[2]};
    auto factor2{primes[3]};
    sub_unit = UnitExponent(num * factor1, den * factor2);
    BOOST_CHECK_EQUAL(sub_unit.get_numerator(), num * factor1);
    BOOST_CHECK_EQUAL(sub_unit.get_denominator(), den * factor2);
    // 4) denominator is zero
    BOOST_CHECK_THROW(UnitExponent(num, 0), RuntimeError);
    BOOST_CHECK_THROW(UnitExponent(num, 0), UnitError);

    // simple cases, just sign things
    sub_unit = UnitExponent(num, -1);
    BOOST_CHECK_EQUAL(sub_unit.get_numerator(), -num);
    BOOST_CHECK_EQUAL(sub_unit.get_denominator(), 1);

    // reducing cases:
    // 1) standard
    sub_unit = UnitExponent(num * factor1, den * factor1);
    BOOST_CHECK_EQUAL(sub_unit.get_numerator(), num);
    BOOST_CHECK_EQUAL(sub_unit.get_denominator(), den);
    // 2) with sign flip
    sub_unit = UnitExponent(num * factor1, -den * factor1);
    BOOST_CHECK_EQUAL(sub_unit.get_numerator(), -num);
    BOOST_CHECK_EQUAL(sub_unit.get_denominator(), den);
  }

  BOOST_AUTO_TEST_CASE(comparison_operators) {
    UnitExponent a{-2};
    UnitExponent b{1, 2};
    UnitExponent c{2, 3};
    UnitExponent d{1};

    BOOST_CHECK_LT(a, b);
    BOOST_CHECK_LT(a, c);
    BOOST_CHECK_LT(a, d);

    BOOST_CHECK_LT(b, c);
    BOOST_CHECK_LT(b, d);

    BOOST_CHECK_LT(c, d);

    BOOST_CHECK(not(a < a));
    BOOST_CHECK(not(b < a));
    BOOST_CHECK(not(c < a));
    BOOST_CHECK(not(d < a));

    BOOST_CHECK(not(b < b));
    BOOST_CHECK(not(c < b));
    BOOST_CHECK(not(d < b));

    BOOST_CHECK(not(c < c));
    BOOST_CHECK(not(d < c));

    BOOST_CHECK(not(d < d));
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(unit_operations) {
    const Int num1{1};
    const Int num2{2};
    const UnitExponent a{num1};
    const UnitExponent b{num2};

    BOOST_CHECK_EQUAL(a, a + a);
    BOOST_CHECK_EQUAL(a, a - a);
    BOOST_CHECK_EQUAL(b, a * a);
    BOOST_CHECK_EQUAL(a, b / a);

    BOOST_CHECK_THROW(a + b, UnitError);
    BOOST_CHECK_THROW(a + b, UnitError);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(full_operation) {
    const Unit length{Unit::length()};
    const Unit force{Unit::mass() * Unit::length() /
                        (Unit::time() * Unit::time())};
    const Unit stress{force / length / length};

    const Unit volume{length * length * length};

    const Unit energy{force * length};

    BOOST_CHECK_EQUAL(energy, stress * volume);

    BOOST_CHECK_EQUAL(energy.get_temperature(), UnitExponent(0));
    BOOST_CHECK_EQUAL(energy.get_length(), UnitExponent(2));
    BOOST_CHECK_EQUAL(energy.get_mass(), UnitExponent(1));
    BOOST_CHECK_EQUAL(energy.get_time(), UnitExponent(-2));
    std::string energy_ref{"l²·m·t⁻²"};
    std::stringstream energy_ostream{};
    energy_ostream << energy;
    BOOST_CHECK_EQUAL(energy_ostream.str(), energy_ref);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(tag_discrimination) {
    constexpr Int SomeTag{24};
    const Unit l1{Unit::length()};
    const Unit l2{Unit::length(SomeTag)};
    const Unit t1{Unit::time()};
    const Unit t2{Unit::time(SomeTag)};

    BOOST_CHECK_THROW(l1 + l2, UnitError);
    BOOST_CHECK_THROW(l1 * l2, UnitError);
    BOOST_CHECK_THROW(t1 * l2, UnitError);
    BOOST_CHECK_THROW(t2 + l2, UnitError);

    BOOST_CHECK_NO_THROW(l2 * l2);
    BOOST_CHECK_NO_THROW(t2 * l2);
  }

  /* ---------------------------------------------------------------------- */
  BOOST_AUTO_TEST_CASE(hashability_test) {
    std::map<Unit, int> map{};
    map[Unit::length()] = 24;
  }
  BOOST_AUTO_TEST_SUITE_END();

}  // namespace muGrid

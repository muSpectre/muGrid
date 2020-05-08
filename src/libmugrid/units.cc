/**
 * @file   units.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   23 Apr 2020
 *
 * @brief  dynamic units class based on mass, length, time (could be interpreted
 *         as kg, m, s according to the SI). Useful to avoid bugs due to
 *         multiphisics mixing of domains.
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

#include "units.hh"
#include "iterators.hh"

#include <algorithm>
#include <sstream>
#include <cmath>

namespace muGrid {

  /**
   * computes the greatest common divisor of two integer values using the
   * Binary GCD algorithm. Can hopefully soon be replaced by C++17's std::gcd.
   */
  constexpr Int compute_gcd_worker(const Uint & a, const Uint & b) {
    // simple termination cases
    if (a == b) {
      return a;
    } else if (a == 0) {
      return b;
    } else if (b == 0) {
      return a;
    }

    if (~a & 1) {    // a is even
      if (~b & 1) {  // b is also even
        // If a and b are both even, then gcd(a, b) = 2·gcd(a/2, b/2), because 2
        // is a common divisor.
        return compute_gcd_worker(a >> 1, b >> 1) << 1;
      } else {  // b is odd
        // If a is even and b is odd, then gcd(a, b) = gcd(a/2, b), because 2 is
        // not a common divisor
        return compute_gcd_worker(a >> 1, b);
      }
    } else if (~b & 1) {  // b is even, but a isn't
      // Similarly, if a is odd and b is even, then gcd(a, b) = gcd(a, b/2).
      return compute_gcd_worker(a, b >> 1);
    }

    // If a and b are both odd, and a ≥ b, then gcd(a, b) = gcd((a − b)/2, b).
    if (a > b) {
      return compute_gcd_worker((a - b) >> 1, b);
    } else {
      // If both are odd and a < b, then gcd(a, b) = gcd((b − a)/2, a).
      return compute_gcd_worker((b - a) >> 1, a);
    }
  }

  /**
   * computes the greatest common divisor of two integer values using the
   * Binary GCD algorithm. Can hopefully soon be replaced by C++17's std::gcd.
   */
  Int compute_gcd(const Int & a_signed, const Int & b_signed) {
    // store absolute values of a and b
    Uint a(std::abs(a_signed));
    Uint b(std::abs(b_signed));
    return compute_gcd_worker(a, b);
  }

  /* ---------------------------------------------------------------------- */
  UnitError::UnitError(const std::string & what) : Parent{what} {}

  /* ---------------------------------------------------------------------- */
  UnitExponent::UnitExponent(const Int & numerator, const Int & denominator)
      : numerator{numerator}, denominator{denominator} {
    if (this->denominator == 0) {
      throw UnitError("Division by zero");
    }
    this->reduce();
  }

  /* ---------------------------------------------------------------------- */
  bool UnitExponent::operator==(const UnitExponent & other) const {
    return (this->numerator == other.numerator) and
           (this->denominator == other.denominator);
  }

  /* ---------------------------------------------------------------------- */
  bool UnitExponent::operator!=(const UnitExponent & other) const {
    return not this->operator==(other);
  }

  /* ---------------------------------------------------------------------- */
  bool UnitExponent::operator<(const UnitExponent & other) const {
    auto && diff{*this / other};
    return diff.numerator < 0;
  }

  /* ---------------------------------------------------------------------- */
  UnitExponent UnitExponent::operator+(const UnitExponent & other) const {
    if (*this != other) {
      std::stringstream error{};
      error << "Unit clash: you cannot add quantities of x" << *this
            << " to quantities of x" << other;
      throw UnitError(error.str());
    }
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  UnitExponent UnitExponent::operator-(const UnitExponent & other) const {
    if (*this != other) {
      std::stringstream error{};
      error << "Unit clash: you cannot subtract quantities of x" << other
            << " from quantities of x" << *this;
      throw UnitError(error.str());
    }
    return *this;
  }

  /* ---------------------------------------------------------------------- */
  UnitExponent UnitExponent::operator*(const UnitExponent & other) const {
    // need to add the exponents
    auto && num{this->numerator * other.denominator +
                other.numerator * this->denominator};
    auto && den{this->denominator * other.denominator};
    return UnitExponent{num, den};
  }

  /* ---------------------------------------------------------------------- */
  UnitExponent UnitExponent::operator/(const UnitExponent & other) const {
    // need to subtract the exponents
    auto && num{this->numerator * other.denominator -
                other.numerator * this->denominator};
    auto && den{this->denominator * other.denominator};
    return UnitExponent{num, den};
  }

  /* ---------------------------------------------------------------------- */
  const Int & UnitExponent::get_numerator() const { return this->numerator; }

  /* ---------------------------------------------------------------------- */
  const Int & UnitExponent::get_denominator() const {
    return this->denominator;
  }

  /* ---------------------------------------------------------------------- */
  void UnitExponent::reduce() {
    auto && is_negative{std::signbit(this->denominator)};
    auto && gcd{(is_negative ? -1 : 1) *
                compute_gcd(this->numerator, this->denominator)};
    this->numerator /= gcd;
    this->denominator /= gcd;
  }

  std::string superscript(const int & val) {
    std::stringstream normal{};
    normal << val;
    std::stringstream super{};
    for (auto && ch : normal.str()) {
      switch (ch) {
      case '0': {
        super << "⁰";
        break;
      }
      case '1': {
        super << "¹";
        break;
      }
      case '2': {
        super << "²";
        break;
      }
      case '3': {
        super << "³";
        break;
      }
      case '4': {
        super << "⁴";
        break;
      }
      case '5': {
        super << "⁵";
        break;
      }
      case '6': {
        super << "⁶";
        break;
      }
      case '7': {
        super << "⁷";
        break;
      }
      case '8': {
        super << "⁸";
        break;
      }
      case '9': {
        super << "⁹";
        break;
      }
      case '-': {
        super << "⁻";
        break;
      }
      default:
        throw RuntimeError("unknown character");
        break;
      }
    }
    return super.str();
  }
  /* ---------------------------------------------------------------------- */
  std::ostream & operator<<(std::ostream & os, const UnitExponent & unit) {
    if (unit.get_denominator() == 1) {
      if (unit.numerator != 1) {
        os << superscript(unit.numerator);
      }
    } else {
      os << "^(" << unit.numerator << "/" << unit.denominator << ")";
    }
    return os;
  }

  /* ---------------------------------------------------------------------- */
  Unit::Unit(const UnitExponent & length, const UnitExponent & mass,
             const UnitExponent & time, const UnitExponent & temperature,
             const UnitExponent & current,
             const UnitExponent & luminous_intensity,
             const UnitExponent & amount, const Int & tag)
      : units{length, mass, time, temperature, current, luminous_intensity,
              amount},
        tag{tag} {}

  /* ---------------------------------------------------------------------- */
  Unit Unit::unitless(const Int & new_tag) {
    Unit tmp{new_tag};
    return tmp;
  }

  /* ---------------------------------------------------------------------- */
  Unit Unit::length(const Int & new_tag) {
    Unit tmp{new_tag};
    tmp.get_length() = UnitExponent(1);
    return tmp;
  }

  /* ---------------------------------------------------------------------- */
  Unit Unit::mass(const Int & new_tag) {
    Unit tmp{new_tag};
    tmp.get_mass() = UnitExponent(1);
    return tmp;
  }

  /* ---------------------------------------------------------------------- */
  Unit Unit::time(const Int & new_tag) {
    Unit tmp{new_tag};
    tmp.get_time() = UnitExponent(1);
    return tmp;
  }

  /* ---------------------------------------------------------------------- */
  Unit Unit::temperature(const Int & new_tag) {
    Unit tmp{new_tag};
    tmp.get_temperature() = UnitExponent(1);
    return tmp;
  }

  /* ---------------------------------------------------------------------- */
  Unit Unit::current(const Int & new_tag) {
    Unit tmp{new_tag};
    tmp.get_current() = UnitExponent(1);
    return tmp;
  }

  /* ---------------------------------------------------------------------- */
  Unit Unit::luminous_intensity(const Int & new_tag) {
    Unit tmp{new_tag};
    tmp.get_luminous_intensity() = UnitExponent(1);
    return tmp;
  }

  /* ---------------------------------------------------------------------- */
  Unit Unit::amount(const Int & new_tag) {
    Unit tmp{new_tag};
    tmp.get_amount() = UnitExponent(1);
    return tmp;
  }

  /* ---------------------------------------------------------------------- */
  const UnitExponent & Unit::get_length() const { return this->units[0]; }
  /* ---------------------------------------------------------------------- */
  const UnitExponent & Unit::get_mass() const { return this->units[1]; }
  /* ---------------------------------------------------------------------- */
  const UnitExponent & Unit::get_time() const { return this->units[2]; }
  /* ---------------------------------------------------------------------- */
  const UnitExponent & Unit::get_temperature() const { return this->units[3]; }
  /* ---------------------------------------------------------------------- */
  const UnitExponent & Unit::get_current() const { return this->units[4]; }
  /* ---------------------------------------------------------------------- */
  const UnitExponent & Unit::get_luminous_intensity() const {
    return this->units[5];
  }
  /* ---------------------------------------------------------------------- */
  const UnitExponent & Unit::get_amount() const { return this->units[6]; }

  /* ---------------------------------------------------------------------- */
  UnitExponent & Unit::get_length() { return this->units[0]; }
  /* ---------------------------------------------------------------------- */
  UnitExponent & Unit::get_mass() { return this->units[1]; }
  /* ---------------------------------------------------------------------- */
  UnitExponent & Unit::get_time() { return this->units[2]; }
  /* ---------------------------------------------------------------------- */
  UnitExponent & Unit::get_temperature() { return this->units[3]; }
  /* ---------------------------------------------------------------------- */
  UnitExponent & Unit::get_current() { return this->units[4]; }
  /* ---------------------------------------------------------------------- */
  UnitExponent & Unit::get_luminous_intensity() { return this->units[5]; }
  /* ---------------------------------------------------------------------- */
  UnitExponent & Unit::get_amount() { return this->units[6]; }

  /* ---------------------------------------------------------------------- */
  void Unit::check_tags(const Unit & other) const {
    if (this->tag != other.tag) {
      std::stringstream error_message{};
      error_message << "Mismatched tags! The left-hand side unit '" << *this
                    << "' is tagged " << this->tag
                    << " but the right-hand side unit '" << other
                    << "' is tagged " << other.tag;
      throw UnitError(error_message.str());
    }
  }

  /* ---------------------------------------------------------------------- */
  bool Unit::operator==(const Unit & other) const {
    return this->tag == other.tag and this->units == other.units;
  }

  /* ---------------------------------------------------------------------- */
  bool Unit::operator<(const Unit & other) const {
    if (this->tag != other.tag) {
      return this->tag < other.tag;
    } else {
      return this->units < other.units;
    }
  }

  /* ---------------------------------------------------------------------- */
  Unit Unit::operator+(const Unit & other) const {
    this->check_tags(other);
    Unit tmp{this->tag};
    try {
      for (int i{0}; i < NbUnits; ++i) {
        tmp.units[i] = this->units[i] + other.units[i];
      }
    } catch (const UnitError & error) {
      std::stringstream error_message{};
      error_message << "Unit clash: you cannot add quantities of " << *this
                    << " to quantities of " << other;
      throw UnitError(error_message.str());
    }
    return tmp;
  }

  /* ---------------------------------------------------------------------- */
  Unit Unit::operator-(const Unit & other) const {
    this->check_tags(other);
    Unit tmp{this->tag};
    try {
      for (int i{0}; i < NbUnits; ++i) {
        tmp.units[i] = this->units[i] - other.units[i];
      }
    } catch (const UnitError & error) {
      std::stringstream error_message{};
      error_message << "Unit clash: you cannot subtract quantities of " << other
                    << " from quantities of " << *this;
      throw UnitError(error_message.str());
    }
    return tmp;
  }

  /* ---------------------------------------------------------------------- */
  Unit Unit::operator*(const Unit & other) const {
    this->check_tags(other);
    Unit tmp{this->tag};
    for (int i{0}; i < NbUnits; ++i) {
      tmp.units[i] = this->units[i] * other.units[i];
    }
    return tmp;
  }

  /* ---------------------------------------------------------------------- */
  Unit Unit::operator/(const Unit & other) const {
    this->check_tags(other);
    Unit tmp{this->tag};
    for (int i{0}; i < NbUnits; ++i) {
      tmp.units[i] = this->units[i] / other.units[i];
    }
    return tmp;
  }

  /* ---------------------------------------------------------------------- */
  Unit::Unit(const Int & tag)
      : units{[]() {
          std::array<UnitExponent, NbUnits> tmp{};
          tmp.fill(UnitExponent(0));
          return tmp;
        }()},
        tag{tag} {}

  /* ---------------------------------------------------------------------- */
  std::ostream & operator<<(std::ostream & os, const Unit & unit) {
    static const std::vector<std::string> names{"l", "m",  "t",  "T",
                                                "I", "Iᵥ", "mol"};
    bool first{true};
    for (auto && name_val : akantu::zip(names, unit.units)) {
      const auto & name{std::get<0>(name_val)};
      const auto & val{std::get<1>(name_val)};
      if (val.get_numerator() != 0) {
        if (first) {
          first = false;
        } else {
          os << "·";
        }
        os << name << val;
      }
    }
    if (unit.tag) {
      os << ", tag(" << unit.tag << ')';
    }
    return os;
  }

}  // namespace muGrid

/**
 * @file   units.hh
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

#ifndef SRC_LIBMUGRID_UNITS_HH_
#define SRC_LIBMUGRID_UNITS_HH_

#include "grid_common.hh"
#include "exception.hh"

#include <array>
#include <iostream>

namespace muGrid {
  /**
   * computes the greatest common divisor of two integer values using the
   * Binary GCD algorithm. Can hopefully soon be replaced by C++17's std::gcd.
   */
  Int compute_gcd(const Int & a_signed, const Int & b_signed);

  /* ---------------------------------------------------------------------- */
  class UnitError : public RuntimeError {
   public:
    using Parent = RuntimeError;
    //! Default constructor
    UnitError() = delete;

    //! Constructor with message
    explicit UnitError(const std::string & what);

    //! Copy constructor
    UnitError(const UnitError & other) = default;

    //! Move constructor
    UnitError(UnitError && other) = default;

    //! Destructor
    virtual ~UnitError() = default;

    //! Copy assignment operator
    UnitError & operator=(const UnitError & other) = default;

    //! Move assignment operator
    UnitError & operator=(UnitError && other) = default;
  };

  /**
   * Run-time representation of a rational exponent for base units. This can be
   * used to compose any unit. e.g., speed would be  length per time, i.e.
   * length¹ · time⁻¹. Note that the rational exponent allows to express more
   * exotic units such as for instance the 1/√length in fracture toughness
   * (length^(⁻¹/₂)). The rational exponent is always reduced to the simplest
   * form with a positive denominator. A zero denominator results in a UnitError
   * being thrown.
   */
  class UnitExponent {
   public:
    //! default constructor (needed for default initialisation of std::array)
    UnitExponent() = default;

    //! constructor
    explicit UnitExponent(const Int & numerator, const Int & denominator = 1);

    //! Copy constructor
    UnitExponent(const UnitExponent & other) = default;

    //! Move constructor
    UnitExponent(UnitExponent && other) = default;

    //! Destructor
    virtual ~UnitExponent() = default;

    //! Copy assignment operator
    UnitExponent & operator=(const UnitExponent & other) = default;

    //! Move assignment operator
    UnitExponent & operator=(UnitExponent && other) = default;

    //! comparison operator
    bool operator==(const UnitExponent & other) const;

    //! comparison operator
    bool operator!=(const UnitExponent & other) const;

    //! comparison operator (required for use as key in std::map)
    bool operator<(const UnitExponent & other) const;

    //! addition
    UnitExponent operator+(const UnitExponent & other) const;

    //! subtraction
    UnitExponent operator-(const UnitExponent & other) const;

    //! multiplication (exponent addition)
    UnitExponent operator*(const UnitExponent & other) const;

    //! division (exponent subtraction)
    UnitExponent operator/(const UnitExponent & other) const;

    friend std::ostream & operator<<(std::ostream &, const UnitExponent & unit);

    const Int & get_numerator() const;
    const Int & get_denominator() const;

   protected:
    void reduce();
    Int numerator{};
    Int denominator{};
  };

  /* ---------------------------------------------------------------------- */
  std::ostream & operator<<(std::ostream & os, const UnitExponent & unit);

  /**
   * A wrapper class around 5 DynBaseUnit objects representing length, mass,
   * time, temperature, and change. This should be sufficient to handle all of
   * µSpectre's needs, but could easily be extended by mol and luminous
   * intensity to represent the full SI.
   */
  class Unit {
   public:
    /**
     * constructor from subunits ond optional integer tag can be used to handle
     * cases of additional incompatibilities, e.g., for species. Imagine a
     * coupled diffusion calculation, where one constitutive law handles a
     * concentration of mol·Na/l, and another one a concentration of mol·Ka/l.
     * Both have the same SI unit, but are not compatible. Use different tags
     * for units that need to be mutually incompatible for whatever reason
     */
    Unit(const UnitExponent & length, const UnitExponent & mass,
         const UnitExponent & time, const UnitExponent & temperature,
         const UnitExponent & current, const UnitExponent & luminous_intensity,
         const UnitExponent & amount, const Int & tag = 0);

    //! factory function for base unit length
    static Unit unitless(const Int & tag = 0);

    //! factory function for base unit length
    static Unit length(const Int & tag = 0);

    //! factory function for base unit mass
    static Unit mass(const Int & tag = 0);

    //! factory function for base unit time
    static Unit time(const Int & tag = 0);

    //! factory function for base unit temperature
    static Unit temperature(const Int & tag = 0);

    //! factory function for base unit current
    static Unit current(const Int & tag = 0);

    //! factory function for base unit luminous intensity
    static Unit luminous_intensity(const Int & tag = 0);

    //! factory function for base unit amount of matter
    static Unit amount(const Int & tag = 0);

    //! returns a const reference to length
    const UnitExponent & get_length() const;

    //! returns a const reference to mass
    const UnitExponent & get_mass() const;

    //! returns a const reference to time
    const UnitExponent & get_time() const;

    //! returns a const reference to temperature
    const UnitExponent & get_temperature() const;

    //! returns a const reference to current
    const UnitExponent & get_current() const;

    //! returns a const reference to luminous intensity
    const UnitExponent & get_luminous_intensity() const;

    //! returns a const reference to amount of matter
    const UnitExponent & get_amount() const;

    //! Copy constructor
    Unit(const Unit & other) = default;

    //! Move constructor
    Unit(Unit && other) = default;

    //! Destructor
    virtual ~Unit() = default;

    //! Copy assignment operator
    Unit & operator=(const Unit & other) = default;

    //! Move assignment operator
    Unit & operator=(Unit && other) = default;

    //! comparison operator
    bool operator==(const Unit & other) const;

    //! comparison (required for use as key in std::map)
    bool operator<(const Unit & other) const;

    //! addition
    Unit operator+(const Unit & other) const;

    //! subtraction
    Unit operator-(const Unit & other) const;

    //! multiplication
    Unit operator*(const Unit & other) const;

    //! division
    Unit operator/(const Unit & other) const;

    friend std::ostream & operator<<(std::ostream &, const Unit & unit);

   protected:
    //! returns a  reference to length
    UnitExponent & get_length();

    //! returns a  reference to mass
    UnitExponent & get_mass();

    //! returns a  reference to time
    UnitExponent & get_time();

    //! returns a  reference to temperature
    UnitExponent & get_temperature();

    //! returns a  reference to current
    UnitExponent & get_current();

    //! returns a  reference to luminous intensity
    UnitExponent & get_luminous_intensity();

    //! returns a  reference to amount of matter
    UnitExponent & get_amount();

    //! Throws a UnitError if *this and other have mismatched tags
    void check_tags(const Unit & other) const;

    static constexpr int NbUnits{7};

    //! tagged almost-default constructor
    explicit Unit(const Int & tag);

    std::array<UnitExponent, NbUnits> units{};
    Int tag;
  };

  /* ---------------------------------------------------------------------- */
  std::ostream & operator<<(std::ostream & os, const Unit & unit);

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_UNITS_HH_


/**
 * @file   physics_domain.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   03 Jun 2020
 *
 * @brief  Implementation of PhysicsDomain member functions
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

#include "physics_domain.hh"

#include <sstream>

namespace muGrid {
  PhysicsDomain::PhysicsDomain(const Uint & rank, const Unit & input,
                               const Unit & output, const std::string & name)
      : Parent{rank, input, output}, domain_name{name} {
    Unit product{this->input() * this->output()};
    const Unit energy_density{Unit::mass() /
                              (Unit::length() * Unit::time() * Unit::time())};
    if (product != energy_density) {
      std::stringstream message{};
      message << "Unit mismatch: the input units(" << input
              << ") multiplied with output units (" << output
              << ") should result in energy density (" << energy_density
              << "), but they result in (" << product << ").";
      std::cout << "WARNING: " << message.str() << std::endl;
    }
  }

  /* ---------------------------------------------------------------------- */
  PhysicsDomain::PhysicsDomain(const PhysicsDomain & other)
      : Parent{other.rank(), other.input(), other.output()},
        domain_name{other.get_name()} {};

  /* ---------------------------------------------------------------------- */
  bool PhysicsDomain::operator<(const PhysicsDomain & other) const {
    return static_cast<const Parent &>(*this) <
           static_cast<const Parent &>(other);
  }

  /* ---------------------------------------------------------------------- */
  PhysicsDomain PhysicsDomain::mechanics(const Int & tag) {
    return PhysicsDomain{secondOrder, Unit::strain(tag), Unit::stress(tag),
                         "mechanics"};
  }

  /* ---------------------------------------------------------------------- */
  PhysicsDomain PhysicsDomain::heat(const Int & tag) {
    return PhysicsDomain{firstOrder, Unit::temperature(tag) / Unit::length(tag),
                         Unit::mass(tag) / Unit::time(tag) / Unit::time(tag) /
                             Unit::time(tag),
                         "heat"};
  }

  /* ---------------------------------------------------------------------- */
  const std::string & PhysicsDomain::get_name() const {
    return this->domain_name;
  }
  /* ---------------------------------------------------------------------- */
  bool PhysicsDomain::operator==(const PhysicsDomain & other) const {
    return static_cast<const Parent &>(*this) ==
           static_cast<const Parent &>(other);
  }

  /* ---------------------------------------------------------------------- */
  const Uint & PhysicsDomain::rank() const { return std::get<0>(*this); }

  /* ---------------------------------------------------------------------- */
  const Unit & PhysicsDomain::input() const { return std::get<1>(*this); }

  /* ---------------------------------------------------------------------- */
  const Unit & PhysicsDomain::output() const { return std::get<2>(*this); }

  /* ---------------------------------------------------------------------- */
  std::ostream & operator<<(std::ostream & os, const PhysicsDomain & domain) {
    os << "Domain(rank: " << domain.rank() << ", input: " << domain.input()
       << ", output: " << domain.output() << ")";
    return os;
  }

}  // namespace muGrid

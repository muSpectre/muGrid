/**
 * @file   physics_domain.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   03 Jun 2020
 *
 * @brief  Helper class to identify and separate physics domains based on the
 *         rank and physical units of the input and output fields of
 *         constitutive models. I.e., a mechanics domain problem is rank 2
 *         (strain, stress) with units pressure, and unitless.
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

#include "grid_common.hh"
#include "units.hh"

#include <iostream>
#include <tuple>

#ifndef SRC_LIBMUGRID_PHYSICS_DOMAIN_HH_
#define SRC_LIBMUGRID_PHYSICS_DOMAIN_HH_
namespace muGrid {

  /* ---------------------------------------------------------------------- */
  class PhysicsDomain : private std::tuple<Uint, Unit, Unit> {
    using Parent = std::tuple<Uint, Unit, Unit>;

   public:
    //! Deleted default constructor
    PhysicsDomain() = delete;

    //! constructor from rank, input- and output units, with validity check
    PhysicsDomain(const Uint & rank, const Unit & input, const Unit & output,
                  const std::string & name = "");

    //! Copy constructor
    PhysicsDomain(const PhysicsDomain & other);

    //! Move constructor
    PhysicsDomain(PhysicsDomain && other) = default;

    //! Destructor
    virtual ~PhysicsDomain() = default;

    //! Copy assignment operator
    PhysicsDomain & operator=(const PhysicsDomain & other) = default;

    //! Move assignment operator
    PhysicsDomain & operator=(PhysicsDomain && other) = default;

    //! for usage as keys in maps
    bool operator<(const PhysicsDomain & other) const;

    //! factory function for mechanics domain
    static PhysicsDomain mechanics(const Int & tag = 0);

    //! factory function for heat diffusion domain
    static PhysicsDomain heat(const Int & tag = 0);

    //! comparison operator
    bool operator==(const PhysicsDomain & other) const;

    //! return tensorial rank of this physics domain
    const Uint & rank() const;

    //! return units of input variable
    const Unit & input() const;

    //! return units of output variable
    const Unit & output() const;

    const std::string & get_name() const;

   protected:
    friend std::ostream & operator<<(std::ostream &, const PhysicsDomain &);
    std::string domain_name;
  };

  /* ---------------------------------------------------------------------- */
  std::ostream & operator<<(std::ostream & os, const PhysicsDomain & domain);

}  // namespace muGrid
#endif  // SRC_LIBMUGRID_PHYSICS_DOMAIN_HH_

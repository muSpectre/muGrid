/**
 * file   field_helpers.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   30 Aug 2018
 *
 * @brief  helper functions that needed to be sequestered to avoid circular 
 *         inclusions
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

#ifndef FIELD_HELPERS_H
#define FIELD_HELPERS_H

#include <memory>

namespace muSpectre {

  /**
   * Factory function, guarantees that only fields get created that
   * are properly registered and linked to a collection.
   */
  template <class FieldType, class FieldCollection, typename... Args>
  inline FieldType &
  make_field(std::string unique_name,
             FieldCollection & collection,
             Args&&... args) {
    std::unique_ptr<FieldType> ptr{
      new FieldType(unique_name, collection, args...)};
    auto& retref{*ptr};
    collection.register_field(std::move(ptr));
    return retref;
  }



}  // muSpectre

#endif /* FIELD_HELPERS_H */

/**
 * @file   type_descriptor.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   29 Dec 2025
 *
 * @brief  Runtime implementations for TypeDescriptor utilities
 *
 * Copyright © 2024 Lars Pastewka
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

#include "type_descriptor.hh"
#include "exception.hh"

#include <complex>
#include <cstddef>
#include <sstream>

namespace muGrid {

TypeDescriptor typeid_to_descriptor(const std::type_info & type_id) {
#define MUGRID_TD_TYPEID_ROW(tag, type, name) \
    if (type_id == typeid(type)) {            \
        return TypeDescriptor::tag;           \
    }
    MUGRID_SCALAR_TYPES(MUGRID_TD_TYPEID_ROW)
#undef MUGRID_TD_TYPEID_ROW
    std::stringstream err{};
    err << "Unsupported type for TypeDescriptor: " << type_id.name()
        << ". Only Int, Uint, Real, Complex, and Index_t are supported.";
    throw RuntimeError(err.str());
}

const char * type_descriptor_name(TypeDescriptor td) {
    switch (td) {
        case TypeDescriptor::Unknown:
            return "Unknown";
#define MUGRID_TD_NAME_ROW(tag, type, name) \
    case TypeDescriptor::tag:               \
        return name;
        MUGRID_SCALAR_TYPES(MUGRID_TD_NAME_ROW)
#undef MUGRID_TD_NAME_ROW
    }
    return "Invalid";
}

}  // namespace muGrid

/**
 * @file   type_descriptor.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   29 Dec 2024
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
#include <sstream>

namespace muGrid {

TypeDescriptor typeid_to_descriptor(const std::type_info & type_id) {
    if (type_id == typeid(char)) {
        return TypeDescriptor::Char;
    } else if (type_id == typeid(signed char)) {
        return TypeDescriptor::SignedChar;
    } else if (type_id == typeid(unsigned char)) {
        return TypeDescriptor::UnsignedChar;
    } else if (type_id == typeid(short)) {
        return TypeDescriptor::Short;
    } else if (type_id == typeid(unsigned short)) {
        return TypeDescriptor::UnsignedShort;
    } else if (type_id == typeid(int)) {
        return TypeDescriptor::Int;
    } else if (type_id == typeid(unsigned int)) {
        return TypeDescriptor::UnsignedInt;
    } else if (type_id == typeid(long)) {
        return TypeDescriptor::Long;
    } else if (type_id == typeid(unsigned long)) {
        return TypeDescriptor::UnsignedLong;
    } else if (type_id == typeid(long long)) {
        return TypeDescriptor::LongLong;
    } else if (type_id == typeid(unsigned long long)) {
        return TypeDescriptor::UnsignedLongLong;
    } else if (type_id == typeid(float)) {
        return TypeDescriptor::Float;
    } else if (type_id == typeid(double)) {
        return TypeDescriptor::Double;
    } else if (type_id == typeid(std::complex<double>)) {
        return TypeDescriptor::Complex;
    } else {
        std::stringstream err{};
        err << "Unsupported type for TypeDescriptor: " << type_id.name();
        throw RuntimeError(err.str());
    }
}

const char * type_descriptor_name(TypeDescriptor td) {
    switch (td) {
        case TypeDescriptor::Unknown:
            return "Unknown";
        case TypeDescriptor::Char:
            return "char";
        case TypeDescriptor::SignedChar:
            return "signed char";
        case TypeDescriptor::UnsignedChar:
            return "unsigned char";
        case TypeDescriptor::Short:
            return "short";
        case TypeDescriptor::UnsignedShort:
            return "unsigned short";
        case TypeDescriptor::Int:
            return "int";
        case TypeDescriptor::UnsignedInt:
            return "unsigned int";
        case TypeDescriptor::Long:
            return "long";
        case TypeDescriptor::UnsignedLong:
            return "unsigned long";
        case TypeDescriptor::LongLong:
            return "long long";
        case TypeDescriptor::UnsignedLongLong:
            return "unsigned long long";
        case TypeDescriptor::Float:
            return "float";
        case TypeDescriptor::Double:
            return "double";
        case TypeDescriptor::Complex:
            return "complex<double>";
        default:
            return "Invalid";
    }
}

}  // namespace muGrid

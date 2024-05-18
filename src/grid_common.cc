/**
 * @file   grid_common.cc
 *
 * @author Indre Joedicke <indre.joedicke@imtek.uni-freiburg.de>
 *
 * @date   17 Feb 2020
 *
 * @brief  Implementation of grid utilities
 *
 * @section LICENSE
 *
 * Copyright © 2020 Indre Joedicke
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
namespace muGrid {
  /* ---------------------------------------------------------------------- */
  bool operator<(const Verbosity v1, const Verbosity v2) {
    using T = std::underlying_type_t<Verbosity>;
    return static_cast<T>(v1) < static_cast<T>(v2);
  }

  bool operator>(const Verbosity v1, const Verbosity v2) {
    using T = std::underlying_type_t<Verbosity>;
    return static_cast<T>(v1) > static_cast<T>(v2);
  }

  bool operator<=(const Verbosity v1, const Verbosity v2) {
    using T = std::underlying_type_t<Verbosity>;
    return static_cast<T>(v1) <= static_cast<T>(v2);
  }

  bool operator>=(const Verbosity v1, const Verbosity v2) {
    using T = std::underlying_type_t<Verbosity>;
    return static_cast<T>(v1) >= static_cast<T>(v2);
  }

  std::ostream & operator<<(std::ostream & os,
                            const IterUnit & sub_division) {
    switch (sub_division) {
    case IterUnit::Pixel: {
      os << "free number of points";
      break;
    }
    case IterUnit::SubPt: {
      os << "Sub point";
      break;
    }
    default:
      throw RuntimeError("unknown pixel subdivision scheme");
      break;
    }
    return os;
  }

  std::ostream & operator<<(std::ostream & os,
                            const StorageOrder & storage_order) {
    switch (storage_order) {
    case StorageOrder::ColMajor: {
      os << "column-major";
      break;
    }
    case StorageOrder::RowMajor: {
      os << "row-major";
      break;
    }
    case StorageOrder::Unknown: {
      os << "unknown";
      break;
    }
    case StorageOrder::Automatic: {
      os << "automatic";
      break;
    }
    default:
      throw RuntimeError("unknown storage order specification");
      break;
    }
    return os;
  }

}  // namespace muGrid

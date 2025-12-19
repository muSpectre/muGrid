/**
 * @file   kokkos_profiling_workaround.cc
 *
 * @author Lars Pastewka <lars.pastewka@imtek.uni-freiburg.de>
 *
 * @date   19 Dec 2024
 *
 * @brief  Workaround for Kokkos ABI issue with Apple Clang 17
 *
 * On macOS with Apple Clang 17, there's an ABI mismatch where the Kokkos
 * library exports functions with const std::string& parameters, but template
 * instantiations generate calls expecting std::string by value.
 *
 * This file provides wrapper functions that forward from by-value to
 * by-reference versions.
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
 * with proprietary FFT implementations or the TBB library (or a modified
 * version of that library), containing parts covered by the terms of the
 * respective license agreement, the licensors of this Program grant you
 * additional permission to convey the resulting work.
 *
 */

#ifdef __APPLE__

#include <Kokkos_Core.hpp>
#include <impl/Kokkos_Profiling.hpp>
#include <string>

namespace Kokkos {
namespace Tools {

// Wrapper for beginFence: forwards from by-value to by-reference
void beginFence(std::string name, const uint32_t deviceId, uint64_t* handle) {
  // Call the real implementation with const reference
  beginFence(static_cast<const std::string&>(name), deviceId, handle);
}

// Wrapper for beginDeepCopy: forwards from by-value to by-reference
void beginDeepCopy(const SpaceHandle dst_space, std::string dst_label,
                   const void* dst_ptr, const SpaceHandle src_space,
                   std::string src_label, const void* src_ptr,
                   const uint64_t size) {
  // Call the real implementation with const references
  beginDeepCopy(dst_space, static_cast<const std::string&>(dst_label), dst_ptr,
                src_space, static_cast<const std::string&>(src_label), src_ptr,
                size);
}

}  // namespace Tools
}  // namespace Kokkos

#endif  // __APPLE__

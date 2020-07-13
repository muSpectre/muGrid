/**
 * @file   raw_memory_operations.cc
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   29 May 2020
 *
 * @brief  implementation of functions for unsafe raw memory operations. Use
 *         these only when  necessary
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

#include <algorithm>

#include "ccoord_operations.hh"

#include "raw_memory_operations.hh"

using muGrid::CcoordOps::compute_axes_order;

namespace muGrid {
  namespace raw_mem_ops {

    CartesianContainer::CartesianContainer(const Shape_t & shape,
                                           const Shape_t & strides)
        : shape{shape}, axes_order{compute_axes_order(shape, strides)} {}

  }  // namespace raw_mem_ops
}  // namespace muGrid

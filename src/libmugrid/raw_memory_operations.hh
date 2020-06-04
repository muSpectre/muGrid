/**
 * @file   raw_memory_operations.hh
 *
 * @author Till Junge <till.junge@altermail.ch>
 *
 * @date   29 May 2020
 *
 * @brief  functions for unsafe raw memory operations. Use these only when
 *         necessary
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
#include "exception.hh"

#include <vector>

#ifndef SRC_LIBMUGRID_RAW_MEMORY_OPERATIONS_HH_
#define SRC_LIBMUGRID_RAW_MEMORY_OPERATIONS_HH_

namespace muGrid {
  namespace raw_mem_ops {
    void strided_copy(const std::vector<Index_t> & logical_shape,
                      const std::vector<Index_t> & input_strides,
                      const std::vector<Index_t> & output_strides,
                      const void * input_data, void * output_data,
                      const size_t & size);
    /**
     * copies between structured arrays of arbitrary strides. The algorithm
     * isn't smart and assumes that at least one of the arrays has column-major
     * storage
     */
    template <typename T>
    void strided_copy(const std::vector<Index_t> & logical_shape,
                      const std::vector<Index_t> & input_strides,
                      const std::vector<Index_t> & output_strides,
                      const T * input_data, T * output_data) {
      strided_copy(logical_shape, input_strides, output_strides, input_data,
                   output_data, sizeof(T));
    }
  }  // namespace raw_mem_ops
}  // namespace muGrid

#endif  // SRC_LIBMUGRID_RAW_MEMORY_OPERATIONS_HH_

/**
 * @file   pixels.cc
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   01 Oct 2019
 *
 * @brief  pre-compilable pixel operations
 *
 * Copyright © 2019 Till Junge
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
#include <iostream>

#include "core/exception.hh"
#include "grid/pixels.hh"

namespace muGrid {

    namespace CcoordOps {
        size_t get_buffer_size(const DynGridIndex & nb_grid_pts,
                               const DynGridIndex & strides) {
            const Dim_t & dim{nb_grid_pts.get_dim()};
            if (strides.get_dim() != dim) {
                std::stringstream error{};
                error << "Dimension mismatch between nb_grid_pts (= "
                      << nb_grid_pts << ") and strides (= " << strides << ")";
                throw RuntimeError(error.str());
            }
            size_t buffer_size{0};
            // We need to loop over the dimensions because the largest stride
            // can occur anywhere. (It depends on the storage order.)
            for (Dim_t i{0}; i < dim; ++i) {
                buffer_size =
                    std::max(buffer_size,
                             static_cast<size_t>(nb_grid_pts[i] * strides[i]));
            }
            return buffer_size;
        }

        size_t get_buffer_size(const Shape_t & nb_grid_pts,
                               const Shape_t & strides) {
            const size_t & dim{nb_grid_pts.size()};
            if (strides.size() != dim) {
                std::stringstream error{};
                error << "Dimension mismatch between nb_grid_pts (= "
                      << nb_grid_pts << ") and strides (= " << strides << ")";
                throw RuntimeError(error.str());
            }
            size_t buffer_size{0};
            // We need to loop over the dimensions because the largest stride
            // can occur anywhere. (It depends on the storage order.)
            for (size_t i{0}; i < dim; ++i) {
                buffer_size =
                    std::max(buffer_size,
                             static_cast<size_t>(nb_grid_pts[i] * strides[i]));
            }
            return buffer_size;
        }
    }  // namespace CcoordOps

}  // namespace muGrid
